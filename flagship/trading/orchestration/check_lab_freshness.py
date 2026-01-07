"""
AlphaLab parquet 新鲜度检查（bars/signal/dataset）+ 自动修复。

目标：
- 确认 AlphaLab 日线 parquet 已覆盖“最新交易日”（expected_date）
- 确认 daily_signal.parquet 与 live 模型文件与 expected_date 对齐
- （可选）检查 dataset 目录下 parquet 的更新时间

自动修复（--fix）：
- 若 expected_date 当天 daily_selection 缺失：自动跑 run_daily_selection(target_date=expected_date)
- 若 bars 缺失/滞后：自动跑 ensure_data_completeness(target_date=expected_date, lookback_days=N)
- 若 live model 缺失/过旧：自动跑 train_daily_model(target_date=expected_date+1)
- 若 signal 缺失/滞后：自动跑 run_live_inference(target_date=expected_date)

注意：
- expected_date 默认由 SPY 的日线 parquet 的 max(datetime) 推断；若没有 SPY 则回退到 QQQ，再回退到 today-1。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.logger import logger

from flagship.universe.pg_ticker_db import get_pg_connection, get_selected_symbols_in_range
from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter

from flagship.trading.orchestration.run_daily_selection import run_daily_selection
from flagship.trading.orchestration.ensure_data_completeness import check_and_backfill_data
from flagship.trading.orchestration.train_daily_model import train_daily_model
from flagship.trading.orchestration.run_live_inference import run_live_inference


EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FreshnessResult:
    expected_date: date
    selection_count: int
    bars_total: int
    bars_missing: int
    bars_stale: int
    signal_ok: bool
    model_ok: bool
    dataset_ok: bool


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _find_index_file(daily_path: Path, symbol_root: str) -> Path | None:
    matches = sorted(daily_path.glob(f"{symbol_root}.*.parquet"))
    if matches:
        return matches[0]
    return None


def _infer_expected_date_from_lab(lab: AlphaLab) -> date:
    """
    优先从 SPY 的 parquet 推断最新交易日；若不存在则回退到 QQQ；最后回退到 today-1。
    """
    daily_path = lab.daily_path
    for root in ("SPY", "QQQ"):
        file_path = _find_index_file(daily_path, root)
        if file_path and file_path.exists():
            try:
                last_dt = (
                    pl.scan_parquet(file_path)
                    .select(pl.col("datetime").max())
                    .collect()
                    .item()
                )
                if last_dt:
                    return last_dt.date()
            except Exception as exc:
                logger.warning(f"[check_lab_freshness] 推断 expected_date 失败 {file_path}: {exc}")
                continue

    return date.today() - timedelta(days=1)


def _load_daily_selection(trade_date: date) -> list[str]:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT vt_symbol FROM daily_selection WHERE trade_date = %s",
                (trade_date,),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]


def _max_bar_date(file_path: Path) -> date | None:
    last_dt = (
        pl.scan_parquet(file_path)
        .select(pl.col("datetime").max())
        .collect()
        .item()
    )
    if not last_dt:
        return None
    return last_dt.date()


def _check_daily_bars(
    lab: AlphaLab,
    vt_symbols: Iterable[str],
    expected_date: date,
) -> tuple[int, int, int]:
    """
    Returns:
        (total, missing_count, stale_count)
    """
    daily_path = lab.daily_path
    total = 0
    missing = 0
    stale = 0

    for vt_symbol in vt_symbols:
        total += 1
        file_path = daily_path / f"{vt_symbol}.parquet"
        if not file_path.exists():
            # 兜底：用 symbol_root 找唯一匹配
            symbol_root = vt_symbol.split(".")[0]
            candidates = sorted(daily_path.glob(f"{symbol_root}.*.parquet"))
            if len(candidates) == 1:
                file_path = candidates[0]
            else:
                missing += 1
                continue

        try:
            max_dt = _max_bar_date(file_path)
            if max_dt is None or max_dt < expected_date:
                stale += 1
        except Exception:
            stale += 1

    return total, missing, stale


def _check_signal_file(lab_path: Path, expected_date: date) -> bool:
    signal_path = lab_path / "signal" / "daily_signal.parquet"
    if not signal_path.exists():
        return False
    try:
        last_dt = (
            pl.scan_parquet(signal_path)
            .select(pl.col("datetime").max())
            .collect()
            .item()
        )
        return bool(last_dt) and last_dt.date() == expected_date
    except Exception:
        return False


def _check_live_model_file(lab_path: Path, expected_date: date) -> bool:
    model_path = lab_path / "model" / "flagship_alpha_mom_live_lgb.pkl"
    if not model_path.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime).date()
        # 模型更新时间不早于 expected_date（保守检查）
        return mtime >= expected_date
    except Exception:
        return False


def _check_dataset_dir(lab_path: Path, expected_date: date) -> bool:
    dataset_dir = lab_path / "dataset"
    if not dataset_dir.exists():
        # 对 live pipeline 来说不强制
        return True
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        return True
    latest_mtime = max(datetime.fromtimestamp(p.stat().st_mtime) for p in parquet_files)
    return latest_mtime.date() >= expected_date


def _emit_freshness_metrics(result: FreshnessResult, *, ok: bool) -> None:
    """
    Emit DoD-style freshness metrics using Prometheus textfile collector.

    Design:
    - Keep metrics cardinality stable (no per-date labels); export expected_date as a timestamp gauge.
    - Safe best-effort: never raise.
    """
    try:
        writer = TextfileMetricsWriter("flagship_lab_freshness.prom")
        now_ts = float(time.time())
        expected_ts = float(datetime.combine(result.expected_date, datetime.min.time(), tzinfo=EASTERN).timestamp())

        samples = [
            Sample("flagship_lab_freshness_last_update_timestamp_seconds", now_ts),
            Sample("flagship_lab_expected_date_timestamp_seconds", expected_ts),
            Sample("flagship_lab_freshness_ok", 1.0 if ok else 0.0),
            Sample("flagship_lab_selection_count", float(result.selection_count)),
            Sample("flagship_lab_bars_total", float(result.bars_total)),
            Sample("flagship_lab_bars_missing", float(result.bars_missing)),
            Sample("flagship_lab_bars_stale", float(result.bars_stale)),
            Sample("flagship_lab_signal_ok", 1.0 if result.signal_ok else 0.0),
            Sample("flagship_lab_model_ok", 1.0 if result.model_ok else 0.0),
            Sample("flagship_lab_dataset_ok", 1.0 if result.dataset_ok else 0.0),
        ]

        writer.write(
            samples=samples,
            help_map={
                "flagship_lab_freshness_last_update_timestamp_seconds": "Last freshness check update timestamp (epoch seconds).",
                "flagship_lab_expected_date_timestamp_seconds": "Expected DATA_DATE (previous close) midnight timestamp in America/New_York.",
                "flagship_lab_freshness_ok": "Overall freshness OK (DoD).",
                "flagship_lab_selection_count": "daily_selection count for expected_date.",
                "flagship_lab_bars_total": "Total symbols checked (U_t).",
                "flagship_lab_bars_missing": "Missing daily bar parquet count (U_t).",
                "flagship_lab_bars_stale": "Stale daily bar parquet count (max(date) < expected_date) (U_t).",
                "flagship_lab_signal_ok": "Signal parquet exists and matches expected_date.",
                "flagship_lab_model_ok": "Live model file exists and is not older than expected_date.",
                "flagship_lab_dataset_ok": "Dataset directory freshness check (best-effort).",
            },
            type_map={
                "flagship_lab_freshness_last_update_timestamp_seconds": "gauge",
                "flagship_lab_expected_date_timestamp_seconds": "gauge",
                "flagship_lab_freshness_ok": "gauge",
                "flagship_lab_selection_count": "gauge",
                "flagship_lab_bars_total": "gauge",
                "flagship_lab_bars_missing": "gauge",
                "flagship_lab_bars_stale": "gauge",
                "flagship_lab_signal_ok": "gauge",
                "flagship_lab_model_ok": "gauge",
                "flagship_lab_dataset_ok": "gauge",
            },
        )
    except Exception as exc:
        logger.warning(f"[check_lab_freshness] emit metrics failed: {exc}")


def check_lab_freshness(
    lab_path: Path,
    expected_date: date | None = None,
    train_lookback_days: int = 180,
    fix: bool = True,
    check_model: bool = True,
    check_signals: bool = True,
    check_datasets: bool = True,
) -> FreshnessResult:
    lab = AlphaLab(str(lab_path))

    if expected_date is None:
        expected_date = _infer_expected_date_from_lab(lab)

    logger.info(f"[check_lab_freshness] expected_date={expected_date}")

    # 1) 确保 daily_selection 存在（U_t）
    ut = _load_daily_selection(expected_date)
    if not ut and fix:
        logger.warning(f"[check_lab_freshness] daily_selection 缺失: {expected_date}，尝试自动生成")
        run_daily_selection(target_date=expected_date, lab_path=lab_path)
        ut = _load_daily_selection(expected_date)

    # 2) 训练窗口 universe（U_train）：用于覆盖训练所需静态 universe
    valid_end = expected_date
    train_start = valid_end - timedelta(days=train_lookback_days)
    u_train = get_selected_symbols_in_range(start_date=train_start, end_date=valid_end)

    # NOTE:
    # - 对“日常交易流程”而言，最关键的是 U_t（当日选股池）对应的 bars/signal/model 对齐 expected_date。
    # - U_train 是训练窗口的静态 universe，里面可能包含停牌/退市/无成交的标的，导致 max(date) < expected_date；
    #   这种情况不应阻塞当天交易（只会减少训练样本），因此我们对 U_train 的 stale 只做 warning，不作为失败条件。
    symbols_to_check = sorted(set(ut))

    # 如果没有任何 symbol，说明上游选股/数据完全断了
    if not symbols_to_check:
        if fix:
            logger.error("[check_lab_freshness] 无法获得任何需要检查的 vt_symbol（U_t 与 U_train 均为空）")
        raise RuntimeError("No vt_symbols to check (U_t and U_train are empty).")

    # U_t bars（强制要求对齐）
    total, missing, stale = _check_daily_bars(lab, symbols_to_check, expected_date)

    # U_train bars（仅提示，不阻塞）
    train_total = 0
    train_missing = 0
    train_stale = 0
    if u_train:
        train_total, train_missing, train_stale = _check_daily_bars(lab, u_train, expected_date)

    # 3) bars 自动修复：用 ensure_data_completeness 对 U_t 做补齐（最关键）
    if fix and (missing > 0 or stale > 0):
        logger.warning(
            f"[check_lab_freshness] bars 缺失/滞后 (missing={missing}, stale={stale})，尝试自动补齐 U_t"
        )
        # ensure_data_completeness 依赖 U_t（daily_selection）
        check_and_backfill_data(target_date=expected_date, lab_path=lab_path, lookback_days=train_lookback_days)
        total, missing, stale = _check_daily_bars(lab, symbols_to_check, expected_date)

    if train_total > 0 and (train_missing > 0 or train_stale > 0):
        logger.warning(
            f"[check_lab_freshness] U_train bars 存在缺失/滞后："
            f"train_total={train_total}, train_missing={train_missing}, train_stale={train_stale} "
            f"(不会阻塞当天交易，但可能降低训练样本数)"
        )

    # 4) model/signal/dataset 检查与修复
    model_ok = True
    if check_model:
        model_ok = _check_live_model_file(lab_path, expected_date)
        if fix and not model_ok:
            logger.warning("[check_lab_freshness] live model 缺失/过旧，尝试训练")
            # 传入 expected_date+1，保证 valid_end 覆盖到 expected_date（周末也能工作）
            train_daily_model(
                target_date=expected_date + timedelta(days=1),
                lab_path=lab_path,
                output_model_path=lab_path / "model" / "flagship_alpha_mom_live_lgb.pkl",
            )
            model_ok = _check_live_model_file(lab_path, expected_date)

    signal_ok = True
    if check_signals:
        signal_ok = _check_signal_file(lab_path, expected_date)
        if fix and not signal_ok:
            logger.warning("[check_lab_freshness] signal 缺失/过旧，尝试重新推理生成")
            run_live_inference(
                target_date=expected_date,
                lab_path=lab_path,
                model_path=lab_path / "model" / "flagship_alpha_mom_live_lgb.pkl",
                output_file=lab_path / "signal" / "daily_signal.parquet",
            )
            signal_ok = _check_signal_file(lab_path, expected_date)

    dataset_ok = True
    if check_datasets:
        dataset_ok = _check_dataset_dir(lab_path, expected_date)

    result = FreshnessResult(
        expected_date=expected_date,
        selection_count=len(ut),
        bars_total=total,
        bars_missing=missing,
        bars_stale=stale,
        signal_ok=signal_ok,
        model_ok=model_ok,
        dataset_ok=dataset_ok,
    )

    logger.info(
        f"[check_lab_freshness] selection={result.selection_count}, "
        f"bars(selection_total={result.bars_total}, missing={result.bars_missing}, stale={result.bars_stale}), "
        f"train_universe={len(u_train)}, train_bars(missing={train_missing}, stale={train_stale}), "
        f"model_ok={result.model_ok}, signal_ok={result.signal_ok}, dataset_ok={result.dataset_ok}"
    )

    ok = (
        result.selection_count > 0
        and result.bars_missing == 0
        and result.bars_stale == 0
        and (result.signal_ok)
        and (result.model_ok)
        and (result.dataset_ok)
    )
    _emit_freshness_metrics(result, ok=ok)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AlphaLab parquet freshness and auto-fix if needed.")
    parser.add_argument("--lab-path", type=str, default="lab/flagship_alpha_momentum")
    parser.add_argument("--expected-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--train-lookback-days", type=int, default=180)
    parser.add_argument("--check-only", action="store_true", help="Only check, do not auto-fix")
    parser.add_argument("--no-check-model", action="store_true", help="Skip live model freshness check (bars-only mode).")
    parser.add_argument("--no-check-signals", action="store_true")
    parser.add_argument("--no-check-datasets", action="store_true")
    args = parser.parse_args()

    lab_path = Path(args.lab_path)
    if not lab_path.is_absolute():
        lab_path = Path(__file__).resolve().parents[3] / lab_path

    expected_date = _parse_date(args.expected_date) if args.expected_date else None
    fix = not args.check_only

    result = check_lab_freshness(
        lab_path=lab_path,
        expected_date=expected_date,
        train_lookback_days=args.train_lookback_days,
        fix=fix,
        check_model=not args.no_check_model,
        check_signals=not args.no_check_signals,
        check_datasets=not args.no_check_datasets,
    )

    # 若仍不满足 freshness，返回非 0，便于 cron 报警
    ok = (
        result.selection_count > 0
        and result.bars_missing == 0
        and result.bars_stale == 0
        and (result.signal_ok)
        and (result.model_ok)
        and (result.dataset_ok)
    )

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()


