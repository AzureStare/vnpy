"""
增量更新 AlphaLab 的日线/分钟线 Parquet（Polygon REST）。

需求：
- 将日线、分钟线数据增量更新到指定 end_date（例如 2025-12-18）
- 只下载“缺失的时间段”，避免全量重下

说明：
- AlphaLab.save_bar_data 会自动与现有 parquet 合并、去重、排序，因此允许 1~2 天 overlap。
- 分钟线单标的 parquet 可能很大，建议只对策略 universe（如 daily_selection）做增量更新。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal

import polars as pl

import sys

# 动态注入项目根路径（确保 `import flagship` 可用）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData
from vnpy.trader.utility import extract_vt_symbol, ZoneInfo

from flagship.config import create_polygon_client, DEFAULT_LAB_DIR
from flagship.scripts.pg_ticker_db import get_pg_connection, get_ref_tickers


UniverseMode = Literal["daily_selection", "ref_tickers_cs", "lab_existing", "vt_symbols"]


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _max_datetime_from_parquet(file_path: Path) -> datetime | None:
    if not file_path.exists():
        return None
    try:
        max_dt = (
            pl.scan_parquet(file_path)
            .select(pl.col("datetime").max())
            .collect()
            .item()
        )
        return max_dt
    except Exception as exc:
        logger.warning(f"[incremental_update] 读取 parquet max(datetime) 失败 {file_path}: {exc}")
        return None


def _is_parquet_schema_conflict_error(exc: Exception) -> bool:
    """
    AlphaLab.save_bar_data 内部会读旧 parquet 并 concat 新数据。
    当历史 parquet 的 dtype 不一致（例如 open/high/low/close 被写成 Int64）时，polars concat 会报 schema 冲突。
    """
    msg = str(exc)
    return "is incompatible with expected type" in msg or "SchemaError" in msg


def _repair_parquet_schema_inplace(file_path: Path) -> bool:
    """
    修复历史 parquet 的列类型（就地覆写）。

    目标 schema（尽量与 BarData 写入保持一致）：
    - datetime: Datetime（保持不动）
    - open/high/low/close/turnover: Float64
    - volume: Int64
    """
    if not file_path.exists():
        return False

    try:
        df = pl.read_parquet(file_path)
        if df.is_empty():
            return False

        exprs: list[pl.Expr] = []
        for col in ("open", "high", "low", "close", "turnover"):
            if col in df.columns:
                exprs.append(pl.col(col).cast(pl.Float64).alias(col))
        if "volume" in df.columns:
            exprs.append(pl.col("volume").cast(pl.Int64).alias("volume"))

        if not exprs:
            return False

        fixed = df.with_columns(exprs)
        # 写回原文件（保持简单；AlphaLab 会按 datetime 去重）
        fixed.write_parquet(file_path)
        return True
    except Exception as fix_exc:
        logger.warning(f"[incremental_update] schema 修复失败 {file_path}: {fix_exc}")
        return False


def _load_daily_selection_symbols(start_date: date, end_date: date) -> list[str]:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT vt_symbol
                FROM daily_selection
                WHERE trade_date >= %s AND trade_date <= %s
                ORDER BY vt_symbol
                """,
                (start_date, end_date),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]


def _load_ref_tickers_cs_vt_symbols() -> list[str]:
    """
    从 ref_tickers 读取 active common stock，并映射 primary_exchange 到 vnpy exchange name。
    """
    # 与 build_daily_selection_to_postgres.py 保持一致的映射
    exchange_map = {
        "XNAS": "NASDAQ",
        "NASDAQ": "NASDAQ",
        "XNYS": "NYSE",
        "NYSE": "NYSE",
        "XASE": "AMEX",
        "AMEX": "AMEX",
        "BATS": "BATS",
        "IEXG": "IEX",
    }

    tickers = get_ref_tickers(
        market="stocks",
        locale="us",
        ticker_type="CS",
        active=True,
    )
    vt_symbols: list[str] = []
    for t in tickers:
        symbol = t.get("symbol")
        if not symbol:
            continue
        primary_exchange = t.get("primary_exchange", "")
        exchange = exchange_map.get(primary_exchange, "NASDAQ")
        vt_symbols.append(f"{symbol}.{exchange}")
    return sorted(set(vt_symbols))


def _load_lab_existing_vt_symbols(lab: AlphaLab, interval: Interval) -> list[str]:
    folder = lab.daily_path if interval == Interval.DAILY else lab.minute_path
    return sorted(p.stem for p in folder.glob("*.parquet"))


def resolve_universe(
    mode: UniverseMode,
    lab: AlphaLab,
    interval: Interval,
    selection_start: date | None,
    selection_end: date | None,
    vt_symbols: list[str] | None,
) -> list[str]:
    if mode == "vt_symbols":
        if not vt_symbols:
            raise ValueError("mode=vt_symbols requires --vt-symbols")
        return vt_symbols

    if mode == "lab_existing":
        return _load_lab_existing_vt_symbols(lab, interval)

    if mode == "ref_tickers_cs":
        return _load_ref_tickers_cs_vt_symbols()

    if mode == "daily_selection":
        if selection_end is None:
            raise ValueError("mode=daily_selection requires --selection-end (YYYY-MM-DD)")
        if selection_start is None:
            selection_start = selection_end
        return _load_daily_selection_symbols(selection_start, selection_end)

    raise ValueError(f"Unsupported universe mode: {mode}")


def _download_symbol_bars(
    client,
    vt_symbol: str,
    interval: Interval,
    start_date: date,
    end_date: date,
) -> list[BarData]:
    symbol, exchange = extract_vt_symbol(vt_symbol)

    if interval == Interval.DAILY:
        timespan = "day"
        multiplier = 1
    elif interval == Interval.MINUTE:
        timespan = "minute"
        multiplier = 1
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    aggs = client.get_aggs(
        ticker=symbol,
        multiplier=multiplier,
        timespan=timespan,
        from_=start_date.isoformat(),
        to=end_date.isoformat(),
        adjusted=True,
        sort="asc",
        limit=50000,
    )

    if not aggs:
        return []

    utc_tz = timezone.utc
    eastern_tz = ZoneInfo("America/New_York")

    bars: list[BarData] = []
    for agg in aggs:
        utc_datetime = datetime.fromtimestamp(agg.timestamp / 1000, tz=utc_tz)
        bar_datetime = utc_datetime.astimezone(eastern_tz).replace(tzinfo=None)

        # AlphaLab 既有 parquet 中 volume 多为 Int64（历史原因），
        # 为避免 polars concat schema 冲突，这里强制 volume 转为 int。
        volume_int = int(agg.volume) if getattr(agg, "volume", None) is not None else 0
        # NOTE:
        # Polygon SDK 在极少数情况下会返回 int 类型的 OHLC（例如价格刚好是整数），
        # 导致新写入的 DataFrame 推断为 Int64，与历史 parquet 的 Float64 冲突。
        open_val = float(agg.open) if getattr(agg, "open", None) is not None else 0.0
        high_val = float(agg.high) if getattr(agg, "high", None) is not None else 0.0
        low_val = float(agg.low) if getattr(agg, "low", None) is not None else 0.0
        close_val = float(agg.close) if getattr(agg, "close", None) is not None else 0.0

        bars.append(
            BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=bar_datetime,
                interval=interval,
                open_price=open_val,
                high_price=high_val,
                low_price=low_val,
                close_price=close_val,
                volume=volume_int,
                turnover=float(volume_int) * close_val,
                open_interest=0,
                gateway_name="POLYGON",
            )
        )

    return bars


@dataclass(frozen=True)
class UpdateStats:
    total: int
    updated: int
    skipped: int
    failed: int


def incremental_update(
    lab: AlphaLab,
    interval: Interval,
    end_date: date,
    mode: UniverseMode,
    selection_start: date | None = None,
    selection_end: date | None = None,
    vt_symbols: list[str] | None = None,
    overlap_days: int = 1,
    max_symbols: int | None = None,
) -> UpdateStats:
    client = create_polygon_client()

    universe = resolve_universe(
        mode=mode,
        lab=lab,
        interval=interval,
        selection_start=selection_start,
        selection_end=selection_end,
        vt_symbols=vt_symbols,
    )
    if max_symbols is not None:
        universe = universe[:max_symbols]

    folder = lab.daily_path if interval == Interval.DAILY else lab.minute_path

    updated = 0
    skipped = 0
    failed = 0

    for idx, vt_symbol in enumerate(universe, start=1):
        file_path = folder / f"{vt_symbol}.parquet"
        last_dt = _max_datetime_from_parquet(file_path)

        # 日线：如果已经有 end_date 当天的数据，直接跳过，避免对全市场重复打 Polygon 请求。
        # 分钟：同一天仍可能持续产生新分钟，因此不按 date 直接跳过。
        if last_dt is not None and interval == Interval.DAILY and last_dt.date() >= end_date:
            skipped += 1
            continue

        if last_dt is None:
            # 没有历史：从 end_date 往前 overlap_days 作为最小增量（避免全量回灌）
            start_date = end_date - timedelta(days=max(1, overlap_days))
        else:
            # 增量起点：最后一天往前 overlap_days（允许少量重复，AlphaLab 会去重）
            start_date = last_dt.date() - timedelta(days=max(0, overlap_days))

        if start_date > end_date:
            skipped += 1
            continue

        bars: list[BarData] = []
        try:
            bars = _download_symbol_bars(client, vt_symbol, interval, start_date, end_date)
        except Exception as exc:
            failed += 1
            logger.warning(f"[incremental_update] 下载失败 {vt_symbol} ({interval.value}): {exc}")
            continue

        if not bars:
            skipped += 1
        else:
            try:
                lab.save_bar_data(bars)
                updated += 1
            except Exception as save_exc:
                # 兜底：对历史 parquet 做 schema 修复并重试一次
                if _is_parquet_schema_conflict_error(save_exc) and file_path.exists():
                    if _repair_parquet_schema_inplace(file_path):
                        try:
                            lab.save_bar_data(bars)
                            updated += 1
                            logger.info(f"[incremental_update] schema 修复后写入成功 {vt_symbol} ({interval.value})")
                        except Exception as save_exc2:
                            failed += 1
                            logger.warning(
                                f"[incremental_update] 写入失败 {vt_symbol} ({interval.value}) after repair: {save_exc2}"
                            )
                    else:
                        failed += 1
                        logger.warning(
                            f"[incremental_update] 写入失败 {vt_symbol} ({interval.value}) (schema repair failed): {save_exc}"
                        )
                else:
                    failed += 1
                    logger.warning(f"[incremental_update] 写入失败 {vt_symbol} ({interval.value}): {save_exc}")

        if idx % 100 == 0:
            logger.info(
                f"[incremental_update] {interval.value} 进度 {idx}/{len(universe)}: "
                f"updated={updated}, skipped={skipped}, failed={failed}"
            )

    return UpdateStats(total=len(universe), updated=updated, skipped=skipped, failed=failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally update AlphaLab parquet using Polygon REST.")
    parser.add_argument("--lab-path", type=str, default=str(DEFAULT_LAB_DIR))
    parser.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD, e.g. 2025-12-18")
    parser.add_argument("--interval", type=str, choices=["daily", "minute", "both"], default="daily")

    parser.add_argument(
        "--universe",
        type=str,
        choices=["daily_selection", "ref_tickers_cs", "lab_existing", "vt_symbols"],
        default="daily_selection",
        help="Which universe to update (minute updates should avoid lab_existing).",
    )

    parser.add_argument("--selection-start", type=str, help="YYYY-MM-DD (for daily_selection)")
    parser.add_argument("--selection-end", type=str, help="YYYY-MM-DD (for daily_selection)")
    parser.add_argument("--vt-symbols", type=str, help="Comma-separated vt_symbols (for vt_symbols mode)")

    parser.add_argument("--overlap-days", type=int, default=1)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()

    lab_path = Path(args.lab_path)
    if not lab_path.is_absolute():
        lab_path = Path(__file__).resolve().parents[2] / lab_path

    end_date = _parse_date(args.end_date)

    selection_start = _parse_date(args.selection_start) if args.selection_start else None
    selection_end = _parse_date(args.selection_end) if args.selection_end else None

    vt_symbols = None
    if args.vt_symbols:
        vt_symbols = [s.strip() for s in args.vt_symbols.split(",") if s.strip()]

    lab = AlphaLab(str(lab_path))

    if args.interval in ("daily", "both"):
        stats = incremental_update(
            lab=lab,
            interval=Interval.DAILY,
            end_date=end_date,
            mode=args.universe,  # type: ignore[arg-type]
            selection_start=selection_start,
            selection_end=selection_end,
            vt_symbols=vt_symbols,
            overlap_days=args.overlap_days,
            max_symbols=args.max_symbols,
        )
        logger.info(f"[incremental_update] DAILY done: {stats}")

    if args.interval in ("minute", "both"):
        stats = incremental_update(
            lab=lab,
            interval=Interval.MINUTE,
            end_date=end_date,
            mode=args.universe,  # type: ignore[arg-type]
            selection_start=selection_start,
            selection_end=selection_end,
            vt_symbols=vt_symbols,
            overlap_days=args.overlap_days,
            max_symbols=args.max_symbols,
        )
        logger.info(f"[incremental_update] MINUTE done: {stats}")


if __name__ == "__main__":
    main()


