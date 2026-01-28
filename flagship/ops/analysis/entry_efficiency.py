"""
Entry efficiency analysis: open-at-09:30 vs VWAP reclaim entry.

Outputs:
- logs/app/analysis/entry_efficiency_detail.csv
- logs/app/analysis/entry_efficiency_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import polars as pl

from vnpy.trader.logger import logger

from flagship.trading.config import LAB_PATH
from flagship.config import PROJECT_ROOT

EASTERN = ZoneInfo("America/New_York")
SESSION_OPEN_ET = dtime(9, 30)
SESSION_CLOSE_ET = dtime(16, 0)

DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_TOP_N = 8
DEFAULT_HOLD_MINUTES = 2
DEFAULT_CUTOFF_HHMM = "10:30"


@dataclass(frozen=True)
class EntryConfig:
    hold_minutes: int
    cutoff_time: dtime
    require_below_before_reclaim: bool = True
    toe_in_ratio: float = 0.2


def _parse_hhmm(text: str) -> dtime:
    text = str(text or "").strip()
    if not text:
        raise ValueError("cutoff time is empty")
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).time()
        except Exception:
            continue
    raise ValueError(f"invalid cutoff time: {text}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid json (expected dict): {path}")
    return data


def _resolve_minute_file(lab_path: Path, vt_symbol: str) -> Path | None:
    minute_dir = lab_path / "minute"
    direct = minute_dir / f"{vt_symbol}.parquet"
    if direct.exists():
        return direct
    symbol_root = vt_symbol.split(".", 1)[0]
    candidates = sorted(minute_dir.glob(f"{symbol_root}.*.parquet"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _normalize_minute_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    if "open_price" in df.columns and "open" not in df.columns:
        rename_map["open_price"] = "open"
    if "high_price" in df.columns and "high" not in df.columns:
        rename_map["high_price"] = "high"
    if "low_price" in df.columns and "low" not in df.columns:
        rename_map["low_price"] = "low"
    if "close_price" in df.columns and "close" not in df.columns:
        rename_map["close_price"] = "close"
    if rename_map:
        df = df.rename(rename_map)
    return df


def _load_minute_bars_for_date(lab_path: Path, vt_symbol: str, trade_date: date) -> pl.DataFrame:
    file_path = _resolve_minute_file(lab_path, vt_symbol)
    if file_path is None or not file_path.exists():
        return pl.DataFrame()
    df = pl.read_parquet(file_path)
    if df.is_empty():
        return df
    df = _normalize_minute_columns(df)
    if "datetime" not in df.columns:
        return pl.DataFrame()
    return df.filter(pl.col("datetime").dt.date() == trade_date).sort("datetime")


def _slice_rth(df: pl.DataFrame, trade_date: date) -> pl.DataFrame:
    if df.is_empty():
        return df
    session_open = datetime.combine(trade_date, SESSION_OPEN_ET)
    session_close = datetime.combine(trade_date, SESSION_CLOSE_ET)
    return df.filter((pl.col("datetime") >= session_open) & (pl.col("datetime") <= session_close))


def compute_intraday_vwap(minute_df: pl.DataFrame) -> list[float | None]:
    if minute_df.is_empty():
        return []
    highs = minute_df.get_column("high").to_list() if "high" in minute_df.columns else None
    lows = minute_df.get_column("low").to_list() if "low" in minute_df.columns else None
    closes = minute_df.get_column("close").to_list() if "close" in minute_df.columns else None
    volumes = minute_df.get_column("volume").to_list() if "volume" in minute_df.columns else None
    if closes is None or volumes is None:
        return []

    vwap_values: list[float | None] = []
    cumulative_volume = 0.0
    cumulative_pv = 0.0
    for idx, close_value in enumerate(closes):
        high_value = highs[idx] if highs is not None else close_value
        low_value = lows[idx] if lows is not None else close_value
        volume_value = float(volumes[idx] or 0.0)
        typical_price = (float(high_value) + float(low_value) + float(close_value)) / 3.0
        cumulative_pv += typical_price * volume_value
        cumulative_volume += volume_value
        if cumulative_volume <= 0:
            vwap_values.append(None)
        else:
            vwap_values.append(cumulative_pv / cumulative_volume)
    return vwap_values


def detect_vwap_reclaim(
    minute_df: pl.DataFrame, *, hold_minutes: int, cutoff_time: dtime, require_below_before_reclaim: bool = False
) -> tuple[bool, int | None]:
    if minute_df.is_empty() or hold_minutes <= 0:
        return (False, None)
    vwap_values = compute_intraday_vwap(minute_df)
    if not vwap_values:
        return (False, None)

    closes = minute_df.get_column("close").to_list()
    timestamps = minute_df.get_column("datetime").to_list()

    consecutive_above = 0
    saw_below_once = False
    for idx, close_value in enumerate(closes):
        vwap_value = vwap_values[idx]
        if vwap_value is None:
            consecutive_above = 0
            continue
        current_time = timestamps[idx].time()
        if current_time > cutoff_time:
            return (False, None)
        # Strict reclaim: must have seen at least one close below VWAP before we allow confirmation.
        if float(close_value) < float(vwap_value):
            saw_below_once = True
            consecutive_above = 0
            continue

        if float(close_value) > float(vwap_value):
            if require_below_before_reclaim and not saw_below_once:
                consecutive_above = 0
                continue
            consecutive_above += 1
        else:
            consecutive_above = 0
        if consecutive_above >= hold_minutes:
            return (True, idx)
    return (False, None)


def _safe_mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _safe_median(values: Iterable[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _minutes_between(start_dt: datetime, end_dt: datetime) -> int:
    delta = end_dt - start_dt
    return max(0, int(delta.total_seconds() // 60))


def _extract_trade_date(selection_payload: dict[str, Any], fallback_date: date | None) -> date | None:
    as_of_date = selection_payload.get("as_of_date")
    if as_of_date:
        try:
            return date.fromisoformat(str(as_of_date))
        except Exception:
            pass
    return fallback_date


def _load_recent_selection_symbols(
    history_dir: Path, *, lookback: int, top_n: int
) -> list[tuple[date, list[str]]]:
    if not history_dir.exists():
        return []
    items: list[tuple[date, list[str]]] = []
    selection_files = sorted(history_dir.glob("selection_*.json"), reverse=True)
    for selection_path in selection_files:
        if len(items) >= lookback:
            break
        payload = _load_json(selection_path)
        fallback_date = None
        try:
            ymd = selection_path.stem.replace("selection_", "")
            fallback_date = datetime.strptime(ymd, "%Y%m%d").date()
        except Exception:
            fallback_date = None
        trade_date = _extract_trade_date(payload, fallback_date)
        if trade_date is None:
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        vt_symbols: list[str] = []
        for row in rows[: max(0, int(top_n))]:
            vt_symbol = row.get("vt_symbol") if isinstance(row, dict) else None
            if vt_symbol:
                vt_symbols.append(str(vt_symbol))
        if vt_symbols:
            items.append((trade_date, vt_symbols))
    return items


def _compute_summary(detail_rows: list[dict[str, Any]], *, toe_in_ratio: float) -> dict[str, Any]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        trade_date = row.get("trade_date")
        if trade_date:
            by_date.setdefault(str(trade_date), []).append(row)

    def _summarize_variant(
        rows: list[dict[str, Any]],
        *,
        variant_key: str,
        triggered_key: str,
        return_key: str,
        bps_key: str,
    ) -> dict[str, Any]:
        total = len(rows)
        open_returns = [r.get("open_return") for r in rows if r.get("open_return") is not None]
        variant_returns = [r.get(return_key) for r in rows if r.get(return_key) is not None]
        open_bps_per_min = [r.get("open_bps_per_min") for r in rows if r.get("open_bps_per_min") is not None]
        variant_bps_per_min = [r.get(bps_key) for r in rows if r.get(bps_key) is not None]
        triggered = sum(1 for r in rows if r.get(triggered_key) is True)

        open_avg = _safe_mean(open_returns)
        open_med = _safe_median(open_returns)
        v_avg = _safe_mean(variant_returns)
        v_med = _safe_median(variant_returns)

        return {
            "variant": variant_key,
            "total_symbols": total,
            "triggered": triggered,
            "trigger_rate": (triggered / total) if total > 0 else None,
            "open_avg_return": open_avg,
            "open_median_return": open_med,
            "variant_avg_return": v_avg,
            "variant_median_return": v_med,
            "open_avg_bps_per_min": _safe_mean(open_bps_per_min),
            "variant_avg_bps_per_min": _safe_mean(variant_bps_per_min),
            "delta_avg_return": (v_avg - open_avg) if v_avg is not None and open_avg is not None else None,
            "delta_median_return": (v_med - open_med) if v_med is not None and open_med is not None else None,
        }

    def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            # Backward-compatible fields (legacy reclaim)
            **{
                k: v
                for k, v in _summarize_variant(
                    rows,
                    variant_key="legacy_reclaim",
                    triggered_key="reclaim_triggered",
                    return_key="reclaim_return",
                    bps_key="reclaim_bps_per_min",
                ).items()
                if k
                in {
                    "total_symbols",
                    "triggered",
                    "trigger_rate",
                    "open_avg_return",
                    "open_median_return",
                    "variant_avg_return",
                    "variant_median_return",
                    "open_avg_bps_per_min",
                    "variant_avg_bps_per_min",
                    "delta_avg_return",
                    "delta_median_return",
                }
            },
            # Additional variants
            "legacy_reclaim": _summarize_variant(
                rows,
                variant_key="legacy_reclaim",
                triggered_key="reclaim_triggered",
                return_key="reclaim_return",
                bps_key="reclaim_bps_per_min",
            ),
            "strict_reclaim": _summarize_variant(
                rows,
                variant_key="strict_reclaim",
                triggered_key="strict_reclaim_triggered",
                return_key="strict_reclaim_return",
                bps_key="strict_reclaim_bps_per_min",
            ),
            "toe_in_strict_reclaim": _summarize_variant(
                rows,
                variant_key=f"toe_in_strict_reclaim(toe_in_ratio={toe_in_ratio:.2f})",
                triggered_key="strict_reclaim_triggered",
                return_key="toe_in_return",
                bps_key="toe_in_bps_per_min",
            ),
        }

    daily_summary = {d: _summarize(rows) for d, rows in sorted(by_date.items(), reverse=True)}
    overall = _summarize(detail_rows)
    return {
        "generated_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "params": {
            "toe_in_ratio": float(toe_in_ratio),
        },
        "daily": daily_summary,
        "overall": overall,
    }


def run_analysis(
    *,
    lab_path: Path,
    output_dir: Path,
    lookback: int,
    top_n: int,
    entry_config: EntryConfig,
) -> tuple[Path, Path]:
    history_dir = output_dir / "history"
    selections = _load_recent_selection_symbols(history_dir, lookback=lookback, top_n=top_n)
    if not selections:
        raise RuntimeError(f"no selection history found under {history_dir}")

    detail_rows: list[dict[str, Any]] = []

    for trade_date, vt_symbols in selections:
        for vt_symbol in vt_symbols:
            minute_df = _load_minute_bars_for_date(lab_path, vt_symbol, trade_date)
            rth_df = _slice_rth(minute_df, trade_date)
            if rth_df.is_empty():
                detail_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "vt_symbol": vt_symbol,
                        "symbol": vt_symbol.split(".", 1)[0],
                        "data_status": "missing_minute_bars",
                    }
                )
                continue

            rth_df = rth_df.select(["datetime", "open", "high", "low", "close", "volume"])
            timestamps = rth_df.get_column("datetime").to_list()
            open_prices = rth_df.get_column("open").to_list()
            close_prices = rth_df.get_column("close").to_list()

            open_entry_time = timestamps[0]
            open_entry_price = float(open_prices[0]) if open_prices else None

            close_time = timestamps[-1]
            close_price = float(close_prices[-1]) if close_prices else None

            open_return = None
            open_hold_minutes = None
            open_bps_per_min = None
            if open_entry_price and close_price and open_entry_price > 0:
                open_return = (close_price - open_entry_price) / open_entry_price
                open_hold_minutes = _minutes_between(open_entry_time, close_time)
                if open_hold_minutes and open_hold_minutes > 0:
                    open_bps_per_min = (open_return * 10000.0) / float(open_hold_minutes)

            reclaim_triggered, reclaim_idx = detect_vwap_reclaim(
                rth_df,
                hold_minutes=entry_config.hold_minutes,
                cutoff_time=entry_config.cutoff_time,
                require_below_before_reclaim=False,
            )
            strict_reclaim_triggered, strict_reclaim_idx = detect_vwap_reclaim(
                rth_df,
                hold_minutes=entry_config.hold_minutes,
                cutoff_time=entry_config.cutoff_time,
                require_below_before_reclaim=bool(entry_config.require_below_before_reclaim),
            )
            reclaim_entry_price = None
            reclaim_entry_time = None
            reclaim_confirm_time = None
            reclaim_return = None
            reclaim_hold_minutes = None
            reclaim_bps_per_min = None

            if reclaim_triggered and reclaim_idx is not None:
                reclaim_confirm_time = timestamps[reclaim_idx]
                entry_idx = min(reclaim_idx + 1, len(open_prices) - 1)
                reclaim_entry_time = timestamps[entry_idx]
                reclaim_entry_price = float(open_prices[entry_idx]) if open_prices else None
                if reclaim_entry_price and close_price and reclaim_entry_price > 0:
                    reclaim_return = (close_price - reclaim_entry_price) / reclaim_entry_price
                    reclaim_hold_minutes = _minutes_between(reclaim_entry_time, close_time)
                    if reclaim_hold_minutes and reclaim_hold_minutes > 0:
                        reclaim_bps_per_min = (reclaim_return * 10000.0) / float(reclaim_hold_minutes)

            strict_reclaim_entry_price = None
            strict_reclaim_entry_time = None
            strict_reclaim_confirm_time = None
            strict_reclaim_return = None
            strict_reclaim_hold_minutes = None
            strict_reclaim_bps_per_min = None

            if strict_reclaim_triggered and strict_reclaim_idx is not None:
                strict_reclaim_confirm_time = timestamps[strict_reclaim_idx]
                entry_idx = min(strict_reclaim_idx + 1, len(open_prices) - 1)
                strict_reclaim_entry_time = timestamps[entry_idx]
                strict_reclaim_entry_price = float(open_prices[entry_idx]) if open_prices else None
                if strict_reclaim_entry_price and close_price and strict_reclaim_entry_price > 0:
                    strict_reclaim_return = (close_price - strict_reclaim_entry_price) / strict_reclaim_entry_price
                    strict_reclaim_hold_minutes = _minutes_between(strict_reclaim_entry_time, close_time)
                    if strict_reclaim_hold_minutes and strict_reclaim_hold_minutes > 0:
                        strict_reclaim_bps_per_min = (strict_reclaim_return * 10000.0) / float(strict_reclaim_hold_minutes)

            toe_in_ratio = float(entry_config.toe_in_ratio)
            toe_in_ratio = max(0.0, min(1.0, toe_in_ratio))

            toe_in_return = None
            toe_in_bps_per_min = None
            toe_in_hold_minutes = None
            toe_in_entry_price = None
            if toe_in_ratio > 0 and open_return is not None:
                # Portfolio-style return with partial exposure:
                # - toe-in portion earns open_return
                # - remainder earns strict_reclaim_return if triggered; else remainder is skipped
                if strict_reclaim_return is not None:
                    toe_in_return = toe_in_ratio * float(open_return) + (1.0 - toe_in_ratio) * float(strict_reclaim_return)
                    toe_in_entry_price = None
                else:
                    toe_in_return = toe_in_ratio * float(open_return)
                    toe_in_entry_price = None
                toe_in_hold_minutes = open_hold_minutes
                if toe_in_hold_minutes and toe_in_hold_minutes > 0:
                    toe_in_bps_per_min = (toe_in_return * 10000.0) / float(toe_in_hold_minutes)

            detail_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "vt_symbol": vt_symbol,
                    "symbol": vt_symbol.split(".", 1)[0],
                    "data_status": "ok",
                    "open_entry_time": open_entry_time.isoformat(),
                    "open_entry_price": open_entry_price,
                    "open_return": open_return,
                    "open_hold_minutes": open_hold_minutes,
                    "open_bps_per_min": open_bps_per_min,
                    "reclaim_triggered": reclaim_triggered,
                    "reclaim_confirm_time": reclaim_confirm_time.isoformat()
                    if reclaim_confirm_time
                    else None,
                    "reclaim_entry_time": reclaim_entry_time.isoformat() if reclaim_entry_time else None,
                    "reclaim_entry_price": reclaim_entry_price,
                    "reclaim_return": reclaim_return,
                    "reclaim_hold_minutes": reclaim_hold_minutes,
                    "reclaim_bps_per_min": reclaim_bps_per_min,
                    "strict_reclaim_triggered": strict_reclaim_triggered,
                    "strict_reclaim_confirm_time": strict_reclaim_confirm_time.isoformat()
                    if strict_reclaim_confirm_time
                    else None,
                    "strict_reclaim_entry_time": strict_reclaim_entry_time.isoformat()
                    if strict_reclaim_entry_time
                    else None,
                    "strict_reclaim_entry_price": strict_reclaim_entry_price,
                    "strict_reclaim_return": strict_reclaim_return,
                    "strict_reclaim_hold_minutes": strict_reclaim_hold_minutes,
                    "strict_reclaim_bps_per_min": strict_reclaim_bps_per_min,
                    "toe_in_ratio": toe_in_ratio,
                    "toe_in_return": toe_in_return,
                    "toe_in_bps_per_min": toe_in_bps_per_min,
                    "toe_in_hold_minutes": toe_in_hold_minutes,
                    "close_time": close_time.isoformat(),
                    "close_price": close_price,
                }
            )

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    detail_path = analysis_dir / "entry_efficiency_detail.csv"
    summary_path = analysis_dir / "entry_efficiency_summary.json"

    pl.DataFrame(detail_rows).write_csv(detail_path)
    summary_payload = _compute_summary(detail_rows, toe_in_ratio=float(entry_config.toe_in_ratio))
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[entry_efficiency] wrote detail: {detail_path}")
    logger.info(f"[entry_efficiency] wrote summary: {summary_path}")
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Entry efficiency analysis for VWAP reclaim.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--hold-minutes", type=int, default=DEFAULT_HOLD_MINUTES)
    parser.add_argument("--cutoff", type=str, default=DEFAULT_CUTOFF_HHMM, help="ET cutoff time, e.g. 10:30")
    parser.add_argument(
        "--require-below-before-reclaim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict reclaim: require at least one close below VWAP before reclaim can confirm (default: true).",
    )
    parser.add_argument(
        "--toe-in-ratio",
        type=float,
        default=0.2,
        help="Toe-in ratio at 09:30 open for toe_in+strict_reclaim variant (default: 0.2). Use 0 to disable.",
    )
    parser.add_argument("--lab-path", type=str, default=str(LAB_PATH))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "logs" / "app"))
    args = parser.parse_args()

    lab_path = Path(args.lab_path)
    if not lab_path.is_absolute():
        lab_path = PROJECT_ROOT / lab_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    entry_config = EntryConfig(
        hold_minutes=int(args.hold_minutes),
        cutoff_time=_parse_hhmm(args.cutoff),
        require_below_before_reclaim=bool(args.require_below_before_reclaim),
        toe_in_ratio=float(args.toe_in_ratio),
    )

    run_analysis(
        lab_path=lab_path,
        output_dir=output_dir,
        lookback=int(args.lookback),
        top_n=int(args.top_n),
        entry_config=entry_config,
    )


if __name__ == "__main__":
    main()
