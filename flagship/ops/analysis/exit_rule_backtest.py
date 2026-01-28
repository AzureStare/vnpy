"""
Exit rule backtest using minute bars and strict VWAP reclaim entry.

Outputs:
- logs/app/analysis/exit_rule_backtest_detail.csv
- logs/app/analysis/exit_rule_backtest_summary.json
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

from flagship.trading.calendar import is_market_closed_day
from flagship.config import PROJECT_ROOT
from flagship.trading.config import LAB_PATH
from flagship.ops.analysis.entry_efficiency import (
    EntryConfig,
    compute_intraday_vwap,
    detect_vwap_reclaim,
    _load_minute_bars_for_date,
    _load_recent_selection_symbols,
    _slice_rth,
)

EASTERN = ZoneInfo("America/New_York")
SESSION_OPEN_ET = dtime(9, 30)
SESSION_CLOSE_ET = dtime(16, 0)

DEFAULT_LOOKBACK_DAYS = 15
DEFAULT_TOP_N = 50
DEFAULT_HOLD_MINUTES = 2
DEFAULT_CUTOFF_HHMM = "10:30"
DEFAULT_MAX_HOLDING_DAYS = 10
DEFAULT_TOE_IN_RATIO = 0.2


@dataclass(frozen=True)
class ExitRuleConfig:
    name: str
    hard_stop_loss_pct: float
    stop_loss_atr_multiplier: float
    profit_threshold_trend: float
    profit_threshold_win: float
    spike_threshold: float
    trailing_stop_pct: float
    trend_ema_period: int
    trailing_ema_period: int
    vwap_exit_confirm_minutes: int


@dataclass
class PositionState:
    size_fraction: float
    avg_cost: float
    entry_time: datetime
    entry_index: int
    entry_high: float
    entry_day: date


@dataclass
class ExitResult:
    exit_time: datetime | None
    exit_price: float | None
    exit_reason: str | None
    return_pct: float | None
    hold_minutes: int | None
    mae: float | None
    mfe: float | None


def _parse_hhmm(text: str) -> dtime:
    s = str(text or "").strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    raise ValueError(f"invalid cutoff time: {text}")


def _minutes_between(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds() // 60))


def _ema_series(values: list[float], period: int) -> list[float | None]:
    if not values or period <= 1:
        return [v for v in values]
    alpha = 2.0 / float(period + 1)
    ema_values: list[float | None] = []
    ema = None
    for v in values:
        if ema is None:
            ema = float(v)
        else:
            ema = alpha * float(v) + (1.0 - alpha) * float(ema)
        ema_values.append(float(ema))
    return ema_values


def _load_minute_bars_range(
    lab_path: Path, vt_symbol: str, start_date: date, max_holding_days: int
) -> pl.DataFrame:
    days_collected = 0
    cur = start_date
    frames: list[pl.DataFrame] = []
    while days_collected < max_holding_days:
        if is_market_closed_day(cur):
            cur += timedelta(days=1)
            continue
        minute_df = _load_minute_bars_for_date(lab_path, vt_symbol, cur)
        rth_df = _slice_rth(minute_df, cur)
        if not rth_df.is_empty():
            frames.append(rth_df.with_columns(pl.lit(cur).alias("trade_date")))
        days_collected += 1
        cur += timedelta(days=1)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical").sort("datetime")


def _compute_vwap_by_day(df: pl.DataFrame) -> list[float | None]:
    if df.is_empty():
        return []
    if "trade_date" not in df.columns:
        return compute_intraday_vwap(df)
    vwap_all: list[float | None] = [None] * df.height
    dates = df.get_column("trade_date").to_list()
    start = 0
    while start < len(dates):
        cur_date = dates[start]
        end = start
        while end < len(dates) and dates[end] == cur_date:
            end += 1
        vwap_slice = compute_intraday_vwap(df.slice(start, end - start))
        for idx, v in enumerate(vwap_slice):
            vwap_all[start + idx] = v
        start = end
    return vwap_all


def _find_reclaim_entry(
    df: pl.DataFrame, entry_config: EntryConfig
) -> tuple[bool, int | None]:
    triggered, confirm_idx = detect_vwap_reclaim(
        df,
        hold_minutes=entry_config.hold_minutes,
        cutoff_time=entry_config.cutoff_time,
        require_below_before_reclaim=bool(entry_config.require_below_before_reclaim),
    )
    if not triggered or confirm_idx is None:
        return (False, None)
    return (True, min(confirm_idx + 1, df.height - 1))


def _init_position(
    timestamps: list[datetime],
    prices: list[float],
    trade_dates: list[date],
    entry_idx: int,
    size_fraction: float,
) -> PositionState:
    entry_time = timestamps[entry_idx]
    entry_price = float(prices[entry_idx])
    entry_day = trade_dates[entry_idx]
    return PositionState(
        size_fraction=float(size_fraction),
        avg_cost=entry_price,
        entry_time=entry_time,
        entry_index=entry_idx,
        entry_high=entry_price,
        entry_day=entry_day,
    )


def _simulate_exit(
    *,
    timestamps: list[datetime],
    trade_dates: list[date],
    open_prices: list[float],
    high_prices: list[float],
    low_prices: list[float],
    close_prices: list[float],
    vwap_values: list[float | None],
    ema_map: dict[str, list[float | None]],
    entry_config: EntryConfig,
    rule_config: ExitRuleConfig,
    entry_variant: str,
    entry_idx: int,
    reclaim_entry_idx: int | None,
    max_holding_days: int,
) -> ExitResult:
    if entry_idx >= len(timestamps):
        return ExitResult(None, None, "invalid_entry", None, None, None, None)

    toe_in_ratio = max(0.0, min(1.0, float(entry_config.toe_in_ratio)))
    day_open_by_date: dict[date, float] = {}
    for idx, d in enumerate(trade_dates):
        if d not in day_open_by_date:
            day_open_by_date[d] = float(open_prices[idx])

    # Initialize position
    if entry_variant == "toe_in_strict_reclaim":
        position = _init_position(
            timestamps, open_prices, trade_dates, entry_idx, size_fraction=toe_in_ratio
        )
    else:
        position = _init_position(
            timestamps, open_prices, trade_dates, entry_idx, size_fraction=1.0
        )

    days_held = 1
    last_date = position.entry_day
    spike_triggered = False
    day_low = low_prices[position.entry_index]
    max_runup = None
    max_drawdown = None

    for idx in range(position.entry_index, len(timestamps)):
        dt = timestamps[idx]
        d = trade_dates[idx]
        close_px = float(close_prices[idx])
        low_px = float(low_prices[idx])
        high_px = float(high_prices[idx])
        day_open = float(day_open_by_date.get(d, open_prices[idx]))

        if d != last_date:
            days_held += 1
            last_date = d
            day_low = low_px
            spike_triggered = False
            if days_held > max_holding_days:
                prev_idx = max(position.entry_index, idx - 1)
                exit_dt = timestamps[prev_idx]
                exit_px = float(close_prices[prev_idx])
                ret = (exit_px / position.avg_cost - 1.0) if position.avg_cost > 0 else None
                hold_minutes = _minutes_between(position.entry_time, exit_dt)
                return ExitResult(exit_dt, exit_px, "time_stop", ret, hold_minutes, max_drawdown, max_runup)

        if entry_variant == "toe_in_strict_reclaim" and reclaim_entry_idx is not None:
            if idx == reclaim_entry_idx and position.size_fraction < 1.0:
                new_price = float(open_prices[idx])
                new_fraction = 1.0 - position.size_fraction
                if new_fraction > 0:
                    position.avg_cost = (
                        position.avg_cost * position.size_fraction + new_price * new_fraction
                    )
                    position.size_fraction = 1.0

        position.entry_high = max(position.entry_high, close_px)
        day_low = min(day_low, low_px)

        if position.size_fraction <= 0:
            continue

        profit_pct = (close_px / position.avg_cost - 1.0) if position.avg_cost > 0 else None
        if profit_pct is not None:
            max_runup = max(max_runup or profit_pct, profit_pct)
            max_drawdown = min(max_drawdown or profit_pct, profit_pct)

        day_return = (close_px / day_open - 1.0) if day_open > 0 else 0.0
        if day_return > rule_config.spike_threshold:
            spike_triggered = True

        exit_reason = None

        # Spike guard
        if spike_triggered and close_px <= day_low:
            exit_reason = "spike_fade"

        if exit_reason is None:
            if profit_pct is None:
                exit_reason = None
            elif profit_pct < rule_config.profit_threshold_trend and days_held < 3:
                hard_stop = position.avg_cost * (1.0 - rule_config.hard_stop_loss_pct)
                if close_px < hard_stop:
                    exit_reason = "hard_stop"
            elif (
                rule_config.profit_threshold_trend
                <= profit_pct
                < rule_config.profit_threshold_win
            ):
                trend_ema = ema_map.get(f"ema{rule_config.trend_ema_period}", [None])[idx]
                if trend_ema is not None and close_px < float(trend_ema):
                    exit_reason = f"ema{rule_config.trend_ema_period}_trend_exit"
            else:
                trailing_stop = position.entry_high * (1.0 - rule_config.trailing_stop_pct)
                trailing_ema = ema_map.get(f"ema{rule_config.trailing_ema_period}", [None])[idx]
                stop_line = trailing_stop
                if trailing_ema is not None:
                    stop_line = max(stop_line, float(trailing_ema))
                if close_px < stop_line:
                    exit_reason = "trailing_stop"

        if exit_reason is None and rule_config.name == "vwap_exit":
            vwap_value = vwap_values[idx]
            if vwap_value is not None and close_px < float(vwap_value):
                confirm = 1
                for j in range(idx + 1, len(timestamps)):
                    if trade_dates[j] != d:
                        break
                    if vwap_values[j] is None:
                        break
                    if float(close_prices[j]) < float(vwap_values[j]):
                        confirm += 1
                    else:
                        break
                    if confirm >= rule_config.vwap_exit_confirm_minutes:
                        exit_reason = "vwap_exit"
                        break

        if exit_reason:
            exit_dt = dt
            exit_px = close_px
            ret = (exit_px / position.avg_cost - 1.0) if position.avg_cost > 0 else None
            hold_minutes = _minutes_between(position.entry_time, exit_dt)
            return ExitResult(exit_dt, exit_px, exit_reason, ret, hold_minutes, max_drawdown, max_runup)

    # Exit at last available close
    exit_dt = timestamps[-1]
    exit_px = float(close_prices[-1])
    ret = (exit_px / position.avg_cost - 1.0) if position.avg_cost > 0 else None
    hold_minutes = _minutes_between(position.entry_time, exit_dt)
    return ExitResult(exit_dt, exit_px, "end_of_data", ret, hold_minutes, max_drawdown, max_runup)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [r["return_pct"] for r in rows if r.get("return_pct") is not None]
    hold_minutes = [r["hold_minutes"] for r in rows if r.get("hold_minutes") is not None]
    if not returns:
        return {
            "samples": 0,
            "avg_return": None,
            "median_return": None,
            "win_rate": None,
            "avg_hold_minutes": None,
            "left_tail_p5": None,
        }
    returns_sorted = sorted(float(v) for v in returns)
    n = len(returns_sorted)
    avg_return = sum(returns_sorted) / float(n)
    median_return = returns_sorted[n // 2] if n % 2 else (returns_sorted[n // 2 - 1] + returns_sorted[n // 2]) / 2.0
    win_rate = float(sum(1 for v in returns_sorted if v > 0)) / float(n)
    left_tail_idx = max(0, int(n * 0.05) - 1)
    left_tail_p5 = returns_sorted[left_tail_idx]
    avg_hold = None
    if hold_minutes:
        avg_hold = sum(float(v) for v in hold_minutes) / float(len(hold_minutes))
    return {
        "samples": int(n),
        "avg_return": float(avg_return),
        "median_return": float(median_return),
        "win_rate": float(win_rate),
        "avg_hold_minutes": float(avg_hold) if avg_hold is not None else None,
        "left_tail_p5": float(left_tail_p5),
    }


def run_analysis(
    *,
    lab_path: Path,
    output_dir: Path,
    lookback: int,
    top_n: int,
    entry_config: EntryConfig,
    max_holding_days: int,
) -> tuple[Path, Path]:
    history_dir = output_dir / "history"
    selections = _load_recent_selection_symbols(history_dir, lookback=lookback, top_n=top_n)
    if not selections:
        raise RuntimeError(f"no selection history found under {history_dir}")

    rules = [
        ExitRuleConfig(
            name="ema10_ladder",
            hard_stop_loss_pct=0.07,
            stop_loss_atr_multiplier=2.5,
            profit_threshold_trend=0.05,
            profit_threshold_win=0.15,
            spike_threshold=0.10,
            trailing_stop_pct=0.10,
            trend_ema_period=10,
            trailing_ema_period=5,
            vwap_exit_confirm_minutes=entry_config.hold_minutes,
        ),
        ExitRuleConfig(
            name="ema20_ladder",
            hard_stop_loss_pct=0.07,
            stop_loss_atr_multiplier=2.5,
            profit_threshold_trend=0.05,
            profit_threshold_win=0.15,
            spike_threshold=0.10,
            trailing_stop_pct=0.10,
            trend_ema_period=20,
            trailing_ema_period=5,
            vwap_exit_confirm_minutes=entry_config.hold_minutes,
        ),
        ExitRuleConfig(
            name="vwap_exit",
            hard_stop_loss_pct=0.07,
            stop_loss_atr_multiplier=2.5,
            profit_threshold_trend=0.05,
            profit_threshold_win=0.15,
            spike_threshold=0.10,
            trailing_stop_pct=0.10,
            trend_ema_period=10,
            trailing_ema_period=5,
            vwap_exit_confirm_minutes=entry_config.hold_minutes,
        ),
    ]

    detail_rows: list[dict[str, Any]] = []
    for trade_date, vt_symbols in selections:
        for vt_symbol in vt_symbols:
            minute_df = _load_minute_bars_range(lab_path, vt_symbol, trade_date, max_holding_days)
            if minute_df.is_empty():
                detail_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "vt_symbol": vt_symbol,
                        "symbol": vt_symbol.split(".", 1)[0],
                        "entry_variant": "strict_reclaim",
                        "exit_rule": None,
                        "data_status": "missing_minute_bars",
                    }
                )
                continue

            minute_df = minute_df.select(
                ["datetime", "trade_date", "open", "high", "low", "close", "volume"]
            )
            timestamps = minute_df.get_column("datetime").to_list()
            trade_dates = minute_df.get_column("trade_date").to_list()
            open_prices = minute_df.get_column("open").to_list()
            high_prices = minute_df.get_column("high").to_list()
            low_prices = minute_df.get_column("low").to_list()
            close_prices = minute_df.get_column("close").to_list()

            vwap_values = _compute_vwap_by_day(minute_df)
            ema5 = _ema_series([float(v) for v in close_prices], 5)
            ema10 = _ema_series([float(v) for v in close_prices], 10)
            ema20 = _ema_series([float(v) for v in close_prices], 20)
            ema_map = {"ema5": ema5, "ema10": ema10, "ema20": ema20}

            # Entry via strict reclaim on the entry day only
            entry_day_df = minute_df.filter(pl.col("trade_date") == trade_date)
            reclaim_triggered, reclaim_entry_idx = _find_reclaim_entry(entry_day_df, entry_config)
            if not reclaim_triggered or reclaim_entry_idx is None:
                detail_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "vt_symbol": vt_symbol,
                        "symbol": vt_symbol.split(".", 1)[0],
                        "entry_variant": "strict_reclaim",
                        "exit_rule": None,
                        "data_status": "no_reclaim",
                    }
                )
                continue

            entry_idx = reclaim_entry_idx
            reclaim_entry_time = entry_day_df.get_column("datetime")[entry_idx]
            reclaim_entry_price = float(entry_day_df.get_column("open")[entry_idx])

            # Map entry_idx to full range index
            full_entry_idx = timestamps.index(reclaim_entry_time)

            for rule in rules:
                for entry_variant in ("strict_reclaim", "toe_in_strict_reclaim"):
                    entry_time = reclaim_entry_time
                    entry_price = reclaim_entry_price
                    if entry_variant == "toe_in_strict_reclaim":
                        entry_time = timestamps[0]
                        entry_price = float(open_prices[0])
                    result = _simulate_exit(
                        timestamps=timestamps,
                        trade_dates=trade_dates,
                        open_prices=open_prices,
                        high_prices=high_prices,
                        low_prices=low_prices,
                        close_prices=close_prices,
                        vwap_values=vwap_values,
                        ema_map=ema_map,
                        entry_config=entry_config,
                        rule_config=rule,
                        entry_variant=entry_variant,
                        entry_idx=full_entry_idx if entry_variant == "strict_reclaim" else 0,
                        reclaim_entry_idx=full_entry_idx if entry_variant == "toe_in_strict_reclaim" else None,
                        max_holding_days=max_holding_days,
                    )
                    detail_rows.append(
                        {
                            "trade_date": trade_date.isoformat(),
                            "vt_symbol": vt_symbol,
                            "symbol": vt_symbol.split(".", 1)[0],
                            "entry_variant": entry_variant,
                            "entry_time": entry_time.isoformat(),
                            "entry_price": entry_price,
                            "exit_rule": rule.name,
                            "exit_time": result.exit_time.isoformat() if result.exit_time else None,
                            "exit_price": result.exit_price,
                            "exit_reason": result.exit_reason,
                            "return_pct": result.return_pct,
                            "hold_minutes": result.hold_minutes,
                            "mae": result.mae,
                            "mfe": result.mfe,
                            "data_status": "ok",
                        }
                    )

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    detail_path = analysis_dir / "exit_rule_backtest_detail.csv"
    summary_path = analysis_dir / "exit_rule_backtest_summary.json"

    pl.DataFrame(detail_rows).write_csv(detail_path)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "params": {
            "lookback_days": int(lookback),
            "top_n": int(top_n),
            "toe_in_ratio": float(entry_config.toe_in_ratio),
            "max_holding_days": int(max_holding_days),
        },
        "overall": {},
    }
    for rule in rules:
        for entry_variant in ("strict_reclaim", "toe_in_strict_reclaim"):
            rows = [
                r
                for r in detail_rows
                if r.get("data_status") == "ok"
                and r.get("exit_rule") == rule.name
                and r.get("entry_variant") == entry_variant
            ]
            summary["overall"][f"{rule.name}:{entry_variant}"] = _summarize(rows)

    # Recommend by avg_return
    best_key = None
    best_avg = None
    for k, v in summary["overall"].items():
        avg = v.get("avg_return")
        if avg is None:
            continue
        if best_avg is None or float(avg) > float(best_avg):
            best_avg = float(avg)
            best_key = k
    summary["recommended"] = {
        "metric": "avg_return",
        "rule_key": best_key,
        "avg_return": best_avg,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[exit_rule_backtest] wrote detail: {detail_path}")
    logger.info(f"[exit_rule_backtest] wrote summary: {summary_path}")
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Exit rule backtest (minute bars + VWAP reclaim entry).")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--hold-minutes", type=int, default=DEFAULT_HOLD_MINUTES)
    parser.add_argument("--cutoff", type=str, default=DEFAULT_CUTOFF_HHMM, help="ET cutoff time, e.g. 10:30")
    parser.add_argument("--toe-in-ratio", type=float, default=DEFAULT_TOE_IN_RATIO)
    parser.add_argument("--max-holding-days", type=int, default=DEFAULT_MAX_HOLDING_DAYS)
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
        require_below_before_reclaim=True,
        toe_in_ratio=float(args.toe_in_ratio),
    )

    run_analysis(
        lab_path=lab_path,
        output_dir=output_dir,
        lookback=int(args.lookback),
        top_n=int(args.top_n),
        entry_config=entry_config,
        max_holding_days=int(args.max_holding_days),
    )


if __name__ == "__main__":
    main()
