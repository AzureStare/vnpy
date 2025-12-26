"""
Backtest session alignment utilities for Flagship Alpha-Momentum.

Design goals (DDD / Clean Code):
- Use intention-revealing names (avoid abbreviations like RTH, avoid generic names like support).
- Encapsulate market session filtering and signal alignment for minute backtests.
- Do NOT modify vn.py framework code; extend via wrappers/subclasses only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from typing import Any

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, BacktestingEngine


# --- Market session (domain concept) ---

REGULAR_TRADING_HOURS_START: dtime = dtime(9, 30)
REGULAR_TRADING_HOURS_END: dtime = dtime(16, 0)


def is_in_regular_trading_hours(dt_naive: datetime) -> bool:
    """Whether a naive ET datetime is inside Regular Trading Hours (09:30-16:00)."""
    t = dt_naive.time()
    return REGULAR_TRADING_HOURS_START <= t <= REGULAR_TRADING_HOURS_END


# --- CLI parsing helpers ---

def is_date_only_str(value: str) -> bool:
    value = value.strip()
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def parse_date_or_datetime(value: str) -> datetime:
    """Parse YYYY-MM-DD or ISO datetime string into naive datetime."""
    return datetime.fromisoformat(value.strip())


# --- Signal alignment ---

def signal_is_daily_snapshot(signal_df: pl.DataFrame) -> bool:
    """
    True if signal has only one unique time-of-day across all rows.
    Common case: daily snapshot signal at 00:00.
    """
    if signal_df.is_empty() or "datetime" not in signal_df.columns:
        return False
    try:
        n_unique = signal_df.select(pl.col("datetime").dt.time().n_unique()).item()
        return bool(n_unique == 1)
    except Exception:
        return False


def build_signal_by_trade_date(signal_df: pl.DataFrame) -> dict[date, pl.DataFrame]:
    """
    Build {trade_date: df_for_that_date} mapping.
    Compatible with polars versions that return tuple keys from partition_by(as_dict=True).
    """
    parts = (
        signal_df
        .with_columns(pl.col("datetime").dt.date().alias("_trade_date"))
        .partition_by("_trade_date", as_dict=True)
    )
    out: dict[date, pl.DataFrame] = {}
    for k, v in parts.items():
        key = k[0] if isinstance(k, tuple) and len(k) == 1 else k
        out[key] = v.drop("_trade_date")
    return out


@dataclass(frozen=True)
class SignalResolver:
    """
    Resolve signal rows for a given bar datetime.

    Rules:
    - Minute-level signal: try exact datetime match first.
    - Daily snapshot (datetime=00:00): use per-trade-date snapshot.
    - If exact match misses (e.g. missing minutes), fall back to trade-date snapshot.
    """

    signal_df: pl.DataFrame
    is_daily_snapshot: bool
    by_trade_date: dict[date, pl.DataFrame]

    @classmethod
    def from_signal_df(cls, signal_df: pl.DataFrame) -> "SignalResolver":
        snap = signal_is_daily_snapshot(signal_df)
        by_date = build_signal_by_trade_date(signal_df) if (not signal_df.is_empty()) else {}
        return cls(signal_df=signal_df, is_daily_snapshot=snap, by_trade_date=by_date)

    def get_for_datetime(self, dt_now: datetime) -> pl.DataFrame:
        dt_now = dt_now.replace(tzinfo=None)

        # Minute-level signals: exact match first.
        if not self.is_daily_snapshot:
            exact = self.signal_df.filter(pl.col("datetime") == dt_now)
            if not exact.is_empty():
                return exact

        # Snapshot or exact miss: fall back to date snapshot.
        day_df = self.by_trade_date.get(dt_now.date())
        if day_df is not None:
            return day_df

        return pl.DataFrame()


# --- vn.py extensions (no framework edits) ---

class RegularTradingHoursFilteredAlphaLab:
    """
    AlphaLab wrapper that filters minute bars to Regular Trading Hours.
    Avoids monkey-patching AlphaLab methods.
    """

    def __init__(self, base: AlphaLab, *, rth_only: bool) -> None:
        self._base = base
        self._rth_only = rth_only

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[Any]:
        bars = self._base.load_bar_data(vt_symbol, interval, start, end)
        if self._rth_only and interval == Interval.MINUTE:
            bars = [
                bar
                for bar in bars
                if is_in_regular_trading_hours(bar.datetime.replace(tzinfo=None))
            ]
        return bars

    def load_contract_setttings(self) -> dict:
        return self._base.load_contract_setttings()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)


class SignalAwareBacktestingEngine(BacktestingEngine):
    """
    BacktestingEngine with override-based signal resolution (no monkey-patch).
    """

    def __init__(self, lab: Any, *, signal_resolver: SignalResolver | None = None) -> None:
        super().__init__(lab)
        self._signal_resolver: SignalResolver | None = signal_resolver

    def get_signal(self) -> pl.DataFrame:
        if self._signal_resolver is None:
            return super().get_signal()

        if not self.datetime:
            return pl.DataFrame()

        dt_now: datetime = self.datetime.replace(tzinfo=None)
        return self._signal_resolver.get_for_datetime(dt_now)


