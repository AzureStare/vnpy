from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval


@dataclass(frozen=True)
class ResampleSpec:
    name: str
    every: str  # polars duration, e.g. "30m", "2h", "4h"
    max_bars: int = 60


DEFAULT_RESAMPLES: tuple[ResampleSpec, ...] = (
    ResampleSpec("30m", "30m", 60),
    ResampleSpec("2h", "2h", 60),
    ResampleSpec("4h", "4h", 60),
)


def _ensure_datetime(df: pl.DataFrame) -> pl.DataFrame:
    if "datetime" in df.columns and df["datetime"].dtype != pl.Datetime:
        return df.with_columns(pl.col("datetime").cast(pl.Datetime))
    return df


def load_minute_bars(
    *,
    lab: AlphaLab,
    vt_symbols: list[str],
    end_date: date,
    lookback_days: int,
) -> pl.DataFrame:
    """
    Load minute bars for vt_symbols in [end_date - lookback_days, end_date] (calendar days).
    Use as input for resampling.
    """
    start_date = end_date - timedelta(days=max(1, int(lookback_days)))
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.MINUTE,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        extended_days=0,
    )
    if df is None:
        return pl.DataFrame()
    df = _ensure_datetime(df)
    if df.is_empty():
        return df
    # Normalize expected columns
    cols = df.columns
    price_col = "close_price" if "close_price" in cols else ("close" if "close" in cols else None)
    if price_col and price_col != "close_price":
        df = df.rename({price_col: "close_price"})
    if "open_price" not in cols and "open" in cols:
        df = df.rename({"open": "open_price"})
    if "high_price" not in cols and "high" in cols:
        df = df.rename({"high": "high_price"})
    if "low_price" not in cols and "low" in cols:
        df = df.rename({"low": "low_price"})
    return df


def resample_ohlcv(
    df_minute: pl.DataFrame,
    *,
    spec: ResampleSpec,
) -> pl.DataFrame:
    """
    Resample minute OHLCV into a coarser timeframe per vt_symbol.
    Output columns: vt_symbol, datetime, open, high, low, close, volume
    """
    if df_minute.is_empty():
        return pl.DataFrame()
    df = df_minute.sort(["vt_symbol", "datetime"])
    out = (
        df.group_by_dynamic(
            index_column="datetime",
            every=str(spec.every),
            by="vt_symbol",
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.col("open_price").first().alias("open"),
                pl.col("high_price").max().alias("high"),
                pl.col("low_price").min().alias("low"),
                pl.col("close_price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls(["close"])
        .sort(["vt_symbol", "datetime"])
    )
    if spec.max_bars > 0:
        out = out.group_by("vt_symbol", maintain_order=True).tail(int(spec.max_bars))
    return out


def pack_bars_for_llm(df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Convert OHLCV frame into a compact list of dicts for LLM prompts.
    """
    if df.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    for r in df.iter_rows(named=True):
        dt = r.get("datetime")
        ts = dt.isoformat() if isinstance(dt, datetime) else str(dt)
        rows.append(
            {
                "t": ts,
                "o": float(r.get("open")) if r.get("open") is not None else None,
                "h": float(r.get("high")) if r.get("high") is not None else None,
                "l": float(r.get("low")) if r.get("low") is not None else None,
                "c": float(r.get("close")) if r.get("close") is not None else None,
                "v": float(r.get("volume")) if r.get("volume") is not None else None,
            }
        )
    return rows

