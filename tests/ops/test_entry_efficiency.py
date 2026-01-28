from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from flagship.ops.analysis.entry_efficiency import detect_vwap_reclaim


def _build_minute_df(closes: list[float]) -> pl.DataFrame:
    base_time = datetime(2026, 1, 2, 9, 30)
    datetimes = [base_time + timedelta(minutes=i) for i in range(len(closes))]
    volumes = [100 for _ in closes]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
        }
    )


def test_detect_vwap_reclaim_triggered() -> None:
    minute_df = _build_minute_df([10, 10, 12, 12, 12])
    triggered, idx = detect_vwap_reclaim(
        minute_df,
        hold_minutes=2,
        cutoff_time=datetime(2026, 1, 2, 10, 30).time(),
        require_below_before_reclaim=False,
    )
    assert triggered is True
    assert idx == 3


def test_detect_vwap_reclaim_not_triggered() -> None:
    minute_df = _build_minute_df([10, 10, 10, 10])
    triggered, idx = detect_vwap_reclaim(
        minute_df,
        hold_minutes=2,
        cutoff_time=datetime(2026, 1, 2, 10, 30).time(),
        require_below_before_reclaim=False,
    )
    assert triggered is False
    assert idx is None


def test_detect_vwap_reclaim_strict_requires_below_first() -> None:
    # Starts above VWAP -> legacy might trigger, strict must not.
    minute_df = _build_minute_df([12, 12, 12, 12, 12])
    legacy, legacy_idx = detect_vwap_reclaim(
        minute_df,
        hold_minutes=2,
        cutoff_time=datetime(2026, 1, 2, 10, 30).time(),
        require_below_before_reclaim=False,
    )
    strict, strict_idx = detect_vwap_reclaim(
        minute_df,
        hold_minutes=2,
        cutoff_time=datetime(2026, 1, 2, 10, 30).time(),
        require_below_before_reclaim=True,
    )
    assert legacy in (True, False)  # legacy behavior depends on VWAP path
    assert legacy_idx is None or isinstance(legacy_idx, int)
    assert strict is False
    assert strict_idx is None
