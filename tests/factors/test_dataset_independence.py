from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.dataset import Segment
from flagship.factors.alpha_momentum.v5_dataset import FlagshipAlphaMomentumV5Dataset
from flagship.factors.alpha_momentum.v7_dataset import FlagshipAlphaMomentumV7Dataset


def _build_sample_df(start_date: datetime, days: int = 20) -> pl.DataFrame:
    rows: list[dict] = []
    symbols = ["AAA.NASDAQ", "BBB.NASDAQ", "SPY.NASDAQ"]

    for i in range(days):
        dt = start_date + timedelta(days=i)
        base = 100 + i
        for sym in symbols:
            offset = 0 if sym == "AAA.NASDAQ" else (10 if sym == "BBB.NASDAQ" else 5)
            open_price = base + offset
            close_price = open_price + 1.0
            high_price = open_price + 5.0
            low_price = open_price - 5.0
            volume = 1_000_000 + offset * 1000 + i * 10
            turnover = close_price * volume
            rows.append(
                {
                    "datetime": dt,
                    "vt_symbol": sym,
                    "open": float(open_price),
                    "high": float(high_price),
                    "low": float(low_price),
                    "close": float(close_price),
                    "volume": float(volume),
                    "turnover": float(turnover),
                }
            )
    return pl.DataFrame(rows)


def test_dataset_independence_and_minimal_processing() -> None:
    assert not issubclass(FlagshipAlphaMomentumV7Dataset, FlagshipAlphaMomentumV5Dataset)

    start = datetime(2024, 1, 1)
    df = _build_sample_df(start, days=20)

    v5 = FlagshipAlphaMomentumV5Dataset(
        df=df,
        train_period=("2024-01-01", "2024-01-10"),
        valid_period=("2024-01-11", "2024-01-15"),
        test_period=("2024-01-16", "2024-01-20"),
    )
    v5.prepare_data(filters=None)
    v5.process_data()
    v5_infer = v5.fetch_infer(Segment.TRAIN)
    assert {"alpha_mom", "alpha_vwap", "alpha_trend", "score", "label"}.issubset(v5_infer.columns)

    v7 = FlagshipAlphaMomentumV7Dataset(
        df=df,
        train_period=("2024-01-01", "2024-01-10"),
        valid_period=("2024-01-11", "2024-01-15"),
        test_period=("2024-01-16", "2024-01-20"),
    )
    v7.prepare_data(filters=None)
    v7.process_data()
    v7_infer = v7.fetch_infer(Segment.TEST)
    assert {"rs_score", "beta", "atr_percent", "score", "label"}.issubset(v7_infer.columns)
