from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from flagship.factors.alpha_momentum.v7_dataset import FlagshipAlphaMomentumV7Dataset


def _forward_ret_5d(closes: list[float]) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(closes)):
        j = i + 5
        if j >= len(closes):
            out.append(None)
        else:
            out.append(float(closes[j] / closes[i] - 1.0))
    return out


def test_v7_post_process_adds_label_excess_5d() -> None:
    base = datetime(2025, 1, 1)
    dates = [base + timedelta(days=i) for i in range(10)]

    spy_close = [100.0 + float(i) for i in range(10)]
    aaa_close = [50.0 + float(i) for i in range(10)]

    spy_ret_5d = _forward_ret_5d(spy_close)
    aaa_ret_5d = _forward_ret_5d(aaa_close)

    rows: list[dict[str, object]] = []
    for dt, close, ret5 in zip(dates, spy_close, spy_ret_5d):
        rows.append(
            {
                "datetime": dt,
                "vt_symbol": "SPY.NASDAQ",
                "close_price": close,
                "volume": 1_000_000.0,
                "atr_14": close * 0.05,  # atr_percent=5% (>3% filter)
                "alpha_mom": 1.0,
                "alpha_vwap": 0.5,
                "return_5d": 0.01,
                "ret_5d": ret5,
            }
        )

    for dt, close, ret5 in zip(dates, aaa_close, aaa_ret_5d):
        rows.append(
            {
                "datetime": dt,
                "vt_symbol": "AAA.NASDAQ",
                "close_price": close,
                "volume": 2_000_000.0,
                "atr_14": close * 0.05,
                "alpha_mom": 1.2,
                "alpha_vwap": 0.4,
                "return_5d": 0.02,
                "ret_5d": ret5,
            }
        )

    df = pl.DataFrame(rows)
    dataset = FlagshipAlphaMomentumV7Dataset(
        df=df,
        train_period=("2025-01-01", "2025-01-03"),
        valid_period=("2025-01-04", "2025-01-06"),
        test_period=("2025-01-07", "2025-01-10"),
    )

    out = dataset._post_process_v7(df)
    assert "label_excess_5d" in out.columns

    # Pick first date where forward 5d return exists
    d0 = dates[0]
    got = (
        out.filter((pl.col("vt_symbol") == "AAA.NASDAQ") & (pl.col("datetime") == d0))
        .select("label_excess_5d")
        .item()
    )

    expected = float(aaa_ret_5d[0] - spy_ret_5d[0])  # both not None for day0
    assert got is not None
    assert abs(float(got) - expected) < 1e-8

