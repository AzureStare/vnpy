import polars as pl
from datetime import date

from flagship.monitoring.backfill_daily_ranking_returns import _compute_returns_for_date


def test_compute_returns_for_date_missing_forward() -> None:
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    prices_aaa = [100.0, 110.0, 121.0, 133.1]
    prices_spy = [100.0, 102.0, 104.04, 106.1208]

    df = pl.DataFrame(
        {
            "vt_symbol": ["AAA"] * len(dates) + ["SPY.NASDAQ"] * len(dates),
            "datetime": dates + dates,
            "close_price": prices_aaa + prices_spy,
        }
    ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))

    out = _compute_returns_for_date(df, trade_date=date(2024, 1, 5), horizons=[1, 3], spy_symbol="SPY.NASDAQ")
    row = out.filter(pl.col("vt_symbol") == "AAA").to_dicts()[0]

    assert abs(float(row["trail_ret_1d"]) - 0.1) < 1e-9
    assert abs(float(row["trail_ret_3d"]) - 0.331) < 1e-9
    assert row["fwd_ret_1d"] is None
    assert row["fwd_ret_3d"] is None
