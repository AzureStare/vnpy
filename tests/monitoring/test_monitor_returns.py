from datetime import datetime

import polars as pl

from flagship.monitoring.app_console_snapshot import _add_return_columns, _pick_price_column




def test_add_return_columns_trailing_and_forward() -> None:
    # 测试场景：两只标的（AAA 与 SPY）给定 6 个交易日收盘价
    # 输入：vt_symbol+datetime+close_price（datetime 为可解析字符串）
    # 期望：输出包含 trail_ret_1d/3d、fwd_ret_1d/3d/5d；
    #      在 signal_date=2024-01-05 上的历史/未来收益与超额收益满足断言
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]
    prices_aaa = [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]
    prices_spy = [100.0, 105.0, 110.25, 115.7625, 121.550625, 127.62815625]

    df = pl.DataFrame(
        {
            "vt_symbol": ["AAA"] * len(dates) + ["SPY.NASDAQ"] * len(dates),
            "datetime": dates + dates,
            "close_price": prices_aaa + prices_spy,
        }
    ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))

    price_col = _pick_price_column(df)
    out = _add_return_columns(df, price_col, windows=[1, 3, 5])

    def _row(symbol: str, dt: str) -> dict:
        dt_value = datetime.fromisoformat(dt)
        row = (
            out.filter((pl.col("vt_symbol") == symbol) & (pl.col("datetime") == dt_value))
            .select(["trail_ret_1d", "trail_ret_3d", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d"])
            .to_dicts()
        )
        return row[0]

    signal_date = "2024-01-05"
    aaa = _row("AAA", signal_date)
    spy = _row("SPY.NASDAQ", signal_date)

    assert abs(float(aaa["trail_ret_1d"]) - 0.1) < 1e-9
    assert abs(float(aaa["trail_ret_3d"]) - 0.331) < 1e-9
    assert abs(float(aaa["fwd_ret_1d"]) - 0.1) < 1e-9
    assert aaa["fwd_ret_3d"] is None
    assert aaa["fwd_ret_5d"] is None

    trail_excess_1d = float(aaa["trail_ret_1d"]) - float(spy["trail_ret_1d"])
    trail_excess_3d = float(aaa["trail_ret_3d"]) - float(spy["trail_ret_3d"])
    assert abs(trail_excess_1d - 0.05) < 1e-9
    assert abs(trail_excess_3d - 0.173375) < 1e-9
