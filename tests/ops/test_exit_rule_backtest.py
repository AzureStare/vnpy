from __future__ import annotations

from datetime import date, datetime, time as dtime

import polars as pl

from flagship.ops.analysis.entry_efficiency import EntryConfig
from flagship.ops.analysis.exit_rule_backtest import (
    ExitRuleConfig,
    _compute_vwap_by_day,
    _find_reclaim_entry,
    _simulate_exit,
)


def _minute_df(
    trade_date: date,
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pl.DataFrame:
    timestamps = [
        datetime.combine(trade_date, dtime(9, 30)).replace(minute=30 + i) for i in range(len(closes))
    ]
    highs = highs or [c for c in closes]
    lows = lows or [c for c in closes]
    return pl.DataFrame(
        {
            "datetime": timestamps,
            "trade_date": [trade_date] * len(closes),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * len(closes),
        }
    )


def test_find_reclaim_entry_strict() -> None:
    # 测试场景：严格回踩入场（require_below_before_reclaim=True）
    # 输入：分钟级 close 序列从 10 -> 11（制造回踩后上穿）
    # 期望：触发入场 triggered=True，且 entry_idx >= 2
    trade_date = date(2025, 1, 2)
    df = _minute_df(trade_date, [10.0, 10.0, 11.0, 11.0, 11.0])
    entry_config = EntryConfig(hold_minutes=2, cutoff_time=dtime(10, 30), require_below_before_reclaim=True)
    triggered, entry_idx = _find_reclaim_entry(df, entry_config)
    assert triggered is True
    assert entry_idx is not None
    assert entry_idx >= 2


def test_simulate_exit_vwap_exit() -> None:
    trade_date = date(2025, 1, 3)
    closes = [10.0, 9.6, 9.6, 9.6, 11.0]
    highs = [c + 1.0 for c in closes]
    df = _minute_df(trade_date, closes, highs=highs)
    vwap_values = _compute_vwap_by_day(df)
    timestamps = df.get_column("datetime").to_list()
    trade_dates = df.get_column("trade_date").to_list()
    open_prices = df.get_column("open").to_list()
    high_prices = df.get_column("high").to_list()
    low_prices = df.get_column("low").to_list()
    close_prices = df.get_column("close").to_list()

    entry_config = EntryConfig(hold_minutes=2, cutoff_time=dtime(10, 30), require_below_before_reclaim=True)
    rule = ExitRuleConfig(
        name="vwap_exit",
        hard_stop_loss_pct=0.07,
        stop_loss_atr_multiplier=2.5,
        profit_threshold_trend=0.05,
        profit_threshold_win=0.15,
        spike_threshold=0.10,
        trailing_stop_pct=0.10,
        trend_ema_period=10,
        trailing_ema_period=5,
        vwap_exit_confirm_minutes=2,
    )

    result = _simulate_exit(
        timestamps=timestamps,
        trade_dates=trade_dates,
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        vwap_values=vwap_values,
        ema_map={
            "ema5": [None] * len(closes),
            "ema10": [None] * len(closes),
            "ema20": [None] * len(closes),
        },
        entry_config=entry_config,
        rule_config=rule,
        entry_variant="strict_reclaim",
        entry_idx=0,
        reclaim_entry_idx=None,
        max_holding_days=1,
    )

    assert result.exit_reason == "vwap_exit"
    assert result.exit_time is not None
