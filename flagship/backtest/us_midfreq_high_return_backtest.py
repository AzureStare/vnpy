from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import polars as pl

from flagship.config import PROJECT_ROOT

# Ensure we import the local vnpy package (with AlphaLab stubs) instead of site-packages
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy


def load_signal(lab: AlphaLab, name: str) -> pl.DataFrame:
    """
    Load model signal for backtesting.

    Expect a Parquet file saved by AlphaLab.save_signal with
    at least the following columns:
        - datetime: timestamp without timezone
        - vt_symbol: contract code, e.g. 'AAPL.NASDAQ'
        - signal: numeric score, higher is better
    """
    signal_df: pl.DataFrame | None = lab.load_signal(name)
    if signal_df is None:
        raise RuntimeError(f"Signal file not found for name={name!r}")

    required_columns = {"datetime", "vt_symbol", "signal"}
    missing = required_columns.difference(signal_df.columns)
    if missing:
        raise RuntimeError(f"Signal DataFrame missing columns: {sorted(missing)}")

    # Ensure proper dtypes and sorting
    signal_df = signal_df.with_columns(
        pl.col("datetime").cast(pl.Datetime, strict=False),
        pl.col("vt_symbol").cast(pl.Utf8),
        pl.col("signal").cast(pl.Float64),
    ).sort(["datetime", "vt_symbol"])

    return signal_df


def infer_vt_symbols_from_lab(lab: AlphaLab, interval: Interval) -> list[str]:
    """Infer vt_symbols from lab storage when signal/universe is missing."""
    folder = lab.minute_path if interval == Interval.MINUTE else lab.daily_path
    return sorted([file.stem for file in folder.glob("*.parquet")])


def generate_naive_signal(
    lab: AlphaLab,
    vt_symbols: list[str],
    interval: Interval,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """
    Build a placeholder signal using simple rolling returns so that
    the backtest pipeline can be executed even before模型完成训练.
    """
    rows: list[dict] = []
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(vt_symbol, interval, start, end)
        if not bars:
            continue

        df = pl.DataFrame(
            {
                "datetime": [bar.datetime.replace(tzinfo=None) for bar in bars],
                "close": [bar.close_price for bar in bars],
            }
        ).sort("datetime")

        df = df.with_columns(
            pl.col("close")
            .pct_change()
            .fill_null(0.0)
            .alias("ret")
        ).with_columns(
            pl.col("ret")
            .rolling_mean(15, min_periods=5)
            .alias("score")
        )

        vt_series = (
            df.select(["datetime", "score"])
            .drop_nulls("score")
            .with_columns(pl.lit(vt_symbol).alias("vt_symbol"))
        )

        rows.extend(vt_series.to_dicts())

    if not rows:
        raise RuntimeError("Unable to generate fallback signal; no bar data available.")

    return pl.DataFrame(rows).select(["datetime", "vt_symbol", "score"]).rename({"score": "signal"})


def run_backtest() -> None:
    """
    Run backtest for the US mid-frequency high-return equity strategy.

    This script reuses the vn.py alpha-research pipeline:
        1) Use AlphaLab to manage data/model/signal files.
        2) Load precomputed model signal (Score_t in the design document)
           as the 'signal' column.
        3) Use BacktestingEngine + EquityDemoStrategy for portfolio
           construction and rebalancing.

    Before running this script, prepare the following:
        - Minute/Daily bar data under lab_path (AlphaLab.save_bar_data).
        - Universe components saved via AlphaLab.save_component_data
          if you plan to use an index universe.
        - Model signal saved via AlphaLab.save_signal with name
          'us_midfreq_high_return' (or adjust 'name' below).
    """
    # ---- User configuration (adjust these paths and parameters) ----
    project_root = Path(__file__).resolve().parents[2]
    lab_path = project_root.joinpath("lab/us_midfreq_high_return")

    name: str = "us_midfreq_high_return"

    # Backtest universe and benchmark; adjust according to your data.
    # For example, you can use an index symbol whose components you
    # have stored via AlphaLab.save_component_data.
    index_symbol: str | None = None  # e.g. "SPY.US"
    vt_symbols_override: list[str] | None = ["AAPL.NASDAQ", "MSFT.NASDAQ"]

    start: datetime = datetime(2024, 1, 2)
    end: datetime = datetime(2024, 1, 10)
    interval: Interval = Interval.MINUTE  # or Interval.DAILY

    initial_capital: int = 1_000_000
    risk_free: float = 0.02
    annual_days: int = 252

    # Strategy parameters for EquityDemoStrategy.
    # These roughly correspond to:
    #   - top_k: target number of holdings (N_hold in the document).
    #   - n_drop: number of lowest-score holdings to drop at each rebalance.
    #   - min_days: minimum holding period to avoid over-trading.
    strategy_setting: dict = {
        "top_k": 30,
        "n_drop": 5,
        "min_days": 1,
        "cash_ratio": 1.2,      # allow modest leverage at portfolio level
        "min_volume": 1,
        "open_rate": 0.0005,
        "close_rate": 0.0015,
        "min_commission": 1,
        "price_add": 0.0005,
    }

    # ---- Initialize lab and load signal ----
    lab = AlphaLab(str(lab_path))

    vt_symbols: list[str] = []
    signal_df: pl.DataFrame | None = None

    try:
        signal_df = load_signal(lab, name)
        vt_symbols = sorted(signal_df["vt_symbol"].unique().to_list())
    except RuntimeError:
        if vt_symbols_override:
            vt_symbols = vt_symbols_override
        elif index_symbol:
            vt_symbols = lab.load_component_symbols(index_symbol, start, end)
        else:
            vt_symbols = infer_vt_symbols_from_lab(lab, interval)

        signal_df = generate_naive_signal(lab, vt_symbols, interval, start, end)
        print("[INFO] Using fallback rolling-return signal; replace with trained model output when available.")

    if index_symbol is not None:
        vt_symbols = lab.load_component_symbols(index_symbol, start, end)

    if not vt_symbols:
        raise RuntimeError("Empty vt_symbols universe for backtest")

    # ---- Configure backtest engine ----
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=interval,
        start=start,
        end=end,
        capital=initial_capital,
        risk_free=risk_free,
        annual_days=annual_days,
    )

    # ---- Add strategy and run backtest ----
    engine.add_strategy(EquityDemoStrategy, strategy_setting, signal_df)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    # Print key statistics
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Show net value and drawdown chart
    engine.show_chart()


if __name__ == "__main__":
    run_backtest()


