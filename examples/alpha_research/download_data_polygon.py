"""
Download Polygon Level-1 data, aggregate to 1-minute bars, and run
Stage-I preprocessing (quality checks + anomaly handling) for the
US mid-frequency high-return strategy.

参考文档：
    docs/strategy/美股中频交易策略(高收益).md - 阶段 I

当前脚本聚焦于：
    1. 通过 vn.py datafeed 抓取 Polygon Tick/分钟数据；
    2. 将 Level-1 数据按 1 分钟聚合，生成基础指标；
    3. 执行缺失值、异常值、数据一致性与时间序列完整性检查；
    4. 生成 `BarData` 列表，供 AlphaLab.save_bar_data 写入实验室。

Step 3 中会调用 AlphaLab 将本脚本产出的 `BarData` 写入 lab 目录。
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import csv
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Sequence

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.object import BarData, HistoryRequest, TickData
from vnpy.trader.utility import extract_vt_symbol


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
LAB_PATH = Path("lab/us_midfreq_high_return")
DEFAULT_VT_SYMBOLS: Sequence[str] = [
    "AAPL.NASDAQ",
    "MSFT.NASDAQ",
]
DEFAULT_START: datetime = datetime(2024, 1, 2)
DEFAULT_END: datetime = datetime(2024, 1, 5)

# Rolling windows required by Stage I (multi-window features)
ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# Liquidity/selection constraints (aligned with strategy doc, realistic 200 万 USD 版本)
MIN_ADV_USD: float = 3e8          # 日均成交额 ADV ≥ 3 亿美元
MIN_PRICE_USD: float = 20.0       # 最低股价 20 美元
MAX_PRICE_USD: float = 500.0      # 最高股价 500 美元
MIN_LISTING_YEARS: int = 3        # 连续上市时间 ≥ 3 年
ADV_LOOKBACK_DAYS: int = 60       # 计算 ADV 的窗口（交易日），约 3 个月


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polygon Level-1 Stage I pipeline (dynamic universe)."
    )
    parser.add_argument(
        "--lab-path",
        type=Path,
        default=LAB_PATH,
        help="目标 AlphaLab 路径，默认 lab/us_midfreq_high_return",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START.date().isoformat(),
        help="开始日期，ISO 格式，如 2024-01-02",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END.date().isoformat(),
        help="结束日期，ISO 格式，如 2024-01-05",
    )
    parser.add_argument(
        "--index-symbol",
        type=str,
        help="指数代码，若 AlphaLab 存有对应 component 则读取其成分股",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="外部股票池文件，每行一个 vt_symbol（或 symbol,exchange）",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="额外指定的 vt_symbol，可重复提供，例如 --symbol AAPL.NASDAQ",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="限制本次下载的最大标的数量，避免一次抓取过多数据",
    )
    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    """Parse ISO date string to naive datetime at 00:00."""
    return datetime.fromisoformat(date_str)


def load_symbols_from_file(file_path: Path) -> list[str]:
    """Load vt_symbols from file; supports `vt`, or `symbol,exchange` formats."""
    if not file_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {file_path}")

    vt_symbols: list[str] = []
    with file_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            row = [item.strip() for item in row if item.strip()]
            if not row:
                continue
            if len(row) == 1 and "." in row[0]:
                vt_symbols.append(row[0])
            elif len(row) >= 2:
                vt_symbols.append(f"{row[0]}.{row[1]}")

    return vt_symbols


def resolve_symbol_universe(
    lab: AlphaLab,
    start: datetime,
    end: datetime,
    args: argparse.Namespace
) -> list[tuple[str, Exchange]]:
    """Build symbol list from index component, file, manual override, fallback."""
    vt_symbols: list[str] = []

    if args.index_symbol:
        vt_symbols.extend(lab.load_component_symbols(args.index_symbol, start, end))

    if args.symbols_file:
        vt_symbols.extend(load_symbols_from_file(args.symbols_file))

    if args.symbol:
        vt_symbols.extend(args.symbol)

    if not vt_symbols:
        vt_symbols.extend(DEFAULT_VT_SYMBOLS)

    deduped: list[str] = sorted({vt.strip() for vt in vt_symbols if vt.strip()})

    pairs: list[tuple[str, Exchange]] = []
    for vt_symbol in deduped:
        symbol, exchange = extract_vt_symbol(vt_symbol)
        pairs.append((symbol, exchange))

    if args.max_symbols:
        pairs = pairs[: args.max_symbols]

    return pairs


def filter_universe_by_liquidity_and_price(
    datafeed,
    universe: Sequence[tuple[str, Exchange]],
    ref_date: datetime,
) -> list[tuple[str, Exchange]]:
    """
    Filter candidate universe by ADV、price range、listing age.

    逻辑与策略文档现实版股票池约束一致：
    - ADV ≥ 3 亿美元；
    - 股价在 [20, 200] 美元；
    - 连续上市时间 ≥ 3 年。
    """
    if not universe:
        return []

    filtered: list[tuple[str, Exchange]] = []

    # 单次请求窗口覆盖：min_listing_years + ADV_LOOKBACK_DAYS，避免多次重复拉取
    history_days: int = max(
        ADV_LOOKBACK_DAYS * 2, MIN_LISTING_YEARS * 365 + 30  # 简单冗余，覆盖非交易日
    )
    start_hist = ref_date - timedelta(days=history_days)
    end_hist = ref_date

    for symbol, exchange in universe:
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            start=start_hist,
            end=end_hist,
            interval=Interval.DAILY,
        )
        bars = datafeed.query_bar_history(req) or []
        if not bars:
            print(f"[INFO] Skip {symbol}.{exchange.value}: no daily history for liquidity filter.")
            continue

        # 按 ADV_LOOKBACK_DAYS 计算近段时间的日均成交额（成交额 ≈ 收盘价 × 成交量）
        recent_bars = bars[-ADV_LOOKBACK_DAYS:] if len(bars) > ADV_LOOKBACK_DAYS else bars
        turnovers = [
            (bar.close_price or 0) * (bar.volume or 0) for bar in recent_bars
        ]
        valid_turnovers = [t for t in turnovers if t > 0]
        if not valid_turnovers:
            print(f"[INFO] Skip {symbol}.{exchange.value}: no valid turnover for ADV.")
            continue

        adv = sum(valid_turnovers) / float(len(valid_turnovers))
        last_close = recent_bars[-1].close_price or 0

        # 简单估算连续上市时长（用可获取到的最早日线时间差近似）
        listing_days = (bars[-1].datetime.date() - bars[0].datetime.date()).days

        if adv < MIN_ADV_USD:
            print(
                f"[INFO] Filtered out {symbol}.{exchange.value}: "
                f"ADV={adv:,.0f} < {MIN_ADV_USD:,.0f}."
            )
            continue

        if not (MIN_PRICE_USD <= last_close <= MAX_PRICE_USD):
            print(
                f"[INFO] Filtered out {symbol}.{exchange.value}: "
                f"price={last_close:.2f} not in [{MIN_PRICE_USD}, {MAX_PRICE_USD}]."
            )
            continue

        if listing_days < MIN_LISTING_YEARS * 365:
            print(
                f"[INFO] Filtered out {symbol}.{exchange.value}: "
                f"listing_days={listing_days} < {MIN_LISTING_YEARS * 365}."
            )
            continue

        filtered.append((symbol, exchange))

    print(
        f"[INFO] Liquidity/price filter kept {len(filtered)} / {len(universe)} symbols "
        f"for Stage I download."
    )
    return filtered


# -----------------------------------------------------------------------------
# Data acquisition helpers
# -----------------------------------------------------------------------------
def init_polygon_datafeed() -> tuple[bool, object]:
    """Initialize Polygon datafeed."""
    datafeed = get_datafeed()
    ok: bool = datafeed.init()
    return ok, datafeed


def query_ticks(
    datafeed,
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime,
) -> list[TickData]:
    """Fetch raw Level-1 ticks if available."""
    req = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        interval=Interval.TICK,
    )
    ticks = datafeed.query_tick_history(req)
    return ticks or []


def query_minute_bars(
    datafeed,
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime,
) -> list[BarData]:
    """Fallback: directly request minute bars."""
    req = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        interval=Interval.MINUTE,
    )
    bars = datafeed.query_bar_history(req)
    return bars or []


# -----------------------------------------------------------------------------
# Stage I preprocessing
# -----------------------------------------------------------------------------
def ticks_to_polars(ticks: Sequence[TickData]) -> pl.DataFrame:
    """Convert TickData list to polars DataFrame with Level-1 columns."""
    if not ticks:
        return pl.DataFrame()

    rows: list[dict] = []
    for tick in ticks:
        rows.append(
            {
                "datetime": tick.datetime.replace(tzinfo=None),
                "bid_price": tick.bid_price_1,
                "ask_price": tick.ask_price_1,
                "bid_size": tick.bid_volume_1,
                "ask_size": tick.ask_volume_1,
                "last_price": tick.last_price,
                "last_volume": tick.last_volume,
            }
        )

    return pl.DataFrame(rows)


def bars_to_polars(bars: Sequence[BarData]) -> pl.DataFrame:
    """Convert BarData list to polars DataFrame for quality checks."""
    if not bars:
        return pl.DataFrame()

    rows: list[dict] = []
    for bar in bars:
        rows.append(
            {
                "datetime": bar.datetime.replace(tzinfo=None),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
        )

    return pl.DataFrame(rows)


def aggregate_level1_to_minute(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate Level-1 data to 1-minute OHLC bars with additional features.

    聚合公式参考策略文档 1.6 节。
    """
    if df.is_empty():
        return pl.DataFrame()

    df = df.with_columns(
        pl.col("datetime").dt.truncate("1m").alias("minute"),
        pl.max_horizontal(["bid_price", "ask_price"]).alias("pair_high"),
        pl.min_horizontal(["bid_price", "ask_price"]).alias("pair_low"),
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        (pl.col("bid_size") + pl.col("ask_size")).alias("depth_volume"),
        pl.when(pl.col("bid_size") + pl.col("ask_size") > 0)
        .then((pl.col("bid_size") - pl.col("ask_size")) / (pl.col("bid_size") + pl.col("ask_size")))
        .otherwise(0)
        .alias("quote_imbalance"),
        pl.col("spread").diff().abs().fill_null(0).alias("quote_velocity"),
    )

    agg = (
        df.group_by("minute", maintain_order=True)
        .agg(
            [
                pl.col("bid_price").first().alias("open_price"),
                pl.col("pair_high").max().alias("high_price"),
                pl.col("pair_low").min().alias("low_price"),
                pl.col("ask_price").last().alias("close_price"),
                pl.col("depth_volume").sum().alias("volume"),
                pl.col("spread").mean().alias("spread"),
                pl.col("quote_imbalance").mean().alias("quote_imbalance"),
                pl.col("quote_velocity").mean().alias("quote_velocity"),
            ]
        )
        .sort("minute")
    )

    return append_rolling_features(agg)


def append_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add multi-window rolling statistics."""
    if df.is_empty():
        return df

    rolling_cols = []
    for window in ROLLING_WINDOWS:
        rolling_cols.extend(
            [
                pl.col("close_price")
                .rolling_mean(window)
                .alias(f"close_mean_{window}"),
                pl.col("close_price")
                .rolling_std(window)
                .alias(f"close_std_{window}"),
                pl.col("volume")
                .rolling_mean(window)
                .alias(f"volume_mean_{window}"),
            ]
        )

    return df.with_columns(rolling_cols)


def check_missing_rate(df: pl.DataFrame) -> float:
    """Calculate missing rate vs. continuous 1-minute index."""
    if df.is_empty():
        return 1.0

    start = df["minute"][0]
    end = df["minute"][-1]
    expected = int(((end - start).total_seconds() // 60) + 1)
    missing = max(expected - df.height, 0)
    return missing / expected if expected else 0.0


def detect_outliers(df: pl.DataFrame, column: str) -> pl.Series:
    """Z-score > 5 detection."""
    stats = df.select(
        [
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
        ]
    )

    mean = stats["mean"][0]
    std = stats["std"][0] or 1e-9
    return ((pl.col(column) - mean) / std).abs() > 5.0


def clip_extremes(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Winsorize 1% tails + clipping."""
    quantiles = df.select(
        [
            pl.col(column).quantile(0.01).alias("q01"),
            pl.col(column).quantile(0.99).alias("q99"),
        ]
    )

    q01 = quantiles["q01"][0]
    q99 = quantiles["q99"][0]

    return df.with_columns(pl.col(column).clip(q01, q99).alias(column))


def enforce_quality_pipeline(df: pl.DataFrame) -> pl.DataFrame:
    """Apply Stage I checks: missing rate, anomalies, data consistency."""
    if df.is_empty():
        return df

    missing_rate = check_missing_rate(df)
    if missing_rate > 0.05:
        print(f"[WARN] Missing rate {missing_rate:.2%} exceeds 5% threshold.")

    # Outlier detection & clipping
    for col in ("close_price", "volume", "spread"):
        if col not in df.columns:
            continue
        mask = df.select(detect_outliers(df, col).alias("mask"))["mask"]
        if mask.any():
            print(f"[WARN] Detected outliers in {col}; applying clipping.")
            df = clip_extremes(df, col)

    # Data consistency checks (Ask > Bid, Volume >= 0)
    inconsistent = df.filter((pl.col("high_price") < pl.col("low_price")) | (pl.col("volume") < 0))
    if not inconsistent.is_empty():
        print(f"[WARN] Found {inconsistent.height} inconsistent records; correcting volume to abs.")
        df = df.with_columns(pl.col("volume").abs())

    return df


def minute_df_to_bars(
    df: pl.DataFrame,
    symbol: str,
    exchange: Exchange,
) -> list[BarData]:
    """Convert minute-level DataFrame to BarData list."""
    bars: list[BarData] = []

    for row in df.iter_rows(named=True):
        minute: datetime = row["minute"].to_pydatetime() if hasattr(row["minute"], "to_pydatetime") else row["minute"]
        turnover = row["close_price"] * row["volume"]
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=minute,
            interval=Interval.MINUTE,
            volume=row["volume"],
            turnover=turnover,
            open_interest=0,
            open_price=row["open_price"],
            high_price=row["high_price"],
            low_price=row["low_price"],
            close_price=row["close_price"],
            gateway_name="POLYGON",
        )
        bars.append(bar)

    return bars


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def fetch_clean_bars(
    datafeed,
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime,
) -> list[BarData]:
    """
    Fetch raw data, aggregate to 1-minute, run quality pipeline,
    and convert to BarData.
    """
    ticks = query_ticks(datafeed, symbol, exchange, start, end)

    if ticks:
        df = ticks_to_polars(ticks)
        minute_df = aggregate_level1_to_minute(df)
    else:
        print(f"[INFO] Tick data unavailable for {symbol}, fallback to minute bars.")
        bars = query_minute_bars(datafeed, symbol, exchange, start, end)
        df = bars_to_polars(bars)
        if df.is_empty():
            return []
        df = df.rename(
            {
                "open": "open_price",
                "high": "high_price",
                "low": "low_price",
                "close": "close_price",
            }
        )
        df = append_rolling_features(df)
        minute_df = df.rename({"datetime": "minute"})

    clean_df = enforce_quality_pipeline(minute_df)
    return minute_df_to_bars(clean_df, symbol, exchange)


def validate_bar_sequence(bars: Sequence[BarData]) -> None:
    """Basic structural validation for generated bars."""
    if not bars:
        return

    violations = 0
    prev_dt: datetime | None = None
    for bar in bars:
        if bar.open_price > bar.high_price or bar.close_price < bar.low_price:
            violations += 1
        if prev_dt and bar.datetime <= prev_dt:
            violations += 1
        prev_dt = bar.datetime

    if violations:
        print(f"[WARN] Detected {violations} basic OHLC/time violations.")


def resample_minute_to_daily(bars: Sequence[BarData]) -> list[BarData]:
    """Aggregate minute bars into daily bars for AlphaLab daily storage."""
    if not bars:
        return []

    df = pl.DataFrame(
        {
            "datetime": [bar.datetime for bar in bars],
            "open": [bar.open_price for bar in bars],
            "high": [bar.high_price for bar in bars],
            "low": [bar.low_price for bar in bars],
            "close": [bar.close_price for bar in bars],
            "volume": [bar.volume for bar in bars],
            "turnover": [bar.turnover for bar in bars],
        }
    ).with_columns(
        pl.col("datetime").dt.date().alias("date")
    )

    daily = (
        df.group_by("date", maintain_order=True)
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("turnover").sum().alias("turnover"),
            ]
        )
        .sort("date")
    )

    result: list[BarData] = []
    for row in daily.iter_rows(named=True):
        dt = datetime.combine(row["date"], time())
        bar = BarData(
            symbol=bars[0].symbol,
            exchange=bars[0].exchange,
            datetime=dt,
            interval=Interval.DAILY,
            volume=row["volume"],
            turnover=row["turnover"],
            open_interest=0,
            open_price=row["open"],
            high_price=row["high"],
            low_price=row["low"],
            close_price=row["close"],
            gateway_name="POLYGON",
        )
        result.append(bar)

    return result


def log_symbol_summary(vt_symbol: str, bars: Sequence[BarData]) -> None:
    if not bars:
        print(f"[WARN] No cleaned bars for {vt_symbol}")
        return

    start = bars[0].datetime
    end = bars[-1].datetime
    print(f"{vt_symbol}: {len(bars)} minute bars from {start} to {end}")


def main() -> None:
    args = parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end)

    ok, datafeed = init_polygon_datafeed()
    if not ok:
        print("Failed to initialize Polygon datafeed. Check vt_setting.json and API key.")
        return

    lab = AlphaLab(str(args.lab_path))

    # 1) 构建候选股票池（指数成分、外部文件、手动指定等）
    universe = resolve_symbol_universe(lab, start, end, args)
    if not universe:
        print("No symbols resolved for download; please provide --index-symbol or --symbols-file.")
        return

    # 2) 按 ADV / 价格 / 上市时长过滤，确保符合容量与流动性约束
    universe = filter_universe_by_liquidity_and_price(datafeed, universe, start)
    if not universe:
        print(
            "[WARN] All symbols filtered out by liquidity/price constraints; "
            "please relax thresholds or adjust base universe."
        )
        return

    print(
        f"[INFO] Downloading Stage I data for {len(universe)} symbols "
        f"between {start.date()} and {end.date()}."
    )

    total_bars = 0
    for symbol, exchange in universe:
        bars = fetch_clean_bars(datafeed, symbol, exchange, start, end)
        total_bars += len(bars)
        vt_symbol = f"{symbol}.{exchange.value}"
        validate_bar_sequence(bars)
        log_symbol_summary(vt_symbol, bars)

        if bars:
            lab.save_bar_data(bars)
            daily_bars = resample_minute_to_daily(bars)
            if daily_bars:
                lab.save_bar_data(daily_bars)

    print(f"Total aggregated bars: {total_bars}")


if __name__ == "__main__":
    main()

