"""
每日动态股票池筛选脚本。

基于 Flagship Alpha-Momentum 策略文档的筛选逻辑：
- U_t = {i ∈ S | MedVol_{i, t-30:t} * P_{i,t} >= $2.5×10^8 ∩ $20 <= P_{i,t} <= $600}
- 每日滚动更新，输出满足条件的股票列表
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from flagship.config import create_polygon_client
from vnpy.trader.logger import logger
from vnpy.trader.setting import SETTINGS

if TYPE_CHECKING:
    from polygon.rest import RESTClient

from flagship.config import DEFAULT_UNIVERSE_DIR, VT_SETTING_PATH

try:
    from flagship.scripts.pg_ticker_db import (
        get_ref_tickers,
    )
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

DEFAULT_OUTPUT_DIR = DEFAULT_UNIVERSE_DIR


def fetch_daily_bars(
    client: "RESTClient",
    symbol: str,
    trade_date: date,
    lookback_days: int = 60,
) -> list[dict[str, Any]]:
    """
    获取指定日期的历史日线数据（用于计算指标）。
    
    Args:
        client: Polygon RESTClient
        symbol: 股票代码
        trade_date: 交易日期（筛选基准日）
        lookback_days: 回溯天数（默认 60 天，用于计算 30 日成交量中位数和 60 日动量）
    
    Returns:
        日线数据列表，按日期升序排列
    """
    end_date = trade_date
    start_date = trade_date - timedelta(days=lookback_days + 30)  # 多取一些以应对非交易日

    try:
        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date.isoformat(),
            to=end_date.isoformat(),
            limit=50000,
        )

        bars = []
        for agg in aggs:
            bars.append({
                "date": datetime.fromtimestamp(agg.timestamp / 1000).date(),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
                "vwap": agg.vwap if hasattr(agg, "vwap") else None,
            })

        # 按日期排序并过滤到 trade_date 之前
        bars.sort(key=lambda x: x["date"])
        bars = [b for b in bars if b["date"] <= trade_date]
        return bars

    except Exception as exc:
        logger.warning(f"Failed to fetch bars for {symbol} on {trade_date}: {exc}")
        return []


def filter_universe_for_date(
    client: "RESTClient",
    trade_date: date,
    min_adv_usd: float = 2.5e8,
    min_price: float = 20.0,
    max_price: float = 600.0,
    use_postgres: bool = True,
) -> list[dict[str, Any]]:
    """
    为指定日期筛选动态股票池。
    
    筛选条件（基于策略文档）：
    - MedVol_{i, t-30:t} * P_{i,t} >= min_adv_usd（默认 $2.5×10^8）
    - min_price <= P_{i,t} <= max_price（默认 $20 ~ $600）
    
    Returns:
        通过筛选的股票列表，每个元素包含：
        {
            "symbol": str,
            "close": float,
            "adv_usd": float,
            "med_vol": float
        }
    """
    logger.info(f"Filtering universe for {trade_date}...")

    # 从 Postgres 获取所有 US stocks
    if use_postgres and PG_AVAILABLE:
        ref_tickers = get_ref_tickers(
            market="stocks",
            locale="us",
            ticker_type="CS",
            active=True,
        )
        symbols = [t["symbol"] for t in ref_tickers]
        logger.info(f"Loaded {len(symbols)} symbols from Postgres")
    else:
        logger.warning("Postgres not available, falling back to empty list")
        return []

    passed_symbols: list[dict[str, Any]] = []
    metric_fail = liquidity_fail = price_fail = 0

    for idx, symbol in enumerate(symbols, start=1):
        # 获取历史日线数据
        bars = fetch_daily_bars(client, symbol, trade_date, lookback_days=60)
        
        if len(bars) < 30:
            metric_fail += 1
            continue

        # 提取收盘价和成交量
        closes = [b["close"] for b in bars if b.get("close") is not None]
        volumes = [b["volume"] for b in bars if b.get("volume") is not None]

        if len(closes) < 1 or len(volumes) < 30:
            metric_fail += 1
            continue

        # 计算指标
        last_close = closes[-1]
        med_vol = statistics.median(volumes[-30:])  # 过去 30 日成交量中位数
        adv_usd = med_vol * last_close  # 日均成交额

        # 流动性筛选
        if adv_usd < min_adv_usd:
            liquidity_fail += 1
            continue

        # 价格筛选
        if not (min_price <= last_close <= max_price):
            price_fail += 1
            continue

        # 市值不作为筛选条件，仅作为可选信息（如果需要可以后续按需获取）
        passed_symbols.append({
            "symbol": symbol,
            "close": last_close,
            "adv_usd": adv_usd,
            "med_vol": med_vol,
        })

        if idx % 500 == 0:
            logger.info(
                f"Progress {idx}/{len(symbols)}: "
                f"passed={len(passed_symbols)}, "
                f"fail:metric={metric_fail},liquidity={liquidity_fail},price={price_fail}"
            )

    logger.info(
        f"Filter completed for {trade_date}: "
        f"passed={len(passed_symbols)}, "
        f"metric_fail={metric_fail}, liquidity_fail={liquidity_fail}, price_fail={price_fail}"
    )

    return passed_symbols


def save_daily_universe(
    trade_date: date,
    symbols: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    保存每日股票池到文件。
    
    Returns:
        输出文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"universe_{trade_date.isoformat()}.json"

    data = {
        "trade_date": trade_date.isoformat(),
        "symbol_count": len(symbols),
        "symbols": symbols,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(symbols)} symbols to {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily universe based on Flagship Alpha-Momentum strategy criteria."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="交易日期 (YYYY-MM-DD)，例如 2025-11-20",
    )
    parser.add_argument(
        "--min-adv-usd",
        type=float,
        default=2.5e8,
        help="最小日均成交额（美元，默认 2.5×10^8）",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=20.0,
        help="最低股价（默认 20 USD）",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=600.0,
        help="最高股价（默认 600 USD）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录（默认 flagship/data/universe）",
    )
    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="使用 Postgres 读取 ticker 列表和 market_cap",
    )
    args = parser.parse_args()

    # 重新加载 vt_setting.json
    import json as json_lib
    if VT_SETTING_PATH.exists():
        try:
            setting_data = json_lib.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
            SETTINGS.update(setting_data)
        except Exception as exc:
            logger.warning(f"Failed to reload vt_setting.json: {exc}")

    trade_date = datetime.fromisoformat(args.date).date()

    client = create_polygon_client()

    # 筛选股票池
    symbols = filter_universe_for_date(
        client,
        trade_date,
        min_adv_usd=args.min_adv_usd,
        min_price=args.min_price,
        max_price=args.max_price,
        use_postgres=args.use_postgres,
    )

    # 保存结果
    if symbols:
        save_daily_universe(trade_date, symbols, args.output_dir)
        logger.info(f"Daily universe built: {len(symbols)} symbols for {trade_date}")
    else:
        logger.warning(f"No symbols passed filters for {trade_date}")


if __name__ == "__main__":
    main()

