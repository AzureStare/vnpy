"""
从 AlphaLab 日线数据批量生成日度选股结果并保存到 PostgreSQL。
优化版：按股票处理数据，大幅提升多日期范围下的处理速度。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

# 动态注入项目根路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from flagship.config import DEFAULT_LAB_DIR, VT_SETTING_PATH
from flagship.universe.pg_ticker_db import get_pg_connection, get_ref_tickers
from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger
from vnpy.trader.setting import SETTINGS


def create_daily_selection_table() -> None:
    """创建每日选股结果表"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_selection (
                    trade_date DATE NOT NULL,
                    vt_symbol TEXT NOT NULL,
                    close_price DOUBLE PRECISION,
                    adv_usd DOUBLE PRECISION,
                    med_volume BIGINT,
                    market_cap DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (trade_date, vt_symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_daily_selection_date 
                    ON daily_selection(trade_date);
                
                CREATE INDEX IF NOT EXISTS idx_daily_selection_symbol 
                    ON daily_selection(vt_symbol);
            """)
            logger.info("Created table: daily_selection")


def get_valid_common_stocks() -> set[str]:
    """从PostgreSQL获取有效的Common Stock列表"""
    try:
        ref_tickers = get_ref_tickers(market="stocks", locale="us", ticker_type="CS", active=True)
        EXCHANGE_MAP = {"XNAS": "NASDAQ", "NASDAQ": "NASDAQ", "XNYS": "NYSE", "NYSE": "NYSE", "XASE": "AMEX", "AMEX": "AMEX", "BATS": "BATS", "IEXG": "IEX"}
        valid_symbols = set()
        for ticker in ref_tickers:
            symbol = ticker["symbol"]
            primary_exchange = ticker.get("primary_exchange", "")
            exchange = EXCHANGE_MAP.get(primary_exchange, "NASDAQ")
            valid_symbols.add(f"{symbol}.{exchange}")
        logger.info(f"从PostgreSQL加载了 {len(valid_symbols)} 只Common Stock")
        return valid_symbols
    except Exception as exc:
        logger.error(f"从PostgreSQL加载ref_tickers失败: {exc}")
        return set()


def build_selection_optimized(
    start_date: date,
    end_date: date,
    lab_dir: Path,
    min_adv_usd: float = 4.0e7,
    min_price: float = 10.0,
    max_price: float = 1000000.0,
    min_market_cap: float = 2.0e9,
    max_market_cap: float = 100.0e9,
    force: bool = False
) -> None:
    """优化版：支持增量更新，按股票文件读取，大幅提升效率"""
    logger.info("=" * 80)
    logger.info(f"优化版：生成选股结果 ({start_date} 到 {end_date})")
    
    create_daily_selection_table()
    lab = AlphaLab(str(lab_dir))
    valid_common_stocks = get_valid_common_stocks()
    daily_files = sorted(lab.daily_path.glob("*.parquet"))
    
    # 1. 预加载已有的选股结果，用于增量跳过
    existing_keys = set()
    if not force:
        logger.info("正在查询数据库已有的选股记录 (增量模式)...")
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT trade_date, vt_symbol FROM daily_selection 
                    WHERE trade_date >= %s AND trade_date <= %s AND market_cap IS NOT NULL
                """, (start_date, end_date))
                for dt, sym in cur.fetchall():
                    existing_keys.add((dt, sym))
        logger.info(f"数据库已存在 {len(existing_keys)} 条完整记录，将自动跳过。")

    # 2. 批量加载市值数据 (缓存)
    logger.info("正在从数据库预加载市值数据...")
    raw_symbols = [s.split(".")[0] for s in valid_common_stocks]
    market_cap_dict = {} # {symbol: {date: cap}}
    latest_market_caps = {} # {symbol: cap} fallback
    
    # 获取 API Key 用于实时补全历史市值
    api_key = None
    if VT_SETTING_PATH.exists():
        with open(VT_SETTING_PATH, "r") as f:
            api_key = json.load(f).get("datafeed.password")
    
    from flagship.universe.sync_ticker_info import get_market_cap_from_massive_v3
    from flagship.universe.pg_ticker_db import upsert_ticker_detail

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # 尝试获取历史市值
            cur.execute("""
                SELECT symbol, as_of_date, market_cap 
                FROM ticker_daily_fundamentals 
                WHERE symbol = ANY(%s) AND as_of_date >= %s AND as_of_date <= %s
            """, (raw_symbols, start_date - timedelta(days=30), end_date))
            for sym, dt, cap in cur.fetchall():
                if sym not in market_cap_dict: market_cap_dict[sym] = {}
                market_cap_dict[sym][dt] = cap
            
            # 获取每个股票最新的市值作为兜底
            cur.execute("""
                SELECT DISTINCT ON (symbol) symbol, market_cap 
                FROM ticker_daily_fundamentals 
                WHERE symbol = ANY(%s)
                ORDER BY symbol, as_of_date DESC
            """, (raw_symbols,))
            for sym, cap in cur.fetchall():
                latest_market_caps[sym] = cap

    all_results = []
    # Cache per-symbol shares estimate to avoid per-day market cap API calls.
    # We infer shares from (market_cap / close) on the first date we successfully fetch market_cap.
    shares_cache: dict[str, float] = {}
    total_processed = 0
    
    # Debug counters
    fail_reasons = {"no_data": 0, "market_cap": 0, "price": 0, "liquidity": 0, "trend": 0, "skipped": 0}
    
    logger.info(f"开始扫描 {len(daily_files)} 个股票文件...")
    
    for idx, file_path in enumerate(daily_files, start=1):
        vt_symbol = file_path.stem
        if vt_symbol not in valid_common_stocks:
            continue
            
        raw_symbol = vt_symbol.split(".")[0]
        ticker_caps = market_cap_dict.get(raw_symbol, {})
        fallback_cap = latest_market_caps.get(raw_symbol)
        
        try:
            df = pl.read_parquet(file_path).sort("datetime")
            if df.is_empty(): 
                fail_reasons["no_data"] += 1
                continue
            
            # 过滤日期
            mask = (pl.col("datetime").dt.date() >= start_date) & (pl.col("datetime").dt.date() <= end_date)
            target_df = df.filter(mask)
            
            if target_df.is_empty(): 
                fail_reasons["no_data"] += 1
                continue

            # 提前计算指标
            df_with_inds = df.with_columns([
                pl.col("close").alias("last_close"),
                pl.col("volume").rolling_median(window_size=30).alias("med_vol_30"),
                pl.col("close").rolling_mean(window_size=50).alias("ma50")
            ])
            target_df = df_with_inds.filter(mask)
            
            for row in target_df.iter_rows(named=True):
                trade_date = row["datetime"].date()
                
                # 增量检查：如果已经有了，直接跳过
                if (trade_date, vt_symbol) in existing_keys:
                    fail_reasons["skipped"] += 1
                    continue

                # 筛选条件
                # 1. 价格
                if not (min_price <= row["last_close"] <= max_price):
                    fail_reasons["price"] += 1
                    continue
                
                # 2. 流动性
                adv_usd = row["med_vol_30"] * row["last_close"] if row["med_vol_30"] else 0
                if adv_usd < min_adv_usd:
                    fail_reasons["liquidity"] += 1
                    continue
                    
                # 3. 趋势
                if row["ma50"] is None or row["last_close"] <= row["ma50"]:
                    fail_reasons["trend"] += 1
                    continue

                # 4. 市值（只在通过价格/流动性/趋势后才触发，显著减少 API 调用）
                close_px = float(row["last_close"])
                mkt_cap: float | None = None

                shares = shares_cache.get(raw_symbol)
                if shares is not None:
                    mkt_cap = shares * close_px
                else:
                    cap0: float | None = None

                    # Prefer exact market cap for this date if we already have it in DB cache
                    cap_cached = ticker_caps.get(trade_date)
                    if cap_cached is not None:
                        cap0 = float(cap_cached)
                    else:
                        # Fetch once from Massive (point-in-time), then infer shares and reuse
                        if api_key:
                            cap_api = get_market_cap_from_massive_v3(raw_symbol, api_key, trade_date)
                            if cap_api is not None:
                                cap0 = float(cap_api)
                                # Cache to DB for auditing
                                upsert_ticker_detail(raw_symbol, trade_date, {"market_cap": cap0})
                                # Cache to in-memory dict for possible reuse within this run
                                ticker_caps[trade_date] = cap0

                    # Last fallback: use latest known market cap if exists
                    if cap0 is None and fallback_cap is not None:
                        cap0 = float(fallback_cap)

                    if cap0 is not None and close_px > 0:
                        shares = cap0 / close_px
                        shares_cache[raw_symbol] = shares
                        mkt_cap = shares * close_px  # equals cap0 for the anchor date

                if mkt_cap is None or not (min_market_cap <= mkt_cap <= max_market_cap):
                    fail_reasons["market_cap"] += 1
                    continue
                
                all_results.append((
                    trade_date,
                    vt_symbol,
                    row["last_close"],
                    adv_usd,
                    int(row["med_vol_30"]),
                    mkt_cap
                ))
            
            total_processed += 1
            if total_processed % 500 == 0:
                logger.info(f"已处理 {total_processed}/{len(daily_files)} 只股票, 累计选中 {len(all_results)} 条, 跳过 {fail_reasons['skipped']} 条")
                
        except Exception as e:
            logger.warning(f"处理 {vt_symbol} 出错: {e}")

    logger.info(f"扫描结束。最终选中 {len(all_results)} 条记录。")
    logger.info(f"最终失败原因统计: {fail_reasons}")

    # 批量写入数据库
    if all_results:
        logger.info(f"正在将 {len(all_results)} 条记录写入数据库...")
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                from psycopg2.extras import execute_values
                # 分块写入，避免事务过大
                batch_size = 10000
                for i in range(0, len(all_results), batch_size):
                    batch = all_results[i : i + batch_size]
                    execute_values(
                        cur,
                        """
                        INSERT INTO daily_selection (trade_date, vt_symbol, close_price, adv_usd, med_volume, market_cap)
                        VALUES %s
                        ON CONFLICT (trade_date, vt_symbol) DO UPDATE SET
                            close_price = EXCLUDED.close_price,
                            adv_usd = EXCLUDED.adv_usd,
                            med_volume = EXCLUDED.med_volume,
                            market_cap = EXCLUDED.market_cap,
                            created_at = CURRENT_TIMESTAMP
                        """,
                        batch
                    )
        logger.info("数据保存完成。")
    else:
        logger.warning("未找到任何符合条件的选股结果。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--lab-path", type=Path, default=DEFAULT_LAB_DIR)
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7")
    parser.add_argument("--min-adv-usd", type=float)
    parser.add_argument("--min-price", type=float)
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--min-market-cap", type=float)
    parser.add_argument("--max-market-cap", type=float)
    
    args = parser.parse_args()
    
    min_adv_usd = args.min_adv_usd
    min_price = args.min_price
    max_price = args.max_price
    min_market_cap = args.min_market_cap
    max_market_cap = args.max_market_cap

    if args.strategy == "v5":
        if min_adv_usd is None: min_adv_usd = 2.5e8
        if min_price is None: min_price = 20.0
        if max_price is None: max_price = 600.0
        if min_market_cap is None: min_market_cap = 0.0 
        if max_market_cap is None: max_market_cap = float('inf')
    elif args.strategy == "v7":
        if min_adv_usd is None: min_adv_usd = 4.0e7
        if min_price is None: min_price = 10.0
        if max_price is None: max_price = 1000000.0
        if min_market_cap is None: min_market_cap = 2.0e9
        if max_market_cap is None: max_market_cap = 1.0e11

    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()
    
    build_selection_optimized(
        start_date=start_date,
        end_date=end_date,
        lab_dir=args.lab_path,
        min_adv_usd=min_adv_usd,
        min_price=min_price,
        max_price=max_price,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
    )


if __name__ == "__main__":
    main()
