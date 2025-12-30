"""
从 AlphaLab 日线数据批量生成日度选股结果并保存到 PostgreSQL。

基于 Flagship Alpha-Momentum v4.0 策略文档的筛选逻辑：
- ADV >= $2.5亿（成交量中位数 × 收盘价）
- 价格范围：$20 - $600
- MA50趋势过滤（在因子计算阶段应用）
- 相对强度过滤（在因子计算阶段应用）
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
from flagship.scripts.pg_ticker_db import get_pg_connection, get_ref_tickers
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
    """
    从PostgreSQL的ref_tickers表获取所有有效的common stock列表。
    
    Returns:
        包含所有ticker_type='CS'的股票symbol集合（格式：SYMBOL.EXCHANGE）
    """
    try:
        ref_tickers = get_ref_tickers(
            market="stocks",
            locale="us",
            ticker_type="CS",  # 只选择Common Stock
            active=True,
        )
        
        # Exchange映射：从Polygon的exchange代码映射到vnpy的Exchange枚举
        EXCHANGE_MAP = {
            "XNAS": "NASDAQ",
            "NASDAQ": "NASDAQ",
            "XNYS": "NYSE",
            "NYSE": "NYSE",
            "XASE": "AMEX",
            "AMEX": "AMEX",
            "BATS": "BATS",
            "IEXG": "IEX",
        }
        
        # 构建vt_symbol集合（symbol + exchange）
        valid_symbols = set()
        for ticker in ref_tickers:
            symbol = ticker["symbol"]
            primary_exchange = ticker.get("primary_exchange", "")
            
            # 映射exchange
            exchange = EXCHANGE_MAP.get(primary_exchange, "NASDAQ")  # 默认NASDAQ
            
            vt_symbol = f"{symbol}.{exchange}"
            valid_symbols.add(vt_symbol)
        
        logger.info(f"从PostgreSQL加载了 {len(valid_symbols)} 只Common Stock")
        return valid_symbols
    
    except Exception as exc:
        logger.error(f"从PostgreSQL加载ref_tickers失败: {exc}")
        logger.warning("将使用空集合，所有股票将被过滤")
        return set()


def filter_universe_from_lab(
    lab: AlphaLab,
    trade_date: date,
    min_adv_usd: float = 4.0e7,
    min_price: float = 10.0,
    max_price: float = 1000000.0,
    min_market_cap: float = 2.0e9,
    max_market_cap: float = 100.0e9,
) -> list[dict[str, Any]]:
    """
    从 AlphaLab 日线数据筛选指定日期的股票池。
    
    根据策略文档 V7.0 要求：
    - ADV >= $4000万（成交量中位数 × 收盘价）
    - 价格范围：>= $10
    - 市值范围：$2B - $100B
    - 基础趋势：Price > MA50
    - 排除 OTC、ETPs（ETF/ETN）、ADRs
    """
    logger.info(f"[{trade_date}] 开始从 lab 筛选股票池 (V7.0 Aggressive)...")
    
    # 先从PostgreSQL获取所有有效的Common Stock列表
    valid_common_stocks = get_valid_common_stocks()
    if not valid_common_stocks:
        logger.warning(f"[{trade_date}] 未找到有效的Common Stock列表，无法进行筛选")
        return []
    
    # 获取所有日线文件
    daily_files = sorted(lab.daily_path.glob("*.parquet"))
    logger.info(f"[{trade_date}] 发现 {len(daily_files)} 个日线文件")
    
    # 批量获取市值数据
    from flagship.scripts.pg_ticker_db import get_pg_connection
    raw_symbols = [s.split(".")[0] for s in valid_common_stocks]
    market_caps = {}
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # 获取每个 symbol 在 trade_date 或之前的最新市值
            # 注意：这在数万只股票上可能较慢，但 valid_common_stocks 只有 5000 只，可以接受
            cur.execute("""
                SELECT DISTINCT ON (symbol) symbol, market_cap
                FROM ticker_daily_fundamentals
                WHERE symbol = ANY(%s) AND as_of_date <= %s
                ORDER BY symbol, as_of_date DESC;
            """, (raw_symbols, trade_date))
            market_caps = {row[0]: row[1] for row in cur.fetchall() if row[1] is not None}
    
    # 计算需要的历史数据范围（需要过去50天计算MA50）
    lookback_start = trade_date - timedelta(days=90)
    
    passed_symbols: list[dict[str, Any]] = []
    metric_fail = liquidity_fail = price_fail = market_cap_fail = not_cs_fail = trend_fail = 0
    
    for idx, file_path in enumerate(daily_files, start=1):
        vt_symbol = file_path.stem
        
        # 只处理PostgreSQL中标记为Common Stock的股票
        if vt_symbol not in valid_common_stocks:
            not_cs_fail += 1
            continue
        
        # 市值筛选
        raw_symbol = vt_symbol.split(".")[0]
        mkt_cap = market_caps.get(raw_symbol)
        if mkt_cap is None or not (min_market_cap <= mkt_cap <= max_market_cap):
            market_cap_fail += 1
            continue

        try:
            # 读取日线数据
            df = pl.read_parquet(file_path)
            if df.is_empty() or "datetime" not in df.columns:
                metric_fail += 1
                continue
            
            # 过滤日期范围
            trade_date_dt = datetime.combine(trade_date, datetime.min.time())
            lookback_start_dt = datetime.combine(lookback_start, datetime.min.time())
            
            df = df.filter(
                (pl.col("datetime") >= pl.lit(lookback_start_dt)) &
                (pl.col("datetime") <= pl.lit(trade_date_dt))
            ).sort("datetime")
            
            # 检查是否有 trade_date 当天的数据
            trade_date_rows = df.filter(pl.col("datetime").dt.date() == trade_date)
            if trade_date_rows.is_empty():
                metric_fail += 1
                continue
            
            # 至少需要50天的数据计算MA50
            if df.height < 50:
                metric_fail += 1
                continue
            
            # 提取收盘价和成交量
            closes = df["close"].to_list()
            volumes = df["volume"].to_list()
            
            # 计算指标（使用 trade_date 当天的收盘价）
            last_close = closes[-1]
            
            # 取过去30天的成交量中位数
            recent_volumes = volumes[-30:]
            med_vol = statistics.median(recent_volumes)
            adv_usd = med_vol * last_close
            
            # 流动性筛选
            if adv_usd < min_adv_usd:
                liquidity_fail += 1
                continue
            
            # 价格筛选
            if not (min_price <= last_close <= max_price):
                price_fail += 1
                continue
            
            # 基础趋势过滤：Price > MA50
            ma50 = sum(closes[-50:]) / 50.0
            if last_close <= ma50:
                trend_fail += 1
                continue
            
            passed_symbols.append({
                "vt_symbol": vt_symbol,
                "close_price": last_close,
                "adv_usd": adv_usd,
                "med_volume": int(med_vol),
            })
        
        except Exception:
            metric_fail += 1
            continue
        
        if idx % 1000 == 0:
            logger.info(
                f"[{trade_date}] Progress {idx}/{len(daily_files)}: "
                f"passed={len(passed_symbols)}, "
                f"fail:metric={metric_fail},liquidity={liquidity_fail},price={price_fail},cap={market_cap_fail},trend={trend_fail}"
            )
    
    logger.info(
        f"[{trade_date}] 筛选完成: passed={len(passed_symbols)}, "
        f"metric_fail={metric_fail}, liquidity_fail={liquidity_fail}, price_fail={price_fail}, market_cap_fail={market_cap_fail}, trend_fail={trend_fail}"
    )
    
    return passed_symbols


def save_selection_to_postgres(
    trade_date: date,
    selections: list[dict[str, Any]],
) -> None:
    """将选股结果保存到PostgreSQL"""
    if not selections:
        logger.warning(f"[{trade_date}] 没有选股结果，跳过保存")
        return
    
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # 先删除该日期的旧数据（如果存在）
            cur.execute(
                "DELETE FROM daily_selection WHERE trade_date = %s",
                (trade_date,)
            )
            
            # 批量插入新数据
            values = [
                (
                    trade_date,
                    sel["vt_symbol"],
                    sel["close_price"],
                    sel["adv_usd"],
                    sel["med_volume"],
                )
                for sel in selections
            ]
            
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO daily_selection (trade_date, vt_symbol, close_price, adv_usd, med_volume)
                VALUES %s
                ON CONFLICT (trade_date, vt_symbol) DO UPDATE SET
                    close_price = EXCLUDED.close_price,
                    adv_usd = EXCLUDED.adv_usd,
                    med_volume = EXCLUDED.med_volume,
                    created_at = CURRENT_TIMESTAMP
                """,
                values,
            )
            
            logger.info(f"[{trade_date}] 已保存 {len(selections)} 只股票到PostgreSQL")


def build_selection_for_date_range(
    start_date: date,
    end_date: date,
    lab_dir: Path,
    min_adv_usd: float = 4.0e7,
    min_price: float = 10.0,
    max_price: float = 1000000.0,
    min_market_cap: float = 2.0e9,
    max_market_cap: float = 100.0e9,
) -> None:
    """
    为日期范围内的每个交易日生成选股结果并保存到PostgreSQL。
    """
    logger.info("=" * 80)
    logger.info(f"批量生成日度选股结果并保存到PostgreSQL (V7.0 Aggressive)")
    logger.info(f"日期范围: {start_date} 到 {end_date}")
    logger.info(f"Lab 目录: {lab_dir}")
    logger.info(f"筛选参数: ADV >= {min_adv_usd}, Price >= {min_price}, Cap: {min_market_cap}-{max_market_cap}")
    logger.info("=" * 80)
    
    # 创建表（如果不存在）
    create_daily_selection_table()
    
    # 初始化 AlphaLab
    lab = AlphaLab(str(lab_dir))
    
    # 遍历每个交易日
    current_date = start_date
    total_days = 0
    processed_days = 0
    skipped_days = 0
    
    while current_date <= end_date:
        # 跳过周末
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        total_days += 1
        
        try:
            # 筛选股票池
            selections = filter_universe_from_lab(
                lab,
                current_date,
                min_adv_usd=min_adv_usd,
                min_price=min_price,
                max_price=max_price,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
            )
            
            if selections:
                # 保存到PostgreSQL
                save_selection_to_postgres(current_date, selections)
                processed_days += 1
                logger.info(f"[{current_date}] 完成: {len(selections)} 只股票")
            else:
                skipped_days += 1
                logger.warning(f"[{current_date}] 未找到符合条件的股票")
        
        except Exception as exc:
            logger.error(f"[{current_date}] 处理失败: {exc}", exc_info=True)
            skipped_days += 1
        
        # 移动到下一个交易日
        current_date += timedelta(days=1)
    
    logger.info("=" * 80)
    logger.info("批量生成完成:")
    logger.info(f"  - 总交易日数: {total_days}")
    logger.info(f"  - 成功处理: {processed_days}")
    logger.info(f"  - 跳过/失败: {skipped_days}")
    logger.info("=" * 80)


def main() -> None:
    # 重新加载 vt_setting.json
    if VT_SETTING_PATH.exists():
        try:
            setting_data = json.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
            SETTINGS.update(setting_data)
        except Exception as exc:
            logger.warning(f"Failed to reload vt_setting.json: {exc}")
    
    parser = argparse.ArgumentParser(
        description="从 AlphaLab 批量生成日度选股结果并保存到PostgreSQL (V7.0 Aggressive)."
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lab-path",
        type=Path,
        default=DEFAULT_LAB_DIR,
        help="AlphaLab 数据目录（默认 lab/flagship_alpha_momentum）",
    )
    parser.add_argument(
        "--min-adv-usd",
        type=float,
        default=4.0e7,
        help="最小日均成交额（美元，默认 4.0×10^7）",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=10.0,
        help="最低股价（默认 10 USD）",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=1000000.0,
        help="最高股价（默认 1,000,000 USD）",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=2.0e9,
        help="最小市值（默认 2.0e9）",
    )
    parser.add_argument(
        "--max-market-cap",
        type=float,
        default=100.0e9,
        help="最大市值（默认 100.0e9）",
    )
    
    args = parser.parse_args()
    
    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()
    
    build_selection_for_date_range(
        start_date=start_date,
        end_date=end_date,
        lab_dir=args.lab_path,
        min_adv_usd=args.min_adv_usd,
        min_price=args.min_price,
        max_price=args.max_price,
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
    )


if __name__ == "__main__":
    main()

