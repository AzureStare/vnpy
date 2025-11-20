"""
同步 Polygon ticker details（每日基本面数据）到 Postgres ticker_daily_fundamentals 表。

基于 Flagship Alpha-Momentum 策略需求：
- 按日期滚动拉取 market_cap 等基本面数据
- 支持批量同步指定日期范围内的所有 tickers
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from polygon.rest import RESTClient

from flagship.config import VT_SETTING_PATH, get_polygon_api_key
from vnpy.trader.logger import logger
from vnpy.trader.setting import SETTINGS
from flagship.scripts.pg_ticker_db import (
    create_ticker_tables,
    get_ref_tickers,
    upsert_ticker_detail,
)


def _fetch_single_ticker_detail(
    api_key: str, symbol: str, as_of_date: date
) -> tuple[str, dict[str, Any] | None, Exception | None]:
    """
    获取单个 ticker 的 detail（用于 multiprocessing）。
    
    Returns:
        (symbol, payload_dict, error)
    """
    try:
        client = RESTClient(api_key)
        detail_obj = client.get_ticker_details(ticker=symbol, date=as_of_date.isoformat())

        # 转换为 dict（处理嵌套对象）
        def to_dict(obj):
            """递归转换对象为 dict，处理嵌套对象和日期"""
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, "__dict__"):
                return {k: to_dict(v) for k, v in vars(obj).items()}
            elif hasattr(obj, "dict"):
                return to_dict(obj.dict())
            elif isinstance(obj, (list, tuple)):
                return [to_dict(item) for item in obj]
            elif hasattr(obj, "isoformat"):  # datetime/date
                return obj.isoformat()
            else:
                return obj

        payload = to_dict(detail_obj)
        return (symbol, payload, None)
    except Exception as exc:
        # 如果是 NOT_FOUND 错误，这是正常的（ticker 在该日期不存在），不记录为错误
        error_str = str(exc)
        if "NOT_FOUND" in error_str or "Ticker not found" in error_str:
            return (symbol, None, None)  # 返回 None error，表示跳过
        return (symbol, None, exc)


def sync_ticker_details_for_date(
    api_key: str,
    symbols: list[str],
    as_of_date: date,
    max_workers: int = 10,
) -> tuple[int, int]:
    """
    为指定日期同步一批 ticker 的 details（使用 multiprocessing 并行获取）。
    
    Args:
        api_key: Polygon API key
        symbols: 股票代码列表
        as_of_date: 日期（当天收盘后的市值）
        max_workers: 并行工作进程数
    
    Returns:
        (success_count, fail_count)
    """
    logger.info(f"Fetching details for {len(symbols)} symbols on {as_of_date} using {max_workers} workers...")
    
    success = 0
    fail = 0
    skipped = 0
    
    # 批量收集结果，然后批量写入数据库（减少数据库连接开销）
    results_batch = []
    batch_size = 100  # 每100条批量写入一次
    
    # 使用 ProcessPoolExecutor 并行获取
    logger.info(f"[sync_ticker_details_for_date] 开始并行获取 {len(symbols)} 个 ticker 的 details")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 创建 future 到 symbol 的映射
        future_to_symbol = {
            executor.submit(_fetch_single_ticker_detail, api_key, symbol, as_of_date): symbol
            for symbol in symbols
        }
        
        completed = 0
        for future in as_completed(future_to_symbol):
            completed += 1
            symbol = future_to_symbol[future]  # 从映射中获取 symbol
            try:
                symbol_result, payload, error = future.result(timeout=30)  # 添加超时
                # 确保 symbol 一致
                if symbol_result != symbol:
                    logger.warning(f"Symbol mismatch: expected {symbol}, got {symbol_result}")
                    symbol = symbol_result
            except Exception as exc:
                logger.error(f"Future result timeout/error for {symbol} on {as_of_date}: {exc}")
                fail += 1
                continue
            
            if error is None and payload is None:
                # NOT_FOUND 错误，ticker 在该日期不存在，跳过
                skipped += 1
            elif error:
                # 其他错误
                error_str = str(error)
                if "NOT_FOUND" in error_str or "Ticker not found" in error_str:
                    skipped += 1
                else:
                    logger.warning(f"Failed to fetch detail for {symbol} on {as_of_date}: {error}")
                    fail += 1
            else:
                # 收集到批量列表
                results_batch.append((symbol, as_of_date, payload))
            
            # 批量写入数据库
            if len(results_batch) >= batch_size:
                batch_success = _batch_upsert_ticker_details(results_batch)
                success += batch_success
                if batch_success != len(results_batch):
                    logger.warning(
                        f"Batch upsert incomplete: {batch_success}/{len(results_batch)} "
                        f"for date {as_of_date}"
                    )
                results_batch = []
            
            if completed % 100 == 0 or completed == len(symbols):
                logger.info(
                    f"Progress {as_of_date}: {completed}/{len(symbols)}, "
                    f"success={success}, skipped={skipped}, fail={fail}"
                )
    
    # 写入剩余的数据
    if results_batch:
        batch_success = _batch_upsert_ticker_details(results_batch)
        success += batch_success
    
    logger.info(
        f"完成 {as_of_date}: success={success}, skipped={skipped}, fail={fail}, "
        f"total={len(symbols)}"
    )
    return success, fail


def _batch_upsert_ticker_details(batch: list[tuple[str, date, dict[str, Any]]]) -> int:
    """
    批量插入或更新 ticker details（使用 execute_batch 提高性能）。
    
    Args:
        batch: [(symbol, as_of_date, detail_data), ...] 列表
    
    Returns:
        成功插入的数量
    """
    import json
    from psycopg2.extras import execute_batch
    from flagship.scripts.pg_ticker_db import get_pg_connection
    
    if not batch:
        return 0
    
    success = 0
    try:
        logger.info(f"[_batch_upsert_ticker_details] 开始批量写入 {len(batch)} 条记录")
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                # 准备批量数据
                values = []
                for symbol, as_of_date, detail_data in batch:
                    values.append((
                        symbol,
                        as_of_date,
                        detail_data.get("market_cap"),
                        detail_data.get("share_class_shares_outstanding"),
                        detail_data.get("weighted_shares_outstanding"),
                        detail_data.get("total_employees"),
                        detail_data.get("sic_code"),
                        detail_data.get("sic_description"),
                        detail_data.get("sector"),
                        detail_data.get("industry"),
                        json.dumps(detail_data),
                    ))
                
                logger.info(f"[_batch_upsert_ticker_details] 准备执行 execute_batch，{len(values)} 条记录")
                # 使用 execute_batch 批量执行（比逐条执行快，支持 ON CONFLICT）
                execute_batch(
                    cur,
                    """
                    INSERT INTO ticker_daily_fundamentals (
                        symbol, as_of_date, market_cap, shares_outstanding,
                        weighted_shares_outstanding, total_employees, sic_code,
                        sic_description, sector, industry, raw, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                        market_cap = EXCLUDED.market_cap,
                        shares_outstanding = EXCLUDED.shares_outstanding,
                        weighted_shares_outstanding = EXCLUDED.weighted_shares_outstanding,
                        total_employees = EXCLUDED.total_employees,
                        sic_code = EXCLUDED.sic_code,
                        sic_description = EXCLUDED.sic_description,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        raw = EXCLUDED.raw,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    values,
                    page_size=100,
                )
                # 上下文管理器会自动提交，但显式提交更安全
                conn.commit()
                success = len(batch)
                logger.info(f"[_batch_upsert_ticker_details] 成功批量写入 {success} 条记录（日期: {batch[0][1] if batch else 'unknown'}）")
    except Exception as exc:
        logger.error(f"Batch upsert failed: {exc}")
        # 如果批量插入失败，回退到逐条插入
        logger.warning("Falling back to individual inserts...")
        for symbol, as_of_date, detail_data in batch:
            try:
                upsert_ticker_detail(symbol, as_of_date, detail_data)
                success += 1
            except Exception as e:
                logger.error(f"Failed to upsert detail for {symbol}: {e}")
    
    return success


def date_range(start: date, end: date) -> list[date]:
    """
    生成日期范围列表（排除周末）。
    
    注意：这里只排除周末，实际交易日可能更少（包含节假日），
    但 Polygon API 会返回该日期最近的有效数据。
    """
    dates = []
    current = start
    while current <= end:
        # 排除周末（Monday=0, Friday=4）
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Polygon ticker details to Postgres ticker_daily_fundamentals table."
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
        "--symbols",
        type=str,
        nargs="*",
        help="指定要同步的 symbol 列表（不指定则同步所有 ref_tickers 中的，建议只同步需要的股票）",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        help="限制同步的 symbol 数量（用于测试，默认全部）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="并行工作进程数（默认 12，适合 M2 Max 12 核心）",
    )
    args = parser.parse_args()

    # 重新加载 vt_setting.json 以确保配置最新
    import json
    if VT_SETTING_PATH.exists():
        try:
            setting_data = json.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
            SETTINGS.update(setting_data)
        except Exception as exc:
            logger.warning(f"Failed to reload vt_setting.json: {exc}")

    # 检查 Postgres 配置
    db_name = SETTINGS.get("database.name", "").lower()
    if db_name != "postgresql":
        logger.error(
            f"SETTINGS['database.name'] must be 'postgresql', got '{db_name}'. "
            f"Please configure vt_setting.json with database.name=postgresql"
        )
        return

    # 自动创建表（如果不存在）
    create_ticker_tables()

    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()
    
    # 获取 API key
    api_key = get_polygon_api_key()

    # 生成日期范围
    dates = date_range(start_date, end_date)
    
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Syncing {len(symbols)} specified symbols")
    else:
        # 从 ref_tickers 读取所有 US stocks
        ref_tickers = get_ref_tickers(market="stocks", locale="us", ticker_type="CS", active=True)
        
        # 根据日期范围过滤：只包含在起始日期之前已上市的 ticker
        # 这样可以减少 NOT_FOUND 错误
        filtered_tickers = []
        for ticker in ref_tickers:
            list_date = ticker.get("list_date")
            if list_date is None or list_date <= start_date:
                filtered_tickers.append(ticker["symbol"])
        
        symbols = filtered_tickers
        
        logger.info(
            f"Filtered tickers: {len(filtered_tickers)}/{len(ref_tickers)} "
            f"(only tickers listed before or on {start_date})"
        )
        
        # 如果指定了限制数量，只取前 N 个（用于测试）
        if args.limit_symbols and args.limit_symbols > 0:
            symbols = symbols[:args.limit_symbols]
            logger.info(f"Limited to first {len(symbols)} symbols (--limit-symbols={args.limit_symbols})")
        else:
            total_calls = len(symbols) * len(dates)
            logger.info(
                f"Will sync {len(symbols)} symbols across {len(dates)} dates = {total_calls:,} API calls"
            )

    if not symbols:
        logger.warning("No symbols to sync")
        return


    dates = date_range(start_date, end_date)
    total_dates = len(dates)
    total_symbols = len(symbols)
    total_api_calls = total_dates * total_symbols
    
    logger.info("=" * 80)
    logger.info(f"开始同步 ticker details")
    logger.info(f"日期范围: {start_date} 到 {end_date}")
    logger.info(f"交易日数: {total_dates}")
    logger.info(f"Ticker 数量: {total_symbols}")
    logger.info(f"预计 API 调用数: {total_api_calls:,}")
    logger.info(f"并行工作进程数: {args.max_workers}")
    logger.info("=" * 80)

    total_success = 0
    total_fail = 0
    processed_dates = 0
    skipped_dates = 0

    # 检查已完成的日期（避免重复处理）
    from flagship.scripts.pg_ticker_db import get_pg_connection
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT as_of_date 
                FROM ticker_daily_fundamentals
                WHERE as_of_date BETWEEN %s AND %s
            """, (start_date, end_date))
            completed_dates = {row[0] for row in cur.fetchall()}
    
    logger.info(f"已完成的日期数: {len(completed_dates)}")
    if completed_dates:
        logger.info(f"已完成的日期范围: {min(completed_dates)} 到 {max(completed_dates)}")

    for idx, as_of_date in enumerate(dates, start=1):
        # 检查该日期是否已完成（如果该日期的记录数足够多，认为已完成）
        if as_of_date in completed_dates:
            with get_pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM ticker_daily_fundamentals 
                        WHERE as_of_date = %s
                    """, (as_of_date,))
                    count = cur.fetchone()[0]
            
            # 如果记录数 >= 4000（足够多，认为数据完整），跳过该日期
            # 或者记录数 >= ticker 数量的 80%（更宽松的阈值）
            if count >= 4000 or count >= len(symbols) * 0.8:
                logger.info(f"[{idx}/{total_dates}] 跳过已完成日期: {as_of_date} ({count:,} 条记录, 阈值: {max(4000, int(len(symbols) * 0.8))})")
                skipped_dates += 1
                continue
        
        logger.info(f"[{idx}/{total_dates}] Processing date: {as_of_date} (当天收盘后的市值)")
        
        # 使用 multiprocessing 并行获取
        success, fail = sync_ticker_details_for_date(
            api_key,
            symbols,
            as_of_date,
            max_workers=args.max_workers,
        )
        total_success += success
        total_fail += fail
        processed_dates += 1
        
        # 每处理 10 个日期输出一次总体进度
        if idx % 10 == 0 or idx == total_dates:
            progress_pct = (idx / total_dates) * 100
            logger.info(
                f"总体进度: {idx}/{total_dates} ({progress_pct:.1f}%) | "
                f"累计成功: {total_success:,} | 累计失败: {total_fail:,} | "
                f"跳过: {skipped_dates}"
            )

    logger.info("=" * 80)
    logger.info(f"同步完成!")
    logger.info(f"处理日期数: {processed_dates}/{total_dates}")
    logger.info(f"跳过日期数: {skipped_dates}")
    logger.info(f"总成功数: {total_success:,}")
    logger.info(f"总失败数: {total_fail:,}")
    logger.info(f"成功率: {(total_success/(total_success+total_fail)*100):.2f}%" if (total_success+total_fail) > 0 else "N/A")
    
    # 最终统计
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM ticker_daily_fundamentals
                WHERE as_of_date BETWEEN %s AND %s
            """, (start_date, end_date))
            final_count = cur.fetchone()[0]
    
    expected_total = len(symbols) * total_dates
    logger.info(f"最终记录数: {final_count:,} / 预期: {expected_total:,} ({final_count/expected_total*100:.2f}%)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

