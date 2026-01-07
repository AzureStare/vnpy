"""
Script to ensure data completeness for selected stocks.
Checks if required history exists in AlphaLab, and downloads missing data if needed.
"""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from flagship.trading.config import LAB_PATH
from flagship.universe.pg_ticker_db import get_pg_connection
from flagship.config import create_polygon_client
from flagship.market_data.update_lab_data_incremental import _download_symbol_bars

def get_daily_selection_from_postgres(trade_date: date) -> list[str]:
    """Fetch symbols selected for a specific date from Postgres"""
    symbols = []
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT vt_symbol FROM daily_selection WHERE trade_date = %s",
                (trade_date,)
            )
            rows = cur.fetchall()
            symbols = [row[0] for row in rows]
    return symbols

def check_and_backfill_data(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    lookback_days: int = 180
) -> None:
    """
    Verify data existence for selected symbols.
    
    Args:
        target_date: The trading date for which we have a selection.
        lab_path: Path to AlphaLab.
        lookback_days: Required history length.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
        
    logger.info(f"[ensure_data_completeness] Checking data for {target_date}")
    
    # 1. Get Universe
    vt_symbols = get_daily_selection_from_postgres(target_date)
    if not vt_symbols:
        logger.warning(f"[ensure_data_completeness] No selection found in Postgres for {target_date}")
        return

    logger.info(f"[ensure_data_completeness] Checking {len(vt_symbols)} symbols...")
    
    # 2. Check Lab Data
    lab = AlphaLab(str(lab_path))
    required_start_date = target_date - timedelta(days=lookback_days)

    missing_or_stale: list[str] = []
    
    for vt_symbol in vt_symbols:
        file_path = lab.daily_path / f"{vt_symbol}.parquet"

        if not file_path.exists():
            missing_or_stale.append(vt_symbol)
            continue
            
        try:
            lf = pl.scan_parquet(file_path)
            min_dt = lf.select(pl.col("datetime").min()).collect().item()
            max_dt = lf.select(pl.col("datetime").max()).collect().item()

            if min_dt is None or max_dt is None:
                missing_or_stale.append(vt_symbol)
                continue
                
            if max_dt.date() < target_date:
                # 最新数据不足（缺失/停更）
                missing_or_stale.append(vt_symbol)
                continue

            if min_dt.date() > required_start_date:
                # 历史长度不足（用于因子/ATR/EMA等）
                missing_or_stale.append(vt_symbol)
                continue

        except Exception as exc:
            logger.warning(f"[ensure_data_completeness] Failed to inspect {vt_symbol}: {exc}")
            missing_or_stale.append(vt_symbol)

    if missing_or_stale:
        logger.warning(
            f"[ensure_data_completeness] Found {len(missing_or_stale)} symbols with missing/stale/short-history data. "
            f"Backfilling {required_start_date} -> {target_date} using incremental downloader..."
        )

        client = create_polygon_client()
        backfilled = 0
        failed = 0
        for idx, vt_symbol in enumerate(missing_or_stale, start=1):
            try:
                bars = _download_symbol_bars(client, vt_symbol, Interval.DAILY, required_start_date, target_date)
            except Exception as exc:
                logger.warning(f"[ensure_data_completeness] Download failed {vt_symbol}: {exc}")
                failed += 1
                continue

            if bars:
                lab.save_bar_data(bars)
                backfilled += 1
            else:
                # Polygon 返回空，可能停牌/退市/不可用
                failed += 1

            if idx % 25 == 0:
                logger.info(
                    f"[ensure_data_completeness] Backfill progress {idx}/{len(missing_or_stale)}: "
                    f"ok={backfilled}, failed={failed}"
                )

        logger.info(f"[ensure_data_completeness] Backfill complete: ok={backfilled}, failed={failed}")
    else:
        logger.info("[ensure_data_completeness] All data is present and up-to-date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure data completeness")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    check_and_backfill_data(target_date=target_dt)

