"""
Script to update live data for paper trading.
Downloads yesterday's daily bars from Polygon to update the dataset.
"""
import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from vnpy.trader.constant import Interval
from flagship.paper_trading.config import LAB_PATH
from flagship.scripts.download_backtest_data import download_bars_for_symbols, load_daily_universe
from flagship.scripts.pg_ticker_db import get_pg_connection

def get_last_trading_day() -> date:
    """
    Get the last trading day.
    Simple logic: Yesterday. 
    (In a real production system, this should check a trading calendar API)
    """
    return date.today() - timedelta(days=1)


def _load_daily_selection_vt_symbols(trade_date: date) -> list[str]:
    """从 Postgres 的 daily_selection 读取某天选股 vt_symbol 列表。"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT vt_symbol FROM daily_selection WHERE trade_date = %s",
                (trade_date,),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]


def update_live_data(
    lab_path: Path = LAB_PATH,
    lookback_days: int = 5,
    universe_file: Path | None = None
) -> None:
    """
    Download recent data to ensure the lab is up to date for inference.
    
    Args:
        lab_path: Path to AlphaLab directory
        lookback_days: Number of days to look back and download (to cover weekends/holidays)
        universe_file: Optional path to a JSON universe file. If None, it might need to 
                       use Postgres daily_selection for the last trading day.
    """
    # 1) Determine the effective end_date by probing daily_selection (handles weekends/holidays)
    probe_date = get_last_trading_day()
    end_date = probe_date
    daily_vt_symbols: list[str] = []
    if universe_file is None:
        for _ in range(max(1, lookback_days + 2)):
            try:
                daily_vt_symbols = _load_daily_selection_vt_symbols(probe_date)
            except Exception as exc:
                logger.warning(f"[update_live_data] 读取 daily_selection 失败 {probe_date}: {exc}")
                daily_vt_symbols = []

            if daily_vt_symbols:
                end_date = probe_date
                break
            probe_date = probe_date - timedelta(days=1)

    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"[update_live_data] Updating data from {start_date} to {end_date}")
    
    # 1. Determine Universe
    symbols: list[str] = []
    if universe_file and universe_file.exists():
        symbols = load_daily_universe(universe_file)
        logger.info(f"[update_live_data] Loaded {len(symbols)} symbols from {universe_file}")
    else:
        # Preferred: Use Postgres daily_selection for end_date (static filtered universe U_t)
        if not daily_vt_symbols:
            daily_vt_symbols = _load_daily_selection_vt_symbols(end_date)

        if not daily_vt_symbols:
            logger.error(f"[update_live_data] daily_selection 为空（{end_date}），无法确定更新范围")
            return

        # download_bars_for_symbols expects raw symbol (no exchange suffix)
        symbols = sorted({vt.split(".")[0] for vt in daily_vt_symbols})
        logger.info(f"[update_live_data] Using daily_selection universe: {len(symbols)} symbols ({end_date})")

    if not symbols:
        logger.error("[update_live_data] No symbols found to update. Aborting.")
        return

    # 2. Download Data
    # Reuse the logic from download_backtest_data
    download_bars_for_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=Interval.DAILY,
        lab_dir=lab_path
    )
    
    logger.info("[update_live_data] Data update complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update live data for paper trading")
    parser.add_argument("--lookback", type=int, default=5, help="Days to look back for download")
    args = parser.parse_args()
    
    update_live_data(lookback_days=args.lookback)

