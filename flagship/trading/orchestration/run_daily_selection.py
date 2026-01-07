"""
Script to automate daily stock selection ($U_t$) based on strategy rules.
Wraps the logic of `build_daily_selection_to_postgres.py` for a single day.
"""
import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.logger import logger
from flagship.trading.config import LAB_PATH
from flagship.universe.build_daily_selection import (
    build_selection_optimized,
)

def run_daily_selection(
    target_date: date | None = None,
    lab_path: Path = LAB_PATH,
    strategy_version: str = "v7"
) -> None:
    """
    Run the universe selection logic for a specific date and save to Postgres.
    
    Args:
        target_date: The date to run selection for. Defaults to yesterday (last complete trading day).
        lab_path: Path to AlphaLab.
        strategy_version: "v5" or "v7".
    """
    if target_date is None:
        # Default to yesterday because we need full daily bars to calculate ADV/MedianVolume correctly
        target_date = date.today() - timedelta(days=1)
        
    logger.info(f"[run_daily_selection] Running selection for {target_date} (Strategy: {strategy_version})")

    # Set Filter Criteria based on strategy
    if strategy_version == "v5":
        min_adv_usd = 2.5e8
        min_price = 20.0
        max_price = 600.0
        min_market_cap = 0.0
        max_market_cap = float('inf')
    else:  # v7
        min_adv_usd = 4.0e7
        min_price = 10.0
        max_price = 1000000.0
        min_market_cap = 2.0e9
        max_market_cap = 1.0e11

    try:
        # Use the optimized selection builder (includes table creation + Postgres upsert).
        build_selection_optimized(
            start_date=target_date,
            end_date=target_date,
            lab_dir=lab_path,
            min_adv_usd=min_adv_usd,
            min_price=min_price,
            max_price=max_price,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            force=False,
        )
        logger.info(f"[run_daily_selection] Selection build completed for {target_date}")
            
    except Exception as e:
        logger.error(f"[run_daily_selection] Failed to run selection: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock selection")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7", help="Strategy version")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        
    run_daily_selection(target_date=target_dt, strategy_version=args.strategy)

