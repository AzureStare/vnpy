"""
Alpaca open rebalance CLI (thin wrapper).

Core logic lives in:
- `flagship/trading/execution/open_rebalance.py`
- `flagship/trading/realtime/polygon_ws.py` (optional trade price cache)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vnpy.trader.logger import logger

# Ensure project root importable under docker/cron
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flagship.config.polygon_config import get_polygon_api_key
from flagship.trading.execution.broker_alpaca import AlpacaAdapter
from flagship.trading.config import DAILY_SIGNAL_FILE
from flagship.trading.execution.open_rebalance import StrategyRunner, execute_rebalance
from flagship.trading.realtime.polygon_ws import PolygonTradePriceCache


def main():
    parser = argparse.ArgumentParser(description="Execute Flagship Alpha-Momentum trades on Alpaca Paper.")
    parser.add_argument(
        "--wait-for-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait until market open before placing orders (default: true).",
    )
    # NOTE:
    # This executor is commonly triggered after market close (e.g. 16:15 ET) to prepare signals,
    # then wait until the next market open to place rebalance orders. 10 hours is NOT enough for
    # an overnight gap (and never enough for a weekend). Use a larger default to avoid silently
    # timing out and skipping the open.
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=72 * 3600,  # ~3 days, covers Fri close -> Mon open and most holidays
        help="Max seconds to wait for next market open before skipping execution.",
    )
    parser.add_argument(
        "--cancel-open-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cancel all open orders before execution (default: true).",
    )
    parser.add_argument(
        "--use-polygon-ws",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Subscribe Polygon WS for tickers (pre-open price monitoring).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["v5", "v7"],
        default="v7",
        help="Strategy version to execute (v5 or v7). Default: v7.",
    )
    args = parser.parse_args()

    logger.info(f"Starting Alpaca Executor (Strategy: {args.strategy})...")
    
    # 1. Setup Adapter
    try:
        adapter = AlpacaAdapter()
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return

    if args.cancel_open_orders:
        adapter.cancel_all_open_orders()

    # 2. Setup Strategy Runner
    runner = OpenRebalanceRunner(adapter, strategy_version=args.strategy)
    
    # 3. Inject Signals
    runner.inject_signal(DAILY_SIGNAL_FILE)
    
    # 4. Run Logic
    targets = runner.run_daily_logic()
    
    if not targets:
        logger.info("No targets generated.")
        return

    # Optional: subscribe Polygon WS (pre-open monitoring)
    ws_cache: PolygonTradePriceCache | None = None
    if args.use_polygon_ws:
        try:
            sig_df = runner.engine.signal_df
            roots = sorted({str(v).split(".")[0] for v in sig_df["vt_symbol"].to_list()})
            ws_cache = PolygonTradePriceCache(get_polygon_api_key(), roots)
            ws_cache.start()
        except Exception as exc:
            logger.warning(f"[PolygonWS] failed to start: {exc}")
            ws_cache = None

    # 5. Wait for market open if needed
    if args.wait_for_open:
        ok = adapter.wait_for_open(max_wait_seconds=args.max_wait_seconds)
        if not ok:
            logger.warning("[AlpacaExecutor] Market did not open within max wait, skip execution.")
            return
    else:
        # Safety: never place orders when market is closed unless user explicitly enables waiting.
        if not adapter.is_market_open():
            logger.warning("[AlpacaExecutor] Market is CLOSED and --no-wait-for-open is set. Skip execution.")
            return

    # 6. Refresh account/buying power right before execution
    info = adapter.get_account_info()
    logger.info(f"[Alpaca] Pre-trade Account: cash={info.cash:.2f}, equity={info.equity:.2f}, buying_power={info.buying_power:.2f}")

    # 7. Execute
    open_execute_rebalance(adapter, targets)
    logger.info("Execution cycle complete.")

    if ws_cache:
        ws_cache.stop()

if __name__ == "__main__":
    main()

