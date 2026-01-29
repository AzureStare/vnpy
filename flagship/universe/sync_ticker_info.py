import json
import sys
import requests
from pathlib import Path
from datetime import date
from typing import Any, Optional

# 仅在“直接运行脚本”（python file.py）时注入项目根路径；以 `python -m` 运行则无需处理
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flagship.config import VT_SETTING_PATH
from flagship.universe.pg_ticker_db import upsert_ref_ticker, upsert_ticker_detail, get_ref_tickers
from vnpy.trader.logger import logger
from polygon.rest import RESTClient

def get_market_cap_from_massive_v3(symbol: str, api_key: str, trade_date: Optional[date] = None) -> Optional[float]:
    """从 Massive.com V3 Ticker Overview API 获取特定日期的市值"""
    date_str = f"&date={trade_date.isoformat()}" if trade_date else ""
    url = f"https://api.massive.com/v3/reference/tickers/{symbol}?apiKey={api_key}{date_str}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if data and data.get("results"):
            mkt_cap = data["results"].get("market_cap")
            return float(mkt_cap) if mkt_cap else None
    except Exception as e:
        logger.error(f"Error fetching {symbol} on {trade_date} from Massive: {e}")
    return None

def sync_all_ticker_details():
    """从 Polygon/Massive 同步所有股票的静态信息和当前基本面（市值等）"""
    if not VT_SETTING_PATH.exists():
        logger.error("vt_setting.json not found")
        return

    with open(VT_SETTING_PATH, "r") as f:
        config = json.load(f)
        api_key = config.get("datafeed.password")

    if not api_key:
        logger.error("API Key not found in vt_setting.json")
        return

    client = RESTClient(api_key)
    today = date.today()

    # 1. 首先通过 Polygon Snapshot 获取全场市值快照 (非常快)
    logger.info("Step 1: Fetching all-tickers snapshot from Polygon...")
    try:
        snapshots = client.get_snapshot_all(market_type="stocks")
        count = 0
        for s in snapshots:
            if hasattr(s, 'market_cap') and s.market_cap:
                upsert_ticker_detail(s.ticker, today, {"market_cap": s.market_cap})
                count += 1
        logger.info(f"Synced {count} market caps from Polygon snapshot.")
    except Exception as e:
        logger.error(f"Polygon snapshot failed: {e}")

    # 2. 同步静态 Ticker 信息
    logger.info("Step 2: Syncing ref_tickers...")
    try:
        tickers = client.list_tickers(market="stocks", type="CS", active=True, limit=1000)
        for t in tickers:
            upsert_ref_ticker(t.__dict__ if hasattr(t, '__dict__') else t)
    except Exception as e:
        logger.error(f"Ref tickers sync failed: {e}")

    # 3. 针对选股池中的标的，通过 Massive V3 补充/更新更准确的市值 (较慢，按需执行)
    # 这里我们只针对已经在库里的 active 股票进行一次轮询
    active_tickers = get_ref_tickers(active=True)
    logger.info(f"Step 3: Syncing accurate market caps from Massive for {len(active_tickers)} tickers...")
    
    success = 0
    for i, t in enumerate(active_tickers):
        symbol = t["symbol"]
        mkt_cap = get_market_cap_from_massive_v3(symbol, api_key)
        if mkt_cap:
            upsert_ticker_detail(symbol, today, {"market_cap": mkt_cap})
            success += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{len(active_tickers)}, Synced: {success}")

    logger.info("All ticker info sync completed.")

if __name__ == "__main__":
    sync_all_ticker_details()

