"""
同步 Polygon reference/tickers 到 Postgres ref_tickers 表。

基于 Flagship Alpha-Momentum 策略需求：
- 只同步 US stocks，type=CS（普通股）
- 全量拉取（不分页限制），覆盖所有 active tickers
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polygon.rest import RESTClient

from flagship.config import VT_SETTING_PATH, create_polygon_client
from vnpy.trader.logger import logger
from vnpy.trader.setting import SETTINGS
from flagship.scripts.pg_ticker_db import (
    create_ticker_tables,
    upsert_ref_ticker,
)


def sync_all_tickers(client: "RESTClient", ticker_type: str = "CS") -> None:
    """
    同步所有 US stocks tickers 到 Postgres。

    Args:
        client: Polygon RESTClient 实例
        ticker_type: ticker 类型过滤（默认 CS=Common Stock）
    """
    logger.info("Starting ticker sync from Polygon to Postgres...")

    iterator = client.list_tickers(
        market="stocks",
        active=True,
        type=ticker_type,
        sort="ticker",
        order="asc",
        limit=1000,
    )

    total_count = 0
    inserted_count = 0

    for item in iterator:
        ticker_obj = item
        payload: dict[str, Any] = {}

        # 转换为 dict（处理 dataclass）
        if hasattr(ticker_obj, "__dict__"):
            payload = dict(vars(ticker_obj))
        elif hasattr(ticker_obj, "dict"):
            payload = ticker_obj.dict()
        else:
            payload = dict(ticker_obj) if isinstance(ticker_obj, dict) else {}

        symbol = payload.get("ticker")
        if not symbol:
            continue

        total_count += 1

        try:
            upsert_ref_ticker(payload)
            inserted_count += 1

            if total_count % 500 == 0:
                logger.info(
                    f"Progress: {total_count} processed, {inserted_count} inserted"
                )
        except Exception as exc:
            logger.warning(f"Failed to upsert {symbol}: {exc}")

    logger.info(
        f"Ticker sync completed: {total_count} total, {inserted_count} inserted"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Polygon reference/tickers to Postgres ref_tickers table."
    )
    parser.add_argument(
        "--ticker-type",
        type=str,
        default="CS",
        help="Ticker type filter (default: CS for Common Stock)",
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

    client = create_polygon_client()

    sync_all_tickers(client, ticker_type=args.ticker_type)


if __name__ == "__main__":
    main()

