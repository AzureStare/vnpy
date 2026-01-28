"""
初始化 Postgres 数据库表（空环境可直接运行）。

默认创建：
- ref_tickers / ticker_daily_fundamentals
- daily_selection
- daily_ranking_history
- daily_ranking_returns
- trading_controls
- users
"""
from __future__ import annotations

import argparse

from vnpy.trader.logger import logger

from flagship.trading.controls import create_trading_controls_tables
from flagship.universe.build_daily_selection import create_daily_selection_table
from flagship.universe.daily_ranking_returns import ensure_daily_ranking_returns_table
from flagship.universe.pg_ticker_db import (
    create_daily_ranking_history_table,
    create_ticker_tables,
    create_users_table,
)


def init_db(
    *,
    include_users: bool,
    include_trading_controls: bool,
    include_daily_selection: bool,
    include_daily_ranking_history: bool,
    include_daily_ranking_returns: bool,
    include_ticker_tables: bool,
) -> None:
    if include_ticker_tables:
        logger.info("[init_db] 创建 ref_tickers / ticker_daily_fundamentals")
        create_ticker_tables()

    if include_daily_selection:
        logger.info("[init_db] 创建 daily_selection")
        create_daily_selection_table()

    if include_daily_ranking_history:
        logger.info("[init_db] 创建 daily_ranking_history")
        create_daily_ranking_history_table()

    if include_daily_ranking_returns:
        logger.info("[init_db] 创建 daily_ranking_returns")
        ensure_daily_ranking_returns_table()

    if include_trading_controls:
        logger.info("[init_db] 创建 trading_controls 表")
        create_trading_controls_tables()

    if include_users:
        logger.info("[init_db] 创建 users")
        create_users_table()


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 Postgres 数据库表")
    parser.add_argument("--skip-users", action="store_true", help="跳过 users 表")
    parser.add_argument("--skip-trading-controls", action="store_true", help="跳过 trading_controls 表")
    parser.add_argument("--skip-daily-selection", action="store_true", help="跳过 daily_selection 表")
    parser.add_argument("--skip-daily-ranking-history", action="store_true", help="跳过 daily_ranking_history 表")
    parser.add_argument("--skip-daily-ranking-returns", action="store_true", help="跳过 daily_ranking_returns 表")
    parser.add_argument("--skip-ticker-tables", action="store_true", help="跳过 ref_tickers / ticker_daily_fundamentals")

    args = parser.parse_args()

    init_db(
        include_users=not args.skip_users,
        include_trading_controls=not args.skip_trading_controls,
        include_daily_selection=not args.skip_daily_selection,
        include_daily_ranking_history=not args.skip_daily_ranking_history,
        include_daily_ranking_returns=not args.skip_daily_ranking_returns,
        include_ticker_tables=not args.skip_ticker_tables,
    )

    logger.info("[init_db] 初始化完成")


if __name__ == "__main__":
    main()
