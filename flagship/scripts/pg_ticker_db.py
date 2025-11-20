"""
Postgres ticker 数据库工具模块。

基于 vnpy.trader.setting.SETTINGS 配置连接 Postgres，提供：
1. DDL 建表（ref_tickers, ticker_daily_fundamentals）
2. 数据同步接口（从 Polygon API 拉取并写入）
3. 查询接口（供 build_daily_universe.py 等脚本使用）
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator

try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    raise ImportError(
        "psycopg2 is required for Postgres support. Install via: pip install psycopg2-binary"
    )

from vnpy.trader.setting import SETTINGS
from vnpy.trader.logger import logger


def get_pg_connection_params() -> dict[str, Any]:
    """从 SETTINGS 读取 Postgres 连接参数。"""
    db_name = SETTINGS.get("database.name", "").lower()
    if db_name != "postgresql":
        raise ValueError(
            f"SETTINGS['database.name'] must be 'postgresql', got '{db_name}'. "
            f"Please configure vt_setting.json with database.name=postgresql"
        )

    return {
        "host": SETTINGS.get("database.host", "localhost"),
        "port": SETTINGS.get("database.port", 5432),
        "database": SETTINGS.get("database.database", "vnpy"),
        "user": SETTINGS.get("database.user", "postgres"),
        "password": SETTINGS.get("database.password", ""),
    }


@contextmanager
def get_pg_connection():
    """获取 Postgres 连接的上下文管理器。"""
    params = get_pg_connection_params()
    conn = psycopg2.connect(**params)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_tables_exist() -> bool:
    """检查表是否已存在。"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ref_tickers'
                );
            """)
            ref_exists = cur.fetchone()[0]
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ticker_daily_fundamentals'
                );
            """)
            detail_exists = cur.fetchone()[0]
            
            return ref_exists and detail_exists


def create_ticker_tables() -> None:
    """创建 ticker 相关表（ref_tickers, ticker_daily_fundamentals）。如果表已存在则跳过。"""
    if check_tables_exist():
        logger.info("Tables ref_tickers and ticker_daily_fundamentals already exist, skipping creation.")
        return
    
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # ref_tickers: 静态 ticker 主表
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ref_tickers (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    locale TEXT,
                    type TEXT,
                    primary_exchange TEXT,
                    currency_name TEXT,
                    currency_symbol TEXT,
                    list_date DATE,
                    delisted_utc TIMESTAMP,
                    active BOOLEAN,
                    cik TEXT,
                    composite_figi TEXT,
                    share_class_figi TEXT,
                    ticker_root TEXT,
                    homepage_url TEXT,
                    raw JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ref_tickers_market_locale 
                    ON ref_tickers(market, locale);
                CREATE INDEX IF NOT EXISTS idx_ref_tickers_type 
                    ON ref_tickers(type);
                CREATE INDEX IF NOT EXISTS idx_ref_tickers_active 
                    ON ref_tickers(active);
                """
            )

            # ticker_daily_fundamentals: 每日滚动基本面数据
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ticker_daily_fundamentals (
                    symbol TEXT NOT NULL,
                    as_of_date DATE NOT NULL,
                    market_cap DOUBLE PRECISION,
                    shares_outstanding BIGINT,
                    weighted_shares_outstanding BIGINT,
                    total_employees INTEGER,
                    sic_code TEXT,
                    sic_description TEXT,
                    sector TEXT,
                    industry TEXT,
                    raw JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, as_of_date)
                );
                CREATE INDEX IF NOT EXISTS idx_ticker_daily_fundamentals_date 
                    ON ticker_daily_fundamentals(as_of_date);
                CREATE INDEX IF NOT EXISTS idx_ticker_daily_fundamentals_symbol 
                    ON ticker_daily_fundamentals(symbol);
                """
            )

            logger.info("Created tables: ref_tickers, ticker_daily_fundamentals")


def upsert_ref_ticker(ticker_data: dict[str, Any]) -> None:
    """插入或更新 ref_tickers 表的一条记录。"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ref_tickers (
                    symbol, name, market, locale, type, primary_exchange,
                    currency_name, currency_symbol, list_date, delisted_utc,
                    active, cik, composite_figi, share_class_figi, ticker_root,
                    homepage_url, raw, updated_at
                ) VALUES (
                    %(ticker)s, %(name)s, %(market)s, %(locale)s, %(type)s,
                    %(primary_exchange)s, %(currency_name)s, %(currency_symbol)s,
                    %(list_date)s, %(delisted_utc)s, %(active)s, %(cik)s,
                    %(composite_figi)s, %(share_class_figi)s, %(ticker_root)s,
                    %(homepage_url)s, %(raw)s, CURRENT_TIMESTAMP
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    market = EXCLUDED.market,
                    locale = EXCLUDED.locale,
                    type = EXCLUDED.type,
                    primary_exchange = EXCLUDED.primary_exchange,
                    currency_name = EXCLUDED.currency_name,
                    currency_symbol = EXCLUDED.currency_symbol,
                    list_date = EXCLUDED.list_date,
                    delisted_utc = EXCLUDED.delisted_utc,
                    active = EXCLUDED.active,
                    cik = EXCLUDED.cik,
                    composite_figi = EXCLUDED.composite_figi,
                    share_class_figi = EXCLUDED.share_class_figi,
                    ticker_root = EXCLUDED.ticker_root,
                    homepage_url = EXCLUDED.homepage_url,
                    raw = EXCLUDED.raw,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                {
                    "ticker": ticker_data.get("ticker"),
                    "name": ticker_data.get("name"),
                    "market": ticker_data.get("market"),
                    "locale": ticker_data.get("locale"),
                    "type": ticker_data.get("type"),
                    "primary_exchange": ticker_data.get("primary_exchange"),
                    "currency_name": ticker_data.get("currency_name"),
                    "currency_symbol": ticker_data.get("currency_symbol"),
                    "list_date": ticker_data.get("list_date"),
                    "delisted_utc": ticker_data.get("delisted_utc"),
                    "active": ticker_data.get("active"),
                    "cik": ticker_data.get("cik"),
                    "composite_figi": ticker_data.get("composite_figi"),
                    "share_class_figi": ticker_data.get("share_class_figi"),
                    "ticker_root": ticker_data.get("ticker_root"),
                    "homepage_url": ticker_data.get("homepage_url"),
                    "raw": json.dumps(ticker_data),
                },
            )


def upsert_ticker_detail(
    symbol: str, as_of_date: date, detail_data: dict[str, Any]
) -> None:
    """插入或更新 ticker_daily_fundamentals 表的一条记录。"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
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
                (
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
                ),
            )


def get_ref_tickers(
    market: str = "stocks",
    locale: str = "us",
    ticker_type: str | None = "CS",
    active: bool = True,
) -> list[dict[str, Any]]:
    """查询 ref_tickers 表，返回符合条件的 ticker 列表。"""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            conditions = ["market = %s", "locale = %s", "active = %s"]
            params: list[Any] = [market, locale, active]

            if ticker_type:
                conditions.append("type = %s")
                params.append(ticker_type)

            query = f"""
                SELECT symbol, name, market, locale, type, primary_exchange,
                       currency_name, list_date, active, raw
                FROM ref_tickers
                WHERE {' AND '.join(conditions)}
                ORDER BY symbol;
            """
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def get_ticker_market_cap(symbol: str, as_of_date: date) -> float | None:
    """查询指定日期 ticker 的市值。"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT market_cap
                FROM ticker_daily_fundamentals
                WHERE symbol = %s AND as_of_date = %s;
                """,
                (symbol, as_of_date),
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else None


def get_ticker_market_caps_batch(
    symbols: list[str], as_of_date: date
) -> dict[str, float]:
    """批量查询多个 ticker 在指定日期的市值。"""
    if not symbols:
        return {}

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, market_cap
                FROM ticker_daily_fundamentals
                WHERE symbol = ANY(%s) AND as_of_date = %s;
                """,
                (symbols, as_of_date),
            )
            return {
                row[0]: float(row[1]) if row[1] is not None else 0.0
                for row in cur.fetchall()
            }

