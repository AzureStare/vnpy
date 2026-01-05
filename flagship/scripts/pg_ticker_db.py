"""
Postgres ticker 数据库工具模块。

基于 vnpy.trader.setting.SETTINGS 配置连接 Postgres，提供：
1. DDL 建表（ref_tickers, ticker_daily_fundamentals）
2. 数据同步接口（从 Polygon API 拉取并写入）
3. 查询接口（供 build_daily_universe.py 等脚本使用）
"""

from __future__ import annotations

import json
import os
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

# Flagship 项目约定：优先使用项目根目录的 vt_setting.json（而不是 ~/.vntrader/vt_setting.json）
try:
    from flagship.config import VT_SETTING_PATH
except Exception:  # pragma: no cover
    VT_SETTING_PATH = None  # type: ignore


def _load_project_vt_setting() -> None:
    """
    将项目根目录的 vt_setting.json 合并进 vnpy.trader.setting.SETTINGS。

    背景：vn.py 默认从 ~/.vntrader/vt_setting.json 读取配置；但本仓库的统一入口是项目根目录 vt_setting.json。
    """
    if VT_SETTING_PATH is None:
        return
    try:
        if VT_SETTING_PATH.exists():
            data = json.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                SETTINGS.update(data)
    except Exception as exc:
        logger.warning(f"[pg_ticker_db] Failed to load project vt_setting.json: {exc}")


def get_pg_connection_params() -> dict[str, Any]:
    """从 SETTINGS 读取 Postgres 连接参数。"""
    _load_project_vt_setting()
    db_name = SETTINGS.get("database.name", "").lower()
    if db_name != "postgresql":
        raise ValueError(
            f"SETTINGS['database.name'] must be 'postgresql', got '{db_name}'. "
            f"Please configure vt_setting.json with database.name=postgresql"
        )

    # Docker/Server 部署场景：允许用环境变量覆盖连接参数（优先级更高）
    env_host = os.getenv("DATABASE_HOST")
    env_port = os.getenv("DATABASE_PORT")
    env_db = os.getenv("DATABASE_DATABASE")
    env_user = os.getenv("DATABASE_USER")
    env_password = os.getenv("DATABASE_PASSWORD")

    return {
        "host": env_host or SETTINGS.get("database.host", "localhost"),
        "port": int(env_port) if env_port else SETTINGS.get("database.port", 5432),
        "database": env_db or SETTINGS.get("database.database", "vnpy"),
        "user": env_user or SETTINGS.get("database.user", "postgres"),
        "password": env_password if env_password is not None else SETTINGS.get("database.password", ""),
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


def get_selected_symbols_in_range(start_date: date, end_date: date) -> list[str]:
    """
    从 daily_selection 表中查询指定日期范围内被选中的去重 vt_symbol 列表。

    用于“静态过滤后的 universe（U）”训练/推理：训练只在该范围内出现过的标的集合上进行。
    """
    if start_date > end_date:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT vt_symbol
                FROM daily_selection
                WHERE trade_date >= %s AND trade_date <= %s
                ORDER BY vt_symbol;
                """,
                (start_date, end_date),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]


def create_users_table() -> None:
    """创建用户表"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user', -- 'admin' or 'user'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("Created table: users")


def add_user(username: str, password_hash: str, role: str = "user") -> bool:
    """添加或更新用户"""
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, password_hash, role) 
                    VALUES (%s, %s, %s) 
                    ON CONFLICT (username) DO UPDATE SET 
                        password_hash=EXCLUDED.password_hash, 
                        role=EXCLUDED.role
                    """,
                    (username, password_hash, role),
                )
        return True
    except Exception as e:
        logger.error(f"Failed to add user: {e}")
        return False


def get_user(username: str) -> dict | None:
    """获取用户信息"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, password_hash, role FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if row:
                return {"username": row[0], "password_hash": row[1], "role": row[2]}
    return None


def list_users() -> list[dict]:
    """列出所有用户"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at")
            rows = cur.fetchall()
            return [{"username": r[0], "role": r[1], "created_at": r[2].isoformat()} for r in rows]


def delete_user(username: str) -> bool:
    """删除用户"""
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username = %s", (username,))
        return True
    except Exception as e:
        logger.error(f"Failed to delete user: {e}")
        return False


def create_daily_ranking_history_table() -> None:
    """创建每日信号排名历史表"""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_ranking_history (
                    trade_date DATE NOT NULL,
                    vt_symbol TEXT NOT NULL,
                    signal DOUBLE PRECISION,
                    close_price DOUBLE PRECISION,
                    adv_usd DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (trade_date, vt_symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_daily_ranking_history_date 
                    ON daily_ranking_history(trade_date);
                
                CREATE INDEX IF NOT EXISTS idx_daily_ranking_history_symbol 
                    ON daily_ranking_history(vt_symbol);
            """)
            logger.info("Created table: daily_ranking_history")


def save_daily_ranking_history(trade_date: date, ranking_data: list[dict[str, Any]]) -> None:
    """保存每日排名历史"""
    if not ranking_data:
        return
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daily_ranking_history WHERE trade_date = %s", (trade_date,))
            values = [
                (trade_date, r["vt_symbol"], r["signal"], r["close_price"], r["adv_usd"])
                for r in ranking_data
            ]
            execute_values(
                cur,
                """
                INSERT INTO daily_ranking_history (trade_date, vt_symbol, signal, close_price, adv_usd)
                VALUES %s
                """,
                values
            )

