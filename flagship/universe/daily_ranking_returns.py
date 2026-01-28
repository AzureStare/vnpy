"""
Daily ranking returns (long table) helpers.

Long-table design makes it easy to add new horizons without schema changes.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from vnpy.trader.logger import logger

from flagship.universe.pg_ticker_db import get_pg_connection


def ensure_daily_ranking_returns_table() -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_ranking_returns (
                    trade_date DATE NOT NULL,
                    vt_symbol VARCHAR(32) NOT NULL,
                    rank_pos INTEGER,
                    signal_score DOUBLE PRECISION,
                    horizon_d INTEGER NOT NULL,
                    ret_type VARCHAR(8) NOT NULL,
                    ret DOUBLE PRECISION,
                    excess_ret DOUBLE PRECISION,
                    close_t DOUBLE PRECISION,
                    close_t_shift DOUBLE PRECISION,
                    spy_close_t DOUBLE PRECISION,
                    spy_close_t_shift DOUBLE PRECISION,
                    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (trade_date, vt_symbol, horizon_d, ret_type)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_ranking_returns_date
                    ON daily_ranking_returns(trade_date);
                CREATE INDEX IF NOT EXISTS idx_daily_ranking_returns_symbol
                    ON daily_ranking_returns(vt_symbol);
                CREATE INDEX IF NOT EXISTS idx_daily_ranking_returns_horizon
                    ON daily_ranking_returns(horizon_d);
                """
            )
    logger.info("[daily_ranking_returns] ensured table daily_ranking_returns exists")


def upsert_daily_ranking_returns(rows: Sequence[Sequence[Any]]) -> None:
    if not rows:
        return
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values

            execute_values(
                cur,
                """
                INSERT INTO daily_ranking_returns (
                    trade_date,
                    vt_symbol,
                    rank_pos,
                    signal_score,
                    horizon_d,
                    ret_type,
                    ret,
                    excess_ret,
                    close_t,
                    close_t_shift,
                    spy_close_t,
                    spy_close_t_shift,
                    computed_at
                )
                VALUES %s
                ON CONFLICT (trade_date, vt_symbol, horizon_d, ret_type) DO UPDATE SET
                    rank_pos = EXCLUDED.rank_pos,
                    signal_score = EXCLUDED.signal_score,
                    ret = EXCLUDED.ret,
                    excess_ret = EXCLUDED.excess_ret,
                    close_t = EXCLUDED.close_t,
                    close_t_shift = EXCLUDED.close_t_shift,
                    spy_close_t = EXCLUDED.spy_close_t,
                    spy_close_t_shift = EXCLUDED.spy_close_t_shift,
                    computed_at = EXCLUDED.computed_at
                """,
                list(rows),
            )

    logger.info(f"[daily_ranking_returns] upserted rows: {len(rows)}")


def fetch_daily_ranking_topn(
    *,
    start_date: str,
    end_date: str,
    top_n: int,
) -> list[dict[str, Any]]:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, vt_symbol, signal_score, rank_pos
                FROM daily_ranking_history
                WHERE trade_date BETWEEN %s AND %s
                  AND rank_pos <= %s
                ORDER BY trade_date ASC, rank_pos ASC
                """,
                (start_date, end_date, int(top_n)),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for trade_date, vt_symbol, signal_score, rank_pos in rows:
        out.append(
            {
                "trade_date": trade_date,
                "vt_symbol": vt_symbol,
                "signal_score": signal_score,
                "rank_pos": rank_pos,
            }
        )
    return out
