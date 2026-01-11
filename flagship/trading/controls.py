"""
Trading controls stored in Postgres.

Currently includes:
- Disabled symbols: blocks new entries (does not force liquidate existing holdings)
- Buy exposure multiplier: scales buy-side budget/qty (buy-only, does not force sell)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vnpy.trader.logger import logger

from flagship.universe.pg_ticker_db import get_pg_connection


DISABLED_SYMBOLS_TABLE = "trading_disabled_symbols"
RISK_CONTROLS_TABLE = "trading_risk_controls"

BUY_EXPOSURE_MULTIPLIER_KEY = "buy_exposure_multiplier"
DEFAULT_BUY_EXPOSURE_MULTIPLIER = 1.0


@dataclass(frozen=True)
class TradingControlsSnapshot:
    disabled_vt_symbols: list[str]
    buy_exposure_multiplier: float


def _normalize_vt_symbol(vt_symbol: str) -> str:
    sym = str(vt_symbol or "").strip()
    if not sym:
        raise ValueError("vt_symbol is required")
    return sym


def create_trading_controls_tables() -> None:
    """
    Create Postgres tables for trading controls.
    Safe to call multiple times.
    """
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {DISABLED_SYMBOLS_TABLE} (
                        vt_symbol TEXT PRIMARY KEY,
                        disabled BOOLEAN NOT NULL DEFAULT TRUE,
                        updated_by TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {RISK_CONTROLS_TABLE} (
                        key TEXT PRIMARY KEY,
                        value DOUBLE PRECISION NOT NULL,
                        updated_by TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

                # Seed default risk control for convenience (idempotent).
                cur.execute(
                    f"""
                    INSERT INTO {RISK_CONTROLS_TABLE} (key, value, updated_by, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (key) DO NOTHING;
                    """,
                    (BUY_EXPOSURE_MULTIPLIER_KEY, float(DEFAULT_BUY_EXPOSURE_MULTIPLIER), "system"),
                )
    except Exception as exc:
        logger.warning(f"[trading_controls] create tables failed: {exc}")


def get_disabled_vt_symbols() -> set[str]:
    """
    Return set of vt_symbols currently disabled (disabled=true).
    On any error, returns empty set.
    """
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT vt_symbol FROM {DISABLED_SYMBOLS_TABLE} WHERE disabled = TRUE;"
                )
                rows = cur.fetchall() or []
        return {str(r[0]) for r in rows if r and r[0]}
    except Exception as exc:
        logger.warning(f"[trading_controls] get_disabled_vt_symbols failed: {exc}")
        return set()


def set_disabled_vt_symbol(*, vt_symbol: str, disabled: bool, updated_by: str | None = None) -> None:
    """
    Upsert a disabled flag for a vt_symbol.
    """
    sym = _normalize_vt_symbol(vt_symbol)
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {DISABLED_SYMBOLS_TABLE} (vt_symbol, disabled, updated_by, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (vt_symbol) DO UPDATE SET
                        disabled = EXCLUDED.disabled,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (sym, bool(disabled), str(updated_by) if updated_by else None),
                )
    except Exception as exc:
        logger.warning(f"[trading_controls] set_disabled_vt_symbol failed: {exc}")
        raise


def get_buy_exposure_multiplier(*, default: float = DEFAULT_BUY_EXPOSURE_MULTIPLIER) -> float:
    """
    Get buy_exposure_multiplier in [0, 1].
    On any error, returns default.
    """
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT value FROM {RISK_CONTROLS_TABLE} WHERE key = %s;",
                    (BUY_EXPOSURE_MULTIPLIER_KEY,),
                )
                row = cur.fetchone()
        if not row:
            return float(default)
        v = float(row[0])
        if not math.isfinite(v):
            return float(default)
        return float(min(1.0, max(0.0, v)))
    except Exception as exc:
        logger.warning(f"[trading_controls] get_buy_exposure_multiplier failed: {exc}")
        return float(default)


def set_buy_exposure_multiplier(*, multiplier: float, updated_by: str | None = None) -> float:
    """
    Set buy_exposure_multiplier in [0, 1].
    Returns the normalized value.
    """
    m = float(multiplier)
    if not math.isfinite(m):
        raise ValueError("multiplier must be a finite number")
    if m < 0.0 or m > 1.0:
        raise ValueError("multiplier must be within [0, 1]")

    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {RISK_CONTROLS_TABLE} (key, value, updated_by, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (BUY_EXPOSURE_MULTIPLIER_KEY, float(m), str(updated_by) if updated_by else None),
                )
    except Exception as exc:
        logger.warning(f"[trading_controls] set_buy_exposure_multiplier failed: {exc}")
        raise

    return float(m)


def get_trading_controls_snapshot() -> TradingControlsSnapshot:
    disabled = sorted(get_disabled_vt_symbols())
    buy_exposure_multiplier = get_buy_exposure_multiplier()
    return TradingControlsSnapshot(
        disabled_vt_symbols=disabled,
        buy_exposure_multiplier=float(buy_exposure_multiplier),
    )

