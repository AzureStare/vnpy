"""
Backfill daily_ranking_returns from daily_ranking_history using AlphaLab daily parquet.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger

from flagship.trading.config import LAB_PATH
from flagship.universe.daily_ranking_returns import (
    ensure_daily_ranking_returns_table,
    fetch_daily_ranking_topn,
    upsert_daily_ranking_returns,
)
from flagship.universe.pg_ticker_db import get_pg_connection


SPY_SYMBOL = "SPY.NASDAQ"


def _parse_horizons(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return sorted(set(out))


def _pick_price_column(df: pl.DataFrame) -> str:
    if "close_price" in df.columns:
        return "close_price"
    if "close" in df.columns:
        return "close"
    raise ValueError("price column not found (expected close_price or close)")


def _compute_returns_for_date(
    bar_df: pl.DataFrame,
    *,
    trade_date: date,
    horizons: Iterable[int],
    spy_symbol: str = SPY_SYMBOL,
) -> pl.DataFrame:
    price_col = _pick_price_column(bar_df)
    df = bar_df.sort(["vt_symbol", "datetime"])

    for h in horizons:
        window = int(h)
        if window <= 0:
            continue
        df = df.with_columns(
            [
                pl.col(price_col).shift(window).over("vt_symbol").alias(f"trail_close_shift_{window}d"),
                pl.col(price_col).shift(-window).over("vt_symbol").alias(f"fwd_close_shift_{window}d"),
                (pl.col(price_col) / pl.col(price_col).shift(window).over("vt_symbol") - 1.0).alias(
                    f"trail_ret_{window}d"
                ),
                (pl.col(price_col).shift(-window).over("vt_symbol") / pl.col(price_col) - 1.0).alias(
                    f"fwd_ret_{window}d"
                ),
            ]
        )

    date_df = df.filter(pl.col("datetime").dt.date() == trade_date)
    if date_df.is_empty():
        return pl.DataFrame()

    spy_row = (
        date_df.filter(pl.col("vt_symbol") == spy_symbol)
        .select(
            [
                "datetime",
                pl.col(price_col).alias("spy_close_t"),
                *[
                    pl.col(f"trail_close_shift_{int(w)}d").alias(f"spy_trail_close_shift_{int(w)}d")
                    for w in horizons
                    if int(w) > 0
                ],
                *[
                    pl.col(f"fwd_close_shift_{int(w)}d").alias(f"spy_fwd_close_shift_{int(w)}d")
                    for w in horizons
                    if int(w) > 0
                ],
                *[
                    pl.col(f"trail_ret_{int(w)}d").alias(f"spy_trail_ret_{int(w)}d")
                    for w in horizons
                    if int(w) > 0
                ],
                *[
                    pl.col(f"fwd_ret_{int(w)}d").alias(f"spy_fwd_ret_{int(w)}d")
                    for w in horizons
                    if int(w) > 0
                ],
            ]
        )
    )

    data = (
        date_df.select(
            [
                "vt_symbol",
                "datetime",
                pl.col(price_col).alias("close_t"),
                *[f"trail_close_shift_{int(w)}d" for w in horizons if int(w) > 0],
                *[f"fwd_close_shift_{int(w)}d" for w in horizons if int(w) > 0],
                *[f"trail_ret_{int(w)}d" for w in horizons if int(w) > 0],
                *[f"fwd_ret_{int(w)}d" for w in horizons if int(w) > 0],
            ]
        )
        .join(spy_row, on="datetime", how="left")
        .filter(pl.col("vt_symbol") != spy_symbol)
    )
    return data


def _build_upsert_rows(
    *,
    returns_df: pl.DataFrame,
    rankings: list[dict[str, Any]],
    horizons: list[int],
    trade_date: date,
) -> list[list[Any]]:
    by_symbol = {row["vt_symbol"]: row for row in returns_df.to_dicts()}
    out: list[list[Any]] = []
    computed_at = datetime.utcnow().isoformat()

    for item in rankings:
        vt_symbol = str(item["vt_symbol"])
        r = by_symbol.get(vt_symbol)
        if not r:
            continue
        for h in horizons:
            h = int(h)
            trail_ret = r.get(f"trail_ret_{h}d")
            fwd_ret = r.get(f"fwd_ret_{h}d")
            trail_excess = None
            fwd_excess = None
            spy_trail_ret = r.get(f"spy_trail_ret_{h}d")
            spy_fwd_ret = r.get(f"spy_fwd_ret_{h}d")
            if trail_ret is not None and spy_trail_ret is not None:
                trail_excess = float(trail_ret) - float(spy_trail_ret)
            if fwd_ret is not None and spy_fwd_ret is not None:
                fwd_excess = float(fwd_ret) - float(spy_fwd_ret)

            out.append(
                [
                    trade_date,
                    vt_symbol,
                    item.get("rank_pos"),
                    item.get("signal_score"),
                    h,
                    "trail",
                    trail_ret,
                    trail_excess,
                    r.get("close_t"),
                    r.get(f"trail_close_shift_{h}d"),
                    r.get("spy_close_t"),
                    r.get(f"spy_trail_close_shift_{h}d"),
                    computed_at,
                ]
            )
            out.append(
                [
                    trade_date,
                    vt_symbol,
                    item.get("rank_pos"),
                    item.get("signal_score"),
                    h,
                    "fwd",
                    fwd_ret,
                    fwd_excess,
                    r.get("close_t"),
                    r.get(f"fwd_close_shift_{h}d"),
                    r.get("spy_close_t"),
                    r.get(f"spy_fwd_close_shift_{h}d"),
                    computed_at,
                ]
            )
    return out


def _compute_missing_trade_dates(
    *,
    history_counts: dict[date, int],
    returns_counts: dict[date, int],
    horizons: list[int],
    min_coverage: float,
) -> list[date]:
    """
    Pure helper: decide which trade_dates are missing/incomplete.

    expected rows per day ~= history_count * len(horizons) * 2 (trail+fwd).
    A day is considered missing if actual < expected * min_coverage.
    """
    horizons = [int(h) for h in horizons if int(h) > 0]
    if not horizons:
        return []

    try:
        cov = float(min_coverage)
    except Exception:
        cov = 0.98
    cov = max(0.0, min(1.0, cov))

    missing: list[date] = []
    for trade_date, n in history_counts.items():
        expected = int(n) * len(horizons) * 2
        if expected <= 0:
            continue
        actual = int(returns_counts.get(trade_date, 0) or 0)
        if actual < int(expected * cov):
            missing.append(trade_date)
    return sorted(set(missing))


def find_missing_trade_dates(
    *,
    start: str,
    end: str,
    horizons: list[int],
    top_n: int,
    min_coverage: float = 0.98,
) -> list[date]:
    """
    Find trade_dates that exist in daily_ranking_history but are missing/incomplete in daily_ranking_returns.
    """
    ensure_daily_ranking_returns_table()

    horizons = [int(h) for h in horizons if int(h) > 0]
    if not horizons:
        return []

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, COUNT(*) AS n
                FROM daily_ranking_history
                WHERE trade_date BETWEEN %s AND %s
                  AND rank_pos <= %s
                GROUP BY trade_date
                ORDER BY trade_date ASC
                """,
                (start, end, int(top_n)),
            )
            history_counts = {row[0]: int(row[1] or 0) for row in cur.fetchall() if row and row[0]}

            cur.execute(
                """
                SELECT trade_date, COUNT(*) AS n
                FROM daily_ranking_returns
                WHERE trade_date BETWEEN %s AND %s
                  AND horizon_d = ANY(%s)
                  AND ret_type IN ('trail', 'fwd')
                GROUP BY trade_date
                """,
                (start, end, horizons),
            )
            returns_counts = {row[0]: int(row[1] or 0) for row in cur.fetchall() if row and row[0]}

    missing = _compute_missing_trade_dates(
        history_counts=history_counts,
        returns_counts=returns_counts,
        horizons=horizons,
        min_coverage=float(min_coverage),
    )
    return missing


def backfill_daily_ranking_returns(
    *,
    start: str,
    end: str,
    horizons: list[int],
    top_n: int,
    lab_path: Path,
    only_trade_dates: Iterable[date] | None = None,
) -> None:
    ensure_daily_ranking_returns_table()

    if only_trade_dates is None:
        rows = fetch_daily_ranking_topn(start_date=start, end_date=end, top_n=top_n)
    else:
        dates = sorted(set(only_trade_dates))
        if not dates:
            logger.info("[backfill_daily_ranking_returns] only_trade_dates is empty, nothing to do")
            return
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT trade_date, vt_symbol, signal_score, rank_pos
                    FROM daily_ranking_history
                    WHERE trade_date = ANY(%s)
                      AND rank_pos <= %s
                    ORDER BY trade_date ASC, rank_pos ASC
                    """,
                    (dates, int(top_n)),
                )
                raw = cur.fetchall()
        rows = []
        for trade_date, vt_symbol, signal_score, rank_pos in raw:
            rows.append(
                {
                    "trade_date": trade_date,
                    "vt_symbol": vt_symbol,
                    "signal_score": signal_score,
                    "rank_pos": rank_pos,
                }
            )

    if not rows:
        logger.warning("[backfill_daily_ranking_returns] no ranking rows found")
        return

    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trade_date"]].append(row)

    lab = AlphaLab(str(lab_path))
    max_h = max(horizons) if horizons else 0
    buffer_days = max(7, max_h * 2)

    for trade_date, rankings in grouped.items():
        symbols = [str(r["vt_symbol"]) for r in rankings]
        if SPY_SYMBOL not in symbols:
            symbols.append(SPY_SYMBOL)

        start_date = trade_date - timedelta(days=buffer_days)
        end_date = trade_date + timedelta(days=buffer_days)

        bar_df = lab.load_bar_df(
            vt_symbols=symbols,
            interval=Interval.DAILY,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            extended_days=0,
        )
        if bar_df is None or bar_df.is_empty():
            logger.warning(f"[backfill_daily_ranking_returns] no bars for {trade_date}")
            continue

        returns_df = _compute_returns_for_date(
            bar_df,
            trade_date=trade_date,
            horizons=horizons,
            spy_symbol=SPY_SYMBOL,
        )
        if returns_df.is_empty():
            logger.warning(f"[backfill_daily_ranking_returns] returns empty for {trade_date}")
            continue

        upsert_rows = _build_upsert_rows(
            returns_df=returns_df,
            rankings=rankings,
            horizons=horizons,
            trade_date=trade_date,
        )
        upsert_daily_ranking_returns(upsert_rows)
        logger.info(f"[backfill_daily_ranking_returns] trade_date={trade_date} rows={len(upsert_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--horizons", type=str, default="1,3,5", help="Comma-separated horizons")
    parser.add_argument("--top-n", type=int, default=50, help="Top N ranking per day")
    parser.add_argument("--lab-path", type=Path, default=LAB_PATH)
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Only backfill missing/incomplete trade_dates (based on daily_ranking_history vs daily_ranking_returns).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.98,
        help="Coverage threshold (0..1). A day is missing if returns_rows < expected_rows * min_coverage.",
    )
    args = parser.parse_args()

    horizons = _parse_horizons(args.horizons)
    if not horizons:
        raise SystemExit("invalid horizons")

    if bool(args.fill_missing):
        missing = find_missing_trade_dates(
            start=str(args.start),
            end=str(args.end),
            horizons=horizons,
            top_n=int(args.top_n),
            min_coverage=float(args.min_coverage),
        )
        if not missing:
            logger.info("[backfill_daily_ranking_returns] fill-missing: no missing trade_dates found")
            return
        logger.info(f"[backfill_daily_ranking_returns] fill-missing: missing trade_dates={len(missing)}")
        backfill_daily_ranking_returns(
            start=str(args.start),
            end=str(args.end),
            horizons=horizons,
            top_n=int(args.top_n),
            lab_path=args.lab_path,
            only_trade_dates=missing,
        )
    else:
        backfill_daily_ranking_returns(
            start=str(args.start),
            end=str(args.end),
            horizons=horizons,
            top_n=int(args.top_n),
            lab_path=args.lab_path,
        )


if __name__ == "__main__":
    main()
