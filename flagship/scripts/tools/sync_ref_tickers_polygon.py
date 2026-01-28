"""
同步 Polygon Reference Tickers → PostgreSQL.ref_tickers

用途：
- 为 universe 构建（ref_tickers_cs）提供权威“Common Stock + Active”列表
- 支持云端/本地重复执行：幂等 upsert（按 symbol 主键）

运行方式（推荐在 docker-compose 的 app 容器内执行，自动使用 DATABASE_* 连接 db）：
  python -m flagship.scripts.tools.sync_ref_tickers_polygon --market stocks --locale us --type CS --active

注意：
- 不会打印任何密钥内容
- 默认只同步 active 的 tickers（可用 --include-inactive 覆盖）
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


from vnpy.trader.logger import logger

from flagship.config import create_polygon_client
from flagship.universe.pg_ticker_db import create_ticker_tables, get_pg_connection


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        d = to_dict()
        return d if isinstance(d, dict) else {}
    # Fallback: best-effort
    d = getattr(obj, "__dict__", None)
    return d if isinstance(d, dict) else {}


def _chunked(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size <= 0:
        batch_size = 1000
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@dataclass(frozen=True)
class SyncConfig:
    market: str
    locale: str
    ticker_type: str
    active: bool
    include_inactive: bool
    page_limit: int
    batch_size: int
    max_items: int
    dry_run: bool


def _build_config(args: argparse.Namespace) -> SyncConfig:
    page_limit = int(args.page_limit)
    if page_limit <= 0:
        page_limit = 1000
    if page_limit > 1000:
        page_limit = 1000

    return SyncConfig(
        market=str(args.market),
        locale=str(args.locale),
        ticker_type=str(args.type),
        active=bool(args.active),
        include_inactive=bool(args.include_inactive),
        page_limit=page_limit,
        batch_size=int(args.batch_size),
        max_items=int(args.max_items),
        dry_run=bool(args.dry_run),
    )


def _collect_tickers(cfg: SyncConfig) -> list[dict[str, Any]]:
    client = create_polygon_client()

    active = None
    if cfg.include_inactive:
        active = None
    else:
        active = bool(cfg.active)

    # list_tickers signature lacks locale, but supports arbitrary params
    params: dict[str, Any] = {}
    if cfg.locale:
        params["locale"] = cfg.locale

    logger.info(
        f"[sync_ref_tickers] fetching: market={cfg.market}, locale={cfg.locale}, type={cfg.ticker_type}, "
        f"active={active}, limit(per_page)={cfg.page_limit}"
    )

    items: list[dict[str, Any]] = []
    it = client.list_tickers(
        market=cfg.market,
        type=cfg.ticker_type,
        active=active,
        limit=cfg.page_limit,
        params=params or None,
    )

    for t in it:
        d = _as_dict(t)
        if not d:
            continue
        items.append(d)
        if cfg.max_items > 0 and len(items) >= cfg.max_items:
            break

    logger.info(f"[sync_ref_tickers] fetched {len(items)} tickers")
    return items


def _normalize_for_db(cfg: SyncConfig, raw: dict[str, Any]) -> dict[str, Any]:
    # Polygon response uses "ticker" key; our DB uses symbol
    symbol = raw.get("ticker") or raw.get("symbol") or raw.get("id")
    if not symbol:
        return {}

    # Ensure stable locale fallback
    locale = raw.get("locale") or cfg.locale

    normalized = {
        "symbol": symbol,
        "name": raw.get("name"),
        "market": raw.get("market") or cfg.market,
        "locale": locale,
        "type": raw.get("type") or cfg.ticker_type,
        "primary_exchange": raw.get("primary_exchange") or raw.get("primaryExchange"),
        "currency_name": raw.get("currency_name") or raw.get("currencyName"),
        "currency_symbol": raw.get("currency_symbol") or raw.get("currencySymbol"),
        "list_date": raw.get("list_date") or raw.get("listDate"),
        "delisted_utc": raw.get("delisted_utc") or raw.get("delistedUtc"),
        "active": raw.get("active"),
        "cik": raw.get("cik"),
        "composite_figi": raw.get("composite_figi") or raw.get("compositeFigi"),
        "share_class_figi": raw.get("share_class_figi") or raw.get("shareClassFigi"),
        "ticker_root": raw.get("ticker_root") or raw.get("tickerRoot"),
        "homepage_url": raw.get("homepage_url") or raw.get("homepageUrl"),
        "raw": raw,
    }
    return normalized


def _upsert_batch(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    try:
        from psycopg2.extras import execute_values, Json
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("psycopg2-binary is required to sync ref_tickers") from exc

    sql = """
    INSERT INTO ref_tickers (
        symbol, name, market, locale, type, primary_exchange,
        currency_name, currency_symbol, list_date, delisted_utc,
        active, cik, composite_figi, share_class_figi, ticker_root,
        homepage_url, raw, updated_at
    ) VALUES %s
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
    """

    values = []
    for r in rows:
        values.append(
            (
                r.get("symbol"),
                r.get("name"),
                r.get("market"),
                r.get("locale"),
                r.get("type"),
                r.get("primary_exchange"),
                r.get("currency_name"),
                r.get("currency_symbol"),
                r.get("list_date"),
                r.get("delisted_utc"),
                r.get("active"),
                r.get("cik"),
                r.get("composite_figi"),
                r.get("share_class_figi"),
                r.get("ticker_root"),
                r.get("homepage_url"),
                Json(r.get("raw") or {}),
            )
        )

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, page_size=len(values))

    return len(values)


def sync_ref_tickers(cfg: SyncConfig) -> None:
    # Ensure tables exist
    create_ticker_tables()

    fetched = _collect_tickers(cfg)
    normalized_rows = []
    for item in fetched:
        r = _normalize_for_db(cfg, item)
        if r:
            normalized_rows.append(r)

    logger.info(f"[sync_ref_tickers] normalized rows={len(normalized_rows)}")

    if cfg.dry_run:
        logger.info("[sync_ref_tickers] dry-run enabled, skip upsert")
        return

    total = 0
    started = time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[sync_ref_tickers] upserting... started_at={started}")

    for batch in _chunked(normalized_rows, cfg.batch_size):
        n = _upsert_batch(batch)
        total += n
        logger.info(f"[sync_ref_tickers] upserted {total}/{len(normalized_rows)}")

    logger.info(f"[sync_ref_tickers] done, total_upserted={total}")

    # Sanity check: count
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(1) FROM ref_tickers;")
            cnt = cur.fetchone()[0]
            logger.info(f"[sync_ref_tickers] db_count(ref_tickers)={cnt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Polygon reference tickers into PostgreSQL.ref_tickers")
    parser.add_argument("--market", default="stocks", help="Polygon market filter (default: stocks)")
    parser.add_argument("--locale", default="us", help="Polygon locale filter (default: us)")
    parser.add_argument("--type", default="CS", help="Ticker type filter (default: CS)")
    parser.add_argument("--active", action="store_true", help="Only active tickers (default)")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive tickers as well")
    parser.add_argument("--page-limit", type=int, default=1000, help="Polygon per-page limit (max 1000)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Postgres upsert batch size")
    parser.add_argument("--max-items", type=int, default=0, help="Max tickers to fetch (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch & normalize only; do not write to DB")

    args = parser.parse_args()
    if not args.include_inactive and not args.active:
        # Keep behavior explicit: default active=True unless user asked include_inactive
        args.active = True
    return args


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)
    logger.info(
        f"[sync_ref_tickers] cfg={json.dumps(cfg.__dict__, ensure_ascii=False)}"
    )
    sync_ref_tickers(cfg)


if __name__ == "__main__":
    main()


