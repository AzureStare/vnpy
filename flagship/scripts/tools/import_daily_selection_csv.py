"""
导入 daily_selection CSV 到 Postgres（支持重建表结构）。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Iterator

from vnpy.trader.logger import logger

from flagship.universe.build_daily_selection import create_daily_selection_table
from flagship.universe.pg_ticker_db import get_pg_connection

REQUIRED_COLUMNS = {
    "trade_date",
    "vt_symbol",
    "close_price",
    "adv_usd",
    "med_volume",
    "created_at",
    "market_cap",
}

DEFAULT_BATCH_SIZE = 2000


@dataclass(frozen=True)
class ImportConfig:
    csv_path: Path
    recreate_table: bool
    batch_size: int


def _parse_date(value: str) -> date:
    value = str(value or "").strip()
    if not value:
        raise ValueError("trade_date is required")
    return datetime.fromisoformat(value).date()


def _parse_datetime(value: str) -> datetime:
    value = str(value or "").strip()
    if not value:
        return datetime.utcnow()
    return datetime.fromisoformat(value)


def _parse_float(value: str) -> float | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value.upper() in {"NULL", "N/A", "NA", "NONE"}:
        return None
    return float(value)


def _parse_int(value: str) -> int | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value.upper() in {"NULL", "N/A", "NA", "NONE"}:
        return None
    return int(float(value))


def _normalize_header(fieldnames: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in fieldnames:
        key = str(name or "").strip().lower()
        mapping[key] = name
    return mapping


def _iter_rows(cfg: ImportConfig) -> Iterator[tuple]:
    with cfg.csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames, "CSV header is required"
        header_map = _normalize_header(reader.fieldnames)
        missing = REQUIRED_COLUMNS - set(header_map.keys())
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

        for row in reader:
            trade_date = _parse_date(row.get(header_map["trade_date"], ""))
            vt_symbol = str(row.get(header_map["vt_symbol"], "") or "").strip()
            if not vt_symbol:
                raise ValueError("vt_symbol is required")
            close_price = _parse_float(row.get(header_map["close_price"], ""))
            adv_usd = _parse_float(row.get(header_map["adv_usd"], ""))
            med_volume = _parse_int(row.get(header_map["med_volume"], ""))
            created_at = _parse_datetime(row.get(header_map["created_at"], ""))
            market_cap = _parse_float(row.get(header_map["market_cap"], ""))

            yield (
                trade_date,
                vt_symbol,
                close_price,
                adv_usd,
                med_volume,
                market_cap,
                created_at,
            )


def _chunked(rows: Iterator[tuple], batch_size: int) -> Iterator[list[tuple]]:
    assert batch_size > 0, "batch_size must be > 0"
    batch: list[tuple] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _recreate_daily_selection_table() -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS daily_selection;")
    create_daily_selection_table()


def _insert_batch(rows: list[tuple]) -> int:
    if not rows:
        return 0
    try:
        from psycopg2.extras import execute_values
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("psycopg2-binary is required to import daily_selection") from exc

    sql = """
    INSERT INTO daily_selection (
        trade_date, vt_symbol, close_price, adv_usd, med_volume, market_cap, created_at
    ) VALUES %s
    ON CONFLICT (trade_date, vt_symbol) DO UPDATE SET
        close_price = EXCLUDED.close_price,
        adv_usd = EXCLUDED.adv_usd,
        med_volume = EXCLUDED.med_volume,
        market_cap = EXCLUDED.market_cap,
        created_at = EXCLUDED.created_at;
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=len(rows))
    return len(rows)


def import_csv(cfg: ImportConfig) -> None:
    if cfg.recreate_table:
        logger.info("[import_daily_selection_csv] 重建 daily_selection 表结构")
        _recreate_daily_selection_table()
    else:
        create_daily_selection_table()

    total = 0
    for batch in _chunked(_iter_rows(cfg), cfg.batch_size):
        total += _insert_batch(batch)
        logger.info("[import_daily_selection_csv] 已导入 %d 行", total)

    logger.info("[import_daily_selection_csv] 导入完成，总行数=%d", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="导入 daily_selection CSV")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV 文件路径")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="重建 daily_selection 表（会先 DROP TABLE）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="批量写入行数",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    cfg = ImportConfig(
        csv_path=csv_path,
        recreate_table=bool(args.recreate),
        batch_size=int(args.batch_size),
    )
    import_csv(cfg)


if __name__ == "__main__":
    main()
