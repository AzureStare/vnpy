"""
Diagnostics for v7 VALID set quality:
- missingness/dropna ratios
- label_excess_5d distribution and day-by-day drift

This is intended to explain model training warnings like:
  [train_daily_model] Validation score ... is low.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.dataset import Segment
from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger

from flagship.factors.alpha_momentum.v7_dataset import FlagshipAlphaMomentumV7Dataset
from flagship.config import PROJECT_ROOT
from flagship.trading.config import LAB_PATH
from flagship.universe.pg_ticker_db import get_selected_symbols_in_range


FEATURE_COLS_V7 = [
    "alpha_mom",
    "alpha_vwap",
    "alpha_trend",
    "rs_60d",
    "beta",
    "atr_percent",
    "return_5d",
]


@dataclass(frozen=True)
class ValidDiagnostics:
    universe_symbols: int
    raw_rows: int
    valid_rows_total: int
    valid_unique_dates: int
    missing_cols: list[str]
    null_counts: dict[str, int]
    lr_dropna_before_excl_spy: int
    lr_dropna_after: int
    lr_kept_pct: float | None
    label_stats: dict[str, float | None]
    drift_last10: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _compute(
    *,
    target_date: date,
    train_start: date,
    valid_start: date,
    valid_end: date,
    lab_path: Path,
    lookback_days: int,
) -> ValidDiagnostics:
    lab = AlphaLab(str(lab_path))

    vt_symbols = get_selected_symbols_in_range(start_date=train_start, end_date=valid_end)
    if "SPY.NASDAQ" not in vt_symbols:
        vt_symbols.append("SPY.NASDAQ")

    data_load_start = valid_start - timedelta(days=int(lookback_days))
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=data_load_start.isoformat(),
        end=valid_end.isoformat(),
        extended_days=0,
    )
    if raw_df is None:
        raw_df = pl.DataFrame()

    train_period = ((valid_start - timedelta(days=30)).isoformat(), (valid_start - timedelta(days=8)).isoformat())
    valid_period = (valid_start.isoformat(), valid_end.isoformat())
    test_period = (target_date.isoformat(), (target_date + timedelta(days=5)).isoformat())

    dataset = FlagshipAlphaMomentumV7Dataset(df=raw_df, train_period=train_period, valid_period=valid_period, test_period=test_period)
    dataset.prepare_data(filters=None)
    dataset.process_data()

    valid_df = dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"])

    need = ["vt_symbol", "datetime", "label_excess_5d", *FEATURE_COLS_V7]
    missing_cols = [c for c in need if c not in valid_df.columns]
    select_cols = [c for c in need if c in valid_df.columns]
    valid_df = valid_df.select(select_cols)

    null_counts: dict[str, int] = {}
    for c in ["label_excess_5d", *FEATURE_COLS_V7]:
        if c not in valid_df.columns:
            continue
        null_counts[c] = int(valid_df.select(pl.col(c).null_count()).item())

    req_cols = [c for c in (["label_excess_5d", *FEATURE_COLS_V7]) if c in valid_df.columns]
    pre = valid_df.filter(pl.col("vt_symbol") != "SPY.NASDAQ")
    post = pre.drop_nulls(subset=req_cols)
    kept_pct = None
    if pre.height > 0:
        kept_pct = float(post.height) / float(pre.height)

    label_stats: dict[str, float | None] = {
        "mean": None,
        "std": None,
        "p05": None,
        "p50": None,
        "p95": None,
        "pos_rate": None,
    }
    drift_last10: list[dict[str, Any]] = []

    if "label_excess_5d" in valid_df.columns:
        lab_nonnull = valid_df.filter(pl.col("label_excess_5d").is_not_null())
        if lab_nonnull.height > 0:
            d = lab_nonnull.select(
                pl.mean("label_excess_5d").alias("mean"),
                pl.std("label_excess_5d").alias("std"),
                pl.quantile("label_excess_5d", 0.05).alias("p05"),
                pl.quantile("label_excess_5d", 0.50).alias("p50"),
                pl.quantile("label_excess_5d", 0.95).alias("p95"),
                (pl.col("label_excess_5d") > 0).mean().alias("pos_rate"),
            ).to_dicts()[0]
            for k in list(label_stats.keys()):
                label_stats[k] = _safe_float(d.get(k))

            drift = (
                lab_nonnull
                .with_columns(pl.col("datetime").dt.date().alias("d"))
                .group_by("d")
                .agg(
                    pl.len().alias("n"),
                    pl.mean("label_excess_5d").alias("mean"),
                    pl.quantile("label_excess_5d", 0.05).alias("p05"),
                    (pl.col("label_excess_5d") > 0).mean().alias("pos_rate"),
                )
                .sort("d")
            )
            drift_last10 = []
            for row in drift.tail(10).to_dicts():
                d0 = row.get("d")
                if isinstance(d0, date):
                    row["d"] = d0.isoformat()
                drift_last10.append(row)

    return ValidDiagnostics(
        universe_symbols=int(len(vt_symbols)),
        raw_rows=int(raw_df.height),
        valid_rows_total=int(valid_df.height),
        valid_unique_dates=int(valid_df.select(pl.col("datetime").dt.date().n_unique()).item() or 0),
        missing_cols=missing_cols,
        null_counts=null_counts,
        lr_dropna_before_excl_spy=int(pre.height),
        lr_dropna_after=int(post.height),
        lr_kept_pct=kept_pct,
        label_stats=label_stats,
        drift_last10=drift_last10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VALID set diagnostics for v7 model training.")
    parser.add_argument("--lab-path", type=str, default=str(LAB_PATH))
    parser.add_argument("--metrics-path", type=str, default=str(PROJECT_ROOT / "logs" / "app" / "model_metrics.json"))
    parser.add_argument("--lookback-days", type=int, default=140)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    lab_path = Path(args.lab_path)
    if not lab_path.is_absolute():
        lab_path = PROJECT_ROOT / lab_path

    metrics_path = Path(args.metrics_path)
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path

    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    target_date = _parse_date(str(metrics.get("target_date") or date.today().isoformat()))
    valid_period = metrics.get("valid_period") or []
    train_period = metrics.get("train_period") or []

    if isinstance(valid_period, list) and len(valid_period) == 2:
        valid_start = _parse_date(str(valid_period[0]))
        valid_end = _parse_date(str(valid_period[1]))
    else:
        valid_end = target_date - timedelta(days=1)
        valid_start = valid_end - timedelta(days=60)

    if isinstance(train_period, list) and len(train_period) == 2:
        train_start = _parse_date(str(train_period[0]))
    else:
        train_start = valid_start - timedelta(days=1095)

    diag = _compute(
        target_date=target_date,
        train_start=train_start,
        valid_start=valid_start,
        valid_end=valid_end,
        lab_path=lab_path,
        lookback_days=int(args.lookback_days),
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "target_date": target_date.isoformat(),
        "valid_period": [valid_start.isoformat(), valid_end.isoformat()],
        "universe_symbols": diag.universe_symbols,
        "raw_rows": diag.raw_rows,
        "valid_rows_total": diag.valid_rows_total,
        "valid_unique_dates": diag.valid_unique_dates,
        "missing_cols": diag.missing_cols,
        "null_counts": diag.null_counts,
        "lr_dropna_before_excl_spy": diag.lr_dropna_before_excl_spy,
        "lr_dropna_after": diag.lr_dropna_after,
        "lr_kept_pct": diag.lr_kept_pct,
        "label_excess_5d_stats": diag.label_stats,
        "label_excess_5d_drift_last10": diag.drift_last10,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[valid_set_diagnostics] wrote {out}")


if __name__ == "__main__":
    main()

