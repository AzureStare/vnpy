"""
查看 AlphaLab signal parquet 的内容（schema/行数/时间范围/样例行），并可导出为 CSV。

用法：
python -m flagship.scripts.tools.inspect_signal_parquet \
  --lab-path lab/flagship_alpha_momentum \
  --signal-name flagship_alpha_momentum_20230217_20251231_lgb_signal

导出全量 CSV（包含 parquet 的所有列）：
python -m flagship.scripts.tools.inspect_signal_parquet \
  --lab-path lab/flagship_alpha_momentum \
  --signal-name flagship_alpha_momentum_20230217_20251231_lgb_signal \
  --out-csv /tmp/flagship_signal.csv

导出最近 10000 条（按 datetime 排序后取最近）：
python -m flagship.scripts.tools.inspect_signal_parquet \
  --lab-path lab/flagship_alpha_momentum \
  --signal-name flagship_alpha_momentum_20230217_20251231_lgb_signal \
  --out-csv /tmp/flagship_signal_recent_10000.csv \
  --out-csv-recent 10000

或者直接给 parquet 路径：
python -m flagship.scripts.tools.inspect_signal_parquet \
  --path lab/flagship_alpha_momentum/signal/flagship_alpha_momentum_20230217_20251231_lgb_signal.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _resolve_path(*, lab_path: str, signal_name: str, path: str | None) -> Path:
    if path:
        return Path(path)
    return Path(lab_path) / "signal" / f"{signal_name}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AlphaLab signal parquet")
    parser.add_argument("--lab-path", type=str, default="lab/flagship_alpha_momentum")
    parser.add_argument("--signal-name", type=str, default=None)
    parser.add_argument("--path", type=str, default=None, help="直接指定 parquet 路径（优先级最高）")
    parser.add_argument("--head", type=int, default=10, help="打印前 N 行（默认 10）")
    parser.add_argument("--out-csv", type=str, default=None, help="导出为 CSV 的输出路径（导出所有列）")
    parser.add_argument(
        "--out-csv-tail",
        type=int,
        default=None,
        help="仅导出最后 N 行（不额外排序；与 --out-csv-recent 互斥）",
    )
    parser.add_argument(
        "--out-csv-recent",
        type=int,
        default=None,
        help="按 datetime 排序后仅导出最近 N 行（与 --out-csv-tail 互斥）",
    )
    args = parser.parse_args()

    if not args.path and not args.signal_name:
        raise SystemExit("Either --path or --signal-name is required")

    if args.out_csv_tail is not None and args.out_csv_recent is not None:
        raise SystemExit("Only one of --out-csv-tail / --out-csv-recent can be set")

    parquet_path = _resolve_path(lab_path=args.lab_path, signal_name=str(args.signal_name), path=args.path)
    if not parquet_path.exists():
        raise SystemExit(f"Signal parquet not found: {parquet_path}")

    df = pl.read_parquet(parquet_path)
    print(f"[inspect_signal_parquet] path={parquet_path}")
    print(f"[inspect_signal_parquet] rows={df.height:,} cols={df.width}")
    print("[inspect_signal_parquet] schema:")
    for name, dtype in df.schema.items():
        print(f"  - {name}: {dtype}")

    if "datetime" in df.columns:
        dt_min = df.select(pl.col("datetime").min()).item()
        dt_max = df.select(pl.col("datetime").max()).item()
        print(f"[inspect_signal_parquet] datetime_range=[{dt_min}, {dt_max}]")

    if "vt_symbol" in df.columns:
        unique_symbols = df.select(pl.col("vt_symbol").n_unique()).item()
        print(f"[inspect_signal_parquet] vt_symbol_n_unique={unique_symbols:,}")

    if "signal" in df.columns:
        stats = df.select(
            [
                pl.col("signal").min().alias("signal_min"),
                pl.col("signal").max().alias("signal_max"),
                pl.col("signal").mean().alias("signal_mean"),
                pl.col("signal").std().alias("signal_std"),
                pl.col("signal").null_count().alias("signal_nulls"),
            ]
        ).to_dicts()[0]
        print(f"[inspect_signal_parquet] signal_stats={stats}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = df
        if args.out_csv_recent is not None:
            n = max(int(args.out_csv_recent), 1)
            if "datetime" not in export_df.columns:
                raise SystemExit("--out-csv-recent requires datetime column")
            export_df = export_df.sort(["datetime", "vt_symbol"]).tail(n) if "vt_symbol" in export_df.columns else export_df.sort("datetime").tail(n)
        elif args.out_csv_tail is not None:
            n = max(int(args.out_csv_tail), 1)
            export_df = export_df.tail(n)

        export_df.write_csv(out_path)
        print(
            f"[inspect_signal_parquet] exported_csv={out_path} rows={export_df.height:,} cols={export_df.width}"
        )

    head_n = max(int(args.head), 1)
    show_cols = [c for c in ["datetime", "vt_symbol", "signal", "score", "rank_5d", "ret_5d"] if c in df.columns]
    if not show_cols:
        show_cols = df.columns[: min(12, df.width)]
    print(f"[inspect_signal_parquet] head({head_n}) cols={show_cols}")
    print(df.select(show_cols).head(head_n))


if __name__ == "__main__":
    main()

