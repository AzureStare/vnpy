"""
Python orchestrator for Flagship daily paper trading pipeline.

This file consolidates the core logic previously implemented in:
- legacy `run_full_daily_cycle.sh` (removed)

Responsibilities (post-close pipeline):
- Update market indices (SPY/QQQ/VIX)
- Incrementally update daily bars (full market)
- Infer DATA_DATE (previous trading day close)
- Build daily_selection (U_t) and ensure data completeness
- Incrementally update minute bars for the selected universe
- Train model (weekly on Mondays or if missing)
- Generate daily signals for DATA_DATE
- Refresh Ops Console snapshots + report (non-blocking)

NOTE:
- Open rebalance execution is intentionally NOT part of this runner when using
  a separate always-on `ExecutorDaemon` to avoid duplicate order placement.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger

# Ensure project root importable under cron
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter
from flagship.trading.config import LAB_PATH, LIVE_MODEL_PATH, DAILY_SIGNAL_FILE
from flagship.trading.calendar import (
    describe_trading_date,
    infer_data_date_from_lab,
    is_market_closed_day,
)
from flagship.trading.orchestration.update_market_indices import update_market_indices
from flagship.market_data.update_lab_data_incremental import incremental_update
from flagship.trading.orchestration.run_daily_selection import run_daily_selection
from flagship.trading.orchestration.ensure_data_completeness import check_and_backfill_data
from flagship.trading.orchestration.check_lab_freshness import check_lab_freshness
from flagship.trading.orchestration.train_daily_model import train_daily_model
from flagship.trading.orchestration.run_live_inference import run_live_inference
from flagship.monitoring.app_console_snapshot import (
    DEFAULT_OUTPUT_DIR as SNAPSHOT_OUTPUT_DIR,
    DEFAULT_SELECTION_TOP_N,
    snapshot_market_status,
    snapshot_orders,
    snapshot_performance,
    snapshot_portfolio,
    snapshot_selection,
)
from flagship.monitoring.app_console_report import generate_report
from flagship.ops.reporting.daily_trade_recap import generate_trade_recap


EASTERN = ZoneInfo("America/New_York")


@dataclass
class DailyCycleMetrics:
    trading_date: date
    data_date: str = "unknown"
    holiday_mode: str = "0"

    running: int = 1
    success: int = 0

    start_ts: float = 0.0
    last_step: str = "start"
    last_step_duration_s: float = 0.0

    writer: TextfileMetricsWriter | None = None

    def start(self) -> None:
        self.start_ts = time.time()
        self.running = 1
        self.success = 0
        self.last_step = "start"
        self.last_step_duration_s = 0.0
        self._write()

    def mark_done(self) -> None:
        self.running = 0
        self.success = 1
        self.last_step = "done"
        self.last_step_duration_s = 0.0
        self._write()

    def mark_failed(self, *, step: str) -> None:
        self.running = 0
        self.success = 0
        self.last_step = step
        self._write()

    def _write(self) -> None:
        if self.writer is None:
            self.writer = TextfileMetricsWriter("flagship_daily_cycle.prom")

        now_ts = time.time()
        total_s = max(0.0, float(now_ts - float(self.start_ts or now_ts)))

        samples = [
            Sample(
                name="flagship_daily_cycle_running",
                value=float(self.running),
                labels={
                    "trading_date": self.trading_date.isoformat(),
                    "data_date": self.data_date,
                    "holiday_mode": self.holiday_mode,
                },
            ),
            Sample(
                name="flagship_daily_cycle_success",
                value=float(self.success),
                labels={
                    "trading_date": self.trading_date.isoformat(),
                    "data_date": self.data_date,
                },
            ),
            Sample(
                name="flagship_daily_cycle_last_update_timestamp_seconds",
                value=float(now_ts),
            ),
            Sample(
                name="flagship_daily_cycle_last_step_duration_seconds",
                value=float(self.last_step_duration_s),
                labels={"step": self.last_step},
            ),
            Sample(
                name="flagship_daily_cycle_total_duration_seconds",
                value=float(total_s),
            ),
        ]

        self.writer.write(
            samples=samples,
            help_map={
                "flagship_daily_cycle_running": "Whether daily cycle pipeline is currently running.",
                "flagship_daily_cycle_success": "Whether last daily cycle completed successfully.",
                "flagship_daily_cycle_last_update_timestamp_seconds": "Last metrics update timestamp (epoch seconds).",
                "flagship_daily_cycle_last_step_duration_seconds": "Duration of last completed step (seconds).",
                "flagship_daily_cycle_total_duration_seconds": "Total duration since pipeline start (seconds).",
            },
            type_map={
                "flagship_daily_cycle_running": "gauge",
                "flagship_daily_cycle_success": "gauge",
                "flagship_daily_cycle_last_update_timestamp_seconds": "gauge",
                "flagship_daily_cycle_last_step_duration_seconds": "gauge",
                "flagship_daily_cycle_total_duration_seconds": "gauge",
            },
        )


@contextmanager
def _step(metrics: DailyCycleMetrics, step_name: str):
    step_start = time.time()
    metrics.last_step = step_name
    metrics.last_step_duration_s = 0.0
    metrics._write()
    try:
        yield
    finally:
        metrics.last_step = step_name
        metrics.last_step_duration_s = max(0.0, float(time.time() - step_start))
        metrics._write()


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _default_trading_date_et() -> date:
    return datetime.now(EASTERN).date()


def run_daily_cycle(
    *,
    trading_date: date,
    strategy_version: str,
    lab_path: Path,
    output_dir: Path,
    selection_top_n: int,
) -> None:
    metrics = DailyCycleMetrics(trading_date=trading_date)
    metrics.holiday_mode = "1" if is_market_closed_day(trading_date) else "0"
    metrics.start()

    logger.info(f"[DailyCycle] trading_date={trading_date} {describe_trading_date(trading_date)}")

    lab = AlphaLab(str(lab_path))

    try:
        # 1) Market indices
        with _step(metrics, "1_update_market_indices"):
            update_market_indices(lab_path=lab_path, lookback_days=5)

        # 2) Daily full market incremental update (ref_tickers_cs)
        with _step(metrics, "2_update_daily_full_market"):
            incremental_update(
                lab=lab,
                interval=Interval.DAILY,
                end_date=trading_date,
                mode="ref_tickers_cs",  # type: ignore[arg-type]
                overlap_days=1,
            )

        # Infer DATA_DATE from lab (previous trading day close)
        metrics.data_date = infer_data_date_from_lab(lab_path).isoformat()
        metrics._write()
        data_date = _parse_date(metrics.data_date)
        logger.info(f"[DailyCycle] inferred data_date={data_date} (from lab index parquet)")

        # Holiday/Weekend: update-only mode
        if metrics.holiday_mode == "1":
            with _step(metrics, "holiday_check_freshness"):
                check_lab_freshness(
                    lab_path=lab_path,
                    expected_date=data_date,
                    fix=True,
                    check_model=False,
                    check_signals=False,
                    check_datasets=False,
                )
            metrics.mark_done()
            return

        # 3) Daily selection (U_t)
        with _step(metrics, "3_daily_selection"):
            run_daily_selection(
                target_date=data_date,
                lab_path=lab_path,
                strategy_version=strategy_version,
            )

        # 4) Ensure data completeness for U_t
        with _step(metrics, "4_ensure_data_completeness"):
            check_and_backfill_data(target_date=data_date, lab_path=lab_path, lookback_days=180)

        # 4.8) Minute bars for daily_selection universe (only for DATA_DATE selection)
        with _step(metrics, "4_8_update_minute_selection"):
            incremental_update(
                lab=lab,
                interval=Interval.MINUTE,
                end_date=trading_date,
                mode="daily_selection",  # type: ignore[arg-type]
                selection_start=data_date,
                selection_end=data_date,
                overlap_days=1,
            )

        # 4.5) Freshness check & auto-fix
        with _step(metrics, "4_5_check_lab_freshness"):
            check_lab_freshness(lab_path=lab_path, expected_date=data_date, fix=True)

        # 5) Train model (weekly on Mondays, or if missing)
        should_train = (not LIVE_MODEL_PATH.exists()) or (trading_date.weekday() == 0)
        if should_train:
            with _step(metrics, "5_train_model"):
                train_daily_model(
                    target_date=trading_date,
                    lab_path=lab_path,
                    output_model_path=LIVE_MODEL_PATH,
                    strategy_version=strategy_version,
                )

        # 6) Run inference (always generate signals for DATA_DATE)
        with _step(metrics, "6_run_inference"):
            run_live_inference(
                target_date=data_date,
                lab_path=lab_path,
                model_path=LIVE_MODEL_PATH,
                output_file=DAILY_SIGNAL_FILE,
                strategy_version=strategy_version,
            )

        # 6.5) Re-check freshness (signals + model; skip datasets)
        with _step(metrics, "6_5_recheck_lab_freshness"):
            check_lab_freshness(
                lab_path=lab_path,
                expected_date=data_date,
                fix=True,
                check_datasets=False,
            )

        # 8.5) Refresh Ops Console snapshots (non-blocking)
        with _step(metrics, "8_5_refresh_ops_console"):
            try:
                snapshot_portfolio(output_dir)
                snapshot_selection(output_dir, lab_path, selection_top_n)
                snapshot_orders(output_dir)
                snapshot_performance(output_dir)
                snapshot_market_status(output_dir)
            except Exception as exc:
                logger.warning(f"[DailyCycle] snapshot failed (non-blocking): {exc}")

        # 8.6) Generate Ops Console report (non-blocking)
        with _step(metrics, "8_6_generate_ops_report"):
            try:
                generate_report(output_dir)
            except Exception as exc:
                logger.warning(f"[DailyCycle] report generation failed (non-blocking): {exc}")

        # 8.7) Generate daily trade recap (non-blocking)
        with _step(metrics, "8_7_generate_trade_recap"):
            try:
                # Archive EOD portfolio snapshot for next-day realized PnL cost basis fallback
                src = output_dir / "portfolio.json"
                if src.exists():
                    dst = output_dir / f"portfolio_{trading_date.strftime('%Y%m%d')}.json"
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                    try:
                        dst.chmod(0o644)
                    except Exception:
                        pass

                generate_trade_recap(
                    trade_date=trading_date,
                    output_dir=output_dir,
                    log_dir=PROJECT_ROOT / "logs",
                    strategy_version=strategy_version,
                )
            except Exception as exc:
                logger.warning(f"[DailyCycle] trade recap generation failed (non-blocking): {exc}")

        metrics.mark_done()
    except Exception:
        metrics.mark_failed(step=metrics.last_step)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flagship daily paper trading pipeline (Python orchestrator).")
    parser.add_argument("--trading-date", type=str, help="YYYY-MM-DD (ET). Defaults to now in America/New_York.")
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7")
    parser.add_argument("--lab-path", type=Path, default=LAB_PATH, help="AlphaLab path (default: flagship lab).")
    parser.add_argument("--output-dir", type=Path, default=SNAPSHOT_OUTPUT_DIR, help="Ops snapshot output directory.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_SELECTION_TOP_N, help="Top N selection rows by signal.")
    args = parser.parse_args()

    trading_date = _parse_date(args.trading_date) if args.trading_date else _default_trading_date_et()

    lab_path = args.lab_path
    if not lab_path.is_absolute():
        lab_path = PROJECT_ROOT / lab_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    run_daily_cycle(
        trading_date=trading_date,
        strategy_version=args.strategy,
        lab_path=lab_path,
        output_dir=output_dir,
        selection_top_n=int(args.top_n),
    )


if __name__ == "__main__":
    main()


