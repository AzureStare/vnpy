"""
Always-on executor daemon for Flagship Alpha-Momentum (open rebalance).

Goal:
- Run continuously
- Use Alpaca clock to detect market open
- Execute ONE rebalance per trading day at/after open
- Emit Prometheus textfile metrics for heartbeat/health

Important safety:
- Idempotent by trading date via a local state file
- Only executes when Alpaca clock reports market is OPEN
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from vnpy.trader.logger import logger

# Ensure project root importable under docker/cron
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter
from flagship.config.polygon_config import get_polygon_api_key
from flagship.trading.execution.broker_alpaca import AlpacaAdapter
from flagship.trading.config import DAILY_SIGNAL_FILE, LAB_PATH
from flagship.trading.calendar import infer_data_date_from_lab
from flagship.trading.orchestration.check_lab_freshness import check_lab_freshness
from flagship.trading.execution.open_rebalance import StrategyRunner, execute_rebalance
from flagship.trading.realtime.polygon_ws import PolygonTradePriceCache


EASTERN = ZoneInfo("America/New_York")


@dataclass
class DaemonState:
    last_rebalance_trade_date: str | None = None
    last_rebalance_timestamp: float | None = None
    last_signal_date: str | None = None


def _ensure_tz_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _clock_now_utc(clock: object) -> datetime:
    raw = getattr(clock, "timestamp", None)
    if isinstance(raw, datetime):
        return _ensure_tz_aware(raw).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _clock_next_open_utc(clock: object) -> datetime | None:
    raw = getattr(clock, "next_open", None)
    if isinstance(raw, datetime):
        return _ensure_tz_aware(raw).astimezone(timezone.utc)
    return None


def _load_state(path: Path) -> DaemonState:
    if not path.exists():
        return DaemonState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DaemonState(
            last_rebalance_trade_date=data.get("last_rebalance_trade_date"),
            last_rebalance_timestamp=data.get("last_rebalance_timestamp"),
            last_signal_date=data.get("last_signal_date"),
        )
    except Exception as exc:
        logger.warning(f"[ExecutorDaemon] failed to load state {path}: {exc}")
        return DaemonState()


def _save_state(path: Path, state: DaemonState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "last_rebalance_trade_date": state.last_rebalance_trade_date,
                    "last_rebalance_timestamp": state.last_rebalance_timestamp,
                    "last_signal_date": state.last_signal_date,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:
        logger.warning(f"[ExecutorDaemon] failed to save state {path}: {exc}")


def _signal_max_date(signal_path: Path) -> date | None:
    if not signal_path.exists():
        return None
    try:
        last_dt = (
            pl.scan_parquet(signal_path)
            .select(pl.col("datetime").max())
            .collect()
            .item()
        )
        if isinstance(last_dt, datetime):
            return last_dt.date()
    except Exception:
        return None
    return None


def _emit_metrics(
    writer: TextfileMetricsWriter,
    *,
    now_ts: float,
    is_open: bool,
    seconds_to_open: float | None,
    trade_date: date,
    state: DaemonState,
    errors_total: int,
) -> None:
    samples = [
        Sample("flagship_executor_daemon_heartbeat_timestamp_seconds", float(now_ts)),
        Sample("flagship_executor_daemon_market_open", 1.0 if is_open else 0.0),
        Sample(
            "flagship_executor_daemon_errors_total",
            float(errors_total),
        ),
    ]

    if seconds_to_open is not None:
        samples.append(Sample("flagship_executor_daemon_seconds_to_open", float(seconds_to_open)))

    if state.last_rebalance_timestamp is not None:
        samples.append(
            Sample(
                "flagship_executor_daemon_last_rebalance_timestamp_seconds",
                float(state.last_rebalance_timestamp),
            )
        )

    # Export current ET date as a stable timestamp (00:00 ET)
    trade_date_midnight = datetime.combine(trade_date, datetime.min.time(), tzinfo=EASTERN).timestamp()
    samples.append(Sample("flagship_executor_daemon_trade_date_timestamp_seconds", float(trade_date_midnight)))

    writer.write(
        samples=samples,
        help_map={
            "flagship_executor_daemon_heartbeat_timestamp_seconds": "Executor daemon heartbeat (epoch seconds).",
            "flagship_executor_daemon_market_open": "Whether Alpaca clock reports market is open.",
            "flagship_executor_daemon_seconds_to_open": "Seconds to next market open (0 if open/unknown).",
            "flagship_executor_daemon_last_rebalance_timestamp_seconds": "Last successful open rebalance timestamp (epoch seconds).",
            "flagship_executor_daemon_trade_date_timestamp_seconds": "Current trading date midnight timestamp in America/New_York.",
            "flagship_executor_daemon_errors_total": "Total errors observed in current daemon process.",
        },
        type_map={
            "flagship_executor_daemon_heartbeat_timestamp_seconds": "gauge",
            "flagship_executor_daemon_market_open": "gauge",
            "flagship_executor_daemon_seconds_to_open": "gauge",
            "flagship_executor_daemon_last_rebalance_timestamp_seconds": "gauge",
            "flagship_executor_daemon_trade_date_timestamp_seconds": "gauge",
            "flagship_executor_daemon_errors_total": "gauge",
        },
    )


def _is_ready_to_trade(*, expected_signal_date: date, signal_path: Path) -> bool:
    signal_date = _signal_max_date(signal_path)
    if signal_date is None:
        logger.error(f"[ExecutorDaemon] signal missing: {signal_path}")
        return False
    if signal_date != expected_signal_date:
        logger.error(
            f"[ExecutorDaemon] signal stale/mismatch: signal_date={signal_date} expected={expected_signal_date} path={signal_path}"
        )
        return False

    try:
        # check-only: do not auto-fix at open (avoid heavy work delaying execution)
        result = check_lab_freshness(
            lab_path=LAB_PATH,
            expected_date=expected_signal_date,
            fix=False,
            check_datasets=False,
        )
        ok = (
            result.selection_count > 0
            and result.bars_missing == 0
            and result.bars_stale == 0
            and result.signal_ok
            and result.model_ok
        )
        if not ok:
            logger.error(f"[ExecutorDaemon] lab not ready: {result}")
        return ok
    except Exception as exc:
        logger.error(f"[ExecutorDaemon] readiness check failed: {exc}")
        return False


def run_daemon(
    *,
    strategy_version: str,
    poll_seconds: int,
    state_path: Path,
    cancel_open_orders: bool,
    dry_run: bool,
    use_polygon_ws: bool,
) -> None:
    adapter = AlpacaAdapter()
    runner = StrategyRunner(adapter, strategy_version=strategy_version)

    state = _load_state(state_path)
    metrics_writer = TextfileMetricsWriter("flagship_executor_daemon.prom")

    errors_total = 0
    ws_cache: PolygonTradePriceCache | None = None

    logger.info(f"[ExecutorDaemon] started. state={state_path} strategy={strategy_version} dry_run={dry_run}")

    while True:
        try:
            clock = adapter.client.get_clock()
            now_utc = _clock_now_utc(clock)
            now_et = now_utc.astimezone(EASTERN)
            trade_date = now_et.date()

            is_open = bool(getattr(clock, "is_open", False))
            next_open_utc = _clock_next_open_utc(clock)
            seconds_to_open: float | None = None
            if next_open_utc is not None:
                seconds_to_open = max(0.0, float((next_open_utc - now_utc).total_seconds()))

            _emit_metrics(
                writer=metrics_writer,
                now_ts=float(time.time()),
                is_open=is_open,
                seconds_to_open=seconds_to_open,
                trade_date=trade_date,
                state=state,
                errors_total=errors_total,
            )

            if is_open:
                if state.last_rebalance_trade_date == trade_date.isoformat():
                    time.sleep(max(5, int(poll_seconds)))
                    continue

                expected_signal_date = infer_data_date_from_lab(LAB_PATH)
                if not _is_ready_to_trade(expected_signal_date=expected_signal_date, signal_path=DAILY_SIGNAL_FILE):
                    time.sleep(max(10, int(poll_seconds)))
                    continue

                # Refresh signals right before computing targets (daemon may have run overnight)
                runner.inject_signal(DAILY_SIGNAL_FILE)
                targets = runner.run_daily_logic()
                if not targets:
                    logger.warning("[ExecutorDaemon] no targets generated, skip.")
                    state.last_rebalance_trade_date = trade_date.isoformat()
                    state.last_rebalance_timestamp = float(time.time())
                    state.last_signal_date = expected_signal_date.isoformat()
                    _save_state(state_path, state)
                    continue

                # Optional Polygon WS for monitoring (not required for order placement)
                if use_polygon_ws and ws_cache is None:
                    try:
                        sig_df = runner.engine.signal_df
                        roots = sorted({str(v).split(".")[0] for v in sig_df["vt_symbol"].to_list()})
                        ws_cache = PolygonTradePriceCache(get_polygon_api_key(), roots)
                        ws_cache.start()
                    except Exception as exc:
                        logger.warning(f"[ExecutorDaemon] polygon ws start failed: {exc}")
                        ws_cache = None

                if cancel_open_orders:
                    adapter.cancel_all_open_orders()

                info = adapter.get_account_info()
                logger.info(
                    f"[ExecutorDaemon] Pre-trade Account: cash={info.cash:.2f}, equity={info.equity:.2f}, buying_power={info.buying_power:.2f}"
                )

                if dry_run:
                    logger.warning(f"[ExecutorDaemon] DRY RUN: would execute rebalance for {len(targets)} targets.")
                else:
                    execute_rebalance(adapter, targets)

                state.last_rebalance_trade_date = trade_date.isoformat()
                state.last_rebalance_timestamp = float(time.time())
                state.last_signal_date = expected_signal_date.isoformat()
                _save_state(state_path, state)

                logger.info(
                    f"[ExecutorDaemon] rebalance done. trade_date={trade_date} signal_date={expected_signal_date}"
                )

            # Sleep policy:
            # - If next open is far away, sleep up to 60s (still emitting heartbeat each loop).
            if not is_open and seconds_to_open is not None:
                sleep_s = min(60, max(5, int(min(float(poll_seconds), seconds_to_open))))
            else:
                sleep_s = max(5, int(poll_seconds))
            time.sleep(sleep_s)

        except Exception as exc:
            errors_total += 1
            logger.error(f"[ExecutorDaemon] loop error: {exc}")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship Executor Daemon (open rebalance, always-on).")
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Clock polling interval seconds.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("logs") / "executor_daemon_state.json",
        help="Local state file for idempotency (per trading day).",
    )
    parser.add_argument(
        "--cancel-open-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cancel all open orders before execution (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not place orders, only compute and log targets (default: false).",
    )
    parser.add_argument(
        "--use-polygon-ws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subscribe Polygon WS for monitoring (default: true).",
    )
    args = parser.parse_args()

    state_file = args.state_file
    if not state_file.is_absolute():
        state_file = PROJECT_ROOT / state_file

    run_daemon(
        strategy_version=args.strategy,
        poll_seconds=int(args.poll_seconds),
        state_path=state_file,
        cancel_open_orders=bool(args.cancel_open_orders),
        dry_run=bool(args.dry_run),
        use_polygon_ws=bool(args.use_polygon_ws),
    )


if __name__ == "__main__":
    main()


