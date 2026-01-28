"""
Always-on intraday daemon (program-driven supervisor).

Purpose:
- Replace cron-based `start_intraday_runner.sh`
- Ensure intraday exit-only logic is running reliably during the session

Design:
- Runs continuously
- Uses Alpaca clock to determine current ET date and whether market is open
- Starts ONE `intraday_runner.py` child process per trading day after a configurable start time (default 09:35 ET)
- Restarts the child on crash (with a cap/backoff) while market is open
- Emits Prometheus textfile metrics for heartbeat/health
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from vnpy.trader.logger import logger

from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter
from flagship.config import PROJECT_ROOT
from flagship.trading.execution.broker_alpaca import AlpacaAdapter


EASTERN = ZoneInfo("America/New_York")


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


def _parse_hhmm(text: str) -> dtime:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty hhmm")
    try:
        return datetime.strptime(s, "%H:%M").time()
    except Exception as exc:
        raise ValueError(f"invalid HH:MM: {text}") from exc


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_proc_cmdline(pid: int) -> str | None:
    """
    Read /proc/<pid>/cmdline and return a space-joined argv string.
    Returns None on any error.
    """
    try:
        raw = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes()
    except Exception:
        return None

    parts: list[str] = []
    for chunk in raw.split(b"\x00"):
        if not chunk:
            continue
        parts.append(chunk.decode("utf-8", errors="ignore"))
    return " ".join(parts).strip() or None


def _pid_is_intraday_runner(pid: int | None) -> bool:
    """
    IMPORTANT:
    - A numeric pid may be reused after container restart.
    - We must verify the process cmdline to avoid false positives (e.g. pid reused by executor_daemon).
    """
    if not _pid_alive(pid):
        return False
    cmdline = _read_proc_cmdline(int(pid)) if pid else None
    if not cmdline:
        return False
    return ("flagship.trading.intraday.intraday_runner" in cmdline) or ("intraday_runner.py" in cmdline)


@dataclass
class IntradayDaemonState:
    last_started_trade_date: str | None = None
    child_pid: int | None = None
    last_start_timestamp: float | None = None
    last_exit_code: int | None = None
    restarts_today: int = 0


def _load_state(path: Path) -> IntradayDaemonState:
    if not path.exists():
        return IntradayDaemonState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return IntradayDaemonState(
            last_started_trade_date=data.get("last_started_trade_date"),
            child_pid=data.get("child_pid"),
            last_start_timestamp=data.get("last_start_timestamp"),
            last_exit_code=data.get("last_exit_code"),
            restarts_today=int(data.get("restarts_today") or 0),
        )
    except Exception as exc:
        logger.warning(f"[IntradayDaemon] failed to load state {path}: {exc}")
        return IntradayDaemonState()


def _save_state(path: Path, state: IntradayDaemonState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "last_started_trade_date": state.last_started_trade_date,
                    "child_pid": state.child_pid,
                    "last_start_timestamp": state.last_start_timestamp,
                    "last_exit_code": state.last_exit_code,
                    "restarts_today": int(state.restarts_today),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:
        logger.warning(f"[IntradayDaemon] failed to save state {path}: {exc}")


def _emit_metrics(
    writer: TextfileMetricsWriter,
    *,
    now_ts: float,
    is_open: bool,
    seconds_to_open: float | None,
    trade_date: date,
    state: IntradayDaemonState,
    enabled: bool,
) -> None:
    samples = [
        Sample("flagship_intraday_daemon_enabled", 1.0 if enabled else 0.0),
        Sample("flagship_intraday_daemon_heartbeat_timestamp_seconds", float(now_ts)),
        Sample("flagship_intraday_daemon_market_open", 1.0 if is_open else 0.0),
        Sample("flagship_intraday_daemon_child_running", 1.0 if _pid_is_intraday_runner(state.child_pid) else 0.0),
        Sample("flagship_intraday_daemon_restarts_today", float(int(state.restarts_today))),
    ]
    if seconds_to_open is not None:
        samples.append(Sample("flagship_intraday_daemon_seconds_to_open", float(seconds_to_open)))
    if state.last_start_timestamp is not None:
        samples.append(
            Sample("flagship_intraday_daemon_last_start_timestamp_seconds", float(state.last_start_timestamp))
        )
    if state.last_exit_code is not None:
        samples.append(Sample("flagship_intraday_daemon_last_exit_code", float(state.last_exit_code)))

    trade_date_midnight = datetime.combine(trade_date, datetime.min.time(), tzinfo=EASTERN).timestamp()
    samples.append(Sample("flagship_intraday_daemon_trade_date_timestamp_seconds", float(trade_date_midnight)))

    writer.write(
        samples=samples,
        help_map={
            "flagship_intraday_daemon_enabled": "Whether intraday daemon is enabled.",
            "flagship_intraday_daemon_heartbeat_timestamp_seconds": "Intraday daemon heartbeat (epoch seconds).",
            "flagship_intraday_daemon_market_open": "Whether Alpaca clock reports market is open.",
            "flagship_intraday_daemon_seconds_to_open": "Seconds to next market open (0 if open/unknown).",
            "flagship_intraday_daemon_child_running": "Whether intraday_runner child process is alive.",
            "flagship_intraday_daemon_last_start_timestamp_seconds": "Last child start timestamp (epoch seconds).",
            "flagship_intraday_daemon_last_exit_code": "Last observed child exit code.",
            "flagship_intraday_daemon_restarts_today": "Child restarts count for current trade date.",
            "flagship_intraday_daemon_trade_date_timestamp_seconds": "Current trading date midnight timestamp in America/New_York.",
        },
        type_map={
            "flagship_intraday_daemon_enabled": "gauge",
            "flagship_intraday_daemon_heartbeat_timestamp_seconds": "gauge",
            "flagship_intraday_daemon_market_open": "gauge",
            "flagship_intraday_daemon_seconds_to_open": "gauge",
            "flagship_intraday_daemon_child_running": "gauge",
            "flagship_intraday_daemon_last_start_timestamp_seconds": "gauge",
            "flagship_intraday_daemon_last_exit_code": "gauge",
            "flagship_intraday_daemon_restarts_today": "gauge",
            "flagship_intraday_daemon_trade_date_timestamp_seconds": "gauge",
        },
    )


def _start_child(*, trade_date: date, signal_top_n: int) -> subprocess.Popen[bytes]:
    """
    Start intraday_runner as a child process.
    We keep `--stop-after-close` default=true so it exits naturally after the session.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"intraday_runner_{trade_date.strftime('%Y%m%d')}.log"
    out = open(log_path, "ab", buffering=0)

    argv = [
        sys.executable,
        "-m",
        "flagship.trading.intraday.intraday_runner",
        "--mode",
        "exit-only",
        "--use-polygon-ws",
        "--rth-only",
        "--symbols-source",
        "both",
        "--signal-top-n",
        str(int(signal_top_n)),
    ]
    logger.info(f"[IntradayDaemon] starting child: {' '.join(argv)} (log={log_path})")
    # NOTE: close_fds True so the child doesn't inherit daemon fds.
    return subprocess.Popen(argv, cwd=str(PROJECT_ROOT), stdout=out, stderr=out, close_fds=True)


def run_daemon(
    *,
    poll_seconds: int,
    state_path: Path,
    start_time_et: dtime,
    max_restarts_per_day: int,
    signal_top_n: int,
) -> None:
    adapter = AlpacaAdapter()
    state = _load_state(state_path)
    metrics_writer = TextfileMetricsWriter("flagship_intraday_daemon.prom")

    child: subprocess.Popen[bytes] | None = None

    logger.info(
        f"[IntradayDaemon] started. state={state_path} start_hhmm={start_time_et.strftime('%H:%M')} "
        f"poll_seconds={poll_seconds} max_restarts_per_day={max_restarts_per_day} signal_top_n={signal_top_n}"
    )

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

            # Reset daily restart counter when trade date advances
            if state.last_started_trade_date and state.last_started_trade_date != trade_date.isoformat():
                state.restarts_today = 0

            # Guard against pid reuse: clear stale/foreign child pid (e.g. executor_daemon) ASAP
            if state.child_pid is not None and not _pid_is_intraday_runner(state.child_pid):
                logger.warning(
                    f"[IntradayDaemon] stale child_pid detected (pid={state.child_pid}, cmdline={_read_proc_cmdline(int(state.child_pid))}); clearing"
                )
                state.child_pid = None
                _save_state(state_path, state)

            # Observe child exit
            if child is not None:
                rc = child.poll()
                if rc is not None:
                    logger.warning(f"[IntradayDaemon] child exited: rc={rc}")
                    state.last_exit_code = int(rc)
                    state.child_pid = None
                    child = None
                    _save_state(state_path, state)

            # Decide whether we should start (or restart) intraday_runner today
            started_today = state.last_started_trade_date == trade_date.isoformat()
            after_start_time = now_et.time() >= start_time_et

            if is_open and after_start_time:
                if child is None and not _pid_is_intraday_runner(state.child_pid):
                    if started_today and state.restarts_today >= max_restarts_per_day:
                        logger.error(
                            f"[IntradayDaemon] restart cap reached for {trade_date}: restarts_today={state.restarts_today}"
                        )
                    else:
                        proc = _start_child(trade_date=trade_date, signal_top_n=signal_top_n)
                        child = proc
                        state.child_pid = int(proc.pid)
                        state.last_start_timestamp = float(time.time())
                        state.last_started_trade_date = trade_date.isoformat()
                        if started_today:
                            state.restarts_today = int(state.restarts_today) + 1
                        else:
                            state.restarts_today = 0
                        _save_state(state_path, state)

            _emit_metrics(
                writer=metrics_writer,
                now_ts=float(time.time()),
                is_open=is_open,
                seconds_to_open=seconds_to_open,
                trade_date=trade_date,
                state=state,
                enabled=True,
            )

            # Sleep policy:
            # - During open session: poll relatively frequently
            # - Otherwise: sleep up to 60s
            if is_open:
                sleep_s = max(5, int(poll_seconds))
            elif seconds_to_open is not None:
                sleep_s = min(60, max(5, int(min(float(poll_seconds), seconds_to_open))))
            else:
                sleep_s = min(60, max(5, int(poll_seconds)))
            time.sleep(sleep_s)

        except Exception as exc:
            logger.error(f"[IntradayDaemon] loop error: {exc}")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship IntradayDaemon (supervise intraday_runner, no cron).")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Clock polling interval seconds.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("logs") / "intraday_daemon_state.json",
        help="State file for idempotency/restart tracking.",
    )
    parser.add_argument(
        "--start-hhmm",
        type=str,
        default=os.getenv("FLAGSHIP_INTRADAY_START_HHMM", "09:35"),
        help="Start intraday runner after this ET time (HH:MM). Default: 09:35.",
    )
    parser.add_argument(
        "--max-restarts-per-day",
        type=int,
        default=int(os.getenv("FLAGSHIP_INTRADAY_MAX_RESTARTS_PER_DAY", "5")),
        help="Max child restarts per trading day while open. Default: 5.",
    )
    parser.add_argument(
        "--signal-top-n",
        type=int,
        default=int(os.getenv("FLAGSHIP_INTRADAY_SIGNAL_TOPN", "10")),
        help="Top-N signals to include in intraday watchlist (positions always included).",
    )
    args = parser.parse_args()

    state_file = args.state_file
    if not state_file.is_absolute():
        state_file = PROJECT_ROOT / state_file

    run_daemon(
        poll_seconds=int(args.poll_seconds),
        state_path=state_file,
        start_time_et=_parse_hhmm(str(args.start_hhmm)),
        max_restarts_per_day=int(args.max_restarts_per_day),
        signal_top_n=int(args.signal_top_n),
    )


if __name__ == "__main__":
    main()


