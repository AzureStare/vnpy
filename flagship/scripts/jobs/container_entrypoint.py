#!/usr/bin/env python3
"""
Container entrypoint (Python).

Why:
- Avoid bash-based orchestration (hard to test/extend)
- Keep "what runs in production" in a single folder (`flagship/scripts/jobs/`)

Responsibilities:
- Write /etc/cron.d/flagship (daily cycle + portfolio heartbeat; intraday cron optional)
- Start FastAPI (uvicorn)
- Start long-running daemons (ExecutorDaemon / IntradayDaemon) with restart loops
- Run cron in foreground
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter


APP_DIR = Path("/app")
LOG_DIR = APP_DIR / "logs"


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _best_effort_set_timezone(tz_value: str) -> None:
    tz_file = Path("/usr/share/zoneinfo") / tz_value
    if not tz_file.exists():
        return
    try:
        localtime = Path("/etc/localtime")
        if localtime.exists() or localtime.is_symlink():
            localtime.unlink(missing_ok=True)
        os.symlink(str(tz_file), str(localtime))
        Path("/etc/timezone").write_text(tz_value + "\n", encoding="utf-8")
    except Exception:
        return


def _write_cron_file() -> None:
    tz_value = _env("TZ", "America/New_York")
    _best_effort_set_timezone(tz_value)

    schedule = _env("FLAGSHIP_CRON_SCHEDULE", "15 16 * * 1-5")
    cmd = _env(
        "FLAGSHIP_CRON_COMMAND",
        "python -m flagship.trading.orchestration.daily_cycle_runner --strategy v7",
    )

    enable_intraday_daemon = _env("FLAGSHIP_ENABLE_INTRADAY_DAEMON", "1")

    schedule_portfolio = _env("FLAGSHIP_CRON_SCHEDULE_PORTFOLIO", "")
    cmd_portfolio = _env("FLAGSHIP_CRON_COMMAND_PORTFOLIO", "")

    schedule_intraday = _env("FLAGSHIP_CRON_SCHEDULE_INTRADAY", "")
    cmd_intraday = _env("FLAGSHIP_CRON_COMMAND_INTRADAY", "")

    cron_file = Path("/etc/cron.d/flagship")
    lines: list[str] = [
        "SHELL=/bin/bash",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"TZ={tz_value}",
    ]

    pass_through = [
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_DATABASE",
        "DATABASE_USER",
        "DATABASE_PASSWORD",
        "FLAGSHIP_TEXTFILE_DIR",
        "FLAGSHIP_INTRADAY_SIGNAL_TOPN",
        "FLAGSHIP_EXECUTOR_MAX_WAIT_SECONDS",
        "ENABLE_INTRADAY_RUNNER",
        "MASSIVE_API_KEY",
        "POLYGON_API_KEY",
    ]
    for k in pass_through:
        v = _env(k, "")
        if v:
            lines.append(f"{k}={v}")

    # Daily cycle
    lines.append(f"{schedule} root {cmd} >> {LOG_DIR}/cron.log 2>&1")

    # Portfolio heartbeat
    if schedule_portfolio and cmd_portfolio:
        lines.append(f"{schedule_portfolio} root {cmd_portfolio} >> {LOG_DIR}/cron.log 2>&1")

    # Legacy intraday cron (disabled when IntradayDaemon is enabled)
    if enable_intraday_daemon != "1" and schedule_intraday and cmd_intraday:
        lines.append(f"{schedule_intraday} root {cmd_intraday} >> {LOG_DIR}/cron.log 2>&1")

    cron_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cron_file.chmod(0o644)

    print(f"[container_entrypoint] TZ={tz_value}", flush=True)
    print(f"[container_entrypoint] CRON={schedule} {cmd}", flush=True)


def _start_uvicorn() -> subprocess.Popen[bytes]:
    argv = [
        sys.executable,
        "-m",
        "uvicorn",
        "flagship.app_console.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    print("[container_entrypoint] Starting FastAPI server (background)...", flush=True)
    return subprocess.Popen(argv, cwd=str(APP_DIR))


def _supervise(name: str, argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with open(log_path, "ab", buffering=0) as out:
            out.write(f"[container_entrypoint] starting {name}: {' '.join(argv)}\n".encode("utf-8", errors="ignore"))
            proc = subprocess.Popen(argv, cwd=str(APP_DIR), stdout=out, stderr=out, close_fds=True)
            rc = proc.wait()
            out.write(
                f"[container_entrypoint] {name} exited rc={rc}, restarting in 5s...\n".encode("utf-8", errors="ignore")
            )
        time.sleep(5)


def _start_daemons() -> None:
    enable_executor = _env("FLAGSHIP_ENABLE_EXECUTOR_DAEMON", "1")
    if enable_executor == "1":
        poll_seconds = _env("FLAGSHIP_EXECUTOR_DAEMON_POLL_SECONDS", "30")
        dry_run = _env("FLAGSHIP_EXECUTOR_DAEMON_DRY_RUN", "0") == "1"
        argv = [sys.executable, "-m", "flagship.trading.execution.executor_daemon", "--poll-seconds", poll_seconds]
        if dry_run:
            argv.append("--dry-run")
        print("[container_entrypoint] Starting ExecutorDaemon (background)...", flush=True)
        threading.Thread(
            target=_supervise,
            args=("ExecutorDaemon", argv, LOG_DIR / "executor_daemon.log"),
            daemon=True,
        ).start()
    else:
        print(f"[container_entrypoint] ExecutorDaemon disabled (FLAGSHIP_ENABLE_EXECUTOR_DAEMON={enable_executor})", flush=True)

    enable_intraday = _env("FLAGSHIP_ENABLE_INTRADAY_DAEMON", "1")
    if enable_intraday == "1":
        poll_seconds = _env("FLAGSHIP_INTRADAY_DAEMON_POLL_SECONDS", "30")
        argv = [sys.executable, "-m", "flagship.trading.intraday.intraday_daemon", "--poll-seconds", poll_seconds]
        print("[container_entrypoint] Starting IntradayDaemon (background)...", flush=True)
        threading.Thread(
            target=_supervise,
            args=("IntradayDaemon", argv, LOG_DIR / "intraday_daemon.log"),
            daemon=True,
        ).start()
    else:
        print(f"[container_entrypoint] IntradayDaemon disabled (FLAGSHIP_ENABLE_INTRADAY_DAEMON={enable_intraday})", flush=True)


def _heartbeat_loop() -> None:
    writer = TextfileMetricsWriter("flagship_container_entrypoint.prom")
    while True:
        try:
            writer.write(
                samples=[
                    Sample("flagship_container_entrypoint_heartbeat_timestamp_seconds", float(time.time())),
                ],
                help_map={
                    "flagship_container_entrypoint_heartbeat_timestamp_seconds": "Container entrypoint heartbeat (epoch seconds).",
                },
                type_map={
                    "flagship_container_entrypoint_heartbeat_timestamp_seconds": "gauge",
                },
            )
        except Exception:
            pass
        time.sleep(15)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_cron_file()
    _start_uvicorn()
    _start_daemons()
    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    print("[container_entrypoint] Starting cron (foreground)...", flush=True)
    proc = subprocess.Popen(["cron", "-f"])
    rc = proc.wait()
    print(f"[container_entrypoint] cron exited rc={rc}", flush=True)
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()


