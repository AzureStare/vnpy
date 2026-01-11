"""
EC2 增量上传 + 环境构建（本地执行）

目标：
- 用 rsync 做增量传输（可反复执行、可断点续传）
- 默认绕过本机 ~/.ssh/config（避免配置错误导致 ssh/rsync 失败）
- 同步两类内容：
  1) 代码（项目根目录，排除 lab/.venv 等）
  2) AlphaLab 数据（例如 lab/flagship_alpha_momentum）
- 可选：远端 bootstrap（安装 docker/compose）与 docker compose build

安全：
- 不打印任何密钥内容
- 默认不删除远端文件（避免误删）

用法示例：
  # 仅同步代码（推荐先做）
  python flagship/scripts/ec2_deploy.py sync-code --host 18.218.179.137 --identity-file /path/paper-trading.pem

  # 同步数据（耗时较长）
  python flagship/scripts/ec2_deploy.py sync-data --host 18.218.179.137 --identity-file /path/paper-trading.pem

  # bootstrap + build（不自动启动）
  python flagship/scripts/ec2_deploy.py bootstrap --host 18.218.179.137 --identity-file /path/paper-trading.pem
  python flagship/scripts/ec2_deploy.py build --host 18.218.179.137 --identity-file /path/paper-trading.pem
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_cmd(argv: list[str]) -> None:
    cmd = " ".join(shlex.quote(a) for a in argv)
    print(f"+ {cmd}", flush=True)


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    _print_cmd(argv)
    return subprocess.run(argv, text=True, check=check)


def _run_capture(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """
    Run command and capture stdout/stderr (do NOT print argv).

    用于高频进度/校验采样，避免刷屏。
    """
    return subprocess.run(argv, text=True, capture_output=True, check=check)


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")


def _chmod_private_key(path: Path) -> None:
    """
    OpenSSH 对私钥权限要求较严格（通常 0600 或 0400）。
    这里统一设置为 0400，避免交互报错。
    """
    try:
        path.chmod(0o400)
    except PermissionError:
        # 如果用户无权限修改，继续执行，ssh 会给出更明确错误
        pass


def _parse_rsync_version(output: str) -> tuple[int, int, int] | None:
    """
    Return (major, minor, patch) if parseable, else None.
    """
    # Examples:
    #   rsync  version 2.6.9  protocol version 29
    #   rsync  version 3.2.7  protocol version 31
    m = re.search(r"rsync\\s+version\\s+(\\d+)\\.(\\d+)\\.(\\d+)", output)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _rsync_supports_info_progress2() -> bool:
    try:
        proc = subprocess.run(["rsync", "--version"], text=True, capture_output=True, check=True)
    except Exception:
        return False
    ver = _parse_rsync_version(proc.stdout + "\n" + proc.stderr)
    if not ver:
        return False
    major, minor, _patch = ver
    # progress2 landed in 3.1.0
    return (major, minor) >= (3, 1)


def _human_kb(kb: int) -> str:
    if kb <= 0:
        return "0B"
    b = kb * 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while b >= 1024.0 and idx < len(units) - 1:
        b /= 1024.0
        idx += 1
    return f"{b:.2f}{units[idx]}"


def _safe_int(text: str, default: int = 0) -> int:
    text = (text or "").strip()
    if not text:
        return default
    try:
        return int(text)
    except Exception:
        return default


def _du_kb_local(path: Path) -> int:
    if not path.exists():
        return 0
    proc = _run_capture(["du", "-sk", str(path)], check=True)
    # format: "<kb>\t<path>"
    return _safe_int(proc.stdout.split("\t", 1)[0], 0)


def _count_parquet_flat_local(dir_path: Path) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".parquet")


def _count_files_recursive_local(dir_path: Path) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    total = 0
    for _root, _dirs, files in os.walk(dir_path):
        total += len(files)
    return total


@dataclass(frozen=True)
class SshTarget:
    host: str
    user: str
    identity_file: Path
    known_hosts_file: Path
    connect_timeout: int
    strict_host_key: str

    @property
    def user_host(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_base_argv(self) -> list[str]:
        return [
            "ssh",
            "-F",
            "/dev/null",  # bypass ~/.ssh/config (may be broken)
            "-i",
            str(self.identity_file),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            f"StrictHostKeyChecking={self.strict_host_key}",
            "-o",
            f"UserKnownHostsFile={str(self.known_hosts_file)}",
            "-o",
            "LogLevel=ERROR",
        ]

    def rsync_rsh(self) -> str:
        # rsync uses a single string for -e
        parts = self.ssh_base_argv()
        # Remove leading "ssh" for rsync -e program string
        assert parts[0] == "ssh"
        return " ".join(shlex.quote(p) for p in parts)


def _ssh_run(target: SshTarget, remote_cmd: str) -> None:
    argv = target.ssh_base_argv() + [target.user_host, remote_cmd]
    _run(argv, check=True)


def _ssh_capture(target: SshTarget, remote_cmd: str) -> str:
    argv = target.ssh_base_argv() + [target.user_host, remote_cmd]
    proc = _run_capture(argv, check=True)
    return (proc.stdout or "").strip()


def _du_kb_remote(target: SshTarget, remote_path: str) -> int:
    cmd = f"du -sk {shlex.quote(remote_path)} 2>/dev/null | cut -f1"
    out = _ssh_capture(target, cmd)
    return _safe_int(out, 0)


def _count_parquet_flat_remote(target: SshTarget, remote_dir: str) -> int:
    cmd = (
        f"find {shlex.quote(remote_dir)} -maxdepth 1 -type f -name '*.parquet' 2>/dev/null | wc -l"
    )
    out = _ssh_capture(target, cmd)
    return _safe_int(out, 0)


def _count_files_recursive_remote(target: SshTarget, remote_dir: str) -> int:
    cmd = f"find {shlex.quote(remote_dir)} -type f 2>/dev/null | wc -l"
    out = _ssh_capture(target, cmd)
    return _safe_int(out, 0)


def _ensure_remote_dirs(target: SshTarget, remote_root: str) -> None:
    cmd = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(remote_root)}/lab {shlex.quote(remote_root)}/logs; "
        f"ls -ld {shlex.quote(remote_root)} {shlex.quote(remote_root)}/lab {shlex.quote(remote_root)}/logs"
    )
    _ssh_run(target, cmd)


def _print_lab_content_summary(local_lab_dir: Path) -> None:
    """
    1) 同步的数据内容：按关键子目录输出 size/文件数。

    注意：daily/minute/signal 目录只统计 parquet 数量（避免递归 walk 成本过高）。
    """
    print(f"[data] local_lab_dir={local_lab_dir}", flush=True)
    if not local_lab_dir.exists():
        print("[data] local_lab_dir not found", flush=True)
        return

    total_kb = _du_kb_local(local_lab_dir)
    print(f"[data] local_total={_human_kb(total_kb)} (du -sk)", flush=True)

    keys = ["daily", "minute", "model", "dataset", "signal", "report", "component", "logs"]
    for name in keys:
        p = local_lab_dir / name
        if not p.exists():
            continue
        kb = _du_kb_local(p)
        if name in ("daily", "minute", "signal"):
            c = _count_parquet_flat_local(p)
            print(f"  - {name}: size={_human_kb(kb)}, parquet_files={c}", flush=True)
        else:
            c = _count_files_recursive_local(p)
            print(f"  - {name}: size={_human_kb(kb)}, files={c}", flush=True)


def _print_lab_completeness_compare(
    *,
    local_lab_dir: Path,
    target: SshTarget,
    remote_lab_dir: str,
) -> None:
    """
    3) 数据完成度对比：local vs remote（以“remote 是否至少包含 local”为准）。

    注意：默认不启用 --delete，因此 remote 允许“多于 local”的文件。
    """
    local_total_kb = _du_kb_local(local_lab_dir)
    remote_total_kb = _du_kb_remote(target, remote_lab_dir)

    local_daily = _count_parquet_flat_local(local_lab_dir / "daily")
    remote_daily = _count_parquet_flat_remote(target, f"{remote_lab_dir}/daily")

    local_minute = _count_parquet_flat_local(local_lab_dir / "minute")
    remote_minute = _count_parquet_flat_remote(target, f"{remote_lab_dir}/minute")

    local_model = _count_files_recursive_local(local_lab_dir / "model")
    remote_model = _count_files_recursive_remote(target, f"{remote_lab_dir}/model")

    local_dataset = _count_files_recursive_local(local_lab_dir / "dataset")
    remote_dataset = _count_files_recursive_remote(target, f"{remote_lab_dir}/dataset")

    local_signal = _count_parquet_flat_local(local_lab_dir / "signal")
    remote_signal = _count_parquet_flat_remote(target, f"{remote_lab_dir}/signal")

    print("[verify] lab completeness compare (remote >= local => OK):", flush=True)
    print(
        f"  - total_size: local={_human_kb(local_total_kb)} remote={_human_kb(remote_total_kb)} (du -sk)",
        flush=True,
    )

    def _row(name: str, local_n: int, remote_n: int) -> None:
        status = "OK" if remote_n >= local_n else f"MISSING {local_n - remote_n}"
        print(f"  - {name}: local={local_n} remote={remote_n} => {status}", flush=True)

    _row("daily_parquet", local_daily, remote_daily)
    _row("minute_parquet", local_minute, remote_minute)
    _row("signal_parquet", local_signal, remote_signal)
    _row("model_files", local_model, remote_model)
    _row("dataset_files", local_dataset, remote_dataset)


def _print_lab_progress(
    *,
    started_ts: float,
    local_total_kb: int,
    local_daily: int,
    local_minute: int,
    target: SshTarget,
    remote_lab_dir: str,
) -> None:
    """
    2) 数据进度：用远端 du + daily/minute parquet 数量作为进度指示。

    注：macOS 自带 rsync(2.6.9) 无 progress2，因此这里用“远端增长采样”给进度。
    """
    remote_kb = _du_kb_remote(target, remote_lab_dir)
    pct = 0.0 if local_total_kb <= 0 else min(remote_kb / float(local_total_kb), 1.0)

    remote_daily = _count_parquet_flat_remote(target, f"{remote_lab_dir}/daily")
    remote_minute = _count_parquet_flat_remote(target, f"{remote_lab_dir}/minute")

    elapsed = int(time.time() - started_ts)
    print(
        f"[progress] elapsed={elapsed}s remote={_human_kb(remote_kb)}/{_human_kb(local_total_kb)} ({pct:.1%}) "
        f"daily={remote_daily}/{local_daily} minute={remote_minute}/{local_minute}",
        flush=True,
    )


def _rsync_with_retry(
    *,
    source: str,
    dest: str,
    rsh: str,
    excludes: Iterable[str],
    retries: int,
    retry_sleep: int,
    delete: bool,
    progress_hook: Callable[[], None] | None = None,
    progress_interval: int = 0,
) -> None:
    supports_progress2 = _rsync_supports_info_progress2()

    base = [
        "rsync",
        "-a",
        "--partial",
        "--human-readable",
        "--stats",
        "--no-perms",
        "--no-owner",
        "--no-group",
    ]
    if supports_progress2:
        base += ["--info=progress2"]
    else:
        # 兼容 macOS 自带 rsync(2.6.9)：不支持 --info=progress2
        # 不使用 --progress（会逐文件刷屏，parquet 文件很多）
        pass

    if delete:
        base += ["--delete"]

    for ex in excludes:
        base += ["--exclude", ex]

    base += ["-e", rsh, source, dest]

    attempt = 0
    while True:
        attempt += 1
        try:
            _print_cmd(base)
            proc = subprocess.Popen(base)
            if progress_hook and progress_interval > 0:
                while True:
                    try:
                        rc = proc.wait(timeout=progress_interval)
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            progress_hook()
                        except Exception:
                            pass
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, base)
            else:
                rc = proc.wait()
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, base)
            return
        except subprocess.CalledProcessError as exc:
            if attempt > retries:
                raise SystemExit(f"rsync failed after {attempt} attempt(s): exit={exc.returncode}")
            print(
                f"[warn] rsync failed (attempt {attempt}/{retries+1}, exit={exc.returncode}), "
                f"sleep {retry_sleep}s and retry...",
                flush=True,
            )
            time.sleep(retry_sleep)


def cmd_sync_code(args: argparse.Namespace) -> None:
    target = _build_target(args)
    _ensure_remote_dirs(target, args.remote_root)

    local_root = Path(args.local_root).resolve()
    if not local_root.exists():
        raise SystemExit(f"local_root not found: {local_root}")

    excludes = [
        ".git/",
        ".venv/",
        "lab/",
        "logs/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "flagship/data/s3_downloads/",
        # Frontend dependencies are large and must never be synced to server.
        "flagship/app_console/frontend/node_modules/",
    ]
    if not args.include_vt_setting:
        excludes.append("vt_setting.json")

    src = str(local_root) + "/"  # sync contents into remote_root
    dst = f"{target.user_host}:{args.remote_root}/"
    _rsync_with_retry(
        source=src,
        dest=dst,
        rsh=target.rsync_rsh(),
        excludes=excludes,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        delete=args.delete,
    )


def cmd_sync_data(args: argparse.Namespace) -> None:
    target = _build_target(args)
    _ensure_remote_dirs(target, args.remote_root)

    local_lab_dir = Path(args.local_lab_dir).resolve()
    if not local_lab_dir.exists():
        raise SystemExit(f"local_lab_dir not found: {local_lab_dir}")
    if not local_lab_dir.is_dir():
        raise SystemExit(f"local_lab_dir is not a directory: {local_lab_dir}")

    src = str(local_lab_dir)
    dst = f"{target.user_host}:{args.remote_root}/lab/"

    remote_lab_dir = f"{args.remote_root}/lab/{local_lab_dir.name}"

    # 1) 同步的数据内容
    _print_lab_content_summary(local_lab_dir)

    # 同步前对比（用于确认 remote 现状）
    print(f"[data] remote_target={target.user_host}:{remote_lab_dir}", flush=True)
    _print_lab_completeness_compare(local_lab_dir=local_lab_dir, target=target, remote_lab_dir=remote_lab_dir)

    started_ts = time.time()
    local_total_kb = _du_kb_local(local_lab_dir)
    local_daily = _count_parquet_flat_local(local_lab_dir / "daily")
    local_minute = _count_parquet_flat_local(local_lab_dir / "minute")

    def _hook() -> None:
        _print_lab_progress(
            started_ts=started_ts,
            local_total_kb=local_total_kb,
            local_daily=local_daily,
            local_minute=local_minute,
            target=target,
            remote_lab_dir=remote_lab_dir,
        )

    # 首次立即输出一次进度（避免等 interval）
    if args.progress_interval > 0:
        _hook()

    _rsync_with_retry(
        source=src,
        dest=dst,
        rsh=target.rsync_rsh(),
        excludes=[],
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        delete=args.delete,
        progress_hook=_hook if args.progress_interval > 0 else None,
        progress_interval=int(args.progress_interval),
    )

    # 同步后完成度对比
    if not args.no_verify:
        _print_lab_completeness_compare(local_lab_dir=local_lab_dir, target=target, remote_lab_dir=remote_lab_dir)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    target = _build_target(args)
    _ensure_remote_dirs(target, args.remote_root)

    # Detect OS and install docker if missing (Ubuntu/Debian path first).
    bootstrap_cmd = r"""
set -euo pipefail
if command -v docker >/dev/null 2>&1; then
  echo "[bootstrap] docker already installed: $(docker --version)"
else
  echo "[bootstrap] installing docker (apt)..."
  sudo apt-get update -y
  # NOTE:
  # - On some Ubuntu repos (e.g., jammy), docker-compose-plugin may not exist unless Docker official repo is added.
  # - We fallback to docker-compose (v1) package to keep bootstrap minimal and non-interactive.
  sudo apt-get install -y docker.io docker-compose rsync ca-certificates curl
  echo "[bootstrap] docker installed: $(docker --version)"
fi

if docker compose version >/dev/null 2>&1; then
  echo "[bootstrap] docker compose (plugin) ok: $(docker compose version)"
elif command -v docker-compose >/dev/null 2>&1; then
  echo "[bootstrap] docker-compose (v1) ok: $(docker-compose version | head -n 1)"
else
  echo "[bootstrap] docker compose not available (neither plugin nor docker-compose v1)"
  exit 1
fi
"""
    _ssh_run(target, bootstrap_cmd)


def cmd_build(args: argparse.Namespace) -> None:
    target = _build_target(args)

    # Build image only (do not start cron automatically).
    build_cmd = (
        "set -euo pipefail; "
        f"cd {shlex.quote(args.remote_root)}; "
        "if docker compose version >/dev/null 2>&1; then "
        "  sudo docker compose build; "
        "elif command -v docker-compose >/dev/null 2>&1; then "
        "  sudo docker-compose build; "
        "else "
        "  echo '[build] docker compose not available'; exit 1; "
        "fi"
    )
    _ssh_run(target, build_cmd)


def _build_target(args: argparse.Namespace) -> SshTarget:
    identity = Path(args.identity_file).expanduser().resolve()
    _ensure_file_exists(identity)
    _chmod_private_key(identity)

    known_hosts = Path(args.known_hosts_file).expanduser().resolve()
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    # touch
    if not known_hosts.exists():
        known_hosts.write_text("", encoding="utf-8")

    return SshTarget(
        host=str(args.host),
        user=str(args.user),
        identity_file=identity,
        known_hosts_file=known_hosts,
        connect_timeout=int(args.connect_timeout),
        # accept-new: non-interactive, but will fail if host key changes (safer than "no")
        strict_host_key=str(args.strict_host_key),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ec2_deploy",
        description="EC2 增量上传 + 环境构建（rsync + ssh）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", required=True, help="EC2 Public IP / hostname")
        p.add_argument("--user", default="ubuntu", help="SSH username (default: ubuntu)")
        p.add_argument("--identity-file", required=True, help="SSH private key path (.pem)")
        p.add_argument(
            "--remote-root",
            default="/home/ubuntu/vnpy",
            help="Remote project root (default: /home/ubuntu/vnpy)",
        )
        p.add_argument(
            "--connect-timeout",
            type=int,
            default=10,
            help="SSH connect timeout seconds",
        )
        p.add_argument(
            "--known-hosts-file",
            default=str(PROJECT_ROOT / "logs" / "ec2_known_hosts"),
            help="Known hosts file path (default: logs/ec2_known_hosts)",
        )
        p.add_argument(
            "--strict-host-key",
            default="accept-new",
            help="StrictHostKeyChecking (default: accept-new; use 'no' to bypass)",
        )

    p_code = sub.add_parser("sync-code", help="增量同步代码到远端（排除 lab/.venv 等）")
    add_common(p_code)
    p_code.add_argument(
        "--local-root",
        default=str(PROJECT_ROOT),
        help="Local project root (default: repo root)",
    )
    p_code.add_argument(
        "--include-vt-setting",
        action="store_true",
        help="Also sync vt_setting.json (contains secrets; use carefully)",
    )
    p_code.add_argument("--retries", type=int, default=5, help="rsync retry count")
    p_code.add_argument("--retry-sleep", type=int, default=10, help="sleep seconds between retries")
    p_code.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files not present locally (dangerous; off by default)",
    )
    p_code.set_defaults(func=cmd_sync_code)

    p_data = sub.add_parser("sync-data", help="增量同步 AlphaLab 数据到远端（耗时较长）")
    add_common(p_data)
    p_data.add_argument(
        "--local-lab-dir",
        default=str(PROJECT_ROOT / "lab" / "flagship_alpha_momentum"),
        help="Local lab dir to sync (default: lab/flagship_alpha_momentum)",
    )
    p_data.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="数据进度采样间隔（秒）。0=不采样（默认 60）",
    )
    p_data.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过同步后的完成度对比输出",
    )
    p_data.add_argument("--retries", type=int, default=8, help="rsync retry count")
    p_data.add_argument("--retry-sleep", type=int, default=15, help="sleep seconds between retries")
    p_data.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files not present locally (dangerous; off by default)",
    )
    p_data.set_defaults(func=cmd_sync_data)

    p_boot = sub.add_parser("bootstrap", help="远端安装 docker/compose（Ubuntu apt 路径）")
    add_common(p_boot)
    p_boot.set_defaults(func=cmd_bootstrap)

    p_build = sub.add_parser("build", help="远端 docker compose build（只构建不启动）")
    add_common(p_build)
    p_build.set_defaults(func=cmd_build)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    func = getattr(args, "func", None)
    if func is None:
        raise SystemExit("No command selected")
    func(args)


if __name__ == "__main__":
    main()


