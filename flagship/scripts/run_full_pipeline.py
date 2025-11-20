"""
完整数据流程脚本：从 ticker 同步到回测数据下载。

流程：
1. 同步所有 ticker info 到 Postgres（如果尚未完成）
2. 同步 ticker details（market_cap 等）到 Postgres（按日期范围）
3. 构建每日股票池（基于策略筛选条件）
4. 下载回测数据（日线/分钟线）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from flagship.config import PROJECT_ROOT, get_paths
from vnpy.trader.logger import logger

SCRIPTS_DIR = get_paths().scripts_dir


def run_command(cmd: list[str], description: str) -> bool:
    """运行命令并返回是否成功。"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True,
    )
    
    if result.returncode == 0:
        logger.info(f"✓ {description} completed successfully")
        return True
    else:
        logger.error(f"✗ {description} failed with exit code {result.returncode}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full data pipeline: ticker sync → details sync → universe building → data download"
    )
    parser.add_argument(
        "--skip-ticker-sync",
        action="store_true",
        help="跳过 ticker 主表同步（如果已完成）",
    )
    parser.add_argument(
        "--skip-details-sync",
        action="store_true",
        help="跳过 ticker details 同步",
    )
    parser.add_argument(
        "--details-start",
        type=str,
        help="Ticker details 同步起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--details-end",
        type=str,
        help="Ticker details 同步结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--universe-date",
        type=str,
        required=True,
        help="构建股票池的日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download-start",
        type=str,
        required=True,
        help="下载回测数据的起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download-end",
        type=str,
        required=True,
        help="下载回测数据的结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download-interval",
        type=str,
        choices=["daily", "minute"],
        default="daily",
        help="下载的 K线周期（默认 daily）",
    )
    parser.add_argument(
        "--init-tables",
        action="store_true",
        help="初始化数据库表（如果尚未创建）",
    )
    args = parser.parse_args()

    success = True

    # 步骤 1: 同步 ticker 主表
    if not args.skip_ticker_sync:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "sync_tickers_postgres.py"),
            "--ticker-type", "CS",
        ]
        if args.init_tables:
            cmd.append("--init-tables")
        
        if not run_command(cmd, "Sync ticker reference data"):
            success = False
            logger.error("Ticker sync failed, aborting pipeline")
            return
    else:
        logger.info("Skipping ticker sync (--skip-ticker-sync)")

    # 步骤 2: 同步 ticker details
    if not args.skip_details_sync:
        if not args.details_start or not args.details_end:
            logger.warning(
                "Ticker details sync requires --details-start and --details-end. "
                "Skipping details sync."
            )
        else:
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "sync_ticker_details_postgres.py"),
                "--start", args.details_start,
                "--end", args.details_end,
            ]
            if args.init_tables:
                cmd.append("--init-tables")
            
            if not run_command(cmd, "Sync ticker details"):
                success = False
                logger.warning("Ticker details sync failed, but continuing...")
    else:
        logger.info("Skipping ticker details sync (--skip-details-sync)")

    # 步骤 3: 构建每日股票池
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_daily_universe.py"),
        "--date", args.universe_date,
        "--use-postgres",
    ]
    
    if not run_command(cmd, "Build daily universe"):
        success = False
        logger.error("Universe building failed, aborting pipeline")
        return

    # 步骤 4: 下载回测数据
    universe_file = PROJECT_ROOT / "flagship" / "data" / "universe" / f"universe_{args.universe_date}.json"
    if not universe_file.exists():
        logger.error(f"Universe file not found: {universe_file}")
        logger.error("Cannot proceed with data download")
        return

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "download_backtest_data.py"),
        "--universe-file", str(universe_file),
        "--start", args.download_start,
        "--end", args.download_end,
        "--interval", args.download_interval,
    ]
    
    if not run_command(cmd, "Download backtest data"):
        success = False

    if success:
        logger.info("=" * 60)
        logger.info("✓ Full pipeline completed successfully!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("✗ Pipeline completed with errors")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

