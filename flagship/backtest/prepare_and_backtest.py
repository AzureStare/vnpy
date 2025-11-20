"""
准备数据并运行回测的完整流程脚本。

流程：
1. 读取每日股票池文件（universe_YYYY-MM-DD.json）
2. 收集所有需要下载的股票
3. 下载这些股票的历史数据（如果还没有下载）
4. 构建因子信号
5. 运行回测
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger
from vnpy.alpha import AlphaLab, BacktestingEngine
from flagship.strategy.flagship_alpha_momentum_strategy import FlagshipAlphaMomentumStrategy
from flagship.config import (
    DEFAULT_LAB_DIR,
    DEFAULT_UNIVERSE_DIR,
    PROJECT_ROOT,
)
from flagship.factors.build_alpha_momentum_signal import build_signal
from flagship.backtest.flagship_alpha_momentum_backtest import load_signal


def load_daily_universe(universe_file: Path) -> list[str]:
    """
    从每日股票池文件加载股票列表。
    
    Returns:
        股票代码列表（不含交易所后缀）
    """
    logger.info(f"[load_daily_universe] 开始加载股票池文件: {universe_file}")
    
    if not universe_file.exists():
        logger.error(f"[load_daily_universe] 文件不存在: {universe_file}")
        raise FileNotFoundError(f"Universe file not found: {universe_file}")

    logger.debug(f"[load_daily_universe] 读取文件内容...")
    data = json.loads(universe_file.read_text(encoding="utf-8"))
    
    trade_date = data.get("trade_date", "unknown")
    symbol_count = data.get("symbol_count", 0)
    logger.info(f"[load_daily_universe] 文件信息: trade_date={trade_date}, symbol_count={symbol_count}")
    
    symbols = [item["symbol"] for item in data.get("symbols", [])]
    logger.info(f"[load_daily_universe] 成功加载 {len(symbols)} 个股票代码")
    
    if len(symbols) > 0:
        logger.debug(f"[load_daily_universe] 前5个股票: {symbols[:5]}")
    
    return symbols


def collect_symbols_from_universe_files(
    start_date: date,
    end_date: date,
    universe_dir: Path = DEFAULT_UNIVERSE_DIR,
) -> set[str]:
    """
    从日期范围内的所有 universe 文件中收集股票代码。
    
    Returns:
        所有股票代码的集合
    """
    logger.info(f"[collect_symbols_from_universe_files] 开始收集股票代码")
    logger.info(f"[collect_symbols_from_universe_files] 日期范围: {start_date} 到 {end_date}")
    logger.info(f"[collect_symbols_from_universe_files] Universe 目录: {universe_dir}")
    
    all_symbols = set()
    current_date = start_date
    processed_files = 0
    skipped_files = 0
    
    while current_date <= end_date:
        universe_file = universe_dir / f"universe_{current_date.isoformat()}.json"
        logger.debug(f"[collect_symbols_from_universe_files] 检查日期 {current_date}: {universe_file}")
        
        if universe_file.exists():
            symbols = load_daily_universe(universe_file)
            before_count = len(all_symbols)
            all_symbols.update(symbols)
            new_symbols = len(all_symbols) - before_count
            processed_files += 1
            logger.info(f"[collect_symbols_from_universe_files] {current_date}: 新增 {new_symbols} 个股票，累计 {len(all_symbols)} 个")
        else:
            logger.warning(f"[collect_symbols_from_universe_files] Universe 文件不存在: {universe_file}")
            skipped_files += 1
        
        # 移动到下一个交易日（跳过周末）
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:  # 跳过周末
            current_date += timedelta(days=1)
    
    logger.info(f"[collect_symbols_from_universe_files] 收集完成:")
    logger.info(f"[collect_symbols_from_universe_files]   - 处理文件数: {processed_files}")
    logger.info(f"[collect_symbols_from_universe_files]   - 跳过文件数: {skipped_files}")
    logger.info(f"[collect_symbols_from_universe_files]   - 唯一股票数: {len(all_symbols)}")
    
    return all_symbols


def download_data_for_symbols(
    symbols: list[str],
    start_date: date,
    end_date: date,
    lab_dir: Path = DEFAULT_LAB_DIR,
) -> None:
    """
    下载指定股票的历史数据。
    
    如果数据已存在，则跳过下载。
    """
    logger.info(f"[download_data_for_symbols] 开始下载历史数据")
    logger.info(f"[download_data_for_symbols] 股票数量: {len(symbols)}")
    logger.info(f"[download_data_for_symbols] 日期范围: {start_date} 到 {end_date}")
    logger.info(f"[download_data_for_symbols] Lab 目录: {lab_dir}")
    
    # 调用 download_backtest_data.py 脚本
    script_path = PROJECT_ROOT / "flagship" / "scripts" / "download_backtest_data.py"
    logger.debug(f"[download_data_for_symbols] 下载脚本路径: {script_path}")
    
    if not script_path.exists():
        logger.error(f"[download_data_for_symbols] 下载脚本不存在: {script_path}")
        raise FileNotFoundError(f"Download script not found: {script_path}")
    
    # 创建临时 universe 文件用于下载
    temp_universe_file = PROJECT_ROOT / "flagship" / "data" / "universe" / f"temp_backtest_{start_date}_{end_date}.json"
    temp_universe_file.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"[download_data_for_symbols] 创建临时 universe 文件: {temp_universe_file}")
    
    temp_universe_data = {
        "trade_date": start_date.isoformat(),
        "symbol_count": len(symbols),
        "symbols": [{"symbol": s} for s in symbols],
    }
    temp_universe_file.write_text(json.dumps(temp_universe_data, indent=2), encoding="utf-8")
    logger.debug(f"[download_data_for_symbols] 临时文件已创建，包含 {len(symbols)} 个股票")
    
    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--universe-file", str(temp_universe_file),
            "--start", start_date.isoformat(),
            "--end", end_date.isoformat(),
            "--interval", "daily",
            "--lab-dir", str(lab_dir),
        ]
        
        logger.info(f"[download_data_for_symbols] 执行命令: {' '.join(cmd)}")
        logger.info(f"[download_data_for_symbols] 工作目录: {PROJECT_ROOT}")
        
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
        
        if result.returncode != 0:
            logger.error(f"[download_data_for_symbols] 下载失败，退出码: {result.returncode}")
            raise RuntimeError(f"Data download failed with exit code {result.returncode}")
        
        logger.info(f"[download_data_for_symbols] 数据下载完成")
    finally:
        # 清理临时文件
        if temp_universe_file.exists():
            logger.debug(f"[download_data_for_symbols] 清理临时文件: {temp_universe_file}")
            temp_universe_file.unlink()


def run_backtest_with_universe(
    start_date: date,
    end_date: date,
    universe_dir: Path = DEFAULT_UNIVERSE_DIR,
    lab_dir: Path = DEFAULT_LAB_DIR,
    skip_download: bool = False,
    skip_signal: bool = False,
    initial_capital: int = 1_000_000,
    strategy_setting: dict[str, Any] | None = None,
) -> None:
    """
    基于每日股票池文件准备数据并运行回测。
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        universe_dir: 股票池文件目录
        lab_dir: AlphaLab 数据目录
        skip_download: 是否跳过数据下载（如果数据已存在）
        skip_signal: 是否跳过信号构建（如果信号已存在）
        initial_capital: 初始资金
        strategy_setting: 策略参数字典
    """
    # 步骤 1: 收集股票代码（这是投资范围 $U_t$，满足流动性条件的股票池）
    logger.info("=" * 60)
    logger.info("步骤 1: 收集股票代码（投资范围 $U_t$）")
    logger.info("=" * 60)
    logger.info(f"[run_backtest_with_universe] 从 universe 文件收集满足流动性条件的股票...")
    symbols = collect_symbols_from_universe_files(start_date, end_date, universe_dir)
    
    if not symbols:
        logger.warning(f"[run_backtest_with_universe] 未找到 universe 文件，将从信号数据中获取股票列表")
        # 如果 universe 文件不存在，尝试从信号数据中获取股票列表
        lab = AlphaLab(str(lab_dir))
        signal_df = load_signal(lab, "flagship_alpha_momentum")
        if signal_df is not None and not signal_df.is_empty():
            # 获取信号数据中的所有股票代码（去掉交易所后缀）
            vt_symbols = signal_df["vt_symbol"].unique().to_list()
            symbols = list(set([vt_symbol.split(".")[0] for vt_symbol in vt_symbols]))
            logger.info(f"[run_backtest_with_universe] 从信号数据获取到 {len(symbols)} 个股票")
        else:
            logger.error(f"[run_backtest_with_universe] 无法从信号数据获取股票列表")
            raise RuntimeError("No symbols found in universe files or signal data")
    
    logger.info(f"[run_backtest_with_universe] 投资范围 $U_t$ 包含 {len(symbols)} 个股票")
    logger.info(f"[run_backtest_with_universe] 后续将对这些股票计算因子和 Score，然后按 Score 排序选 Top N")
    
    # 步骤 2: 下载数据
    if not skip_download:
        logger.info("=" * 60)
        logger.info("步骤 2: 下载历史数据")
        logger.info("=" * 60)
        download_data_for_symbols(
            list(symbols),
            start_date,
            end_date,
            lab_dir=lab_dir,
        )
    else:
        logger.info("跳过数据下载（--skip-download）")
    
    # 步骤 3: 构建信号（对投资范围 $U_t$ 中的股票计算因子和 Score）
    if not skip_signal:
        logger.info("=" * 60)
        logger.info("步骤 3: 构建因子信号")
        logger.info("=" * 60)
        logger.info(f"[run_backtest_with_universe] 对 {len(symbols)} 个股票计算因子（A/B/C）和综合 Score")
        logger.info(f"[run_backtest_with_universe] 因子计算包括：")
        logger.info(f"[run_backtest_with_universe]   - 因子 A: 波动率调整突破强度 (alpha_mom)")
        logger.info(f"[run_backtest_with_universe]   - 因子 B: 相对异常成交量 (alpha_vol)")
        logger.info(f"[run_backtest_with_universe]   - 因子 C: 聪明资金流向代理 (alpha_flow)")
        logger.info(f"[run_backtest_with_universe]   - 截面 Winsorization + Z-Score 标准化")
        logger.info(f"[run_backtest_with_universe]   - 综合 Score = 0.4*Z_mom + 0.4*Z_flow + 0.2*Z_vol")
        build_signal(
            lab_path=lab_dir,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )
        logger.info(f"[run_backtest_with_universe] 信号构建完成，每个股票都有 Score 值")
    else:
        logger.info("跳过信号构建（--skip-signal）")
    
    # 步骤 4: 运行回测（策略类会按 Score 排序，选 Top N，且 Score > 0.5）
    logger.info("=" * 60)
    logger.info("步骤 4: 运行回测")
    logger.info("=" * 60)
    logger.info(f"[run_backtest_with_universe] 策略选股逻辑：")
    logger.info(f"[run_backtest_with_universe]   1. 按 Score 降序排序")
    logger.info(f"[run_backtest_with_universe]   2. 选取 Top N（默认 12 只）")
    logger.info(f"[run_backtest_with_universe]   3. 门槛过滤：Score > 0.5（高于市场均值 0.5 个标准差）")
    logger.info(f"[run_backtest_with_universe]   4. 如果满足条件的股票不足，则持有现金")
    
    from flagship.backtest.flagship_alpha_momentum_backtest import run_backtest
    
    run_backtest(
        lab_path=lab_dir,
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
        interval=Interval.DAILY,
        signal_name="flagship_alpha_momentum",
        initial_capital=initial_capital,
        risk_free=0.02,
        annual_days=252,
        strategy_setting=strategy_setting,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data and run backtest based on daily universe files."
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="回测起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="回测结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=DEFAULT_UNIVERSE_DIR,
        help="股票池文件目录（默认 flagship/data/universe）",
    )
    parser.add_argument(
        "--lab-dir",
        type=Path,
        default=DEFAULT_LAB_DIR,
        help="AlphaLab 数据目录（默认 lab/flagship_alpha_momentum）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过数据下载（如果数据已存在）",
    )
    parser.add_argument(
        "--skip-signal",
        action="store_true",
        help="跳过信号构建（如果信号已存在）",
    )
    parser.add_argument(
        "--capital",
        type=int,
        default=1_000_000,
        help="初始资金（默认 1,000,000）",
    )
    args = parser.parse_args()

    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()

    run_backtest_with_universe(
        start_date=start_date,
        end_date=end_date,
        universe_dir=args.universe_dir,
        lab_dir=args.lab_dir,
        skip_download=args.skip_download,
        skip_signal=args.skip_signal,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    main()

