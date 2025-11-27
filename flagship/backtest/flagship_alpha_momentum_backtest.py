"""
Flagship Alpha-Momentum 策略回测入口脚本。

支持短期回测（2-3天）用于验证策略逻辑。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import polars as pl

from flagship.config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger
from vnpy.alpha import AlphaLab, BacktestingEngine
from flagship.strategy.flagship_alpha_momentum_strategy import FlagshipAlphaMomentumStrategy


def save_backtest_report(
    lab: AlphaLab,
    engine: BacktestingEngine,
    stats: dict[str, Any],
    start: datetime,
    end: datetime,
    signal_name: str,
) -> None:
    """
    保存回测报告到文件。
    
    保存内容：
    1. 统计指标（JSON）
    2. 每日净值数据（Parquet）
    3. 交易清单（Parquet）
    """
    from datetime import datetime as dt
    
    # 创建报告目录（保存在项目根目录）
    from flagship.config import PROJECT_ROOT
    report_dir = PROJECT_ROOT / "backtest_report"
    report_dir.mkdir(exist_ok=True)
    
    # 生成报告文件名（基于日期范围）
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    report_prefix = f"{signal_name}_{start_str}_{end_str}"
    
    # 1. 保存统计指标（JSON）
    stats_file = report_dir / f"{report_prefix}_stats.json"
    stats_data = {
        "backtest_info": {
            "signal_name": signal_name,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "report_generated_at": dt.now().isoformat(),
        },
        "statistics": stats,
    }
    import json
    with open(stats_file, "w", encoding="UTF-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    logger.info(f"[save_backtest_report] 统计指标已保存: {stats_file}")
    
    # 2. 保存每日净值数据（Parquet）
    if hasattr(engine, 'daily_df') and engine.daily_df is not None:
        daily_file = report_dir / f"{report_prefix}_daily.parquet"
        engine.daily_df.write_parquet(daily_file)
        logger.info(f"[save_backtest_report] 每日净值数据已保存: {daily_file}")
    
    # 3. 保存交易清单（Parquet）
    trade_df = engine.get_trade_list()
    if not trade_df.is_empty():
        trade_file = report_dir / f"{report_prefix}_trades.parquet"
        trade_df.write_parquet(trade_file)
        logger.info(f"[save_backtest_report] 交易清单已保存: {trade_file}")
        logger.info(f"[save_backtest_report] 交易记录数: {len(trade_df)}")
    else:
        logger.warning(f"[save_backtest_report] 无交易记录，跳过交易清单保存")
    
    logger.info(f"[save_backtest_report] 回测报告已保存到: {report_dir}")


def load_signal(lab: AlphaLab, name: str) -> pl.DataFrame:
    """
    加载模型信号用于回测。

    期望一个由 AlphaLab.save_signal 保存的 Parquet 文件，至少包含以下列：
        - datetime: 时间戳（无时区）
        - vt_symbol: 合约代码，例如 'AAPL.NASDAQ'
        - signal: 数值得分，越高越好
    """
    signal_df: pl.DataFrame | None = lab.load_signal(name)
    if signal_df is None:
        raise RuntimeError(f"Signal file not found for name={name!r}")

    required_columns = {"datetime", "vt_symbol", "signal"}
    missing = required_columns.difference(signal_df.columns)
    if missing:
        raise RuntimeError(f"Signal DataFrame missing columns: {sorted(missing)}")

    # 确保正确的数据类型和排序
    signal_df = signal_df.with_columns(
        pl.col("datetime").cast(pl.Datetime, strict=False),
        pl.col("vt_symbol").cast(pl.Utf8),
        pl.col("signal").cast(pl.Float64),
    ).sort(["datetime", "vt_symbol"])

    return signal_df


def infer_vt_symbols_from_lab(lab: AlphaLab, interval: Interval) -> list[str]:
    """当信号/股票池缺失时，从 lab 存储推断 vt_symbols"""
    folder = lab.minute_path if interval == Interval.MINUTE else lab.daily_path
    return sorted([file.stem for file in folder.glob("*.parquet")])


def generate_naive_signal(
    lab: AlphaLab,
    vt_symbols: list[str],
    interval: Interval,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """
    构建占位信号，使用简单的滚动收益率，以便在模型完成训练前也能执行回测流水线。
    """
    rows: list[dict] = []
    for vt_symbol in vt_symbols:
        bars = lab.load_bar_data(vt_symbol, interval, start, end)
        if not bars:
            continue

        df = pl.DataFrame(
            {
                "datetime": [bar.datetime.replace(tzinfo=None) for bar in bars],
                "close": [bar.close_price for bar in bars],
            }
        ).sort("datetime")

        df = df.with_columns(
            pl.col("close")
            .pct_change()
            .fill_null(0.0)
            .alias("ret")
        ).with_columns(
            pl.col("ret")
            .rolling_mean(15, min_samples=5)
            .alias("score")
        )

        vt_series = (
            df.select(["datetime", "score"])
            .drop_nulls("score")
            .with_columns(pl.lit(vt_symbol).alias("vt_symbol"))
        )

        rows.extend(vt_series.to_dicts())

    if not rows:
        raise RuntimeError("Unable to generate fallback signal; no bar data available.")

    return pl.DataFrame(rows).select(["datetime", "vt_symbol", "score"]).rename({"score": "signal"})


def run_backtest(
    lab_path: str | Path,
    start: datetime,
    end: datetime,
    interval: Interval = Interval.DAILY,
    signal_name: str = "flagship_alpha_momentum",
    initial_capital: int = 1_000_000,
    risk_free: float = 0.02,
    annual_days: int = 252,
    strategy_setting: dict[str, Any] | None = None,
    vt_symbols: list[str] | None = None,
) -> None:
    """
    运行 Flagship Alpha-Momentum 策略回测。

    Args:
        lab_path: AlphaLab 数据目录路径
        start: 回测起始日期
        end: 回测结束日期
        interval: K线周期（日线或分钟线）
        signal_name: 信号文件名（默认 "flagship_alpha_momentum"）
        initial_capital: 初始资金
        risk_free: 无风险利率
        annual_days: 年化交易日数
        strategy_setting: 策略参数字典
    """
    logger.info(f"[run_backtest] 开始运行回测")
    logger.info(f"[run_backtest] Lab 路径: {lab_path}")
    logger.info(f"[run_backtest] 回测日期范围: {start} 到 {end}")
    logger.info(f"[run_backtest] K线周期: {interval}")
    logger.info(f"[run_backtest] 信号文件名: {signal_name}")
    logger.info(f"[run_backtest] 初始资金: {initial_capital:,}")
    
    lab = AlphaLab(str(lab_path))
    logger.debug(f"[run_backtest] AlphaLab 初始化完成")

    vt_symbols_list: list[str] = []
    signal_df: pl.DataFrame | None = None

    # 步骤 1: 如果有 universe 文件，先筛选股票池（投资范围 $U_t$）
    universe_dir = Path(lab_path).parent.parent / "flagship" / "data" / "universe"
    if universe_dir.exists():
        logger.info(f"[run_backtest] 发现 universe 目录，尝试加载股票池...")
        from flagship.backtest.prepare_and_backtest import collect_symbols_from_universe_files
        from datetime import date as date_type
        
        try:
            universe_symbols = collect_symbols_from_universe_files(
                start_date=date_type.fromisoformat(start.date().isoformat()),
                end_date=date_type.fromisoformat(end.date().isoformat()),
                universe_dir=universe_dir,
            )
            if universe_symbols:
                logger.info(f"[run_backtest] 从 universe 文件收集到 {len(universe_symbols)} 个股票")
                logger.info(f"[run_backtest] 后续将只对这些股票进行回测（投资范围 $U_t$）")
                # 转换为 vt_symbol 格式（添加交易所后缀）
                # 注意：这里假设都是 NASDAQ，实际应该从数据中推断
                universe_vt_symbols = [f"{s}.NASDAQ" for s in universe_symbols]
            else:
                logger.warning(f"[run_backtest] Universe 文件为空，将使用信号中的所有股票")
                universe_vt_symbols = None
        except Exception as exc:
            logger.warning(f"[run_backtest] 加载 universe 文件失败: {exc}，将使用信号中的所有股票")
            universe_vt_symbols = None
    else:
        logger.info(f"[run_backtest] 未找到 universe 目录，将使用信号中的所有股票")
        universe_vt_symbols = None

    # 尝试加载信号
    logger.info(f"[run_backtest] 尝试加载信号文件: {signal_name}")
    try:
        signal_df = load_signal(lab, signal_name)
        vt_symbols_list = sorted(signal_df["vt_symbol"].unique().to_list())
        logger.info(f"[run_backtest] 成功加载信号: {len(vt_symbols_list)} 个合约, {len(signal_df)} 行数据")
    except RuntimeError as exc:
        logger.warning(f"[run_backtest] 信号文件不存在: {exc}")
        logger.info(f"[run_backtest] 从 lab 推断股票列表...")
        # 如果信号不存在，从 lab 推断股票列表并生成占位信号
        vt_symbols_list = infer_vt_symbols_from_lab(lab, interval)
        if not vt_symbols_list:
            logger.error(f"[run_backtest] 无法推断股票列表，lab 中无数据")
            raise RuntimeError("No vt_symbols found in lab and no signal available")
        
        logger.info(f"[run_backtest] 生成占位信号（基于滚动收益率）...")
        signal_df = generate_naive_signal(lab, vt_symbols_list, interval, start, end)
        logger.warning(f"[run_backtest] 使用占位信号，建议先构建正式信号")

    # 如果指定了 vt_symbols，使用指定的；否则从信号或 lab 推断
    if vt_symbols is not None and len(vt_symbols) > 0:
        # 使用指定的 vt_symbols
        logger.info(f"[run_backtest] 使用指定的 vt_symbols: {len(vt_symbols)} 个")
        vt_symbols_list = vt_symbols
    else:
        # 过滤信号到回测日期范围（使用 <= 而不是 <，因为回测日期是当天的收盘）
        logger.info(f"[run_backtest] 过滤信号到回测日期范围...")
        if signal_df is not None:
            before_filter = len(signal_df)
            logger.debug(f"[run_backtest] 信号数据日期范围: {signal_df['datetime'].min()} 到 {signal_df['datetime'].max()}")
            logger.debug(f"[run_backtest] 回测日期范围: {start} 到 {end}")
            
            # 先尝试过滤到回测日期范围
            filtered_df = signal_df.filter(
                (pl.col("datetime") >= start) & (pl.col("datetime") <= end)
            )
            after_filter = len(filtered_df)
            logger.info(f"[run_backtest] 信号过滤: {before_filter} -> {after_filter} 行")
            
            # 如果过滤后为空，使用最近的信号数据（回测日期之前的最后一个交易日）
            if after_filter == 0:
                logger.warning(f"[run_backtest] 信号数据在回测日期范围内为空，使用最近的信号数据")
                # 获取最近的日期（<= end）
                latest_date = signal_df.filter(pl.col("datetime") <= end)["datetime"].max()
                if latest_date is not None:
                    filtered_df = signal_df.filter(pl.col("datetime") == latest_date)
                    logger.info(f"[run_backtest] 使用最近信号日期: {latest_date}, 行数: {len(filtered_df)}")
                else:
                    logger.error(f"[run_backtest] 无法找到 <= {end} 的信号数据")
                    filtered_df = signal_df
            
            signal_df = filtered_df
            
            # 如果指定了 universe，只保留 universe 中的股票
            if universe_vt_symbols is not None:
                before_universe_filter = len(signal_df)
                signal_df = signal_df.filter(pl.col("vt_symbol").is_in(universe_vt_symbols))
                after_universe_filter = len(signal_df)
                logger.info(f"[run_backtest] Universe 筛选: {before_universe_filter} -> {after_universe_filter} 行")
                logger.info(f"[run_backtest] 投资范围 $U_t$ 包含 {len(universe_vt_symbols)} 个股票，信号中有 {signal_df['vt_symbol'].n_unique()} 个")
            
            vt_symbols_list = sorted(signal_df["vt_symbol"].unique().to_list())
            logger.info(f"[run_backtest] 过滤后合约数: {len(vt_symbols_list)}")
        
        if not vt_symbols_list:
            logger.error(f"[run_backtest] 回测股票池为空")
            raise RuntimeError("Empty vt_symbols universe for backtest")
    
    # 添加 VIX 和 VIX3M 到回测合约列表（策略需要这些数据来计算杠杆）
    vix_symbols = ["VIX.CBOE", "VIX3M.CBOE"]
    for vix_symbol in vix_symbols:
        if vix_symbol not in vt_symbols_list:
            # 检查 lab 中是否有 VIX 数据
            vix_bars = lab.load_bar_data(vix_symbol, Interval.DAILY, start, end)
            if vix_bars:
                vt_symbols_list.append(vix_symbol)
                logger.info(f"[run_backtest] 添加 VIX 数据: {vix_symbol} ({len(vix_bars)} 条数据)")
            else:
                logger.warning(f"[run_backtest] VIX 数据不存在: {vix_symbol}，策略将使用默认杠杆")
    
    vt_symbols_list = sorted(vt_symbols_list)

    # 默认策略参数
    if strategy_setting is None:
        strategy_setting = {
            "top_n": 12,
            "min_score_threshold": 0.5,
            "min_holding_days": 2,
            "max_holding_days": 5,
            "cash_ratio": 0.95,
            "min_volume": 1,
            "open_rate": 0.0005,
            "close_rate": 0.0015,
            "min_commission": 1,
            "price_add": 0.0005,
            "stop_loss_atr_multiplier": 2.5,
            "max_daily_drawdown": 0.04,
        }

    # 配置回测引擎
    logger.info(f"[run_backtest] 初始化回测引擎...")
    logger.info(f"[run_backtest] 回测参数:")
    logger.info(f"[run_backtest]   - 合约数: {len(vt_symbols_list)}")
    logger.info(f"[run_backtest]   - 初始资金: {initial_capital:,}")
    logger.info(f"[run_backtest]   - 无风险利率: {risk_free}")
    logger.info(f"[run_backtest]   - 年化交易日: {annual_days}")
    logger.info(f"[run_backtest] 策略参数: {strategy_setting}")
    
    # 为所有合约添加交易配置（避免 KeyError）- 批量优化版本
    logger.info(f"[run_backtest] 批量添加合约交易配置...")
    import json
    from pathlib import Path
    
    # 批量加载现有配置（容错处理）
    contracts: dict = {}
    contract_path = lab.contract_path
    if contract_path.exists():
        try:
            with open(contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"[run_backtest] 合约配置文件 JSON 格式错误: {e}，将重新创建")
            # 备份损坏的文件
            import shutil
            backup_path = contract_path.with_suffix('.json.corrupted')
            shutil.copy(contract_path, backup_path)
            logger.warning(f"[run_backtest] 已备份损坏文件到: {backup_path}")
            contracts = {}
    
    # 批量更新配置（确保所有必需字段都存在）
    new_count = 0
    updated_count = 0
    required_fields = ["long_rate", "short_rate", "size", "pricetick"]
    default_config = {
        "long_rate": 0.0003,
        "short_rate": 0.0003,
        "size": 1,
        "pricetick": 0.01
    }
    
    for vt_symbol in vt_symbols_list:
        if vt_symbol not in contracts:
            contracts[vt_symbol] = default_config.copy()
            new_count += 1
        else:
            # 检查并补充缺失的字段
            needs_update = False
            for field in required_fields:
                if field not in contracts[vt_symbol]:
                    contracts[vt_symbol][field] = default_config[field]
                    needs_update = True
            if needs_update:
                updated_count += 1
    
    # 一次性保存所有配置
    if new_count > 0 or updated_count > 0:
        with open(contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(
                contracts,
                f,
                indent=4,
                ensure_ascii=False
            )
        if new_count > 0 and updated_count > 0:
            logger.info(f"[run_backtest] 已为 {new_count} 个新合约添加配置，更新了 {updated_count} 个合约的配置（共 {len(vt_symbols_list)} 个合约）")
        elif new_count > 0:
            logger.info(f"[run_backtest] 已为 {new_count} 个新合约添加交易配置（共 {len(vt_symbols_list)} 个合约）")
        else:
            logger.info(f"[run_backtest] 已更新 {updated_count} 个合约的配置（共 {len(vt_symbols_list)} 个合约）")
    else:
        logger.info(f"[run_backtest] 所有 {len(vt_symbols_list)} 个合约的配置已存在且完整")
    
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols_list,
        interval=interval,
        start=start,
        end=end,
        capital=initial_capital,
        risk_free=risk_free,
        annual_days=annual_days,
    )
    logger.info(f"[run_backtest] 回测引擎参数设置完成")

    # 添加策略并运行回测
    logger.info(f"[run_backtest] 添加策略: FlagshipAlphaMomentumStrategy")
    engine.add_strategy(FlagshipAlphaMomentumStrategy, strategy_setting, signal_df)

    logger.info(f"[run_backtest] 加载历史数据...")
    engine.load_data()
    logger.info(f"[run_backtest] 历史数据加载完成")

    logger.info(f"[run_backtest] 开始运行回测...")
    engine.run_backtesting()
    logger.info(f"[run_backtest] 回测执行完成")

    logger.info(f"[run_backtest] 计算回测结果...")
    engine.calculate_result()
    logger.info(f"[run_backtest] 结果计算完成")

    logger.info(f"[run_backtest] 计算统计指标...")
    stats = engine.calculate_statistics()
    logger.info(f"[run_backtest] 统计指标计算完成")

    # 打印关键统计指标
    logger.info("\n" + "=" * 60)
    logger.info("回测统计结果")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"{k}: {v}")

    # 显示交易清单
    logger.info(f"[run_backtest] 显示交易清单...")
    engine.show_trade_list()

    # 显示净值曲线和回撤图表
    logger.info(f"[run_backtest] 显示回测图表...")
    engine.show_chart()
    
    # 保存回测报告
    logger.info(f"[run_backtest] 保存回测报告...")
    save_backtest_report(lab, engine, stats, start, end, signal_name)
    logger.info(f"[run_backtest] 回测流程全部完成")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run backtest for Flagship Alpha-Momentum strategy."
    )
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab 数据目录路径（默认 lab/flagship_alpha_momentum）",
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
        "--interval",
        type=str,
        choices=["daily", "minute"],
        default="daily",
        help="K线周期（默认 daily）",
    )
    parser.add_argument(
        "--signal-name",
        type=str,
        default="flagship_alpha_momentum",
        help="信号文件名（默认 flagship_alpha_momentum）",
    )
    parser.add_argument(
        "--capital",
        type=int,
        default=1_000_000,
        help="初始资金（默认 1,000,000）",
    )
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    interval = Interval.DAILY if args.interval == "daily" else Interval.MINUTE

    run_backtest(
        lab_path=args.lab_path,
        start=start,
        end=end,
        interval=interval,
        signal_name=args.signal_name,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    main()

