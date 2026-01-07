"""
Flagship Alpha-Momentum 策略回测入口脚本。

支持短期回测（2-3天）用于验证策略逻辑。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time as dtime
from pathlib import Path
import sys
from typing import Any

import polars as pl

# 动态注入项目根路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger
from vnpy.trader.setting import SETTINGS
from vnpy.alpha import AlphaLab, BacktestingEngine
from flagship.config import VT_SETTING_PATH
from flagship.universe.pg_ticker_db import get_pg_connection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 延迟导入策略类，以便根据参数选择
def get_strategy_class(strategy_name: str):
    if strategy_name.lower() == "v7":
        from flagship.strategy.flagship_alpha_momentum_strategy_v7 import FlagshipAlphaMomentumStrategy as V7Strategy
        return V7Strategy
    else:
        from flagship.strategy.flagship_alpha_momentum_strategy import FlagshipAlphaMomentumStrategy as V5Strategy
        return V5Strategy

from flagship.backtest.backtest_session_alignment import (
    RegularTradingHoursFilteredAlphaLab,
    SignalAwareBacktestingEngine,
    SignalResolver,
    is_date_only_str,
    is_in_regular_trading_hours,
    parse_date_or_datetime,
    signal_is_daily_snapshot,
)


def _compute_market_independence_metrics(
    lab: AlphaLab,
    daily_df: pl.DataFrame,
    *,
    benchmark_symbol: str = "SPY.NASDAQ",
    start: datetime,
    end: datetime,
    annual_days: int,
) -> dict[str, Any]:
    """
    计算策略相对基准（默认 SPY）市场独立性指标：
    - Beta: cov(Rs, Rb) / var(Rb)
    - Alpha: E[Rs] - Beta * E[Rb]（并给出年化）
    - Correlation: corr(Rs, Rb)

    注意：
    - 使用 engine.daily_df 的 balance 计算策略日收益（pct_change）。
    - 基准使用 lab 日线 close 计算日收益。
    - start/end 可能是 minute+rth-only 的 09:30/16:00，因此这里内部会扩展到整天以包含 daily bar（00:00）。
    """
    if daily_df is None or daily_df.is_empty():
        return {}

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        logger.warning(f"[market_independence] numpy 不可用，跳过 Beta/Alpha 计算: {exc}")
        return {}

    # 策略日收益（由 balance 推导）
    try:
        strategy_ret_df = (
            daily_df
            .select(
                pl.col("date").cast(pl.Date).alias("date"),
                pl.col("balance").cast(pl.Float64).alias("balance"),
            )
            .with_columns(
                (pl.col("balance") / pl.col("balance").shift(1) - 1.0).alias("strategy_ret")
            )
            .drop_nulls(["strategy_ret"])
        )
    except Exception as exc:
        logger.warning(f"[market_independence] 计算策略日收益失败，跳过: {exc}")
        return {}

    # 基准日收益（由 daily close 推导）
    start_daily = datetime.combine(start.date(), dtime(0, 0))
    end_daily = datetime.combine(end.date(), dtime(23, 59, 59))

    bars = lab.load_bar_data(benchmark_symbol, Interval.DAILY, start_daily, end_daily)
    if not bars:
        logger.warning(f"[market_independence] 基准日线数据缺失: {benchmark_symbol}，跳过 Beta/Alpha 计算")
        return {"benchmark_symbol": benchmark_symbol}

    bench_df = pl.DataFrame(
        {
            "date": [bar.datetime.date() for bar in bars],
            "close": [float(bar.close_price) for bar in bars],
        }
    ).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )

    bench_ret_df = (
        bench_df
        .with_columns((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("bench_ret"))
        .drop_nulls(["bench_ret"])
        .select(["date", "bench_ret"])
    )

    merged = (
        strategy_ret_df
        .select(["date", "strategy_ret"])
        .join(bench_ret_df, on="date", how="inner")
        .drop_nulls(["strategy_ret", "bench_ret"])
        .sort("date")
    )

    if merged.is_empty() or merged.height < 3:
        logger.warning(
            f"[market_independence] 回归样本不足（n={merged.height}），跳过 Beta/Alpha 计算"
        )
        return {"benchmark_symbol": benchmark_symbol}

    x = merged["bench_ret"].to_numpy()
    y = merged["strategy_ret"].to_numpy()
    x = x.astype(float)
    y = y.astype(float)

    # Filter NaN/Inf defensively
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        logger.warning(
            f"[market_independence] 有效样本不足（n={x.size}），跳过 Beta/Alpha 计算"
        )
        return {"benchmark_symbol": benchmark_symbol}

    var_x = float(np.var(x, ddof=1))
    if var_x <= 0:
        logger.warning("[market_independence] 基准收益方差为 0，无法计算 Beta/Alpha")
        return {"benchmark_symbol": benchmark_symbol}

    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    beta = cov_xy / var_x
    alpha_daily = float(np.mean(y) - beta * np.mean(x))
    alpha_annual = alpha_daily * float(annual_days)
    corr = float(np.corrcoef(x, y)[0, 1])

    return {
        "benchmark_symbol": benchmark_symbol,
        "beta_vs_benchmark": float(beta),
        "alpha_vs_benchmark_daily": float(alpha_daily),
        "alpha_vs_benchmark_annual": float(alpha_annual),
        "corr_vs_benchmark": float(corr),
        "beta_observations": int(x.size),
    }


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


def load_daily_selection_from_postgres(
    start_date: date,
    end_date: date,
) -> dict[date, list[str]]:
    """
    从PostgreSQL加载每日选股结果。
    
    Returns:
        字典，key为交易日期，value为该日选中的vt_symbol列表
    """
    # 重新加载配置
    if VT_SETTING_PATH.exists():
        try:
            setting_data = json.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
            SETTINGS.update(setting_data)
        except Exception as exc:
            logger.warning(f"Failed to reload vt_setting.json: {exc}")
    
    daily_selections: dict[date, list[str]] = {}
    
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trade_date, vt_symbol
                FROM daily_selection
                WHERE trade_date >= %s AND trade_date <= %s
                ORDER BY trade_date, vt_symbol
            """, (start_date, end_date))
            
            for row in cur.fetchall():
                trade_date, vt_symbol = row
                if trade_date not in daily_selections:
                    daily_selections[trade_date] = []
                daily_selections[trade_date].append(vt_symbol)
    
    logger.info(f"从PostgreSQL加载了 {len(daily_selections)} 个交易日的选股结果")
    return daily_selections


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


def setup_backtest_logging(
    lab_path: str | Path,
    start: datetime,
    end: datetime,
    signal_name: str,
) -> Path:
    """
    设置回测日志文件。
    
    Returns:
        日志文件路径
    """
    # 确保 lab_path 是 Path 对象
    lab_path_obj = Path(lab_path)
    if not lab_path_obj.is_absolute():
        # 如果是相对路径，基于项目根目录
        lab_path_obj = PROJECT_ROOT / lab_path_obj
    
    log_dir = lab_path_obj / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名：backtest_YYYYMMDD_HHMMSS_start_end_signal.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    signal_short = signal_name.replace("flagship_alpha_momentum", "fam").replace("_signal", "")
    
    log_filename = f"backtest_{timestamp}_{start_str}_{end_str}_{signal_short}.log"
    log_path = log_dir / log_filename
    
    # 使用 loguru 的 add 方法添加文件输出
    # loguru 的格式：{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}
    logger.add(
        sink=str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        encoding="utf-8",
        enqueue=True,  # 异步写入，提高性能
    )
    
    return log_path


def run_backtest(
    lab_path: str | Path,
    start: datetime,
    end: datetime,
    interval: Interval = Interval.MINUTE,
    signal_name: str = "flagship_alpha_momentum",
    initial_capital: int = 1_000_000,
    risk_free: float = 0.02,
    annual_days: int = 252,
    strategy_setting: dict[str, Any] | None = None,
    vt_symbols: list[str] | None = None,
    min_score_threshold: float = 0.5,
    use_postgres_selection: bool = True,
    commission_rate: float | None = None,
    rth_only: bool | None = None,
    strategy_version: str = "v5",
) -> dict[str, Any] | None:
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
        commission_rate: 交易佣金费率（默认为 None，优先 from strategy_setting 读取，否则默认为 0.0）
        strategy_version: 策略版本 ("v5" 或 "v7")
    """
    # 设置日志文件（必须在记录日志之前）
    log_path = setup_backtest_logging(lab_path, start, end, signal_name)
    
    logger.info("=" * 80)
    logger.info(f"[run_backtest] 开始运行回测")
    logger.info(f"[run_backtest] 策略版本: {strategy_version}")
    logger.info(f"[run_backtest] Lab 路径: {lab_path}")
    logger.info(f"[run_backtest] 回测日期范围: {start} 到 {end}")
    logger.info(f"[run_backtest] K线周期: {interval}")
    logger.info(f"[run_backtest] 信号文件名: {signal_name}")
    logger.info(f"[run_backtest] 初始资金: {initial_capital:,}")
    logger.info(f"[run_backtest] 使用PostgreSQL选股: {use_postgres_selection}")
    logger.info(f"[run_backtest] 日志文件: {log_path}")
    logger.info("=" * 80)

    # minute 回测默认启用 RTH-only（除非显式关闭）
    if rth_only is None:
        rth_only = interval == Interval.MINUTE
    logger.info(f"[run_backtest] rth_only={rth_only} (interval={interval})")
    
    lab = AlphaLab(str(lab_path))
    logger.debug(f"[run_backtest] AlphaLab 初始化完成")
    lab_for_engine = lab
    if interval == Interval.MINUTE and rth_only:
        lab_for_engine = RegularTradingHoursFilteredAlphaLab(lab, rth_only=True)
        logger.info("[run_backtest] 已启用 RTH-only：minute bars 仅保留 09:30–16:00 (ET)")

    vt_symbols_list: list[str] = []
    signal_df: pl.DataFrame | None = None
    daily_selections: dict[date, list[str]] | None = None
    signal_snapshot: bool = False
    
    # 如果使用PostgreSQL选股，加载每日选股结果
    if use_postgres_selection:
        try:
            daily_selections = load_daily_selection_from_postgres(
                start_date=start.date(),
                end_date=end.date(),
            )
            if daily_selections:
                # 合并所有交易日的选股结果作为候选股票池
                all_selected_symbols = set()
                for symbols in daily_selections.values():
                    all_selected_symbols.update(symbols)
                vt_symbols_list = sorted(list(all_selected_symbols))
                logger.info(f"[run_backtest] 从PostgreSQL加载了 {len(vt_symbols_list)} 只候选股票")
        except Exception as exc:
            logger.warning(f"[run_backtest] 从PostgreSQL加载选股结果失败: {exc}，将使用信号文件或lab推断")
            daily_selections = None

    # Universe 文件选股（legacy）已废弃：当前以 PostgreSQL daily_selection 为主。
    # 如需强制限定回测标的，请通过 run_backtest(vt_symbols=...) 传入。
    # 尝试加载信号
    logger.info(f"[run_backtest] 尝试加载信号文件: {signal_name}")
    signal_df = None
    try:
        signal_df = load_signal(lab, signal_name)
        logger.info(f"[run_backtest] 成功加载信号: {len(signal_df)} 行数据")
    except RuntimeError as exc:
        logger.warning(f"[run_backtest] 信号文件不存在: {exc}")
        logger.info(f"[run_backtest] 从 lab 推断股票列表...")
        # 如果信号不存在，从 lab 推断股票列表并生成占位信号
        inferred_symbols = infer_vt_symbols_from_lab(lab, interval)
        if not inferred_symbols:
            logger.error(f"[run_backtest] 无法推断股票列表，lab 中无数据")
            raise RuntimeError("No vt_symbols found in lab and no signal available")
        
        logger.info(f"[run_backtest] 生成占位信号（基于滚动收益率）...")
        signal_df = generate_naive_signal(lab, inferred_symbols, interval, start, end)
        logger.warning(f"[run_backtest] 使用占位信号，建议先构建正式信号")

    if signal_df is not None and not signal_df.is_empty():
        signal_snapshot = signal_is_daily_snapshot(signal_df)
        logger.info(f"[run_backtest] signal_is_snapshot={signal_snapshot}")

    # 如果指定了 vt_symbols，使用指定的；否则从信号或 lab 推断
    if vt_symbols is not None and len(vt_symbols) > 0:
        # 使用指定的 vt_symbols
        logger.info(f"[run_backtest] 使用指定的 vt_symbols: {len(vt_symbols)} 个")
        vt_symbols_list = vt_symbols
    else:
        # 第一步：过滤信号到回测日期范围
        logger.info(f"[run_backtest] 第一步：过滤信号到回测日期范围...")
        if signal_df is not None:
            before_filter = len(signal_df)
            logger.debug(f"[run_backtest] 信号数据日期范围: {signal_df['datetime'].min()} 到 {signal_df['datetime'].max()}")
            logger.debug(f"[run_backtest] 回测日期范围: {start} 到 {end}")
            
            # 先尝试过滤到回测日期范围
            if interval == Interval.MINUTE and signal_snapshot:
                start_d = start.date()
                end_d = end.date()
                filtered_df = signal_df.filter(
                    (pl.col("datetime").dt.date() >= start_d) & (pl.col("datetime").dt.date() <= end_d)
                )
            else:
                filtered_df = signal_df.filter(
                    (pl.col("datetime") >= start) & (pl.col("datetime") <= end)
                )
            after_filter = len(filtered_df)
            logger.info(f"[run_backtest] 日期过滤: {before_filter} -> {after_filter} 行")
            
            # 如果过滤后为空，使用最近的信号数据（回测日期之前的最后一个交易日）
            if after_filter == 0:
                logger.warning(f"[run_backtest] 信号数据在回测日期范围内为空，使用最近的信号数据")
                # 获取最近的日期（<= end）
                if interval == Interval.MINUTE and signal_snapshot:
                    latest_dt = signal_df.filter(pl.col("datetime").dt.date() <= end.date())["datetime"].max()
                    if latest_dt is not None:
                        filtered_df = signal_df.filter(pl.col("datetime").dt.date() == latest_dt.date())
                        logger.info(f"[run_backtest] 使用最近信号日期: {latest_dt.date()}, 行数: {len(filtered_df)}")
                    else:
                        logger.error(f"[run_backtest] 无法找到 <= {end.date()} 的信号数据")
                        filtered_df = signal_df
                else:
                    latest_dt = signal_df.filter(pl.col("datetime") <= end)["datetime"].max()
                    if latest_dt is not None:
                        filtered_df = signal_df.filter(pl.col("datetime") == latest_dt)
                        logger.info(f"[run_backtest] 使用最近信号日期: {latest_dt}, 行数: {len(filtered_df)}")
                    else:
                        logger.error(f"[run_backtest] 无法找到 <= {end} 的信号数据")
                        filtered_df = signal_df
            
            signal_df = filtered_df

        # 第二步：如果使用 PostgreSQL 选股，按每日选股结果过滤信号（ADV、价格等基础条件已在选股时完成）
        if daily_selections and signal_df is not None:
            logger.info("[run_backtest] 第二步：根据 PostgreSQL 每日选股结果过滤信号（按交易日）...")
            filtered_rows: list[dict[str, Any]] = []
            for row in signal_df.iter_rows(named=True):
                trade_date = row["datetime"].date()
                vt_symbol = row["vt_symbol"]
                if trade_date in daily_selections and vt_symbol in daily_selections[trade_date]:
                    filtered_rows.append(row)

            if filtered_rows:
                signal_df = pl.DataFrame(filtered_rows)
                vt_symbols_list = sorted(signal_df["vt_symbol"].unique().to_list())
                logger.info(
                    f"[run_backtest] PostgreSQL选股过滤后: {len(signal_df)} 行信号, {len(vt_symbols_list)} 只股票"
                )
            else:
                logger.warning("[run_backtest] PostgreSQL选股过滤后信号为空，将使用原始信号的股票列表")
                vt_symbols_list = sorted(signal_df["vt_symbol"].unique().to_list())
                logger.info(f"[run_backtest] 使用信号中的所有股票: {len(vt_symbols_list)} 只")
        else:
            if signal_df is not None:
                vt_symbols_list = sorted(signal_df["vt_symbol"].unique().to_list())
                logger.info(f"[run_backtest] 未使用PostgreSQL选股，使用信号中的所有股票: {len(vt_symbols_list)} 只")
            else:
                logger.error(f"[run_backtest] 无法确定股票列表")
                raise RuntimeError("Cannot determine vt_symbols list")
        
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
            "top_n": 5,  # 修改为 Top 5
            "min_score_threshold": min_score_threshold,
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
    else:
        strategy_setting = dict(strategy_setting)
        strategy_setting["min_score_threshold"] = strategy_setting.get(
            "min_score_threshold", min_score_threshold
        )

    # 确定佣金费率
    # 优先级: 1. strategy_setting['commission_rate']  2. commission_rate 参数  3. 默认值 0.0 (Alpaca)
    effective_commission_rate = 0.0
    if "commission_rate" in strategy_setting:
        effective_commission_rate = float(strategy_setting["commission_rate"])
    elif commission_rate is not None:
        effective_commission_rate = commission_rate
    
    logger.info(f"[run_backtest] 交易佣金费率: {effective_commission_rate}")

    # 配置回测引擎
    logger.info(f"[run_backtest] 初始化回测引擎...")
    logger.info(f"[run_backtest] 回测参数:")
    logger.info(f"[run_backtest]   - 合约数: {len(vt_symbols_list)}")
    logger.info(f"[run_backtest]   - 初始资金: {initial_capital:,}")
    logger.info(f"[run_backtest]   - 无风险利率: {risk_free}")
    logger.info(f"[run_backtest]   - 年化交易日: {annual_days}")
    logger.info(f"[run_backtest] 策略参数: {strategy_setting}")
    
    # 为所有合约添加交易配置（避免 KeyError）- 批量写入版本
    logger.info("[run_backtest] 批量添加合约交易配置...")

    contracts: dict[str, Any] = {}
    contract_path = lab.contract_path
    if contract_path.exists():
        try:
            with open(contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning(f"[run_backtest] 合约配置文件 JSON 损坏: {exc}，将重新创建")
            # 备份损坏文件
            try:
                import shutil

                backup_path = contract_path.with_suffix(".json.corrupted")
                shutil.copy(contract_path, backup_path)
                logger.warning(f"[run_backtest] 已备份损坏文件到: {backup_path}")
            except Exception:
                pass
            contracts = {}
        except OSError as exc:
            logger.warning(f"[run_backtest] 读取合约配置失败: {exc}，将重新创建")
            contracts = {}

    required_fields = ["long_rate", "short_rate", "size", "pricetick"]
    default_config = {
        "long_rate": float(effective_commission_rate),
        "short_rate": float(effective_commission_rate),
        "size": 1,
        "pricetick": 0.01,
    }

    new_count = 0
    updated_count = 0
    for vt_symbol in vt_symbols_list:
        if vt_symbol not in contracts or not isinstance(contracts.get(vt_symbol), dict):
            contracts[vt_symbol] = dict(default_config)
            new_count += 1
            continue

        needs_update = False
        for field in required_fields:
            if field not in contracts[vt_symbol]:
                contracts[vt_symbol][field] = default_config[field]
                needs_update = True

        # 确保费率与本次回测一致
        if contracts[vt_symbol].get("long_rate") != default_config["long_rate"]:
            contracts[vt_symbol]["long_rate"] = default_config["long_rate"]
            needs_update = True
        if contracts[vt_symbol].get("short_rate") != default_config["short_rate"]:
            contracts[vt_symbol]["short_rate"] = default_config["short_rate"]
            needs_update = True

        if needs_update:
            updated_count += 1

    if new_count > 0 or updated_count > 0:
        with open(contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(contracts, f, indent=4, ensure_ascii=False)
        logger.info(
            f"[run_backtest] 合约配置写入完成：新增 {new_count}，更新 {updated_count}，总计 {len(vt_symbols_list)}"
        )
    else:
        logger.info(f"[run_backtest] 合约配置已完整（共 {len(vt_symbols_list)}）")
    
    # 分钟回测：signal 解析器（精确匹配优先，按交易日回退）
    signal_resolver: SignalResolver | None = None
    if interval == Interval.MINUTE and signal_df is not None and not signal_df.is_empty():
        try:
            signal_resolver = SignalResolver.from_signal_df(signal_df)
            logger.info(
                f"[run_backtest] SignalResolver ready: "
                f"snapshot={signal_resolver.is_daily_snapshot}, "
                f"days={len(signal_resolver.by_trade_date)}"
            )
        except Exception as exc:
            logger.warning(f"[run_backtest] SignalResolver 初始化失败，将回退到原始 get_signal: {exc}")
            signal_resolver = None

    engine = SignalAwareBacktestingEngine(lab_for_engine, signal_resolver=signal_resolver)
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
    strategy_class = get_strategy_class(strategy_version)
    logger.info(f"[run_backtest] 添加策略: {strategy_class.__name__} (Version: {strategy_version})")
    engine.add_strategy(strategy_class, strategy_setting, signal_df)

    logger.info(f"[run_backtest] 加载历史数据...")
    engine.load_data()
    logger.info(f"[run_backtest] 历史数据加载完成")

    # 自动验证：RTH-only 时间轴检查
    if interval == Interval.MINUTE and rth_only:
        if getattr(engine, "dts", None):
            dts = list(engine.dts)  # type: ignore[attr-defined]
            dts.sort()
            first_dt = dts[0]
            last_dt = dts[-1]
            outside_rth_count = sum(1 for dt in dts if not is_in_regular_trading_hours(dt))
            logger.info(
                f"[run_backtest] RTH-only timeline: first_dt={first_dt}, last_dt={last_dt}, "
                f"outside_rth_count={outside_rth_count}"
            )
        else:
            logger.warning("[run_backtest] engine.dts 为空，无法验证 RTH-only 时间轴")

    logger.info(f"[run_backtest] 开始运行回测...")
    engine.run_backtesting()
    logger.info(f"[run_backtest] 回测执行完成")

    # 自动验证：第一笔成交时间（应 >= 09:30，且通常为 09:31）
    trade_dts = [t.datetime for t in engine.trades.values() if t.datetime]  # type: ignore[attr-defined]
    if trade_dts:
        first_trade_time = min(trade_dts)
        logger.info(f"[run_backtest] first_trade_time={first_trade_time}")
    else:
        logger.info("[run_backtest] first_trade_time=N/A（无成交）")

    logger.info(f"[run_backtest] 计算回测结果...")
    daily_df = engine.calculate_result()
    logger.info(f"[run_backtest] 结果计算完成")

    if daily_df is None:
        logger.warning("[run_backtest] 无成交记录，跳过统计与图表输出")
        return None

    logger.info(f"[run_backtest] 计算统计指标...")
    stats = engine.calculate_statistics()
    logger.info(f"[run_backtest] 统计指标计算完成")

    # 市场独立性（Alpha/Beta vs Benchmark）
    try:
        market_metrics = _compute_market_independence_metrics(
            lab=lab,
            daily_df=engine.daily_df,
            benchmark_symbol="SPY.NASDAQ",
            start=start,
            end=end,
            annual_days=annual_days,
        )
        if market_metrics:
            stats.update(market_metrics)
            logger.info(
                "[run_backtest] market_independence: "
                f"benchmark={market_metrics.get('benchmark_symbol')}, "
                f"beta={market_metrics.get('beta_vs_benchmark')}, "
                f"alpha_annual={market_metrics.get('alpha_vs_benchmark_annual')}, "
                f"corr={market_metrics.get('corr_vs_benchmark')}, "
                f"n={market_metrics.get('beta_observations')}"
            )
    except Exception as exc:
        logger.warning(f"[run_backtest] market_independence 计算失败，跳过: {exc}")

    # 确定报告文件夹名称（FROM_TO_Scenario格式）并获取regime信息
    from flagship.backtest.index_regime_windows import REGIME_WINDOWS
    report_folder_name = None
    matched_regime = None
    for regime in REGIME_WINDOWS:
        if regime.start == start.date() and regime.end == end.date():
            # 格式：20240102_20240412_regime01
            report_folder_name = f"{regime.start.strftime('%Y%m%d')}_{regime.end.strftime('%Y%m%d')}_regime{regime.id:02d}"
            matched_regime = regime
            break
    
    if report_folder_name is None:
        # 如果没有匹配的regime，使用日期范围
        report_folder_name = f"{start.date().strftime('%Y%m%d')}_{end.date().strftime('%Y%m%d')}_backtest"
    
    report_dir = lab.lab_path / "report" / report_folder_name
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[run_backtest] 报告文件夹: {report_dir}")

    # 打印关键统计指标
    logger.info("\n" + "=" * 60)
    logger.info("回测统计结果")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"{k}: {v}")

    # 获取交易清单DataFrame
    trade_list_df = engine.get_trade_list()
    trade_list_path = report_dir / "trade_list.csv"
    if trade_list_df is not None and not trade_list_df.is_empty():
        # 使用UTF-8-BOM编码保存CSV，确保Excel等工具正确显示中文
        # Polars的write_csv不支持BOM，需要手动添加
        import io
        csv_buffer = io.StringIO()
        trade_list_df.write_csv(csv_buffer)
        csv_content = csv_buffer.getvalue()
        
        # 添加UTF-8 BOM标记
        with open(trade_list_path, 'w', encoding='utf-8-sig') as f:
            f.write(csv_content)
        logger.info(f"[run_backtest] 交易清单已保存（UTF-8-BOM）: {trade_list_path}")
    else:
        logger.warning("[run_backtest] 交易清单为空")
        trade_list_path = None

    # 生成包含图表的HTML报告
    logger.info(f"[run_backtest] 生成HTML回测报告（包含图表）...")
    try:
        html_report_path = report_dir / "report.html"
        
        # 尝试加载数据集名称（用于因子有效性计算）
        dataset_name = None
        if hasattr(engine.strategy, 'dataset_name'):
            dataset_name = engine.strategy.dataset_name
        else:
            # 尝试从信号文件名推断数据集名称
            if signal_name and "regime" in signal_name:
                # 如果是lgb信号（LightGBM训练的信号），使用对应的lgb数据集
                if "_lgb_signal" in signal_name:
                    # 提取regime ID，例如 flagship_alpha_mom_regime02_lgb_signal -> flagship_alpha_mom_regime02_lgb
                    import re
                    match = re.search(r'regime(\d+)', signal_name)
                    if match:
                        regime_id = int(match.group(1))
                        dataset_name = f"flagship_alpha_mom_regime{regime_id:02d}_lgb"
                    else:
                        # 回退到v5数据集（因为lgb信号是基于v5因子训练的）
                        dataset_name = "flagship_alpha_momentum_v5"
                # v5信号使用v5数据集
                elif "_v5_" in signal_name or signal_name.endswith("_v5_signal"):
                    dataset_name = "flagship_alpha_momentum_v5"
                # 其他情况默认使用v5数据集（不再使用v4）
                else:
                    dataset_name = "flagship_alpha_momentum_v5"
            else:
                # 默认使用v5数据集
                dataset_name = "flagship_alpha_momentum_v5"
        
        # 保存统计结果到临时文件（用于生成报告）
        stats_path = report_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        generate_html_report_with_charts(
            engine=engine,
            stats=stats,
            trade_list_path=trade_list_path,
            output_path=html_report_path,
            signal_name=signal_name,
            dataset_name=dataset_name,
            lab=lab,
            regime=matched_regime,
        )
        logger.info(f"[run_backtest] HTML报告已生成: {html_report_path}")
    except Exception as exc:
        logger.warning(f"[run_backtest] 生成HTML报告失败: {exc}", exc_info=True)
    
    logger.info("=" * 80)
    logger.info(f"[run_backtest] 回测流程全部完成")
    logger.info(f"[run_backtest] 日志文件已保存: {log_path}")
    logger.info("=" * 80)
    
    # loguru 会自动管理 sink，不需要手动移除
    # 如果需要移除，可以使用 logger.remove(sink_id)，但通常不需要
    
    return stats


def generate_performance_summary(trade_list_path: Path) -> dict:
    """从交易清单生成表现摘要"""
    if not trade_list_path.exists():
        return {}
    
    df = pl.read_csv(trade_list_path)
    
    if df.is_empty():
        return {}
    
    # 使用小写列名（实际 DataFrame 的列名）
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "Net PnL"
    pnl_pct_col = "net_pnl_pct" if "net_pnl_pct" in df.columns else "Net PnL %"
    balance_entry_col = "balance_at_entry" if "balance_at_entry" in df.columns else "Balance at Entry"
    position_size_col = "position_size" if "position_size" in df.columns else "Position Size"
    positions_entry_col = "positions_at_entry" if "positions_at_entry" in df.columns else "Positions at Entry"
    
    # 计算总交易数
    total_trades = len(df)
    
    # 计算盈利交易数
    profitable_trades = df.filter(pl.col(pnl_col) > 0)
    winning_trades_count = len(profitable_trades)
    winning_trades_pct = (winning_trades_count / total_trades * 100) if total_trades > 0 else 0
    
    # 计算总盈亏
    total_pnl = df[pnl_col].sum()
    total_pnl_pct = (total_pnl / df[balance_entry_col].sum() * 100) if df[balance_entry_col].sum() > 0 else 0
    
    # 计算毛利润和毛亏损
    gross_profit = profitable_trades[pnl_col].sum() if not profitable_trades.is_empty() else 0
    losing_trades = df.filter(pl.col(pnl_col) < 0)
    gross_loss = abs(losing_trades[pnl_col].sum()) if not losing_trades.is_empty() else 0
    
    # 计算盈利因子
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    # 计算平均盈亏
    avg_pnl = df[pnl_col].mean()
    avg_pnl_pct = (avg_pnl / df[balance_entry_col].mean() * 100) if df[balance_entry_col].mean() > 0 else 0
    
    # 计算平均盈利交易和平均亏损交易
    avg_winning_trade = profitable_trades[pnl_col].mean() if not profitable_trades.is_empty() else 0
    avg_losing_trade = losing_trades[pnl_col].mean() if not losing_trades.is_empty() else 0
    
    # 计算最大盈利和最大亏损交易
    max_profit_trade = df[pnl_col].max()
    max_loss_trade = df[pnl_col].min()
    
    # 计算最大持仓数量
    max_position_size = df[position_size_col].max()
    
    # 计算最大合同持有量（从 Positions at Entry/Exit 中提取）
    max_contracts_held = 0
    if positions_entry_col in df.columns:
        for pos_str in df[positions_entry_col]:
            if pos_str and isinstance(pos_str, str):
                pos_count = len([s for s in pos_str.split(", ") if s.strip()])
                max_contracts_held = max(max_contracts_held, pos_count)
    
    return {
        "total_trades": total_trades,
        "winning_trades": {
            "count": winning_trades_count,
            "percentage": winning_trades_pct,
        },
        "total_pnl": {
            "value": float(total_pnl),
            "percentage": float(total_pnl_pct),
        },
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor),
        "average_pnl": {
            "value": float(avg_pnl),
            "percentage": float(avg_pnl_pct),
        },
        "average_winning_trade": float(avg_winning_trade),
        "average_losing_trade": float(avg_losing_trade),
        "max_profit_trade": float(max_profit_trade),
        "max_loss_trade": float(max_loss_trade),
        "max_position_size": float(max_position_size),
        "max_contracts_held": int(max_contracts_held),
    }


def generate_html_report_with_charts(
    engine: BacktestingEngine,
    stats: dict[str, Any],
    trade_list_path: Path | None,
    output_path: Path,
    signal_name: str,
    dataset_name: str,
    lab: AlphaLab,
    regime: Any | None = None,
) -> None:
    """生成包含图表的HTML报告（包含所有原有内容）"""
    from datetime import datetime
    import json
    
    # 1. 生成plotly图表
    df = engine.daily_df
    
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=["Balance", "Drawdown", "Daily Pnl"],
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
    )
    
    # Balance曲线
    balance_line = go.Scatter(
        x=df["date"],
        y=df["balance"],
        mode="lines",
        name="Balance",
        line=dict(color="blue", width=2),
    )
    
    # Drawdown曲线
    drawdown_scatter = go.Scatter(
        x=df["date"],
        y=df["drawdown"],
        fillcolor="rgba(255, 0, 0, 0.3)",
        fill='tozeroy',
        mode="lines",
        name="Drawdown",
        line=dict(color="red", width=1),
    )
    
    # Daily PnL柱状图
    colors = ['green' if x >= 0 else 'red' for x in df["net_pnl"]]
    pnl_bar = go.Bar(
        x=df["date"],
        y=df["net_pnl"],
        name="Daily Pnl",
        marker=dict(color=colors),
    )
    
    fig.add_trace(balance_line, row=1, col=1)
    fig.add_trace(drawdown_scatter, row=2, col=1)
    fig.add_trace(pnl_bar, row=3, col=1)
    
    fig.update_layout(
        height=1200,
        width=1400,
        title_text=f"回测图表 - {stats.get('start_date', 'N/A')} 至 {stats.get('end_date', 'N/A')}",
        title_x=0.5,
        showlegend=True,
    )
    
    # 更新x轴标签
    fig.update_xaxes(title_text="日期", row=3, col=1)
    fig.update_yaxes(title_text="余额", row=1, col=1)
    fig.update_yaxes(title_text="回撤", row=2, col=1)
    fig.update_yaxes(title_text="每日盈亏", row=3, col=1)
    
    # 将图表转换为HTML（包含plotly.js CDN）
    chart_html = fig.to_html(include_plotlyjs='cdn', div_id="backtest-charts", full_html=False)
    
    # 2. 生成因子有效性数据（可选，如果模块不存在则跳过）
    
    factor_validity = {}
    try:
        from vnpy.alpha.dataset import Segment
        from flagship.model.evaluate_signal_quality import evaluate_metrics  # type: ignore
        # 加载数据集（如果不存在则报错，不再回退）
        dataset = None
        if dataset_name:
            try:
                dataset = lab.load_dataset(dataset_name)
                if dataset is None:
                    raise ValueError(f"数据集 {dataset_name} 不存在")
            except Exception as e:
                logger.error(f"[generate_html_report_with_charts] 无法加载数据集 {dataset_name}: {e}")
                raise ValueError(f"数据集 {dataset_name} 不存在或加载失败，请确保使用v5数据集") from e
        
        signal_df = lab.load_signal(signal_name)
        if dataset and signal_df is not None and not signal_df.is_empty():
            factor_validity = evaluate_metrics(dataset, signal_df, Segment.TEST, quantiles=5)
    except ImportError:
        logger.debug("[generate_html_report_with_charts] evaluate_signal_quality 模块不存在，跳过因子有效性计算")
    except ValueError as exc:
        # ValueError表示数据集不存在，这是严重错误，记录并继续（不中断报告生成）
        logger.error(f"[generate_html_report_with_charts] {exc}")
    except Exception as exc:
        logger.warning(f"[generate_html_report_with_charts] 计算因子有效性失败: {exc}")
    
    # 3. 生成交易表现摘要
    performance_summary = {}
    if trade_list_path and trade_list_path.exists():
        try:
            performance_summary = generate_performance_summary(trade_list_path)
        except Exception as exc:
            logger.warning(f"[generate_html_report_with_charts] 生成交易表现摘要失败: {exc}")
    
    # 4. 读取交易清单（用于显示前20条）
    trade_list_preview_html = ""
    if trade_list_path and trade_list_path.exists():
        try:
            trade_df = pl.read_csv(trade_list_path)
            if not trade_df.is_empty():
                # 选择关键列
                display_cols = []
                col_mapping = {
                    "Trade ID": "trade_id",
                    "Symbol": "vt_symbol",
                    "Direction": "direction",
                    "Entry Time": "entry_datetime",
                    "Exit Time": "exit_datetime",
                    "Entry Price": "entry_price",
                    "Exit Price": "exit_price",
                    "Position Size": "position_size",
                    "Net PnL": "net_pnl",
                    "Net PnL %": "net_pnl_pct",
                }
                
                headers = []
                available_cols = []
                for display_name, col_name in col_mapping.items():
                    if col_name in trade_df.columns:
                        headers.append(display_name)
                        available_cols.append(col_name)
                
                if headers:
                    trade_list_preview_html = "<h2>交易清单</h2>\n"
                    trade_list_preview_html += f"<p>完整的交易清单请查看 CSV 文件: <a href='{trade_list_path.name}'>{trade_list_path.name}</a></p>\n"
                    trade_list_preview_html += "<h3>交易清单预览（前20条）</h3>\n"
                    trade_list_preview_html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; font-size: 12px;'>\n"
                    trade_list_preview_html += "<tr>" + "".join([f"<th style='text-align: left; background-color: #f0f0f0;'>{h}</th>" for h in headers]) + "</tr>\n"
                    
                    preview_df = trade_df.head(20).select(available_cols)
                    for row in preview_df.iter_rows(named=True):
                        trade_list_preview_html += "<tr>"
                        for col in available_cols:
                            val = row.get(col)
                            if val is None:
                                trade_list_preview_html += "<td></td>"
                            elif isinstance(val, float):
                                if col in ["entry_price", "exit_price", "net_pnl"]:
                                    trade_list_preview_html += f"<td style='text-align: right;'>{val:,.2f}</td>"
                                elif col == "net_pnl_pct":
                                    trade_list_preview_html += f"<td style='text-align: right;'>{val:.2f}%</td>"
                                elif col == "position_size":
                                    trade_list_preview_html += f"<td style='text-align: right;'>{val:,.0f}</td>"
                                else:
                                    trade_list_preview_html += f"<td style='text-align: right;'>{val:.4f}</td>"
                            elif isinstance(val, (int, str)):
                                trade_list_preview_html += f"<td>{val}</td>"
                            else:
                                trade_list_preview_html += f"<td>{str(val)}</td>"
                        trade_list_preview_html += "</tr>\n"
                    trade_list_preview_html += "</table>\n"
        except Exception as exc:
            logger.warning(f"[generate_html_report_with_charts] 读取交易清单失败: {exc}")
    
    # 5. 生成HTML内容
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 生成regime信息HTML
    regime_html = ""
    if regime is not None:
        # 使用更突出的格式显示regime信息
        regime_html = f"""
    <div style="background-color: #e8f4f8; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0;">
        <h2 style="color: #1976D2; margin-top: 0;">市场状态 (Market Regime)</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 5px; width: 150px;"><strong>Regime ID</strong>:</td>
                <td style="padding: 5px;">{regime.id}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>标签</strong>:</td>
                <td style="padding: 5px;">{regime.label}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>市场特征</strong>:</td>
                <td style="padding: 5px; color: #d32f2f; font-weight: bold;">{regime.feature}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>起始日期</strong>:</td>
                <td style="padding: 5px;">{regime.start}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>结束日期</strong>:</td>
                <td style="padding: 5px;">{regime.end}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>开盘价</strong>:</td>
                <td style="padding: 5px;">{regime.open_price:,.2f}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>收盘价</strong>:</td>
                <td style="padding: 5px;">{regime.close_price:,.2f}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>涨跌幅</strong>:</td>
                <td style="padding: 5px; color: {'#d32f2f' if regime.pct_change < 0 else '#388e3c'}; font-weight: bold;">
                    {regime.pct_change:+.2f}%
                </td>
            </tr>
        </table>
    </div>
"""
    
    # 格式化数值
    def format_value(v: Any) -> str:
        if isinstance(v, (int, float)):
            if abs(v) >= 1e6:
                return f"{v/1e6:.2f}M"
            elif abs(v) >= 1e3:
                return f"{v/1e3:.2f}K"
            elif isinstance(v, float):
                return f"{v:.4f}"
            else:
                return str(v)
        return str(v)
    
    # 因子验证HTML
    factor_validity_html = ""
    if factor_validity:
        factor_validity_html = "<h2>因子验证预测能力</h2>\n"
        factor_validity_html += f"<p><strong>样本数</strong>: {factor_validity.get('sample_size', 'N/A'):,}</p>\n"
        coverage = factor_validity.get('coverage', 0)
        if isinstance(coverage, (int, float)):
            factor_validity_html += f"<p><strong>覆盖率</strong>: {coverage * 100:.2f}%</p>\n"
        
        ic = factor_validity.get("ic", {})
        if ic and ic.get('mean') is not None:
            factor_validity_html += "<h3>IC 指标（Information Coefficient）</h3>\n"
            factor_validity_html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>\n"
            factor_validity_html += "<tr><th>指标</th><th>数值</th></tr>\n"
            factor_validity_html += f"<tr><td>IC 均值</td><td style='text-align: right;'>{ic.get('mean', 0):.4f}</td></tr>\n"
            factor_validity_html += f"<tr><td>IC 标准差</td><td style='text-align: right;'>{ic.get('std', 0):.4f}</td></tr>\n"
            if ic.get('t_value') is not None:
                factor_validity_html += f"<tr><td>IC t 值</td><td style='text-align: right;'>{ic.get('t_value', 0):.4f}</td></tr>\n"
            if ic.get('icir') is not None:
                factor_validity_html += f"<tr><td><strong>IC IR</strong></td><td style='text-align: right;'><strong>{ic.get('icir', 0):.4f}</strong></td></tr>\n"
            factor_validity_html += "</table>\n"
            factor_validity_html += "<p><em>IC IR = IC均值 / IC标准差，衡量因子预测能力的稳定性</em></p>\n"
        
        rank_ic = factor_validity.get("rank_ic", {})
        if rank_ic and rank_ic.get('mean') is not None:
            factor_validity_html += "<h3>Rank IC 指标（Rank Information Coefficient）</h3>\n"
            factor_validity_html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>\n"
            factor_validity_html += "<tr><th>指标</th><th>数值</th></tr>\n"
            factor_validity_html += f"<tr><td>Rank IC 均值</td><td style='text-align: right;'>{rank_ic.get('mean', 0):.4f}</td></tr>\n"
            factor_validity_html += f"<tr><td>Rank IC 标准差</td><td style='text-align: right;'>{rank_ic.get('std', 0):.4f}</td></tr>\n"
            if rank_ic.get('t_value') is not None:
                factor_validity_html += f"<tr><td>Rank IC t 值</td><td style='text-align: right;'>{rank_ic.get('t_value', 0):.4f}</td></tr>\n"
            if rank_ic.get('icir') is not None:
                factor_validity_html += f"<tr><td><strong>Rank IC IR</strong></td><td style='text-align: right;'><strong>{rank_ic.get('icir', 0):.4f}</strong></td></tr>\n"
            factor_validity_html += "</table>\n"
            factor_validity_html += "<p><em>Rank IC IR = Rank IC均值 / Rank IC标准差，衡量因子排序能力的稳定性</em></p>\n"
        
        quantile_returns = factor_validity.get("quantile_returns", [])
        if quantile_returns:
            factor_validity_html += "<h3>分位数收益分析</h3>\n"
            factor_validity_html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>\n"
            factor_validity_html += "<tr><th>分位数</th><th>平均收益</th><th>样本数</th></tr>\n"
            for qr in quantile_returns:
                factor_validity_html += f"<tr><td>{qr.get('quantile', 'N/A')}</td><td style='text-align: right;'>{qr.get('avg_return', 0):.4f}</td><td style='text-align: right;'>{qr.get('count', 0)}</td></tr>\n"
            factor_validity_html += "</table>\n"
        
        turnover = factor_validity.get("top_quantile_turnover", {})
        if turnover:
            factor_validity_html += "<h3>Top 分位换手率</h3>\n"
            factor_validity_html += f"<p><strong>平均换手率</strong>: {turnover.get('average', 'N/A')}</p>\n"
            factor_validity_html += f"<p><strong>统计天数</strong>: {turnover.get('count', 0)}</p>\n"
    
    # 回测统计指标HTML
    backtest_stats_html = "<h2>回测统计指标</h2>\n"
    backtest_stats_html += f"<p><strong>起始日期</strong>: {stats.get('start_date', 'N/A')}</p>\n"
    backtest_stats_html += f"<p><strong>结束日期</strong>: {stats.get('end_date', 'N/A')}</p>\n"
    total_days = stats.get('total_days', 0)
    if isinstance(total_days, str):
        total_days = int(total_days)
    backtest_stats_html += f"<p><strong>总交易日</strong>: {total_days}</p>\n"
    profit_days = stats.get('profit_days', 0)
    if isinstance(profit_days, str):
        profit_days = int(profit_days)
    backtest_stats_html += f"<p><strong>盈利交易日</strong>: {profit_days}</p>\n"
    loss_days = stats.get('loss_days', 0)
    if isinstance(loss_days, str):
        loss_days = int(loss_days)
    backtest_stats_html += f"<p><strong>亏损交易日</strong>: {loss_days}</p>\n"
    backtest_stats_html += f"<p><strong>起始资金</strong>: {float(stats.get('capital', 0)):,.2f}</p>\n"
    backtest_stats_html += f"<p><strong>结束资金</strong>: {float(stats.get('end_balance', 0)):,.2f}</p>\n"
    backtest_stats_html += f"<p><strong>总收益率</strong>: {float(stats.get('total_return', 0)):.2f}%</p>\n"
    backtest_stats_html += f"<p><strong>年化收益</strong>: {float(stats.get('annual_return', 0)):.2f}%</p>\n"
    backtest_stats_html += f"<p><strong>最大回撤</strong>: {float(stats.get('max_drawdown', 0)):,.2f} ({float(stats.get('max_ddpercent', 0)):.2f}%)</p>\n"
    max_dd_duration = stats.get('max_drawdown_duration', 0)
    if isinstance(max_dd_duration, str):
        max_dd_duration = int(max_dd_duration)
    backtest_stats_html += f"<p><strong>最大回撤持续时间</strong>: {max_dd_duration} 天</p>\n"
    backtest_stats_html += f"<p><strong>买入持有回报</strong>: {float(stats.get('buy_hold_return', 0)):.2f}%</p>\n"
    backtest_stats_html += f"<p><strong>平均股权上涨</strong>: {float(stats.get('avg_equity_runup', 0)):,.2f} ({float(stats.get('avg_equity_runup_pct', 0)):.2f}%)</p>\n"
    backtest_stats_html += f"<p><strong>平均股权上涨持续时间</strong>: {float(stats.get('avg_equity_runup_duration', 0)):.1f} 天</p>\n"
    backtest_stats_html += f"<p><strong>最大股权上涨</strong>: {float(stats.get('max_equity_runup', 0)):,.2f} ({float(stats.get('max_equity_runup_pct', 0)):.2f}%)</p>\n"
    backtest_stats_html += f"<p><strong>平均股权回撤</strong>: {float(stats.get('avg_equity_drawdown', 0)):,.2f} ({float(stats.get('avg_equity_drawdown_pct', 0)):.2f}%)</p>\n"
    backtest_stats_html += f"<p><strong>平均股权回撤持续时间</strong>: {float(stats.get('avg_equity_drawdown_duration', 0)):.1f} 天</p>\n"
    backtest_stats_html += f"<p><strong>Sharpe Ratio</strong>: {float(stats.get('sharpe_ratio', 0)):.4f}</p>\n"
    backtest_stats_html += f"<p><strong>收益回撤比</strong>: {float(stats.get('return_drawdown_ratio', 0)):.4f}</p>\n"

    # 市场独立性（vs Benchmark）
    benchmark_symbol = stats.get("benchmark_symbol", None)
    if benchmark_symbol:
        backtest_stats_html += "<h3>市场独立性（vs Benchmark）</h3>\n"
        backtest_stats_html += f"<p><strong>Benchmark</strong>: {benchmark_symbol}</p>\n"

        beta = stats.get("beta_vs_benchmark", None)
        alpha_annual = stats.get("alpha_vs_benchmark_annual", None)
        corr = stats.get("corr_vs_benchmark", None)
        n_obs = stats.get("beta_observations", None)

        if isinstance(beta, (int, float)):
            backtest_stats_html += f"<p><strong>Beta</strong>: {float(beta):.4f}</p>\n"
        else:
            backtest_stats_html += "<p><strong>Beta</strong>: N/A</p>\n"

        if isinstance(alpha_annual, (int, float)):
            backtest_stats_html += f"<p><strong>Alpha (annualized)</strong>: {float(alpha_annual) * 100:.2f}%</p>\n"
        else:
            backtest_stats_html += "<p><strong>Alpha (annualized)</strong>: N/A</p>\n"

        if isinstance(corr, (int, float)):
            backtest_stats_html += f"<p><strong>Correlation</strong>: {float(corr):.4f}</p>\n"
        else:
            backtest_stats_html += "<p><strong>Correlation</strong>: N/A</p>\n"

        if isinstance(n_obs, int):
            backtest_stats_html += f"<p><strong>Observations</strong>: {n_obs}</p>\n"
    
    # 交易表现摘要HTML
    performance_html = ""
    if performance_summary:
        performance_html = "<h2>交易表现摘要</h2>\n"
        performance_html += f"<p><strong>总交易数</strong>: {performance_summary.get('total_trades', 'N/A')}</p>\n"
        performance_html += f"<p><strong>盈利交易</strong>: {performance_summary.get('winning_trades', 'N/A')} ({performance_summary.get('win_rate', 0) * 100:.2f}%)</p>\n"
        performance_html += f"<p><strong>总盈亏</strong>: {float(performance_summary.get('total_net_pnl', 0)):,.2f} ({float(performance_summary.get('total_net_pnl_pct', 0)):.2f}%)</p>\n"
        performance_html += f"<p><strong>毛利润</strong>: {float(performance_summary.get('gross_profit', 0)):,.2f}</p>\n"
        performance_html += f"<p><strong>毛亏损</strong>: {float(performance_summary.get('gross_loss', 0)):,.2f}</p>\n"
        performance_html += f"<p><strong>盈利因子</strong>: {float(performance_summary.get('profit_factor', 0)):.4f}</p>\n"
        performance_html += f"<p><strong>平均盈亏</strong>: {float(performance_summary.get('avg_net_pnl', 0)):,.2f} ({float(performance_summary.get('avg_net_pnl_pct', 0)):.2f}%)</p>\n"
        performance_html += f"<p><strong>平均盈利交易</strong>: {float(performance_summary.get('avg_winning_trade', 0)):,.2f}</p>\n"
        performance_html += f"<p><strong>平均亏损交易</strong>: {float(performance_summary.get('avg_losing_trade', 0)):,.2f}</p>\n"
        performance_html += f"<p><strong>最大盈利交易</strong>: {float(performance_summary.get('max_winning_trade', 0)):,.2f}</p>\n"
        performance_html += f"<p><strong>最大亏损交易</strong>: {float(performance_summary.get('max_losing_trade', 0)):,.2f}</p>\n"
        performance_html += f"<p><strong>最大持仓数量</strong>: {int(performance_summary.get('max_position_size', 0)):,}</p>\n"
        performance_html += f"<p><strong>最大合同持有量</strong>: {performance_summary.get('max_contracts_held', 'N/A')}</p>\n"
    
    # 完整的HTML报告
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Flagship Alpha-Momentum 综合回测报告</title>
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Ubuntu", sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #777;
            margin-top: 20px;
        }}
        table {{
            margin: 20px 0;
            border-collapse: collapse;
            width: 100%;
        }}
        th {{
            background-color: #f0f0f0;
            font-weight: bold;
            padding: 8px;
            text-align: left;
        }}
        td {{
            padding: 8px;
        }}
        .note {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Flagship Alpha-Momentum 综合回测报告</h1>
    <p><strong>报告生成时间</strong>: {report_date}</p>
    <p><strong>数据集</strong>: {dataset_name}</p>
    <p><strong>信号名称</strong>: {signal_name}</p>
    
    {regime_html}
    
    <div class="note">
        <strong>注意</strong>: 本报告中的价格数据为未调整价格（unadjusted prices），与Polygon API的 <code>adjusted=true</code> 参数返回的调整后价格可能不同。调整后价格会考虑拆股、分红等因素对历史价格的影响。
    </div>
    
    {factor_validity_html}
    
    {backtest_stats_html}
    
    {performance_html}
    
    <h2>回测图表</h2>
    {chart_html}
    
    {trade_list_preview_html}
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


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
        default="minute",
        help="K线周期（默认 minute，分钟线回测）",
    )
    parser.add_argument(
        "--rth-only",
        dest="rth_only",
        action="store_true",
        default=None,
        help="仅在 RTH（09:30-16:00 ET）回放并交易（minute 默认启用）",
    )
    parser.add_argument(
        "--no-rth-only",
        dest="rth_only",
        action="store_false",
        help="关闭 RTH-only（允许盘前/盘后分钟进入回测）",
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
    parser.add_argument(
        "--min-score-threshold",
        type=float,
        default=0.5,
        help="Score 选股阈值（默认 0.5）",
    )
    parser.add_argument(
        "--use-postgres-selection",
        action="store_true",
        default=True,
        help="使用PostgreSQL每日选股结果（默认启用）",
    )
    parser.add_argument(
        "--no-postgres-selection",
        dest="use_postgres_selection",
        action="store_false",
        help="不使用PostgreSQL每日选股结果",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        help="交易佣金费率 (例如 0.0001 表示万分之一)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["v5", "v7"],
        default="v5",
        help="策略版本 (默认 v5)",
    )
    args = parser.parse_args()

    interval = Interval.DAILY if args.interval == "daily" else Interval.MINUTE

    # 解析日期/时间
    start_in = parse_date_or_datetime(args.start)
    end_in = parse_date_or_datetime(args.end)

    # rth_only 默认行为：minute 启用，daily 关闭
    rth_only = args.rth_only
    if rth_only is None:
        rth_only = interval == Interval.MINUTE

    # minute+rth-only：若输入为 YYYY-MM-DD，自动扩展到 09:30–16:00
    if interval == Interval.MINUTE and rth_only and is_date_only_str(args.start):
        start = datetime.combine(start_in.date(), dtime(9, 30))
    else:
        start = start_in

    if interval == Interval.MINUTE and rth_only and is_date_only_str(args.end):
        end = datetime.combine(end_in.date(), dtime(16, 0))
    else:
        end = end_in

    # 如果信号名包含 v7 且 strategy 是默认值 v5，自动切换到 v7
    strategy_version = args.strategy
    if "v7" in args.signal_name.lower() and strategy_version == "v5":
        strategy_version = "v7"

    run_backtest(
        lab_path=args.lab_path,
        start=start,
        end=end,
        interval=interval,
        signal_name=args.signal_name,
        initial_capital=args.capital,
        min_score_threshold=args.min_score_threshold,
        use_postgres_selection=args.use_postgres_selection,
        commission_rate=args.commission_rate,
        rth_only=rth_only,
        strategy_version=strategy_version,
    )


if __name__ == "__main__":
    main()

