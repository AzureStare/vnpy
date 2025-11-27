"""
从 S3 flatfiles (CSV.gz) 导入数据到 AlphaLab，并在导入时转换为美东时间。

S3 flatfiles 中的时间戳是 UTC 时间（纳秒级），需要转换为美东时间（EST/EDT）。
"""

from __future__ import annotations

import argparse
import gzip
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# 先计算项目根目录（从本文件位置向上 3 级：flagship/scripts/import_s3_to_lab.py -> flagship/scripts -> flagship -> vnpy）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 将项目根目录添加到 Python 路径
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData
from vnpy.trader.utility import ZoneInfo
from vnpy.alpha.lab import AlphaLab

from flagship.config import DEFAULT_LAB_DIR

EXCHANGE_MAP = {
    "XNAS": Exchange.NASDAQ,
    "XNYS": Exchange.NYSE,
    "XASE": Exchange.AMEX,
    "ARCX": Exchange.NYSE,
    "BATS": Exchange.BATS,
    "IEXG": Exchange.IEX,
}


def get_exchange_for_symbol(symbol: str) -> Exchange:
    """
    根据股票代码推断交易所。
    
    默认返回 NASDAQ，可以根据需要扩展逻辑。
    """
    # 可以根据需要添加更多逻辑，例如从数据库查询
    return Exchange.NASDAQ


def parse_s3_csv_file(
    csv_file: Path,
    interval: Interval,
    exchange_map: dict[str, Exchange] | None = None,
) -> list[BarData]:
    """
    解析 S3 flatfile CSV 文件并转换为 BarData 列表。
    
    Args:
        csv_file: CSV.gz 文件路径
        exchange_map: 股票代码到交易所的映射（可选）
    
    Returns:
        BarData 列表，时间戳已转换为美东时间
    """
    if exchange_map is None:
        exchange_map = {}
    
    # 时区转换设置
    utc_tz = timezone.utc
    eastern_tz = ZoneInfo("America/New_York")
    
    bars: list[BarData] = []
    
    try:
        # 读取 CSV.gz 文件
        with gzip.open(csv_file, "rt") as f:
            df = pl.read_csv(f)
        
        if df.is_empty():
            logger.warning(f"文件 {csv_file} 为空")
            return bars
        
        # 检查必需的列
        required_cols = {"ticker", "volume", "open", "close", "high", "low", "window_start"}
        missing_cols = required_cols.difference(set(df.columns))
        if missing_cols:
            logger.error(f"文件 {csv_file} 缺少必需的列: {missing_cols}")
            return bars
        
        # 处理每一行
        for row in df.iter_rows(named=True):
            symbol = row["ticker"]
            exchange = exchange_map.get(symbol, get_exchange_for_symbol(symbol))
            
            # 时间戳是纳秒级，转换为秒
            timestamp_ns = row["window_start"]
            if isinstance(timestamp_ns, str):
                timestamp_ns = int(timestamp_ns)
            timestamp_s = timestamp_ns / 1e9
            
            # 转换为 UTC datetime
            utc_datetime = datetime.fromtimestamp(timestamp_s, tz=utc_tz)
            # 转换为美东时间，然后移除时区信息
            eastern_datetime = utc_datetime.astimezone(eastern_tz).replace(tzinfo=None)
            
            # 创建 BarData
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=eastern_datetime,
                interval=interval,  # 使用传入的 interval 参数
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=int(row["volume"]),
                turnover=float(row["volume"]) * float(row["close"]),
                open_interest=0,
                gateway_name="S3_FLATFILE"
            )
            bars.append(bar)
        
        logger.debug(f"从 {csv_file} 解析了 {len(bars)} 条数据")
        
    except Exception as exc:
        logger.error(f"解析文件 {csv_file} 失败: {exc}")
    
    return bars


def import_s3_to_lab(
    s3_dir: Path,
    lab_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    interval: Interval = Interval.DAILY,
) -> None:
    """
    从 S3 flatfiles 目录导入数据到 AlphaLab。
    
    Args:
        s3_dir: S3 flatfiles 根目录（例如 flagship/data/s3_downloads/bars/day）
        lab_dir: AlphaLab 数据目录
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        interval: K线周期（默认 DAILY,可选 MINUTE）
    """
    logger.info(f"[import_s3_to_lab] 开始导入 S3 数据")
    logger.info(f"[import_s3_to_lab] S3 目录: {s3_dir}")
    logger.info(f"[import_s3_to_lab] Lab 目录: {lab_dir}")
    logger.info(f"[import_s3_to_lab] K线周期: {interval.value}")
    
    if not s3_dir.exists():
        raise FileNotFoundError(f"S3 目录不存在: {s3_dir}")
    
    # 初始化 AlphaLab
    lab = AlphaLab(str(lab_dir))
    logger.debug(f"[import_s3_to_lab] AlphaLab 初始化完成")
    
    # 根据 interval 确定子目录
    if interval == Interval.DAILY:
        subdir = "day"
    elif interval == Interval.MINUTE:
        subdir = "minute"
    else:
        raise ValueError(f"不支持的 interval: {interval.value}")
    
    s3_bars_dir = s3_dir / subdir
    if not s3_bars_dir.exists():
        raise FileNotFoundError(f"S3 bars 目录不存在: {s3_bars_dir}")
    
    # 收集所有 CSV.gz 文件
    csv_files = sorted(s3_bars_dir.rglob("*.csv.gz"))
    logger.info(f"[import_s3_to_lab] 找到 {len(csv_files)} 个 CSV.gz 文件")
    
    if not csv_files:
        logger.warning(f"[import_s3_to_lab] 未找到 CSV.gz 文件")
        return
    
    # 按股票代码分组处理，分批保存以避免内存溢出
    symbol_bars: dict[str, list[BarData]] = {}
    
    total_files = len(csv_files)
    processed_files = 0
    saved_symbols: set[str] = set()  # 记录已保存的股票
    
    # 每处理多少文件后保存一次（分钟数据更频繁保存）
    save_batch_size = 10 if interval == Interval.MINUTE else 50
    
    def save_batch() -> tuple[int, int]:
        """保存当前批次的数据"""
        if not symbol_bars:
            return 0, 0
        
        success_count = 0
        fail_count = 0
        
        for vt_symbol, bars in symbol_bars.items():
            try:
                # 按时间排序
                bars.sort(key=lambda b: b.datetime)
                
                # 保存到 AlphaLab
                lab.save_bar_data(bars)
                saved_symbols.add(vt_symbol)
                success_count += 1
            except Exception as exc:
                logger.error(f"[import_s3_to_lab] 保存 {vt_symbol} 失败: {exc}")
                fail_count += 1
        
        # 清空已保存的数据
        symbol_bars.clear()
        return success_count, fail_count
    
    for csv_file in csv_files:
        # 检查日期范围（如果指定）
        if start_date or end_date:
            # 从文件名提取日期（例如 2023-01-09.csv.gz）
            try:
                file_date_str = csv_file.stem.replace(".csv", "")
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
                
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
            except ValueError:
                logger.warning(f"无法从文件名解析日期: {csv_file.name}")
                continue
        
        # 解析文件
        bars = parse_s3_csv_file(csv_file, interval)
        
        # 按股票代码分组
        for bar in bars:
            vt_symbol = bar.vt_symbol
            if vt_symbol not in symbol_bars:
                symbol_bars[vt_symbol] = []
            symbol_bars[vt_symbol].append(bar)
        
        processed_files += 1
        
        # 每处理一定数量的文件后保存一次
        if processed_files % save_batch_size == 0:
            logger.info(
                f"[import_s3_to_lab] 进度: {processed_files}/{total_files} "
                f"文件已处理, 当前内存中 {len(symbol_bars)} 个股票"
            )
            success, fail = save_batch()
            logger.info(
                f"[import_s3_to_lab] 批次保存完成: 成功={success}, 失败={fail}, "
                f"累计已保存 {len(saved_symbols)} 个股票"
            )
    
    logger.info(f"[import_s3_to_lab] 文件处理完成: {processed_files}/{total_files}")
    
    # 保存剩余的数据
    if symbol_bars:
        logger.info(f"[import_s3_to_lab] 保存剩余 {len(symbol_bars)} 个股票的数据...")
        success, fail = save_batch()
        logger.info(
            f"[import_s3_to_lab] 最终批次保存: 成功={success}, 失败={fail}"
        )
    
    logger.info(
        f"[import_s3_to_lab] 导入完成: 总计已保存 {len(saved_symbols)} 个股票的数据"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 S3 flatfiles 导入数据到 AlphaLab，并转换为美东时间"
    )
    parser.add_argument(
        "--s3-dir",
        type=Path,
        required=True,
        help="S3 flatfiles 根目录（例如 flagship/data/s3_downloads/bars）",
    )
    parser.add_argument(
        "--lab-dir",
        type=Path,
        default=DEFAULT_LAB_DIR,
        help="AlphaLab 数据目录（默认 lab/flagship_alpha_momentum）",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="起始日期 (YYYY-MM-DD)，可选",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="结束日期 (YYYY-MM-DD)，可选",
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["daily", "minute"],
        default="daily",
        help="K线周期（默认 daily）",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="日志文件路径（可选，默认使用 vnpy logger 配置）",
    )
    args = parser.parse_args()
    
    # 如果指定了日志文件，配置 logger 输出到文件（loguru 使用 add 方法）
    if args.log_file:
        from loguru import logger as loguru_logger
        
        # loguru 使用 add 方法添加文件输出
        loguru_logger.add(
            args.log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            encoding="utf-8",
            rotation="100 MB",  # 当日志文件超过 100MB 时轮转
            retention="7 days",  # 保留 7 天的日志
        )
        logger.info(f"日志将输出到文件: {args.log_file}")
    
    start_date = None
    if args.start:
        start_date = datetime.fromisoformat(args.start).date()
    
    end_date = None
    if args.end:
        end_date = datetime.fromisoformat(args.end).date()
    
    interval = Interval.DAILY if args.interval == "daily" else Interval.MINUTE
    
    import_s3_to_lab(
        s3_dir=args.s3_dir,
        lab_dir=args.lab_dir,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )


if __name__ == "__main__":
    main()

