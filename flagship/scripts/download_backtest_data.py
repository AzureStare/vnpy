"""
基于每日股票池下载回测数据（日线/分钟线）。

从 Polygon API 下载筛选出的股票的历史行情数据，保存为 vnpy.alpha 框架可用的格式。
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import ZoneInfo
from vnpy.alpha.lab import AlphaLab

from flagship.config import (
    DEFAULT_LAB_DIR,
    DEFAULT_UNIVERSE_DIR,
    VT_SETTING_PATH,
    create_polygon_client,
)

try:
    from flagship.scripts.pg_ticker_db import get_ref_tickers
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

EXCHANGE_MAP = {
    "XNAS": Exchange.NASDAQ,
    "XNYS": Exchange.NYSE,
    "XASE": Exchange.AMEX,
    "ARCX": Exchange.NYSE,
    "BATS": Exchange.BATS,
    "IEXG": Exchange.IEX,
}


def load_daily_universe(universe_file: Path) -> list[str]:
    """
    从每日股票池文件加载股票列表。
    
    Returns:
        股票代码列表（不含交易所后缀）
    """
    if not universe_file.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_file}")

    data = json.loads(universe_file.read_text(encoding="utf-8"))
    symbols = [item["symbol"] for item in data.get("symbols", [])]
    logger.info(f"Loaded {len(symbols)} symbols from {universe_file}")
    return symbols


def get_exchange_for_symbol(symbol: str) -> Exchange:
    """
    根据股票代码从 Postgres 查询交易所。
    
    Returns:
        交易所枚举值，如果查询失败则默认返回 NASDAQ
    """
    if PG_AVAILABLE:
        try:
            ref_tickers = get_ref_tickers(
                market="stocks",
                locale="us",
                ticker_type="CS",
                active=True,
            )
            for ticker in ref_tickers:
                if ticker.get("symbol") == symbol:
                    primary_exchange = ticker.get("primary_exchange", "")
                    return EXCHANGE_MAP.get(primary_exchange, Exchange.NASDAQ)
        except Exception as exc:
            logger.warning(f"Failed to query exchange for {symbol}: {exc}")
    
    # 默认返回 NASDAQ
    return Exchange.NASDAQ


def download_bars_for_symbols(
    symbols: list[str],
    start_date: date,
    end_date: date,
    interval: Interval = Interval.DAILY,
    lab_dir: Path = DEFAULT_LAB_DIR,
) -> None:
    """
    下载指定股票列表的历史行情数据并保存到 AlphaLab。
    
    Args:
        symbols: 股票代码列表
        start_date: 起始日期
        end_date: 结束日期
        interval: K线周期（日线或分钟线）
        lab_dir: AlphaLab 数据目录
    """
    logger.info(
        f"[download_bars_for_symbols] 开始下载历史数据"
    )
    logger.info(
        f"[download_bars_for_symbols] 股票数量: {len(symbols)}, "
        f"日期范围: {start_date} 到 {end_date}, "
        f"周期: {interval.value}"
    )

    # 初始化 AlphaLab
    lab = AlphaLab(str(lab_dir))
    logger.debug(f"[download_bars_for_symbols] AlphaLab 初始化完成: {lab_dir}")

    # 创建 Polygon 客户端
    logger.info(f"[download_bars_for_symbols] 创建 Polygon RESTClient...")
    client = create_polygon_client()
    logger.debug(f"[download_bars_for_symbols] Polygon 客户端创建成功")

    success_count = 0
    fail_count = 0

    for idx, symbol in enumerate(symbols, start=1):
        try:
            logger.debug(f"[download_bars_for_symbols] [{idx}/{len(symbols)}] 下载 {symbol}...")
            
            exchange = get_exchange_for_symbol(symbol)
            vt_symbol = f"{symbol}.{exchange.value}"

            # 使用 Polygon RESTClient 下载数据
            if interval == Interval.DAILY:
                timespan = "day"
                multiplier = 1
            else:
                timespan = "minute"
                multiplier = 1

            # 需要多取一些历史数据用于因子计算（至少60天）
            # 但保存时只保存日期范围内的数据
            fetch_start = start_date - timedelta(days=90)  # 多取一些以应对非交易日
            
            logger.debug(f"[download_bars_for_symbols] {symbol}: 从 {fetch_start} 到 {end_date} 获取数据")
            
            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=fetch_start.isoformat(),
                to=end_date.isoformat(),
                adjusted=True,
                sort="asc",
                limit=50000,
            )

            if not aggs:
                logger.warning(f"[download_bars_for_symbols] {symbol}: 未获取到数据")
                fail_count += 1
                continue

            logger.debug(f"[download_bars_for_symbols] {symbol}: 获取到 {len(aggs)} 条原始数据")

            # 转换为 BarData 列表（保存所有数据，包括历史数据，用于因子计算）
            # Polygon API 返回 UTC 时间戳，需要转换为美东时间（EST/EDT）
            utc_tz = timezone.utc
            eastern_tz = ZoneInfo("America/New_York")
            
            bar_data_list = []
            for agg in aggs:
                # Polygon 返回的是 UTC 时间戳（毫秒），先转换为 UTC datetime
                utc_datetime = datetime.fromtimestamp(agg.timestamp / 1000, tz=utc_tz)
                # 转换为美东时间，然后移除时区信息（保存为 naive datetime）
                bar_datetime = utc_datetime.astimezone(eastern_tz).replace(tzinfo=None)
                # 保存所有数据（包括历史数据），用于后续因子计算
                
                raw_open = getattr(agg, "open", None)
                raw_high = getattr(agg, "high", None)
                raw_low = getattr(agg, "low", None)
                raw_close = getattr(agg, "close", None)
                open_val = float(raw_open) if raw_open is not None else 0.0
                high_val = float(raw_high) if raw_high is not None else 0.0
                low_val = float(raw_low) if raw_low is not None else 0.0
                close_val = float(raw_close) if raw_close is not None else 0.0

                raw_volume = getattr(agg, "volume", 0) or 0
                try:
                    volume_int = int(raw_volume)
                except Exception:
                    # 极端情况下 Polygon 返回异常类型，退化为 0
                    volume_int = 0

                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=bar_datetime,
                    interval=interval,
                    open_price=open_val,
                    high_price=high_val,
                    low_price=low_val,
                    close_price=close_val,
                    # NOTE:
                    # - parquet 里历史 volume 通常为 Int64（整型成交量）
                    # - Polygon SDK 有时会返回 float（例如 12345.0），直接拼接会触发 dtype 不匹配
                    volume=volume_int,
                    turnover=float(volume_int) * close_val,
                    open_interest=0,
                    gateway_name="POLYGON"
                )
                bar_data_list.append(bar)

            if not bar_data_list:
                logger.warning(f"[download_bars_for_symbols] {symbol}: 日期范围内无数据")
                fail_count += 1
                continue

            # 保存到 AlphaLab（interval 信息已在 BarData 中）
            logger.debug(f"[download_bars_for_symbols] {symbol}: 保存 {len(bar_data_list)} 条数据")
            lab.save_bar_data(bar_data_list)

            success_count += 1
            if idx % 50 == 0:
                logger.info(
                    f"[download_bars_for_symbols] 进度 {idx}/{len(symbols)}: "
                    f"成功={success_count}, 失败={fail_count}"
                )

        except Exception as exc:
            logger.error(f"[download_bars_for_symbols] {symbol} 下载失败: {exc}")
            fail_count += 1

    logger.info(
        f"[download_bars_for_symbols] 下载完成: "
        f"成功={success_count}, 失败={fail_count}, 总计={len(symbols)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download backtest data for symbols from daily universe files."
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        help="每日股票池 JSON 文件路径（如果指定，则只下载该文件的股票）",
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=DEFAULT_UNIVERSE_DIR,
        help="股票池目录（如果指定 --universe-dir，则下载目录下所有文件的股票）",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["daily", "minute"],
        default="daily",
        help="K线周期（默认 daily）",
    )
    parser.add_argument(
        "--lab-dir",
        type=Path,
        default=DEFAULT_LAB_DIR,
        help="AlphaLab 数据目录（默认 lab/flagship_alpha_momentum）",
    )
    args = parser.parse_args()

    # 重新加载 vt_setting.json（如果需要）
    if VT_SETTING_PATH.exists():
        try:
            setting_data = json.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
            SETTINGS.update(setting_data)
            logger.debug(f"[main] 已重新加载 vt_setting.json")
        except Exception as exc:
            logger.warning(f"[main] 重新加载 vt_setting.json 失败: {exc}")

    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()

    interval = Interval.DAILY if args.interval == "daily" else Interval.MINUTE

    # 收集所有需要下载的股票
    all_symbols = set()

    if args.universe_file:
        symbols = load_daily_universe(args.universe_file)
        all_symbols.update(symbols)
    elif args.universe_dir.exists():
        universe_files = sorted(args.universe_dir.glob("universe_*.json"))
        logger.info(f"Found {len(universe_files)} universe files")
        for universe_file in universe_files:
            symbols = load_daily_universe(universe_file)
            all_symbols.update(symbols)
    else:
        raise ValueError(
            "Either --universe-file or --universe-dir must be specified"
        )

    logger.info(f"Total unique symbols to download: {len(all_symbols)}")

    # 下载数据
    download_bars_for_symbols(
        list(all_symbols),
        start_date,
        end_date,
        interval=interval,
        lab_dir=args.lab_dir,
    )


if __name__ == "__main__":
    main()

