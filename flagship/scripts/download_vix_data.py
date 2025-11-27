"""
下载 VIX 和 VIX3M 指数数据并导入到 AlphaLab。

策略需要 VIX 期限结构比率（VIX/VIX3M）来调整杠杆，因此需要下载这两个指数的日线数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Manually calculate PROJECT_ROOT before importing flagship.config
# This script is in flagship/scripts/, so PROJECT_ROOT is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import ZoneInfo
from vnpy.alpha.lab import AlphaLab

from flagship.config import (
    DEFAULT_LAB_DIR,
    VT_SETTING_PATH,
    create_polygon_client,
)

# VIX 相关指数的 Polygon ticker 符号
# Polygon 指数数据使用 "I:" 前缀
VIX_INDICES = {
    "VIX": {
        "ticker": "I:VIX",  # Polygon 指数 ticker 格式
        "vt_symbol": "VIX.CBOE",
        "exchange": Exchange.CBOE,
        "name": "VIX 波动率指数",
    },
    "VIX3M": {
        "ticker": "I:VIX3M",  # VIX 3个月期货指数
        "vt_symbol": "VIX3M.CBOE",
        "exchange": Exchange.CBOE,
        "name": "VIX 3个月期货指数",
    },
}


def download_vix_indices(
    start_date: date,
    end_date: date,
    lab_dir: Path = DEFAULT_LAB_DIR,
) -> None:
    """
    下载 VIX 和 VIX3M 指数数据并保存到 AlphaLab。
    
    Args:
        start_date: 起始日期
        end_date: 结束日期
        lab_dir: AlphaLab 数据目录
    """
    logger.info(
        f"[download_vix_indices] 开始下载 VIX 指数数据"
    )
    logger.info(
        f"[download_vix_indices] 日期范围: {start_date} 到 {end_date}"
    )

    # 初始化 AlphaLab
    lab = AlphaLab(str(lab_dir))
    logger.debug(f"[download_vix_indices] AlphaLab 初始化完成: {lab_dir}")

    # 创建 Polygon 客户端
    logger.info(f"[download_vix_indices] 创建 Polygon RESTClient...")
    client = create_polygon_client()
    logger.debug(f"[download_vix_indices] Polygon 客户端创建成功")

    success_count = 0
    fail_count = 0

    for idx_name, idx_config in VIX_INDICES.items():
        try:
            ticker = idx_config["ticker"]
            vt_symbol = idx_config["vt_symbol"]
            exchange = idx_config["exchange"]
            name = idx_config["name"]
            
            logger.info(f"[download_vix_indices] 下载 {name} ({ticker})...")
            
            # 多取一些历史数据以应对非交易日
            fetch_start = start_date - timedelta(days=30)
            
            logger.debug(f"[download_vix_indices] {ticker}: 从 {fetch_start} 到 {end_date} 获取数据")
            
            # 使用 Polygon API 下载指数聚合数据
            # 注意：指数数据可能不支持 adjusted 参数
            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=fetch_start.isoformat(),
                    to=end_date.isoformat(),
                    adjusted=False,  # 指数数据通常不需要调整
                    sort="asc",
                    limit=50000,
                )
            except Exception as exc:
                logger.warning(f"[download_vix_indices] {ticker}: Polygon API 调用失败: {exc}")
                # 尝试使用不同的 ticker 格式
                if ticker.startswith("I:"):
                    alt_ticker = ticker[2:]  # 移除 "I:" 前缀
                    logger.info(f"[download_vix_indices] 尝试使用替代 ticker: {alt_ticker}")
                    try:
                        aggs = client.get_aggs(
                            ticker=alt_ticker,
                            multiplier=1,
                            timespan="day",
                            from_=fetch_start.isoformat(),
                            to=end_date.isoformat(),
                            adjusted=False,
                            sort="asc",
                            limit=50000,
                        )
                    except Exception as exc2:
                        logger.error(f"[download_vix_indices] {alt_ticker}: 也失败: {exc2}")
                        fail_count += 1
                        continue
                else:
                    fail_count += 1
                    continue

            if not aggs:
                logger.warning(f"[download_vix_indices] {ticker}: 未获取到数据")
                fail_count += 1
                continue

            logger.debug(f"[download_vix_indices] {ticker}: 获取到 {len(aggs)} 条原始数据")

            # 转换为 BarData 列表
            # Polygon API 返回 UTC 时间戳，需要转换为美东时间（EST/EDT）
            utc_tz = timezone.utc
            eastern_tz = ZoneInfo("America/New_York")
            
            bar_data_list = []
            for agg in aggs:
                # Polygon 返回的是 UTC 时间戳（毫秒），先转换为 UTC datetime
                utc_datetime = datetime.fromtimestamp(agg.timestamp / 1000, tz=utc_tz)
                # 转换为美东时间，然后移除时区信息（保存为 naive datetime）
                bar_datetime = utc_datetime.astimezone(eastern_tz).replace(tzinfo=None)
                
                # 过滤到指定日期范围内
                bar_date = bar_datetime.date()
                if bar_date < start_date or bar_date > end_date:
                    continue
                
                bar = BarData(
                    symbol=idx_name,  # 使用简化的 symbol（VIX 或 VIX3M）
                    exchange=exchange,
                    datetime=bar_datetime,
                    interval=Interval.DAILY,  # VIX 数据只使用日线
                    open_price=agg.open,
                    high_price=agg.high,
                    low_price=agg.low,
                    close_price=agg.close,
                    volume=agg.volume if hasattr(agg, 'volume') and agg.volume else 0,
                    turnover=0.0,  # 指数没有成交额
                    open_interest=0,
                    gateway_name="POLYGON"
                )
                bar_data_list.append(bar)

            if not bar_data_list:
                logger.warning(f"[download_vix_indices] {ticker}: 日期范围内无数据")
                fail_count += 1
                continue

            # 保存到 AlphaLab
            logger.info(f"[download_vix_indices] {ticker}: 保存 {len(bar_data_list)} 条数据到 {vt_symbol}")
            lab.save_bar_data(bar_data_list)

            success_count += 1
            logger.info(
                f"[download_vix_indices] {name} ({vt_symbol}) 下载完成: "
                f"{len(bar_data_list)} 条数据，日期范围: "
                f"{bar_data_list[0].datetime.date()} 到 {bar_data_list[-1].datetime.date()}"
            )

        except Exception as exc:
            logger.error(f"[download_vix_indices] {idx_name} 下载失败: {exc}", exc_info=True)
            fail_count += 1

    logger.info(
        f"[download_vix_indices] 下载完成: "
        f"成功={success_count}, 失败={fail_count}, 总计={len(VIX_INDICES)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download VIX and VIX3M index data from Polygon API."
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

    # 下载 VIX 指数数据
    download_vix_indices(
        start_date,
        end_date,
        lab_dir=args.lab_dir,
    )


if __name__ == "__main__":
    main()

