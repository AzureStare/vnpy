"""
检查数据文件中的时间戳时区情况。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

# 先计算项目根目录（从本文件位置向上 3 级：flagship/scripts/check_timezone.py -> flagship/scripts -> flagship -> vnpy）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 将项目根目录添加到 Python 路径
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("警告: polars 未安装，无法直接读取 parquet 文件")
    print("请安装 polars: pip install polars")

from vnpy.alpha import AlphaLab
from vnpy.trader.utility import ZoneInfo


def check_timezone(lab_path: str | Path) -> None:
    """检查数据文件中的时间戳时区"""
    if not POLARS_AVAILABLE:
        print("无法检查：polars 未安装")
        return
    
    lab = AlphaLab(str(lab_path))
    
    # 检查日线数据
    daily_files = list(lab.daily_path.glob("*.parquet"))
    if not daily_files:
        print(f"未找到日线数据文件在 {lab.daily_path}")
        return
    
    # 读取第一个文件作为示例
    sample_file = daily_files[0]
    print(f"\n检查文件: {sample_file.name}")
    print("=" * 60)
    
    df = pl.read_parquet(sample_file)
    
    # 美东时区用于对比
    eastern_tz = ZoneInfo("America/New_York")
    utc_tz = timezone.utc
    
    if "datetime" not in df.columns:
        print("数据文件中没有 datetime 列")
        return
    
    # 显示前几条数据
    print("\n前5条数据的时间戳:")
    print(df.head(5).select(["datetime"]))
    
    # 检查时间戳的类型
    datetime_col = df["datetime"]
    print(f"\n时间戳数据类型: {datetime_col.dtype}")
    
    # 检查是否有时区信息
    sample_dt = datetime_col[0]
    if isinstance(sample_dt, datetime):
        print(f"\n第一条时间戳: {sample_dt}")
        print(f"是否有 tzinfo: {sample_dt.tzinfo is not None}")
        if sample_dt.tzinfo:
            print(f"时区: {sample_dt.tzinfo}")
    
    # 检查时间范围
    min_dt = datetime_col.min()
    max_dt = datetime_col.max()
    print(f"\n时间范围:")
    print(f"  最早: {min_dt}")
    print(f"  最晚: {max_dt}")
    
    # 检查是否是交易日（美股交易日通常是周一到周五）
    # 如果数据是日线，应该都是交易日
    print(f"\n数据条数: {len(df)}")
    
    # 检查时间戳的小时部分（如果是日线，应该都是 00:00:00 或接近）
    if isinstance(min_dt, datetime):
        print(f"\n第一条时间戳的详细信息:")
        print(f"  日期: {min_dt.date()}")
        print(f"  时间: {min_dt.time()}")
        print(f"  星期: {min_dt.strftime('%A')}")
        
        # 检查是否是美东时间（日线数据应该在美东时间的 00:00:00）
        # 如果数据是 UTC，日线数据可能在 04:00:00 或 05:00:00（取决于夏令时）
        print(f"\n时区分析:")
        print(f"  如果数据是 UTC，日线时间戳应该在 04:00:00 或 05:00:00（美东时间 00:00:00）")
        print(f"  如果数据是美东时间，日线时间戳应该在 00:00:00")
        
        # 尝试判断：如果小时是 0，可能是美东时间；如果是 4 或 5，可能是 UTC
        hour = min_dt.hour
        if hour == 0:
            print(f"  ⚠️  当前数据时间戳小时为 {hour}，可能是美东时间（正确）")
        elif hour in [4, 5]:
            print(f"  ⚠️  当前数据时间戳小时为 {hour}，可能是 UTC 时间（需要转换为美东时间）")
        else:
            print(f"  ⚠️  当前数据时间戳小时为 {hour}，需要进一步检查")
    
    # 检查分钟线数据（如果存在）
    minute_files = list(lab.minute_path.glob("*.parquet"))
    if minute_files:
        print("\n" + "=" * 60)
        print(f"\n检查分钟线数据: {minute_files[0].name}")
        print("=" * 60)
        
        minute_df = pl.read_parquet(minute_files[0])
        if "datetime" in minute_df.columns:
            minute_datetime_col = minute_df["datetime"]
            print(f"\n前5条分钟线数据的时间戳:")
            print(minute_df.head(5).select(["datetime"]))
            
            minute_sample_dt = minute_datetime_col[0]
            if isinstance(minute_sample_dt, datetime):
                print(f"\n第一条分钟线时间戳: {minute_sample_dt}")
                print(f"是否有 tzinfo: {minute_sample_dt.tzinfo is not None}")
                if minute_sample_dt.tzinfo:
                    print(f"时区: {minute_sample_dt.tzinfo}")
                print(f"  日期: {minute_sample_dt.date()}")
                print(f"  时间: {minute_sample_dt.time()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查数据文件中的时间戳时区")
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab 数据目录路径",
    )
    args = parser.parse_args()
    
    check_timezone(args.lab_path)

