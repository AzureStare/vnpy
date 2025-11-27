"""验收分钟数据统计脚本"""
import polars as pl
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def check_minute_data(lab_path: str = "lab/flagship_alpha_momentum"):
    """统计分钟数据的详细信息"""
    minute_dir = Path(lab_path) / "minute"
    
    if not minute_dir.exists():
        print(f"错误：目录不存在 {minute_dir}")
        return
    
    # 获取所有 parquet 文件
    parquet_files = list(minute_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"错误：在 {minute_dir} 中未找到 parquet 文件")
        return
    
    print(f"找到 {len(parquet_files)} 个分钟数据文件\n")
    
    # 统计信息
    total_rows = 0
    total_files_with_data = 0
    symbol_date_ranges = {}
    all_dates = set()
    
    print("正在统计数据...")
    for i, file_path in enumerate(parquet_files):
        try:
            df = pl.read_parquet(file_path)
            row_count = len(df)
            if row_count == 0:
                continue
            
            total_files_with_data += 1
            total_rows += row_count
            
            # 获取标的符号（文件名）
            symbol = file_path.stem
            
            # 获取日期范围
            if "datetime" in df.columns:
                dates = df["datetime"].unique().sort()
                min_date = dates[0]
                max_date = dates[-1]
                
                symbol_date_ranges[symbol] = {
                    "min_date": min_date,
                    "max_date": max_date,
                    "row_count": row_count
                }
                # 收集所有日期
                for date in dates:
                    if isinstance(date, datetime):
                        all_dates.add(date.date())
                    elif hasattr(date, 'date'):
                        all_dates.add(date.date())
                    else:
                        all_dates.add(date)
            
            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i + 1}/{len(parquet_files)} 个文件...")
                
        except Exception as e:
            print(f"  警告：读取 {file_path.name} 时出错: {e}")
            continue
    
    print("\n" + "="*60)
    print("分钟数据统计结果")
    print("="*60)
    print(f"总文件数: {len(parquet_files)}")
    print(f"有数据的文件数: {total_files_with_data}")
    print(f"总分钟数据条数: {total_rows:,}")
    print(f"平均每个文件数据条数: {total_rows / total_files_with_data if total_files_with_data > 0 else 0:,.0f}")
    
    if symbol_date_ranges:
        # 统计日期范围
        all_min_dates = [info["min_date"] for info in symbol_date_ranges.values()]
        all_max_dates = [info["max_date"] for info in symbol_date_ranges.values()]
        
        if all_min_dates and all_max_dates:
            overall_min_date = min(all_min_dates)
            overall_max_date = max(all_max_dates)
            print(f"\n数据时间范围:")
            print(f"  最早日期: {overall_min_date}")
            print(f"  最晚日期: {overall_max_date}")
            
            # 统计唯一日期数量
            if all_dates:
                print(f"  唯一交易日数: {len(all_dates)}")
        
        # 统计每个标的的数据条数分布
        row_counts = [info["row_count"] for info in symbol_date_ranges.values()]
        if row_counts:
            print(f"\n数据条数分布:")
            print(f"  最小值: {min(row_counts):,}")
            print(f"  最大值: {max(row_counts):,}")
            print(f"  中位数: {sorted(row_counts)[len(row_counts)//2]:,}")
            
            # 统计数据条数最多的前10个标的
            sorted_symbols = sorted(symbol_date_ranges.items(), key=lambda x: x[1]["row_count"], reverse=True)
            print(f"\n数据条数最多的前10个标的:")
            for symbol, info in sorted_symbols[:10]:
                print(f"  {symbol}: {info['row_count']:,} 条 ({info['min_date']} 至 {info['max_date']})")
    
    print("="*60)

if __name__ == "__main__":
    check_minute_data()

