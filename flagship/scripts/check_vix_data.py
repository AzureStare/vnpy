"""检查 VIX 和 VIX3M 日线数据"""
import polars as pl
from pathlib import Path

daily_dir = Path('lab/flagship_alpha_momentum/daily')

print("="*60)
print("检查 VIX 和 VIX3M 日线数据")
print("="*60)

# 查找 VIX 相关文件
vix_files = list(daily_dir.glob('VIX*.parquet'))
print(f"\n找到的 VIX 相关文件: {len(vix_files)} 个")
for f in vix_files:
    print(f"  {f.name}")

# 检查 VIX.CBOE
vix_file = daily_dir / "VIX.CBOE.parquet"
print(f"\n{'='*60}")
print("VIX.CBOE 数据:")
print(f"  文件存在: {vix_file.exists()}")

if vix_file.exists():
    df = pl.read_parquet(vix_file)
    print(f"  总条数: {len(df):,}")
    print(f"  日期范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    print(f"  唯一日期数: {df['datetime'].dt.date().n_unique()}")
    print(f"\n  前5条数据:")
    print(df.head(5))
    print(f"\n  后5条数据:")
    print(df.tail(5))
else:
    print("  ⚠️  文件不存在！")

# 检查 VIX3M.CBOE
vix3m_file = daily_dir / "VIX3M.CBOE.parquet"
print(f"\n{'='*60}")
print("VIX3M.CBOE 数据:")
print(f"  文件存在: {vix3m_file.exists()}")

if vix3m_file.exists():
    df3m = pl.read_parquet(vix3m_file)
    print(f"  总条数: {len(df3m):,}")
    print(f"  日期范围: {df3m['datetime'].min()} 至 {df3m['datetime'].max()}")
    print(f"  唯一日期数: {df3m['datetime'].dt.date().n_unique()}")
    print(f"\n  前5条数据:")
    print(df3m.head(5))
    print(f"\n  后5条数据:")
    print(df3m.tail(5))
else:
    print("  ⚠️  文件不存在！")

# 如果两个文件都存在，计算比率
if vix_file.exists() and vix3m_file.exists():
    print(f"\n{'='*60}")
    print("VIX/VIX3M 比率分析:")
    df_vix = pl.read_parquet(vix_file)
    df_vix3m = pl.read_parquet(vix3m_file)
    
    # 合并数据
    merged = df_vix.join(
        df_vix3m,
        on="datetime",
        how="inner",
        suffix="_vix3m"
    ).with_columns(
        (pl.col("close") / pl.col("close_vix3m")).alias("vix_ratio")
    )
    
    print(f"  合并后数据条数: {len(merged):,}")
    print(f"  日期范围: {merged['datetime'].min()} 至 {merged['datetime'].max()}")
    print(f"\n  VIX比率统计:")
    print(f"    平均: {merged['vix_ratio'].mean():.4f}")
    print(f"    最小: {merged['vix_ratio'].min():.4f}")
    print(f"    最大: {merged['vix_ratio'].max():.4f}")
    print(f"    中位数: {merged['vix_ratio'].median():.4f}")
    
    # 统计不同杠杆区间的天数
    ratio_1 = merged.filter(pl.col("vix_ratio") <= 1.0)
    ratio_1_1_1 = merged.filter((pl.col("vix_ratio") > 1.0) & (pl.col("vix_ratio") <= 1.1))
    ratio_above_1_1 = merged.filter(pl.col("vix_ratio") > 1.1)
    
    print(f"\n  杠杆区间统计:")
    print(f"    Ratio <= 1.0 (杠杆1.0): {len(ratio_1):,} 天 ({len(ratio_1)/len(merged)*100:.1f}%)")
    print(f"    1.0 < Ratio <= 1.1 (杠杆0.5): {len(ratio_1_1_1):,} 天 ({len(ratio_1_1_1)/len(merged)*100:.1f}%)")
    print(f"    Ratio > 1.1 (杠杆0.3): {len(ratio_above_1_1):,} 天 ({len(ratio_above_1_1)/len(merged)*100:.1f}%)")
    
    print(f"\n  最近10天的 VIX比率:")
    recent = merged.sort("datetime", descending=True).head(10)
    for row in recent.iter_rows(named=True):
        print(f"    {row['datetime'].date()}: VIX={row['close']:.2f}, VIX3M={row['close_vix3m']:.2f}, Ratio={row['vix_ratio']:.4f}")

print("\n" + "="*60)

