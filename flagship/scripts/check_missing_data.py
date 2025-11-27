"""计算分钟数据缺失情况（多进程版本，优化内存）"""
import polars as pl
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import psutil

def process_single_file(file_path: Path):
    """处理单个文件，返回统计信息（优化内存版本）"""
    try:
        df = pl.read_parquet(file_path)
        row_count = len(df)
        if row_count == 0:
            return None
        
        symbol = file_path.stem
        result = {
            "symbol": symbol,
            "row_count": row_count,
            "min_date": None,
            "max_date": None,
            "unique_dates": 0,  # 只统计数量，不存储所有日期
            "total_minutes": 0  # 总分钟数
        }
        
        if "datetime" in df.columns:
            dates = df["datetime"].unique().sort()
            if len(dates) > 0:
                result["min_date"] = dates[0]
                result["max_date"] = dates[-1]
                
                # 只统计唯一日期数量和总分钟数，不存储具体数据
                unique_dates_set = set()
                for date_val in dates:
                    if isinstance(date_val, datetime):
                        date_key = date_val.date()
                    elif hasattr(date_val, 'date'):
                        date_key = date_val.date()
                    else:
                        continue
                    unique_dates_set.add(date_key)
                
                result["unique_dates"] = len(unique_dates_set)
                result["total_minutes"] = len(dates)
        
        # 释放内存
        del df
        return result
    except Exception:
        # 跳过错误文件
        return None

def process_files_chunk(file_paths_chunk):
    """处理一批文件"""
    results = []
    for file_path in file_paths_chunk:
        result = process_single_file(file_path)
        if result:
            results.append(result)
    return results

def get_optimal_process_count(cpu_usage_target=0.8, memory_limit_gb=40):
    """根据CPU和内存情况确定最优进程数"""
    cpu_count_val = cpu_count()
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)
    
    # CPU限制：使用80%的核心数
    cpu_based_procs = int(cpu_count_val * cpu_usage_target)
    
    # 内存限制：假设每个进程最多使用500MB内存
    # 估算：每个parquet文件平均约1-5MB，处理时可能膨胀到10-50MB
    # 保守估计每个进程需要100MB
    memory_based_procs = int(available_memory_gb * 1024 / 100)  # 每个进程100MB
    
    # 取两者较小值
    optimal_procs = min(cpu_based_procs, memory_based_procs, cpu_count_val - 1)
    optimal_procs = max(1, optimal_procs)  # 至少1个进程
    
    return optimal_procs, cpu_count_val, available_memory_gb

def calculate_missing_data(lab_path: str = "lab/flagship_alpha_momentum", num_processes: int = None):
    """计算缺失的数据量（多进程版本，优化内存）"""
    minute_dir = Path(lab_path) / "minute"
    
    if not minute_dir.exists():
        print(f"错误：目录不存在 {minute_dir}")
        return
    
    # 获取所有 parquet 文件（排除 Mac 隐藏文件）
    all_files = list(minute_dir.glob("*.parquet"))
    parquet_files = [f for f in all_files if not f.name.startswith("._")]
    
    if not parquet_files:
        print(f"错误：在 {minute_dir} 中未找到 parquet 文件")
        return
    
    print(f"找到 {len(parquet_files)} 个分钟数据文件（排除隐藏文件）")
    
    # 确定进程数（考虑CPU和内存）
    if num_processes is None:
        num_processes, cpu_total, mem_available = get_optimal_process_count()
        print(f"系统资源:")
        print(f"  CPU核心数: {cpu_total}")
        print(f"  可用内存: {mem_available:.2f} GB")
        print(f"  推荐进程数: {num_processes} (约 {num_processes/cpu_total*100:.1f}% CPU使用率)\n")
    else:
        cpu_total = cpu_count()
        mem_available = psutil.virtual_memory().available / (1024**3)
        print(f"系统资源:")
        print(f"  CPU核心数: {cpu_total}")
        print(f"  可用内存: {mem_available:.2f} GB")
        print(f"  使用进程数: {num_processes} (约 {num_processes/cpu_total*100:.1f}% CPU使用率)\n")
    
    # 将文件列表分块
    chunk_size = max(1, len(parquet_files) // num_processes)
    file_chunks = [parquet_files[i:i + chunk_size] for i in range(0, len(parquet_files), chunk_size)]
    print(f"文件分块: {len(file_chunks)} 个块，每块约 {chunk_size} 个文件\n")
    
    # 多进程处理（分批处理避免内存溢出）
    all_results = []
    print("开始处理数据...")
    with Pool(processes=num_processes) as pool:
        # 使用imap_unordered提高效率，并显示进度
        chunk_results = pool.imap_unordered(process_files_chunk, file_chunks)
        processed_chunks = 0
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
            processed_chunks += 1
            if processed_chunks % 5 == 0:
                memory = psutil.virtual_memory()
                print(f"  已处理 {processed_chunks}/{len(file_chunks)} 个块，"
                      f"当前内存使用: {memory.percent:.1f}%, "
                      f"已收集 {len(all_results)} 个有效文件结果")
    
    if not all_results:
        print("未找到有效数据")
        return
    
    print(f"\n处理完成！共收集到 {len(all_results)} 个有效文件结果\n")
    
    # 汇总统计（优化内存版本）
    total_rows = sum(r["row_count"] for r in all_results)
    total_files_with_data = len(all_results)
    all_dates = set()
    symbol_date_ranges = {}
    symbol_trading_days = {}  # 每个标的的交易日数
    
    for result in all_results:
        symbol = result["symbol"]
        symbol_date_ranges[symbol] = {
            "min_date": result["min_date"],
            "max_date": result["max_date"],
            "row_count": result["row_count"],
            "unique_dates": result["unique_dates"],
            "total_minutes": result["total_minutes"]
        }
        
        symbol_trading_days[symbol] = result["unique_dates"]
        
        # 收集所有日期（只收集日期，不收集具体分钟）
        if result["min_date"] and result["max_date"]:
            if isinstance(result["min_date"], datetime):
                all_dates.add(result["min_date"].date())
            if isinstance(result["max_date"], datetime):
                all_dates.add(result["max_date"].date())
    
    print("="*60)
    print("分钟数据缺失情况分析")
    print("="*60)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  总文件数: {len(parquet_files)}")
    print(f"  有数据的文件数: {total_files_with_data}")
    print(f"  总分钟数据条数: {total_rows:,}")
    
    if symbol_date_ranges:
        # 统计日期范围
        all_min_dates = [info["min_date"] for info in symbol_date_ranges.values() if info["min_date"]]
        all_max_dates = [info["max_date"] for info in symbol_date_ranges.values() if info["max_date"]]
        
        if all_min_dates and all_max_dates:
            overall_min_date = min(all_min_dates)
            overall_max_date = max(all_max_dates)
            
            print(f"\n数据时间范围:")
            print(f"  最早日期: {overall_min_date}")
            print(f"  最晚日期: {overall_max_date}")
            print(f"  唯一交易日数: {len(all_dates)}")
            
            # 统计实际每天的平均分钟数（基于每个标的的平均值）
            total_minutes_list = [r["total_minutes"] for r in all_results if r["total_minutes"] > 0]
            trading_days_list = [r["unique_dates"] for r in all_results if r["unique_dates"] > 0]
            
            if total_minutes_list and trading_days_list:
                # 计算每个标的平均每天分钟数
                minutes_per_day_per_symbol = []
                for result in all_results:
                    if result["unique_dates"] > 0:
                        avg = result["total_minutes"] / result["unique_dates"]
                        minutes_per_day_per_symbol.append(avg)
                
                if minutes_per_day_per_symbol:
                    avg_minutes_per_day = sum(minutes_per_day_per_symbol) / len(minutes_per_day_per_symbol)
                    max_minutes_per_day = max(minutes_per_day_per_symbol)
                    min_minutes_per_day = min(minutes_per_day_per_symbol)
                    
                    print(f"\n实际数据统计:")
                    print(f"  平均每天分钟数: {avg_minutes_per_day:.1f}")
                    print(f"  最多分钟数/天: {max_minutes_per_day:.1f}")
                    print(f"  最少分钟数/天: {min_minutes_per_day:.1f}")
                    
                    # 计算理论数据量
                    # 使用最大值作为参考（最完整的标的）
                    expected_minutes_per_day = int(max_minutes_per_day)
                    expected_total_minutes = len(all_dates) * total_files_with_data * expected_minutes_per_day
                    actual_total_minutes = total_rows
                    missing_minutes = expected_total_minutes - actual_total_minutes
                    missing_percentage = (missing_minutes / expected_total_minutes * 100) if expected_total_minutes > 0 else 0
                    
                    print(f"\n理论数据量估算:")
                    print(f"  交易日数: {len(all_dates)}")
                    print(f"  标的数量: {total_files_with_data}")
                    print(f"  理论每分钟数/标的/天: {expected_minutes_per_day}")
                    print(f"  理论总分钟数: {expected_total_minutes:,}")
                    print(f"  实际总分钟数: {actual_total_minutes:,}")
                    print(f"  缺失分钟数: {missing_minutes:,}")
                    print(f"  缺失比例: {missing_percentage:.2f}%")
                    
                    # 按标的统计缺失情况
                    symbol_missing_stats = []
                    for symbol, info in symbol_date_ranges.items():
                        trading_days = symbol_trading_days.get(symbol, info.get("unique_dates", 0))
                        
                        if trading_days > 0:
                            expected_symbol_minutes = trading_days * expected_minutes_per_day
                            actual_symbol_minutes = info["row_count"]
                            missing_symbol_minutes = expected_symbol_minutes - actual_symbol_minutes
                            missing_symbol_pct = (missing_symbol_minutes / expected_symbol_minutes * 100) if expected_symbol_minutes > 0 else 0
                            
                            symbol_missing_stats.append({
                                "symbol": symbol,
                                "trading_days": trading_days,
                                "expected": expected_symbol_minutes,
                                "actual": actual_symbol_minutes,
                                "missing": missing_symbol_minutes,
                                "missing_pct": missing_symbol_pct
                            })
                    
                    # 排序并显示缺失最多的前10个标的
                    symbol_missing_stats.sort(key=lambda x: x["missing"], reverse=True)
                    print(f"\n缺失数据最多的前10个标的:")
                    for stat in symbol_missing_stats[:10]:
                        print(f"  {stat['symbol']}: 缺失 {stat['missing']:,} 条 ({stat['missing_pct']:.2f}%), "
                              f"交易日数: {stat['trading_days']}, 实际: {stat['actual']:,}, 理论: {stat['expected']:,}")
                    
                    # 统计完全缺失的标的
                    missing_symbols = len(parquet_files) - total_files_with_data
                    print(f"\n完全缺失数据的标的数: {missing_symbols}")
                    print(f"  有数据: {total_files_with_data}/{len(parquet_files)} ({total_files_with_data/len(parquet_files)*100:.2f}%)")
                    print(f"  无数据: {missing_symbols}/{len(parquet_files)} ({missing_symbols/len(parquet_files)*100:.2f}%)")
    
    print("="*60)

if __name__ == "__main__":
    # 自动计算最优进程数（80% CPU使用率，考虑内存限制）
    calculate_missing_data()

