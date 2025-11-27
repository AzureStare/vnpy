"""监控回测日志文件，实时显示最新日志"""
import sys
import time
from pathlib import Path
from datetime import datetime

def monitor_log(log_file: str | Path, tail_lines: int = 50, refresh_interval: float = 1.0):
    """
    监控日志文件，实时显示最新内容
    
    Args:
        log_file: 日志文件路径
        tail_lines: 显示最后N行
        refresh_interval: 刷新间隔（秒）
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"错误：日志文件不存在: {log_path}")
        print("等待文件创建...")
        # 等待文件创建
        while not log_path.exists():
            time.sleep(1)
        print(f"文件已创建: {log_path}")
    
    print("=" * 80)
    print(f"监控回测日志: {log_path}")
    print(f"刷新间隔: {refresh_interval} 秒")
    print("按 Ctrl+C 退出监控")
    print("=" * 80)
    print()
    
    last_size = 0
    last_lines = []
    
    try:
        while True:
            if log_path.exists():
                current_size = log_path.stat().st_size
                
                # 如果文件有更新
                if current_size != last_size:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    # 只显示新增的行或最后N行
                    if len(lines) > len(last_lines):
                        # 有新内容，显示新增部分
                        new_lines = lines[len(last_lines):]
                        for line in new_lines:
                            print(line.rstrip())
                    elif len(lines) > 0:
                        # 文件可能被重新创建或清空，显示最后N行
                        display_lines = lines[-tail_lines:]
                        for line in display_lines:
                            print(line.rstrip())
                    
                    last_lines = lines
                    last_size = current_size
                else:
                    # 文件没有更新，显示时间戳
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"\r[{current_time}] 等待日志更新...", end='', flush=True)
            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] 等待日志文件创建...", end='', flush=True)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        if log_path.exists():
            print(f"\n日志文件位置: {log_path}")
            print(f"文件大小: {log_path.stat().st_size / 1024:.2f} KB")
            print(f"总行数: {len(last_lines)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="监控回测日志文件")
    parser.add_argument(
        "log_file",
        type=str,
        help="日志文件路径"
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="显示最后N行（默认50）"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="刷新间隔（秒，默认1.0）"
    )
    
    args = parser.parse_args()
    
    monitor_log(args.log_file, tail_lines=args.tail, refresh_interval=args.interval)

