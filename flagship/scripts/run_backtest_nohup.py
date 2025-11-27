"""使用 nohup 方式启动回测并输出日志"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_backtest_nohup(
    lab_path: str = "lab/flagship_alpha_momentum",
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    interval: str = "minute",
    capital: int = 1000000,
):
    """启动回测任务并输出到日志文件"""
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "backtest_report"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"backtest_{timestamp}.log"
    
    print("=" * 60)
    print("启动回测任务")
    print("=" * 60)
    print(f"Lab路径: {lab_path}")
    print(f"日期范围: {start} 到 {end}")
    print(f"K线周期: {interval}")
    print(f"初始资金: {capital:,}")
    print(f"日志文件: {log_file}")
    print("=" * 60)
    print()
    
    # Python 可执行文件路径
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    backtest_script = project_root / "flagship" / "backtest" / "flagship_alpha_momentum_backtest.py"
    
    # 构建命令
    cmd = [
        str(venv_python),
        str(backtest_script),
        "--lab-path", lab_path,
        "--start", start,
        "--end", end,
        "--interval", interval,
        "--capital", str(capital),
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # 启动进程（后台运行）
    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(project_root),
        )
    
    print(f"回测任务已启动")
    print(f"进程ID: {process.pid}")
    print(f"日志文件: {log_file}")
    print()
    print("使用以下命令监控日志:")
    print(f'  Get-Content "{log_file}" -Wait -Tail 50')
    print()
    print("或使用监控脚本:")
    print(f'  python flagship/scripts/monitor_backtest_log.py "{log_file}"')
    print()
    
    # 保存进程信息
    process_info_file = log_dir / "backtest_process_info.txt"
    import json
    process_info = {
        "process_id": process.pid,
        "log_file": str(log_file),
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "lab_path": lab_path,
            "start": start,
            "end": end,
            "interval": interval,
            "capital": capital,
        }
    }
    with open(process_info_file, "w", encoding="utf-8") as f:
        json.dump(process_info, f, indent=2, ensure_ascii=False)
    
    print(f"进程信息已保存到: {process_info_file}")
    return process.pid, log_file

if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="启动回测任务（nohup方式）")
    parser.add_argument("--lab-path", type=str, default="lab/flagship_alpha_momentum")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--interval", type=str, default="minute", choices=["daily", "minute"])
    parser.add_argument("--capital", type=int, default=1000000)
    
    args = parser.parse_args()
    
    pid, log_file = run_backtest_nohup(
        lab_path=args.lab_path,
        start=args.start,
        end=args.end,
        interval=args.interval,
        capital=args.capital,
    )
    
    print(f"\n回测任务已在后台运行，PID: {pid}")
    print(f"查看日志: Get-Content \"{log_file}\" -Wait -Tail 50")

