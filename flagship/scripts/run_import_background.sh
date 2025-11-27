#!/bin/bash
# 后台运行数据导入脚本的辅助脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# 日志文件路径（使用时间戳）
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/import_s3_${TIMESTAMP}.log"

# 参数
S3_DIR="${1:-flagship/data/s3_downloads/bars}"
LAB_DIR="${2:-lab/flagship_alpha_momentum}"
INTERVAL="${3:-minute}"
START_DATE="${4:-2023-02-17}"
END_DATE="${5:-2025-11-20}"

echo "=========================================="
echo "后台运行数据导入脚本"
echo "=========================================="
echo "S3 目录: $S3_DIR"
echo "Lab 目录: $LAB_DIR"
echo "K线周期: $INTERVAL"
echo "日期范围: $START_DATE 到 $END_DATE"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 后台运行脚本
nohup .venv/bin/python3 flagship/scripts/import_s3_to_lab.py \
    --s3-dir "$S3_DIR" \
    --lab-dir "$LAB_DIR" \
    --interval "$INTERVAL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --log-file "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "脚本已在后台运行，PID: $PID"
echo "查看日志: tail -f $LOG_FILE"
echo "查看进程: ps aux | grep $PID"
echo ""
echo "保存以下信息以便后续查看："
echo "  PID: $PID"
echo "  日志: $LOG_FILE"

