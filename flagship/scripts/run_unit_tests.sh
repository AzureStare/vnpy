#!/usr/bin/env bash
set -euo pipefail

# 统一入口：一键运行全量单元测试
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# 优先使用本地 .venv
PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

OUT_DIR="${UNIT_TESTS_OUT_DIR:-logs/test}"
HTML_REPORT="${UNIT_TESTS_HTML_REPORT:-$OUT_DIR/unit_tests.html}"
DURATIONS="${UNIT_TESTS_DURATIONS:-10}"

mkdir -p "$OUT_DIR"

# 确保 pytest-html 可用（用于生成 HTML 报告）
if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("pytest_html") else 1)
PY
then
  "$PYTHON_BIN" -m pip install pytest-html
fi

# 默认输出 HTML + 慢用例统计；如需覆盖可直接传 pytest 参数
"$PYTHON_BIN" -m pytest tests --html="$HTML_REPORT" --self-contained-html --durations="$DURATIONS" "$@"
