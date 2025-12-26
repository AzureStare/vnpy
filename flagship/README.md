# Flagship Alpha-Momentum (FCAM)

本目录是 **Flagship Alpha-Momentum** 策略项目代码，基于 vn.py 的 `vnpy.alpha` 研究/回测框架实现：
- 数据准备与增量更新（Polygon → AlphaLab parquet）
- 因子建模与训练（LightGBM）
- 信号生成与报告输出（HTML）
- 纸面交易（Alpaca Paper）日常自动化 + 盘中分钟级退出守护

## 快速开始（本地）

### 1) Python 版本与依赖

- Python：`>=3.10`（仓库 `pyproject.toml`）
- 安装（在仓库根目录执行）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 研究/回测/模型训练依赖（polars/lightgbm 等）
python -m pip install -e ".[alpha]"

# Paper Trading / 数据源 / 数据库（按需）
python -m pip install alpaca-py polygon-api-client psycopg2-binary
```

### 2) 配置：`vt_setting.json`（项目根目录）

该文件默认在 `.gitignore` 中，不会被提交。

最小配置示例：

```json
{
  "datafeed.name": "polygon",
  "datafeed.password": "YOUR_POLYGON_API_KEY",

  "alpaca.api_key": "YOUR_ALPACA_KEY",
  "alpaca.secret_key": "YOUR_ALPACA_SECRET",

  "database.name": "postgresql",
  "database.host": "localhost",
  "database.port": 5432,
  "database.database": "vnpy",
  "database.user": "vnpy",
  "database.password": "change_me",

  "open-ai.api_key": "YOUR_OPENAI_API_KEY"
}
```

说明：
- Polygon API Key：也可以用环境变量 `POLYGON_API_KEY` 覆盖（优先级更高）。
- Postgres：`flagship/scripts/pg_ticker_db.py` 支持用 `DATABASE_*` 环境变量覆盖连接参数（Docker/EC2 场景）。
- OpenAI：仅用于 `flagship/model/diagnose_factors.py` 的可选 LLM 总结（默认模型 `gpt-5.2`）。

## 目录结构（以仓库根目录为工作目录）

- `flagship/`：策略项目代码
  - `flagship/scripts/`：数据准备、增量更新、pipeline、EC2 部署脚本
  - `flagship/model/`：训练/评估/因子诊断
  - `flagship/backtest/`：回测入口（默认 minute + RTH-only）
  - `flagship/paper_trading/`：Alpaca Paper 全自动日循环 + 盘中 runner
  - `flagship/strategy/`：策略类实现（`FlagshipAlphaMomentumStrategy`）
  - `flagship/docs/`：策略与部署文档（本地/服务器侧文件可能被 `.gitignore` 忽略）
- `lab/flagship_alpha_momentum/`：AlphaLab 输出目录（数据/模型/信号/报告）
  - `daily/`、`minute/`、`dataset/`、`model/`、`signal/`、`report/`、`logs/`

## 常用入口（最常用 6 个命令）

### 1) Regime 训练 +（可选）回测 Pipeline

```bash
python flagship/scripts/run_lgb_pipeline.py --regime-id 1 --run-backtest
```

### 2) 回测入口（分钟线 RTH-only 默认启用）

```bash
python flagship/backtest/flagship_alpha_momentum_backtest.py \
  --start 2024-01-02 \
  --end 2024-04-12 \
  --interval minute \
  --rth-only
```

### 3) 因子诊断报告（相关性/重要性/可选 LLM 总结）

```bash
python flagship/model/diagnose_factors.py \
  --dataset-name flagship_alpha_mom_regime01_lgb \
  --model-name flagship_alpha_mom_regime01_lgb \
  --output-path lab/flagship_alpha_momentum/report/20240102_20240412_regime01/model_diagnostics.html \
  --llm-summary --llm-model gpt-5.2
```

### 4) Paper Trading：每日全自动（串行批处理 + 盘中守护）

```bash
bash flagship/paper_trading/run_full_daily_cycle.sh
```

要点：
- `run_full_daily_cycle.sh` 是 **盘前串行任务**（数据更新 → 选股 → 补数 → 训练 → 推理 → 开盘调仓）。
- 同脚本会启动 **盘中常驻服务** `intraday_runner.py`（分钟级止盈/止损/跟踪止盈等 exit-only）。
- 可用环境变量关闭盘中 runner：`ENABLE_INTRADAY_RUNNER=0`.

### 5) 单独启动盘中 Runner（用于调试/守护）

```bash
python flagship/paper_trading/intraday_runner.py \
  --mode exit-only \
  --rth-only \
  --poll-seconds 20
```

说明：
- 默认使用 Polygon WebSocket（`AM.{ticker}` 分钟聚合）并 **自动重连**。
- 配置 `--poll-seconds` 后，WS 连续失败超过阈值会切换为 polling 兜底。

### 6) EC2 部署（增量同步/启动环境）

```bash
python flagship/scripts/ec2_deploy.py sync-code --host <EC2_IP> --identity-file <PEM_PATH>
python flagship/scripts/ec2_deploy.py sync-data --host <EC2_IP> --identity-file <PEM_PATH>
python flagship/scripts/ec2_deploy.py bootstrap --host <EC2_IP> --identity-file <PEM_PATH>
python flagship/scripts/ec2_deploy.py build --host <EC2_IP> --identity-file <PEM_PATH>
```

## Docker（推荐：Paper/DB 一体化）

仓库根目录提供 `docker-compose.yml`：

```bash
docker compose up -d db
docker compose up -d app
```

说明：
- `app` 容器内默认运行 cron（`America/New_York` 时区），定时执行 `flagship/paper_trading/run_full_daily_cycle.sh`
- Postgres 连接优先读取 `DATABASE_*` 环境变量（见 `docker-compose.yml`）

