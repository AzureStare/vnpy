# Flagship Alpha-Momentum (FCAM)

本目录是 **Flagship Alpha-Momentum** 策略项目代码，基于 vn.py 的 `vnpy.alpha` 研究/回测框架实现：
- 数据准备与增量更新（Polygon → AlphaLab parquet）
- 因子建模与训练（LightGBM）
- 信号生成与报告输出（HTML）
- 模拟交易（Alpaca Paper）日常自动化 + 盘中分钟级退出守护

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
- Postgres：`flagship/universe/pg_ticker_db.py` 支持用 `DATABASE_*` 环境变量覆盖连接参数（Docker/EC2 场景）。
- OpenAI：仅用于 `flagship/model/diagnose_factors.py` 的可选 LLM 总结（默认模型 `gpt-5.2`）。

## 目录结构（以仓库根目录为工作目录）

- `flagship/`：策略项目代码
  - `flagship/scripts/`：数据准备、增量更新、pipeline、部署脚本
  - `flagship/model/`：训练/评估/因子诊断
  - `flagship/factors/`：因子与数据集定义（V5/V7）
  - `flagship/backtest/`：统一回测入口
  - `flagship/strategy/`：策略类实现（`alpha_momentum_v5.py` / `alpha_momentum_v7.py`）
  - `flagship/trading/`：交易域包（orchestration/execution/intraday/realtime）
  - `flagship/docs/`：策略与部署文档
- `lab/flagship_alpha_momentum/`：AlphaLab 输出目录（数据/模型/信号/报告）
  - `daily/`、`minute/`：K 线
  - `dataset/`：AlphaDataset 输出
  - `model/`：训练模型
  - `signal/`：模型信号（用于回测/实盘）
  - `report/`、`logs/`：回测与诊断输出

## 常用入口（最常用 6 个命令）

### 1) 训练 +（可选）回测 Pipeline

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start 2024-01-02 \
  --end 2024-04-12 \
  --run-backtest
```

### 2) 回测入口（统一脚本）

前置条件：
- `lab/flagship_alpha_momentum` 下必须有 **daily/minute bars**，否则无法生成占位信号。
- 若显式传 `--signal-name`，需要先生成对应信号文件（`lab/flagship_alpha_momentum/signal/<name>.parquet`）。

常用方式（推荐）：
1) 先跑 pipeline 生成数据/模型/信号：
```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start 2024-01-02 \
  --end 2024-04-12 \
  --run-backtest
```

2) 直接回测（已有信号或可用占位信号）：
```bash
python -m flagship.backtest.flagship_alpha_momentum_backtest \
  --start 2024-01-02 \
  --end 2024-04-12 \
  --interval minute \
  --rth-only \
  --strategy v7 \
  --no-postgres-selection
```

说明：
- `--strategy v5|v7`：策略版本
- `--signal-name`：指定信号文件名（不指定则使用默认名）
- `--no-postgres-selection`：不依赖 Postgres 选股

### 3) 因子诊断报告（相关性/重要性/可选 LLM 总结）

```bash
python flagship/model/diagnose_factors.py \
  --dataset-name flagship_alpha_momentum_20240102_20240412_lgb \
  --model-name flagship_alpha_momentum_20240102_20240412_lgb \
  --output-path lab/flagship_alpha_momentum/report/20240102_20240412_backtest/model_diagnostics.html \
  --llm-summary --llm-model gpt-5.2
```

### 4) 盘后 Daily Cycle（数据更新/选股/训练/推理/产物）

```bash
python -m flagship.trading.orchestration.daily_cycle_runner --strategy v7
```

要点：
- Daily Cycle 负责 **盘后批处理**（数据更新 → 选股/补数 → 训练（按规则）→ 推理生成信号 → 快照/报告）。
- 开盘调仓由常驻 `ExecutorDaemon` 负责；盘中退出由常驻 `IntradayDaemon` 负责（两者在 Docker entrypoint 内启动）。

### 5) 单独启动盘中 Runner（用于调试）

```bash
python -m flagship.trading.intraday.intraday_runner \
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
- `app` 容器内默认运行 cron（`America/New_York` 时区），定时执行 Daily Cycle（`python -m flagship.trading.orchestration.daily_cycle_runner --strategy v7`）
- Postgres 连接优先读取 `DATABASE_*` 环境变量（见 `docker-compose.yml`）

