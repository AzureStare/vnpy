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

**必选参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `--start` | str | 训练/回测起始日期，格式 `YYYY-MM-DD` |
| `--end` | str | 训练/回测结束日期，格式 `YYYY-MM-DD` |

**可选参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lab-path` | str | `lab/flagship_alpha_momentum` | AlphaLab 数据根目录 |
| `--use-postgres-selection` | flag | True | 使用 PostgreSQL daily_selection 过滤股票 |
| `--selection-strategy` | str | `v7` | 构建 daily_selection 的策略版本，可选 `v5` / `v7` |
| `--dataset-strategy` | str | 同 selection | 生成 AlphaDataset 的策略版本，可选 `v5` / `v7` |
| `--skip-selection` | flag | - | 跳过 daily_selection 构建 |
| `--skip-prepare` | flag | - | 同 `--skip-selection`（兼容旧参数） |
| `--skip-dataset` | flag | - | 跳过 AlphaDataset 生成 |
| `--valid-days` | int | 60 | VALID 窗口天数 |
| `--test-days` | int | 60 | TEST 窗口天数 |
| `--gap-days` | int | 7 | VALID/TEST 前的间隔天数 |
| `--extended-days` | int | 120 | 额外加载历史天数 |
| `--max-workers` | int | None | 特征计算并行进程数 |
| `--skip-diagnostics` | flag | - | 跳过因子诊断报告生成 |
| `--llm-summary` | flag | False | 诊断报告中启用 LLM 总结（需 OpenAI key） |
| `--llm-model` | str | `gpt-5.2` | LLM 模型名 |
| `--llm-max-completion-tokens` | int | 2000 | LLM 最大完成 token 数 |
| `--skip-train` | flag | - | 跳过训练（若模型已存在） |
| `--run-backtest` | flag | - | 训练完成后运行回测 |
| `--backtest-strategy` | str | 同 dataset | 回测策略版本，可选 `v5` / `v7` |
| `--backtest-capital` | float | 1000000.0 | 回测初始资金 |
| `--dataset-name` | str | 按日期生成 | 数据集名称 |
| `--model-name` | str | 按日期生成 | 模型名称 |
| `--signal-name` | str | 按日期生成 | 信号名称 |

**用法示意（`[]` 表示可选参数）**

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  [--lab-path lab/flagship_alpha_momentum] \
  [--use-postgres-selection] \
  [--selection-strategy v5|v7] \
  [--dataset-strategy v5|v7] \
  [--skip-selection] \
  [--skip-prepare] \
  [--skip-dataset] \
  [--valid-days 60] \
  [--test-days 60] \
  [--gap-days 7] \
  [--extended-days 120] \
  [--max-workers 4] \
  [--skip-diagnostics] \
  [--llm-summary --llm-model gpt-5.2 --llm-max-completion-tokens 2000] \
  [--skip-train] \
  [--run-backtest --backtest-strategy v5|v7 --backtest-capital 1000000] \
  [--dataset-name <name>] \
  [--model-name <name>] \
  [--signal-name <name>]
```

**示例：全参数（`[]` 表示可选；基于本地已有数据区间，可直接跑）**

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start 2023-02-17 \
  --end 2025-12-31 \
  [--lab-path lab/flagship_alpha_momentum] \
  [--use-postgres-selection] \
  [--selection-strategy v7] \
  [--dataset-strategy v7] \
  [--skip-selection] \
  [--skip-prepare] \
  [--skip-dataset] \
  [--valid-days 60] \
  [--test-days 60] \
  [--gap-days 7] \
  [--extended-days 120] \
  [--max-workers 4] \
  [--skip-diagnostics] \
  [--llm-summary --llm-model gpt-5.2 --llm-max-completion-tokens 2000] \
  [--skip-train] \
  [--run-backtest --backtest-strategy v7 --backtest-capital 1000000] \
  [--dataset-name flagship_alpha_momentum_20240102_20240412_lgb] \
  [--model-name flagship_alpha_momentum_20240102_20240412_lgb] \
  [--signal-name flagship_alpha_momentum_20240102_20240412_lgb_signal]
```

**示例：最小运行（训练 + 回测）**

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start  2023-02-17  \
  --end 2025-12-31 \
  --run-backtest
```

**示例：跳过选股、指定 lab、多进程、跑回测**

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start  2023-02-17  \
  --end 2025-12-31 \
  --lab-path lab/flagship_alpha_momentum \
  --skip-selection \
  --max-workers 4 \
  --run-backtest \
  --backtest-capital 500000
```

**示例：只生成数据集 + 诊断（不训练、不回测）**

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start 2024-01-02 \
  --end 2024-04-12 \
  --skip-train
```

**示例：启用 LLM 诊断总结**

```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start 2024-01-02 \
  --end 2024-04-12 \
  --llm-summary \
  --llm-model gpt-4o
```

### 2) 回测入口（统一脚本）

前置条件：
- `lab/flagship_alpha_momentum` 下必须有 **daily/minute bars**，否则无法生成占位信号。
- 若显式传 `--signal-name`，需要先生成对应信号文件（`lab/flagship_alpha_momentum/signal/<name>.parquet`）。

常用方式（推荐）：
1) 先跑 pipeline 生成数据/模型/信号：
```bash
python -m flagship.scripts.run_lgb_pipeline \
  --start  2023-02-17  \
  --end 2025-12-31 \
  --run-backtest
```

2) 直接回测（已有信号或可用占位信号）：
```bash
python -m flagship.backtest.flagship_alpha_momentum_backtest \
  --start  2023-02-17  \
  --end 2025-12-31 \
  --interval minute \
  --rth-only \
  --strategy v7 \
  --no-postgres-selection
```

说明：
- `--strategy v5|v7`：策略版本
- `--signal-name`：指定信号文件名（不指定则自动加载 `lab/flagship_alpha_momentum/signal/` 下最近更新的 parquet）
- `--no-postgres-selection`：不依赖 Postgres 选股
- 默认不会生成占位信号：如果 `lab/flagship_alpha_momentum/signal/` 下没有任何可用信号文件，会直接报错提示先生成信号（推荐先跑 `run_lgb_pipeline`）。
- 如需旧行为（信号缺失时用滚动收益率生成占位信号，仅用于调试），加 `--allow-naive-signal`。

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

