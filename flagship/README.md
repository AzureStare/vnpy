# Flagship Capital Alpha-Momentum (FCAM) Strategy

独立策略项目，基于 vnpy 框架实现 Flagship Alpha-Momentum 量化策略。

## 架构说明

- **`vnpy/`**：量化交易框架库（不包含业务逻辑）
- **`flagship/`**：独立策略项目，使用 vnpy 库实现完整的量化策略流程
  - 数据清洗与预处理
  - 因子发现与建模
  - 策略执行与回测

## 项目结构

```
flagship/
├── data/              # 数据目录
│   ├── raw/           # 原始数据（Polygon API 拉取）
│   ├── cleaned/       # 清洗后数据
│   └── universe/      # 动态股票池
├── scripts/           # 数据清洗与预处理脚本
│   ├── __init__.py
│   ├── pg_ticker_db.py                    # Postgres ticker 数据库工具
│   ├── sync_tickers_postgres.py          # 同步 ticker 主表
│   ├── sync_ticker_details_postgres.py   # 同步 ticker 每日基本面
│   ├── build_daily_universe.py           # 构建每日动态股票池（基于策略筛选条件）
│   ├── download_backtest_data.py          # 下载回测数据（日线/分钟线）
│   └── run_full_pipeline.py              # 完整数据流程脚本
├── factors/           # 因子计算模块（待实现）
├── strategy/           # 策略执行模块（待实现）
├── backtest/           # 回测模块（待实现）
└── config/             # 配置文件（待实现）
```

## 数据流程

1. **数据同步**：从 Polygon API 拉取 ticker 和基本面数据到 Postgres
2. **每日动态股票池**：基于 Flagship Alpha-Momentum 策略规则构建每日 universe（$U_t$）
3. **回测数据下载**：基于筛选结果下载历史行情数据（日线/分钟线）
4. **策略回测**：使用 vnpy.alpha 框架进行回测（待实现）

## 配置要求

在项目根目录的 `vt_setting.json` 中配置：

```json
{
  "database.name": "postgresql",
  "database.host": "localhost",
  "database.port": 5432,
  "database.database": "vnpy",
  "database.user": "postgres",
  "database.password": "your_password",
  "datafeed.name": "polygon",
  "datafeed.password": "your_polygon_api_key"
}
```

## 使用流程

### 完整流程（推荐）

使用 `run_full_pipeline.py` 一键执行所有步骤：

```bash
python flagship/scripts/run_full_pipeline.py \
  --init-tables \
  --details-start 2023-01-01 \
  --details-end 2023-12-31 \
  --universe-date 2023-11-20 \
  --download-start 2023-01-01 \
  --download-end 2023-12-31 \
  --download-interval daily
```

### 分步执行

#### 1. 初始化数据库表并同步 ticker 主表

```bash
python flagship/scripts/sync_tickers_postgres.py --init-tables --ticker-type CS
```

#### 2. 同步 ticker 基本面数据（按需，可选）

**注意**：策略筛选条件不包含市值，通常不需要预先同步所有 ticker 的市值。如果确实需要，建议：

```bash
# 只同步特定股票列表（推荐）
python flagship/scripts/sync_ticker_details_postgres.py \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --symbols AAPL MSFT GOOGL AMZN TSLA \
  --init-tables

# 或限制数量用于测试
python flagship/scripts/sync_ticker_details_postgres.py \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --limit-symbols 100 \
  --init-tables
```

#### 3. 构建每日股票池（基于策略筛选条件）

```bash
python flagship/scripts/build_daily_universe.py \
  --date 2023-11-20 \
  --use-postgres \
  --min-adv-usd 2.5e8 \
  --min-price 20 \
  --max-price 600
```

#### 4. 下载回测数据（基于筛选结果）

```bash
python flagship/scripts/download_backtest_data.py \
  --universe-file flagship/data/universe/universe_2023-11-20.json \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --interval daily
```

或者下载整个目录下的所有股票池：

```bash
python flagship/scripts/download_backtest_data.py \
  --universe-dir flagship/data/universe \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --interval daily
```

