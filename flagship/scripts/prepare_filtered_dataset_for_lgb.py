"""
基于 daily_selection 过滤数据集并添加排名标签，用于 LightGBM 训练。

支持按Regime自动划分train/valid/test，并添加T+5日收益率截面排名标签。
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import psycopg2
import polars as pl
from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import Segment
from vnpy.trader.constant import Interval
from flagship.factors.flagship_alpha_momentum_v5 import FlagshipAlphaMomentumV5Dataset
from flagship.backtest.index_regime_windows import REGIME_WINDOWS, get_regime_window
from vnpy.trader.logger import logger


def load_daily_selection_from_postgres(start_date: date, end_date: date) -> pl.DataFrame:
    """从PostgreSQL加载daily_selection数据"""
    settings = json.loads(Path('vt_setting.json').read_text())
    conn = psycopg2.connect(
        host=settings['database.host'],
        port=settings['database.port'],
        dbname=settings['database.database'],
        user=settings['database.user'],
        password=settings['database.password'],
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT to_char(trade_date,'YYYY-MM-DD'), vt_symbol "
        "FROM daily_selection "
        "WHERE trade_date BETWEEN %s AND %s "
        "ORDER BY trade_date, vt_symbol",
        (start_date, end_date)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    sel_df = pl.DataFrame(rows, schema=["trade_date_str", "vt_symbol"], orient="row")
    sel_df = sel_df.with_columns(pl.col("trade_date_str").str.to_datetime().cast(pl.Date).alias("trade_date"))
    return sel_df


def prepare_dataset_for_regime(
    lab: AlphaLab,
    regime_id: int,
    use_postgres_selection: bool = True,
) -> None:
    """
    为指定Regime准备LightGBM训练数据集。
    
    Args:
        lab: AlphaLab实例
        regime_id: Regime编号
        use_postgres_selection: 是否使用PostgreSQL选股过滤
    """
    regime = get_regime_window(regime_id)
    logger.info(f"[prepare_dataset_for_regime] 开始处理 Regime {regime_id}: {regime.label}")
    logger.info(f"[prepare_dataset_for_regime] 日期范围: {regime.start} ~ {regime.end}")
    
    # 根据Regime定义train/valid/test
    # 使用与build_v4_signal.py相同的逻辑：基于交易日索引
    # 先加载数据，然后根据交易日索引划分
    from datetime import timedelta
    
    # 扩展数据加载范围，确保有足够的历史数据计算因子
    # test_period = regime日期范围
    # valid_period = regime开始前45个交易日
    # train_period = valid_period开始前90个交易日
    # 先扩展起始日期以获取足够的历史数据
    test_start = regime.start
    test_end = regime.end
    data_start = regime.start - timedelta(days=180)  # 约6个月的历史数据
    
    logger.info(f"[prepare_dataset_for_regime] 数据加载起始: {data_start}")
    logger.info(f"[prepare_dataset_for_regime] 数据加载结束: {test_end}")
    
    # 加载daily_selection
    sel_df = None
    if use_postgres_selection:
        logger.info(f"[prepare_dataset_for_regime] 从PostgreSQL加载daily_selection...")
        sel_df = load_daily_selection_from_postgres(data_start, test_end)
        logger.info(f"[prepare_dataset_for_regime] 加载了 {len(sel_df)} 行选股数据")
    
    # 加载日线数据
    lab = AlphaLab(str(lab.lab_path))
    if use_postgres_selection and sel_df is not None:
        # 只加载选中的股票
        vt_symbols = sel_df["vt_symbol"].unique().to_list()
        logger.info(f"[prepare_dataset_for_regime] 使用PostgreSQL选股: {len(vt_symbols)} 只股票")
    else:
        # 加载所有日线文件
        daily_files = sorted(lab.daily_path.glob("*.parquet"))
        vt_symbols = [p.stem for p in daily_files]
        logger.info(f"[prepare_dataset_for_regime] 使用所有日线数据: {len(vt_symbols)} 只股票")
    
    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=data_start.isoformat(),
        end=test_end.isoformat(),
        extended_days=0,
    )
    logger.info(f"[prepare_dataset_for_regime] 加载原始日线数据: {len(raw_df)} 行")
    
    # 根据交易日索引划分train/valid/test（与build_v4_signal.py逻辑一致）
    all_dates = sorted(raw_df["datetime"].unique().to_list())
    
    # 找到test_period的索引
    test_start_idx = None
    test_end_idx = None
    for i, dt in enumerate(all_dates):
        if dt.date() == regime.start:
            test_start_idx = i
        if dt.date() == regime.end:
            test_end_idx = i
    
    # 如果找不到结束日期，尝试使用数据中的最后一个日期（处理数据截止日期问题）
    if test_end_idx is None:
        last_date = all_dates[-1].date()
        if last_date >= regime.start:
            # 使用最后一个可用日期
            test_end_idx = len(all_dates) - 1
            logger.warning(f"[prepare_dataset_for_regime] Regime结束日期 {regime.end} 不在数据中，使用数据最后日期 {last_date}")
        else:
            raise RuntimeError(f"数据最后日期 {last_date} 早于Regime开始日期 {regime.start}")
    
    if test_start_idx is None:
        raise RuntimeError(f"无法在数据中找到regime的开始日期: {regime.start}")
    
    # valid_period: test_period之前45个交易日
    valid_days = 45
    valid_start_idx = max(0, test_start_idx - valid_days)
    valid_end_idx = test_start_idx - 1
    
    if valid_end_idx < valid_start_idx:
        raise RuntimeError(f"可用交易日不足，无法定义valid_period")
    
    # train_period: valid_period之前90个交易日
    train_days = 90
    train_start_idx = max(0, valid_start_idx - train_days)
    train_end_idx = valid_start_idx - 1
    
    if train_end_idx < train_start_idx:
        raise RuntimeError(f"可用交易日不足，无法定义train_period")
    
    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")
    
    train_period = (fmt(all_dates[train_start_idx]), fmt(all_dates[train_end_idx]))
    valid_period = (fmt(all_dates[valid_start_idx]), fmt(all_dates[valid_end_idx]))
    test_period = (fmt(all_dates[test_start_idx]), fmt(all_dates[test_end_idx]))
    
    logger.info(f"[prepare_dataset_for_regime] 数据段划分:")
    logger.info(f"  - Train: {train_period[0]} ~ {train_period[1]} ({train_end_idx - train_start_idx + 1} 个交易日)")
    logger.info(f"  - Valid: {valid_period[0]} ~ {valid_period[1]} ({valid_end_idx - valid_start_idx + 1} 个交易日)")
    logger.info(f"  - Test: {test_period[0]} ~ {test_period[1]} ({test_end_idx - test_start_idx + 1} 个交易日)")
    
    # 创建数据集
    dataset = FlagshipAlphaMomentumV5Dataset(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )
    
    logger.info("[prepare_dataset_for_regime] 开始 prepare_data...")
    dataset.prepare_data(filters=None)
    logger.info("[prepare_dataset_for_regime] 开始 process_data...")
    dataset.process_data()
    logger.info("[prepare_dataset_for_regime] 因子计算完成")
    
    # 过滤 daily_selection 并添加排名标签
    logger.info("[prepare_dataset_for_regime] 开始过滤 daily_selection 并添加排名标签...")
    
    # 处理learn_df和infer_df
    for df_name in ["learn_df", "infer_df"]:
        df = getattr(dataset, df_name)
        if df is None or df.is_empty():
            logger.warning(f"[prepare_dataset_for_regime] {df_name} 为空，跳过")
            continue
        
        logger.info(f"[prepare_dataset_for_regime] 处理 {df_name}: {len(df)} 行")
        
        # 过滤daily_selection
        # 注意：daily_selection可能只包含test_period的数据，所以train/valid期间不过滤
        if use_postgres_selection and sel_df is not None:
            df = df.with_columns(pl.col("datetime").cast(pl.Date).alias("trade_date"))
            df_before = len(df)
            
            # 分别处理每个segment
            filtered_dfs = []
            for segment in [Segment.TRAIN, Segment.VALID, Segment.TEST]:
                segment_df = df.filter(
                    (pl.col("trade_date") >= datetime.fromisoformat(dataset.data_periods[segment][0]).date()) &
                    (pl.col("trade_date") <= datetime.fromisoformat(dataset.data_periods[segment][1]).date())
                )
                
                if segment_df.is_empty():
                    continue
                
                # 检查该segment是否有daily_selection数据
                segment_start = segment_df["trade_date"].min()
                segment_end = segment_df["trade_date"].max()
                segment_sel_df = sel_df.filter(
                    (pl.col("trade_date") >= segment_start) & 
                    (pl.col("trade_date") <= segment_end)
                )
                
                if not segment_sel_df.is_empty():
                    # 有daily_selection数据，进行过滤
                    segment_df = segment_df.join(segment_sel_df, on=["trade_date", "vt_symbol"], how="inner")
                    logger.info(f"[prepare_dataset_for_regime] {segment.name} 过滤daily_selection: {len(segment_df)} 行")
                else:
                    # 没有daily_selection数据，保留所有数据（train/valid期间可能没有选股数据）
                    logger.info(f"[prepare_dataset_for_regime] {segment.name} 无daily_selection数据，保留所有数据: {len(segment_df)} 行")
                
                # 统一列：移除trade_date相关列
                drop_cols = []
                if "trade_date" in segment_df.columns:
                    drop_cols.append("trade_date")
                if "trade_date_str" in segment_df.columns:
                    drop_cols.append("trade_date_str")
                if drop_cols:
                    segment_df = segment_df.drop(drop_cols)
                
                filtered_dfs.append(segment_df)
            
            if filtered_dfs:
                df = pl.concat(filtered_dfs)
            else:
                df = pl.DataFrame()
            
            logger.info(f"[prepare_dataset_for_regime] 过滤后 {df_name}: {len(df)} 行 (过滤前 {df_before} 行)")
        
        # 添加T+5日收益率和排名标签
        if "close_price" in df.columns and not df.is_empty():
            # 计算T+5日收益率
            df = df.with_columns(
                ((pl.col("close_price").shift(-5) / pl.col("close_price")) - 1).alias("ret_5d")
            )
            
            # 计算截面排名（按datetime分组，ret_5d降序排名）
            # 排名从0开始（最高收益率为0）
            df = df.sort(["datetime", "ret_5d"], descending=[False, True])
            df = df.with_columns(
                (pl.int_range(pl.len()).over("datetime")).cast(pl.Int32).alias("rank_5d")
            )
            
            logger.info(f"[prepare_dataset_for_regime] {df_name} 添加排名标签完成")
            logger.info(f"[prepare_dataset_for_regime]   - ret_5d非null: {df.filter(pl.col('ret_5d').is_not_null()).height}")
            logger.info(f"[prepare_dataset_for_regime]   - rank_5d范围: {df['rank_5d'].min()} ~ {df['rank_5d'].max()}")
        else:
            if df.is_empty():
                logger.warning(f"[prepare_dataset_for_regime] {df_name} 过滤后为空")
            else:
                logger.warning(f"[prepare_dataset_for_regime] {df_name} 没有 close_price 列，无法添加标签")
        
        # 直接更新数据集属性
        setattr(dataset, df_name, df)
    
    # 保存数据集
    dataset_name = f"flagship_alpha_mom_regime{regime_id:02d}_lgb"
    lab.save_dataset(dataset_name, dataset)
    logger.info(f"[prepare_dataset_for_regime] 数据集已保存: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="准备LightGBM训练数据集，支持按Regime划分"
    )
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab数据根目录",
    )
    parser.add_argument(
        "--regime-id",
        type=int,
        help="Regime编号（1-10），如果指定则按Regime划分train/valid/test",
    )
    parser.add_argument(
        "--use-postgres-selection",
        action="store_true",
        default=True,
        help="使用PostgreSQL daily_selection过滤股票",
    )
    args = parser.parse_args()
    
    lab = AlphaLab(args.lab_path)
    
    if args.regime_id:
        prepare_dataset_for_regime(lab, args.regime_id, args.use_postgres_selection)
    else:
        logger.error("必须指定 --regime-id")
        raise ValueError("必须指定 --regime-id")


if __name__ == "__main__":
    main()

