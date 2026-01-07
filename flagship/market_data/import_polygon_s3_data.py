import boto3
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
import polars as pl
from tqdm import tqdm
import botocore
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData
from vnpy.trader.logger import logger
from flagship.config import DEFAULT_LAB_DIR

def load_s3_config():
    vt_setting_path = PROJECT_ROOT / "vt_setting.json"
    if not vt_setting_path.exists():
        raise FileNotFoundError("vt_setting.json not found")
    
    with open(vt_setting_path, "r") as f:
        config = json.load(f)
    
    return {
        "aws_access_key_id": config.get("polygon.s3.access_key_id"),
        "aws_secret_access_key": config.get("polygon.s3.secret_access_key"),
        "endpoint_url": "https://flatfiles.polygon.io",
        "region_name": "us-east-1" # dummy
    }

def get_s3_client():
    config = load_s3_config()
    # Disable SSL verification as a workaround for the 'self-signed certificate' error 
    # encountered in the Docker environment for flatfiles.polygon.io
    return boto3.client('s3', verify=False, **config)

def download_daily_flat_file(s3_client, target_date: date, local_dir: Path):
    """
    Polygon S3 structure:
    us_equity_daily_bars_v1/2023/01/2023-01-01.parquet (or similar)
    Actually usually it's /us_equity_daily_bars_v1/{year}/{month}/{day}.parquet
    """
    year = target_date.year
    month = f"{target_date.month:02d}"
    day = target_date.strftime("%Y-%m-%d")
    
    # Common Polygon paths
    possible_keys = [
        f"us_equity_daily_bars_v1/{year}/{month}/{day}.parquet",
        f"us_indices_daily_bars_v1/{year}/{month}/{day}.parquet"
    ]
    
    downloaded_files = []
    
    for s3_key in possible_keys:
        local_path = local_dir / Path(s3_key).name
        try:
            logger.info(f"Downloading s3://polygon/{s3_key}")
            s3_client.download_file('polygon', s3_key, str(local_path))
            downloaded_files.append(local_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # logger.warning(f"S3 Key not found: {s3_key}")
                pass
            else:
                logger.error(f"S3 error for {s3_key}: {e}")
                
    return downloaded_files

def process_flat_file(file_path: Path, lab: AlphaLab):
    """
    Process a full market daily parquet file and split into AlphaLab per-symbol parquets.
    Polygon flat file columns usually: ticker, open, high, low, close, volume, vwap, transactions, timestamp
    """
    df = pl.read_parquet(file_path)
    
    # Rename columns to match vn.py BarData if needed
    # Polygon flat files: 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp' (nanoseconds)
    
    if "ticker" not in df.columns:
        logger.warning(f"Skipping {file_path}: 'ticker' column not found.")
        return

    # Convert timestamp (ns) to datetime
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="ns").alias("datetime")
        )
    
    # Map symbols to VT format (assuming US stocks)
    df = df.with_columns(
        pl.col("ticker").alias("vt_symbol") # Simplified, real logic might need exchange mapping
    )

    # Group by symbol and save
    symbols = df["vt_symbol"].unique().to_list()
    
    # Performance note: For full market, saving one by one is slow. 
    # But AlphaLab.save_bar_data expects BarData objects.
    # To optimize, we bypass BarData and write parquet directly if possible, 
    # but for compatibility, we use lab.save_bar_data in batches.
    
    for symbol in symbols:
        symbol_df = df.filter(pl.col("vt_symbol") == symbol)
        
        # Determine exchange (hack for US market if not in file)
        # Usually Polygon flat files don't have exchange, we default to NYSE/NASDAQ based on ticker or suffix
        exchange = "NYSE" # Placeholder
        
        bars = []
        for row in symbol_df.iter_rows(named=True):
            bars.append(BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=Interval.DAILY,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=int(row["volume"]),
                turnover=float(row.get("vwap", row["close"])) * row["volume"],
                gateway_name="POLYGON_S3"
            ))
        
        if bars:
            lab.save_bar_data(bars)

def backfill_via_s3(start_date: date, end_date: date, lab_path: Path = DEFAULT_LAB_DIR):
    lab = AlphaLab(str(lab_path))
    s3_client = get_s3_client()
    
    temp_dir = PROJECT_ROOT / "temp_s3_downloads"
    temp_dir.mkdir(exist_ok=True)
    
    current_date = start_date
    delta = timedelta(days=1)
    
    logger.info(f"Starting S3 backfill from {start_date} to {end_date}")
    
    while current_date <= end_date:
        # Skip weekends (simple)
        if current_date.weekday() < 5:
            files = download_daily_flat_file(s3_client, current_date, temp_dir)
            for f in files:
                process_flat_file(f, lab)
                f.unlink() # Remove temp file
        
        current_date += delta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    backfill_via_s3(start, end)

