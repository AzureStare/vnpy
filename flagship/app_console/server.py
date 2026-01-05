from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import re
from pydantic import BaseModel
from typing import Optional, List
import hashlib

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flagship.scripts.pg_ticker_db import (
    create_users_table, add_user, get_user, list_users, delete_user
)

# Configuration
SECRET_KEY = "flagship-ecc-style-secret-key" # In production use env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

app = FastAPI()

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class UserBase(BaseModel):
    username: str
    role: str = "user"

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    created_at: str

class BacktestRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = PROJECT_ROOT / "logs" / "backtest"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = LOG_DIR / "status.json"

# Utility
def get_password_hash(password: str) -> str:
    salt = "flagship-salt"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return payload
    except JWTError:
        raise credentials_exception

def check_admin(user = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ---------------- Backtest status helpers ----------------
def _is_pid_running(pid: int) -> bool:
    """Return True if pid exists in current process namespace."""
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

# Auth Endpoints
@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or user["password_hash"] != get_password_hash(form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"]}

# User Management (Admin Only)
@app.get("/api/users", response_model=List[UserOut])
async def get_users(admin = Depends(check_admin)):
    return list_users()

@app.post("/api/users")
async def create_new_user(user: UserCreate, admin = Depends(check_admin)):
    hashed = get_password_hash(user.password)
    if add_user(user.username, hashed, user.role):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to create user")

@app.delete("/api/users/{username}")
async def remove_user(username: str, admin = Depends(check_admin)):
    if delete_user(username):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to delete user")

# Backtest Endpoints
def _update_status(
    status: str,
    message: str = "",
    progress: float = 0,
    pid: int | None = None,
    log_file: str | None = None,
):
    with open(STATUS_FILE, "w") as f:
        payload = {
            "status": status,
            "message": message,
            "progress": progress,
            "updated_at": datetime.now().isoformat(),
        }
        if pid is not None:
            payload["pid"] = int(pid)
        if log_file is not None:
            payload["log_file"] = str(log_file)
        json.dump(payload, f)

def run_backtest_task(start_date: str, end_date: str):
    log_path: Path | None = None
    log_fp = None
    try:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"backtest_{run_ts}.log"
        log_fp = open(log_path, "w", encoding="utf-8")
        log_fp.write(f"[{datetime.now().isoformat()}] Backtest start: {start_date} -> {end_date}\n")
        log_fp.flush()

        _update_status(
            "running",
            f"Preparing Backtest: {start_date} to {end_date}",
            0.05,
            pid=None,
            log_file=log_path.name,
        )
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        data_start_date = (start_dt - timedelta(days=1095)).strftime("%Y-%m-%d")
        
        # 1. Update Data ( Skip if data already exists and is recent)
        lab_daily_dir = PROJECT_ROOT / "lab" / "flagship_alpha_momentum" / "daily"
        last_sync_file = LOG_DIR / "last_full_sync.tag"
        
        needs_sync = True
        if lab_daily_dir.exists():
            # Check if we have SPY data covering the end_date as a proxy for market data completeness
            spy_file = lab_daily_dir / "SPY.NASDAQ.parquet"
            if spy_file.exists():
                import polars as pl
                spy_df = pl.read_parquet(spy_file)
                if not spy_df.is_empty():
                    max_date = spy_df["datetime"].max()
                    if max_date and max_date.date() >= end_dt.date():
                        # Data seems to cover the range, check if sync was recent (within 12h)
                        if last_sync_file.exists() and (datetime.now() - datetime.fromtimestamp(last_sync_file.stat().st_mtime)) < timedelta(hours=12):
                            needs_sync = False
                            _update_status("running", "Market data is up-to-date. Skipping sync...", 0.1, log_file=log_path.name)
        
        if needs_sync:
            _update_status("running", "Checking for fast S3 data sync...", 0.08, log_file=log_path.name)
            vt_setting_path = PROJECT_ROOT / "vt_setting.json"
            has_s3 = False
            if vt_setting_path.exists():
                with open(vt_setting_path, "r") as f:
                    conf = json.load(f)
                    if conf.get("polygon.s3.access_key_id") and conf.get("polygon.s3.secret_access_key"):
                        has_s3 = True
            
            if has_s3:
                _update_status("running", f"Syncing full market history via Polygon S3 (Fast)...", 0.1, log_file=log_path.name)
                subprocess.run([
                    "python", "flagship/scripts/import_polygon_s3_data.py", 
                    "--start", data_start_date, "--end", end_date
                ], cwd=PROJECT_ROOT, check=True, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
            else:
                _update_status("running", f"Syncing market data via REST API (Slow)...", 0.1, log_file=log_path.name)
                subprocess.run([
                    "python", "flagship/scripts/update_lab_data_incremental.py", 
                    "--end-date", end_date, "--start-date", data_start_date,
                    "--interval", "daily", "--universe", "ref_tickers_cs"
                ], cwd=PROJECT_ROOT, check=True, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
            
            subprocess.run(
                ["python", "flagship/paper_trading/update_market_indices.py", "--lookback", "1200"],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            last_sync_file.touch() # Mark successful sync
        
        # 2. Selection Table (Optimized: Only backfill if missing for the range)
        _update_status("running", "Backfilling selection universe...", 0.3, log_file=log_path.name)
        sel_proc = subprocess.Popen(
            [
                "python",
                "flagship/scripts/build_daily_selection_to_postgres.py",
                "--start",
                data_start_date,
                "--end",
                end_date,
                "--strategy",
                "v7",
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _update_status("running", "Backfilling selection universe...", 0.3, pid=sel_proc.pid, log_file=log_path.name)

        if sel_proc.stdout:
            for line in sel_proc.stdout:
                log_fp.write(line)
                log_fp.flush()

                # Example: 已处理 500/18335 只股票, ...
                m = re.search(r"已处理\s+(\d+)/(\d+)", line)
                if m:
                    done = int(m.group(1))
                    total = int(m.group(2)) if int(m.group(2)) > 0 else 1
                    p = 0.3 + 0.09 * (done / total)
                    _update_status(
                        "running",
                        f"Backfilling selection universe... {done}/{total}",
                        p,
                        pid=sel_proc.pid,
                        log_file=log_path.name,
                    )
                elif "数据保存完成" in line:
                    _update_status("running", "Backfilling selection universe... done", 0.39, pid=sel_proc.pid, log_file=log_path.name)

        sel_proc.wait()
        if sel_proc.returncode != 0:
            _update_status("failed", f"Selection backfill failed (code {sel_proc.returncode}). See {log_path.name}", 1.0, log_file=log_path.name)
            return
        
        # 3. Rolling Signals & Strategy Backtest (Integrated in V7 Rolling Script)
        _update_status("running", "Running Rolling AI Training & Backtest Engine...", 0.4, log_file=log_path.name)
        process = subprocess.Popen(
            ["python", "flagship/backtest/v7_rolling_backtest.py", "--start", start_date, "--end", end_date],
            cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        _update_status("running", "Running Rolling AI Training & Backtest Engine...", 0.4, pid=process.pid, log_file=log_path.name)
        if process.stdout:
            for line in process.stdout:
                log_fp.write(line)
                log_fp.flush()
                if "Generating Signals:" in line:
                    _update_status("running", f"AI Signal Gen: {line.strip()}", 0.5, pid=process.pid, log_file=log_path.name)
                elif "Running Strategy Backtest Engine" in line:
                    _update_status("running", "Executing Strategy Simulation...", 0.8, pid=process.pid, log_file=log_path.name)

        process.wait()
        if process.returncode != 0:
            _update_status("failed", f"Failed (code {process.returncode}). See {log_path.name}", 1.0, log_file=log_path.name)
            return

        # 5. Promote Model (Option A - Automatic)
        _update_status("running", "Promoting latest backtest model to paper trading...", 0.95, log_file=log_path.name)
        promote_proc = subprocess.run(
            ["python", "flagship/scripts/promote_backtest_model.py"],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        
        if promote_proc.returncode == 0:
            _update_status("completed", "Backtest finished and model promoted to Paper Trading!", 1.0, log_file=log_path.name)
        else:
            _update_status("completed", f"Backtest finished, but model promotion failed: {promote_proc.stderr}", 1.0, log_file=log_path.name)

    except Exception as e:
        _update_status("failed", str(e), 1.0, log_file=log_path.name if log_path else None)
        try:
            if log_fp is not None:
                log_fp.write(f"[{datetime.now().isoformat()}] ERROR: {e}\n")
                log_fp.flush()
        except Exception:
            pass
    finally:
        try:
            if log_fp is not None:
                log_fp.close()
        except Exception:
            pass

@app.post("/api/backtest/run")
async def trigger_backtest(req: BacktestRequest, background_tasks: BackgroundTasks, admin = Depends(check_admin)):
    start_date = req.start_date or "2025-01-01"
    end_date = req.end_date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            if json.load(f).get("status") == "running":
                return {"status": "error", "message": "Already running"}

    background_tasks.add_task(run_backtest_task, start_date, end_date)
    return {"status": "success", "message": "Backtest started."}

@app.get("/api/backtest/status")
async def get_status(user = Depends(get_current_user)):
    if not STATUS_FILE.exists():
        return {"status": "idle"}
    with open(STATUS_FILE, "r") as f:
        data = json.load(f)

    # Auto-reset stale running status:
    # - if pid exists but is not running
    # - if no pid and updated_at is too old (legacy state)
    if data.get("status") == "running":
        pid = data.get("pid")
        if pid is not None:
            try:
                if not _is_pid_running(int(pid)):
                    _update_status("idle", "Stale status reset (process not running).", 0.0)
                    return {"status": "idle"}
            except Exception:
                pass
        else:
            updated_at = data.get("updated_at")
            if isinstance(updated_at, str):
                try:
                    last = datetime.fromisoformat(updated_at)
                    if datetime.now() - last > timedelta(minutes=30):
                        _update_status("idle", "Stale status reset (no pid heartbeat).", 0.0)
                        return {"status": "idle"}
                except Exception:
                    pass

    return data

@app.on_event("startup")
async def startup_event():
    create_users_table()
    add_user("admin", get_password_hash("admin@123"), "admin")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
