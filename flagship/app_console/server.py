from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from jose import JWTError, jwt
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
import statistics
import json
import re
from pydantic import BaseModel
from typing import Optional, List
import hashlib

# Project imports
from flagship.config import PROJECT_ROOT
from flagship.universe.pg_ticker_db import (
    create_users_table, add_user, get_user, list_users, delete_user, get_pg_connection
)
from flagship.trading.controls import (
    create_trading_controls_tables,
    get_trading_controls_snapshot,
    set_buy_exposure_multiplier,
    set_disabled_vt_symbol,
)
from flagship.universe.daily_ranking_returns import ensure_daily_ranking_returns_table

# Configuration
SECRET_KEY = "flagship-ecc-style-secret-key" # In production use env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_users_table()
    create_trading_controls_tables()
    ensure_daily_ranking_returns_table()
    add_user("admin", get_password_hash("admin@123"), "admin")
    yield

app = FastAPI(lifespan=lifespan)

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

class DailyOpsRequest(BaseModel):
    """
    Manual trigger for the daily ops pipeline (daily_cycle_runner).
    """
    trading_date: Optional[str] = None  # YYYY-MM-DD (ET)
    strategy: str = "v7"

class ModelTrainRequest(BaseModel):
    """
    Manual trigger for model training only (train_daily_model).
    """
    trading_date: Optional[str] = None  # YYYY-MM-DD (ET)
    strategy: str = "v7"

class TradeRecapRegenerateRequest(BaseModel):
    """
    Manual regenerate trade recap HTML for a given ET trade_date.
    """
    trade_date: str  # YYYY-MM-DD (ET)
    strategy: str = "v7"

class TradingControlsResponse(BaseModel):
    disabled_vt_symbols: List[str]
    buy_exposure_multiplier: float

class TradingDisableRequest(BaseModel):
    vt_symbol: str
    disabled: bool = True

class TradingExposureRequest(BaseModel):
    multiplier: float

class AccountSummary(BaseModel):
    account_id: str
    display_name: str
    broker: str
    env: str
    status: str
    last_sync_utc: Optional[str] = None
    equity_usd: Optional[float] = None
    cash_usd: Optional[float] = None
    buying_power_usd: Optional[float] = None
    positions_count: int = 0
    open_orders_count: int = 0
    today_pnl_usd: Optional[float] = None
    data_base_path: str

class AccountsResponse(BaseModel):
    generated_at: str
    accounts: List[AccountSummary]

class AccountSnapshotResponse(BaseModel):
    account_id: str
    data_base_path: str
    portfolio: dict
    orders: dict
    performance: dict

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

APP_DATA_DIR = PROJECT_ROOT / "logs" / "app"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

SITE_DIR = PROJECT_ROOT / "flagship" / "app_console" / "site"

DAILY_OPS_DIR = APP_DATA_DIR / "daily_ops"
DAILY_OPS_DIR.mkdir(parents=True, exist_ok=True)
DAILY_OPS_STATUS_FILE = DAILY_OPS_DIR / "status.json"

MODEL_TRAIN_DIR = APP_DATA_DIR / "model_train"
MODEL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_TRAIN_STATUS_FILE = MODEL_TRAIN_DIR / "status.json"

REPORTING_DIR = APP_DATA_DIR / "reporting"
REPORTING_DIR.mkdir(parents=True, exist_ok=True)
TRADE_RECAP_REGEN_DIR = REPORTING_DIR / "trade_recap"
TRADE_RECAP_REGEN_DIR.mkdir(parents=True, exist_ok=True)
TRADE_RECAP_REGEN_STATUS_FILE = TRADE_RECAP_REGEN_DIR / "status.json"

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

def _load_json_file(path: Path) -> dict | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None

def _to_float(v: object) -> float | None:
    try:
        x = float(v)  # type: ignore[arg-type]
        return x if x == x else None
    except Exception:
        return None

def _count_open_orders(orders_payload: dict | None) -> int:
    if not orders_payload:
        return 0
    items = orders_payload.get("orders")
    if not isinstance(items, list):
        return 0
    cnt = 0
    for o in items:
        if not isinstance(o, dict):
            continue
        status = str(o.get("status") or "").lower()
        if status in ("new", "accepted", "pending_new", "submitted", "held", "partially_filled"):
            cnt += 1
    return cnt

def _build_account_summary(account_id: str, display_name: str, broker: str, env: str, base_dir: Path, data_base_path: str) -> AccountSummary | None:
    pf = _load_json_file(base_dir / "portfolio.json") or {}
    od = _load_json_file(base_dir / "orders.json") or {}
    pr = _load_json_file(base_dir / "performance.json") or {}

    acct = pf.get("account") if isinstance(pf.get("account"), dict) else {}
    positions = pf.get("positions") if isinstance(pf.get("positions"), list) else []
    last_sync = pf.get("generated_at") or od.get("generated_at") or pr.get("generated_at")
    status = str(acct.get("status") or "unknown") if isinstance(acct, dict) else "unknown"

    equity = _to_float(acct.get("equity") if isinstance(acct, dict) else None)
    cash = _to_float(acct.get("cash") if isinstance(acct, dict) else None)
    buying_power = _to_float(acct.get("buying_power") if isinstance(acct, dict) else None)

    # Best-effort: infer today's PnL from performance (last - prev)
    today_pnl: float | None = None
    series = pr.get("equity_series")
    if isinstance(series, list) and len(series) >= 2:
        try:
            last = series[-1]
            prev = series[-2]
            if isinstance(last, dict) and isinstance(prev, dict):
                last_eq = _to_float(last.get("equity"))
                prev_eq = _to_float(prev.get("equity"))
                if last_eq is not None and prev_eq is not None:
                    today_pnl = last_eq - prev_eq
        except Exception:
            today_pnl = None

    return AccountSummary(
        account_id=account_id,
        display_name=display_name,
        broker=broker,
        env=env,
        status=status,
        last_sync_utc=str(last_sync) if last_sync else None,
        equity_usd=equity,
        cash_usd=cash,
        buying_power_usd=buying_power,
        positions_count=len(positions) if isinstance(positions, list) else 0,
        open_orders_count=_count_open_orders(od),
        today_pnl_usd=today_pnl,
        data_base_path=data_base_path,
    )

def _load_accounts_snapshot() -> AccountsResponse:
    # Preferred: precomputed accounts.json (generated by app_console_snapshot.py)
    accounts_path = APP_DATA_DIR / "accounts.json"
    raw = _load_json_file(accounts_path)
    if raw and isinstance(raw.get("accounts"), list):
        items: list[AccountSummary] = []
        for it in raw.get("accounts", []):
            if not isinstance(it, dict):
                continue
            try:
                items.append(AccountSummary(**it))
            except Exception:
                continue
        gen = str(raw.get("generated_at") or datetime.utcnow().isoformat())
        if items:
            return AccountsResponse(generated_at=gen, accounts=items)

    # Fallback: build from default single-account snapshots (existing behavior)
    default_id = os.getenv("FLAGSHIP_DEFAULT_ACCOUNT_ID") or "alpaca_paper_main"
    default_name = os.getenv("FLAGSHIP_DEFAULT_ACCOUNT_NAME") or "Alpaca Paper (Main)"
    summary = _build_account_summary(
        account_id=str(default_id),
        display_name=str(default_name),
        broker="alpaca",
        env="paper",
        base_dir=APP_DATA_DIR,
        data_base_path="/data",
    )
    return AccountsResponse(
        generated_at=datetime.utcnow().isoformat(),
        accounts=[summary] if summary else [],
    )

def _load_account_dir(account_id: str) -> tuple[Path, str]:
    if account_id == (os.getenv("FLAGSHIP_DEFAULT_ACCOUNT_ID") or "alpaca_paper_main"):
        return APP_DATA_DIR, "/data"
    # Multi-account convention: logs/app/accounts/<account_id>/
    return APP_DATA_DIR / "accounts" / account_id, f"/data/accounts/{account_id}"

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

# Trading Controls (login required)
@app.get("/api/trading/controls", response_model=TradingControlsResponse)
async def get_trading_controls(user = Depends(get_current_user)):
    snap = get_trading_controls_snapshot()
    return {
        "disabled_vt_symbols": list(snap.disabled_vt_symbols),
        "buy_exposure_multiplier": float(snap.buy_exposure_multiplier),
    }

@app.post("/api/trading/controls/disabled")
async def set_trading_disabled(req: TradingDisableRequest, user = Depends(get_current_user)):
    updated_by = str(user.get("sub") or "")
    try:
        set_disabled_vt_symbol(vt_symbol=req.vt_symbol, disabled=bool(req.disabled), updated_by=updated_by or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success"}

@app.post("/api/trading/controls/exposure")
async def set_trading_exposure(req: TradingExposureRequest, user = Depends(get_current_user)):
    updated_by = str(user.get("sub") or "")
    try:
        m = set_buy_exposure_multiplier(multiplier=float(req.multiplier), updated_by=updated_by or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", "buy_exposure_multiplier": float(m)}

# Accounts (login required)
@app.get("/api/accounts", response_model=AccountsResponse)
async def get_accounts(user = Depends(get_current_user)):
    return _load_accounts_snapshot()

@app.get("/api/accounts/{account_id}/snapshot", response_model=AccountSnapshotResponse)
async def get_account_snapshot(account_id: str, user = Depends(get_current_user)):
    account_id = str(account_id or "").strip()
    if not account_id:
        raise HTTPException(status_code=400, detail="account_id is required")
    base_dir, base_path = _load_account_dir(account_id)
    pf = _load_json_file(base_dir / "portfolio.json") or {}
    od = _load_json_file(base_dir / "orders.json") or {}
    pr = _load_json_file(base_dir / "performance.json") or {}
    return {
        "account_id": account_id,
        "data_base_path": base_path,
        "portfolio": pf,
        "orders": od,
        "performance": pr,
    }


# Monitor (admin-only)
def _parse_date_param(raw: str, field: str) -> date:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field} must be YYYY-MM-DD")

def _get_ranking_returns_available_range() -> tuple[date | None, date | None]:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("select min(trade_date), max(trade_date) from daily_ranking_returns")
            row = cur.fetchone()
    if not row:
        return (None, None)
    a, b = row
    return (a, b)

def _get_latest_ranking_returns_date() -> date | None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("select max(trade_date) from daily_ranking_returns")
            row = cur.fetchone()
    if not row:
        return None
    return row[0]


def _fetch_ranking_returns_day(trade_date: date) -> dict:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, vt_symbol, rank_pos, signal_score, horizon_d, ret_type,
                       ret, excess_ret, close_t, spy_close_t
                FROM daily_ranking_returns
                WHERE trade_date = %s
                ORDER BY rank_pos ASC, vt_symbol ASC
                """,
                (trade_date,),
            )
            rows = cur.fetchall()

    by_symbol: dict[str, dict] = {}
    windows: set[int] = set()
    for (
        _trade_date,
        vt_symbol,
        rank_pos,
        signal_score,
        horizon_d,
        ret_type,
        ret,
        excess_ret,
        close_t,
        spy_close_t,
    ) in rows:
        sym = str(vt_symbol)
        windows.add(int(horizon_d))
        slot = by_symbol.setdefault(
            sym,
            {
                "vt_symbol": sym,
                "rank_pos": rank_pos,
                "signal_score": signal_score,
                "close_t": close_t,
                "spy_close_t": spy_close_t,
            },
        )
        key_ret = f"{ret_type}_ret_{int(horizon_d)}d"
        key_excess = f"{ret_type}_excess_{int(horizon_d)}d"
        slot[key_ret] = ret
        slot[key_excess] = excess_ret

    return {
        "trade_date": trade_date.isoformat(),
        "windows": sorted(windows),
        "rows": list(by_symbol.values()),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


@app.get("/api/monitor/ranking_returns/day")
async def get_ranking_returns_day(trade_date: str | None = None, admin = Depends(check_admin)):
    if trade_date is None or not str(trade_date).strip():
        latest = _get_latest_ranking_returns_date()
        if latest is None:
            a, b = _get_ranking_returns_available_range()
            return {"trade_date": None, "windows": [], "rows": [], "available_start": a, "available_end": b}
        payload = _fetch_ranking_returns_day(latest)
        a, b = _get_ranking_returns_available_range()
        payload["available_start"] = a.isoformat() if a else None
        payload["available_end"] = b.isoformat() if b else None
        return payload

    dt = _parse_date_param(str(trade_date), "trade_date")
    payload = _fetch_ranking_returns_day(dt)
    a, b = _get_ranking_returns_available_range()
    payload["available_start"] = a.isoformat() if a else None
    payload["available_end"] = b.isoformat() if b else None
    return payload


@app.get("/api/monitor/ranking_returns/range")
async def get_ranking_returns_range(
    start: str | None = None,
    end: str | None = None,
    bucket: int = 10,
    admin = Depends(check_admin),
):
    if start is None or not str(start).strip() or end is None or not str(end).strip():
        a, b = _get_ranking_returns_available_range()
        if a is None or b is None:
            return {"start": None, "end": None, "bucket_size": max(1, int(bucket)), "windows": [], "rows": [], "available_start": None, "available_end": None}
        start_dt = a
        end_dt = b
    else:
        start_dt = _parse_date_param(str(start), "start")
        end_dt = _parse_date_param(str(end), "end")
    bucket_size = max(1, int(bucket))

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, vt_symbol, rank_pos, horizon_d, ret_type, ret, excess_ret
                FROM daily_ranking_returns
                WHERE trade_date BETWEEN %s AND %s
                  AND rank_pos IS NOT NULL
                """,
                (start_dt, end_dt),
            )
            rows = cur.fetchall()

    grouped: dict[tuple[str, int, str, str], list[float]] = {}
    stats_map: dict[tuple[str, int, str, str], dict[str, int]] = {}
    windows: set[int] = set()

    for _trade_date, _vt_symbol, rank_pos, horizon_d, ret_type, ret, excess_ret in rows:
        if rank_pos is None:
            continue
        horizon = int(horizon_d)
        windows.add(horizon)
        bucket_idx = (int(rank_pos) - 1) // bucket_size
        bucket_label = f"{bucket_idx * bucket_size + 1}-{(bucket_idx + 1) * bucket_size}"

        for metric_name, value in (("ret", ret), ("excess", excess_ret)):
            key = (bucket_label, horizon, str(ret_type), metric_name)
            if value is None:
                continue
            grouped.setdefault(key, []).append(float(value))
            stats_map.setdefault(key, {"n": 0, "pos": 0})
            stats_map[key]["n"] += 1
            if float(value) > 0:
                stats_map[key]["pos"] += 1

    out_rows: list[dict] = []
    for key, values in grouped.items():
        bucket_label, horizon, ret_type, metric_name = key
        meta = stats_map.get(key) or {"n": 0, "pos": 0}
        n = int(meta["n"])
        pos = int(meta["pos"])
        win_rate = float(pos / n) if n else None
        out_rows.append(
            {
                "bucket": bucket_label,
                "horizon_d": horizon,
                "ret_type": ret_type,
                "metric": metric_name,
                "mean": _mean(values),
                "p50": _median(values),
                "win_rate": win_rate,
                "n": n,
            }
        )

    out_rows.sort(key=lambda r: (int(str(r["bucket"]).split("-")[0]), int(r["horizon_d"]), str(r["ret_type"]), str(r["metric"])))

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "bucket_size": bucket_size,
        "windows": sorted(windows),
        "rows": out_rows,
        "available_start": _get_ranking_returns_available_range()[0].isoformat() if _get_ranking_returns_available_range()[0] else None,
        "available_end": _get_ranking_returns_available_range()[1].isoformat() if _get_ranking_returns_available_range()[1] else None,
    }

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

def _update_daily_ops_status(
    status: str,
    message: str = "",
    progress: float = 0,
    pid: int | None = None,
    log_file: str | None = None,
):
    with open(DAILY_OPS_STATUS_FILE, "w") as f:
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

def _update_model_train_status(
    status: str,
    message: str = "",
    progress: float = 0,
    pid: int | None = None,
    log_file: str | None = None,
):
    with open(MODEL_TRAIN_STATUS_FILE, "w") as f:
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

def _update_trade_recap_regen_status(
    status: str,
    message: str = "",
    progress: float = 0,
    pid: int | None = None,
    log_file: str | None = None,
    trade_date: str | None = None,
):
    with open(TRADE_RECAP_REGEN_STATUS_FILE, "w") as f:
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
        if trade_date is not None:
            payload["trade_date"] = str(trade_date)
        json.dump(payload, f)

def run_daily_ops_task(strategy: str = "v7", trading_date: str | None = None):
    """
    Run daily_cycle_runner as a background task, writing logs into logs/app/daily_ops/.
    """
    log_path: Path | None = None
    log_fp = None
    log_rel: str | None = None
    try:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = DAILY_OPS_DIR / f"daily_ops_{run_ts}.log"
        log_fp = open(log_path, "w", encoding="utf-8")
        log_rel = f"daily_ops/{log_path.name}"
        log_fp.write(f"[{datetime.now().isoformat()}] Daily ops start: strategy={strategy} trading_date={trading_date or 'default'}\n")
        log_fp.flush()

        cmd: list[str] = [
            sys.executable,
            "-m",
            "flagship.trading.orchestration.daily_cycle_runner",
            "--strategy",
            str(strategy),
        ]
        if trading_date:
            cmd += ["--trading-date", str(trading_date)]

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _update_daily_ops_status(
            "running",
            "Daily ops running...",
            0.1,
            pid=proc.pid,
            log_file=log_rel,
        )

        rc = proc.wait()
        if rc == 0:
            _update_daily_ops_status("completed", "Daily ops completed.", 1.0, log_file=log_rel)
        else:
            _update_daily_ops_status("failed", f"Daily ops failed (code {rc}). See {log_rel}", 1.0, log_file=log_rel)
    except Exception as e:
        _update_daily_ops_status("failed", str(e), 1.0, log_file=log_rel)
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

def run_model_train_task(strategy: str = "v7", trading_date: str | None = None):
    """
    Run train_daily_model as a background task, writing logs into logs/app/model_train/.
    """
    log_path: Path | None = None
    log_fp = None
    log_rel: str | None = None
    try:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = MODEL_TRAIN_DIR / f"train_model_{run_ts}.log"
        log_fp = open(log_path, "w", encoding="utf-8")
        log_rel = f"model_train/{log_path.name}"
        log_fp.write(
            f"[{datetime.now().isoformat()}] Model train start: strategy={strategy} trading_date={trading_date or 'default'}\n"
        )
        log_fp.flush()

        cmd: list[str] = [
            sys.executable,
            "-m",
            "flagship.trading.orchestration.train_daily_model",
            "--strategy",
            str(strategy),
        ]
        if trading_date:
            cmd += ["--date", str(trading_date)]

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _update_model_train_status(
            "running",
            "Model training running...",
            0.1,
            pid=proc.pid,
            log_file=log_rel,
        )

        rc = proc.wait()
        if rc == 0:
            _update_model_train_status("completed", "Model training completed.", 1.0, log_file=log_rel)
        else:
            _update_model_train_status("failed", f"Model training failed (code {rc}). See {log_rel}", 1.0, log_file=log_rel)
    except Exception as e:
        _update_model_train_status("failed", str(e), 1.0, log_file=log_rel)
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

def run_trade_recap_regen_task(*, strategy: str, trade_date: str):
    """
    Regenerate trade recap HTML for a given ET trade_date, writing logs into logs/app/reporting/trade_recap/.
    """
    log_path: Path | None = None
    log_fp = None
    log_rel: str | None = None
    try:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_date = re.sub(r"[^0-9]", "", str(trade_date))[:8] or "unknown"
        log_path = TRADE_RECAP_REGEN_DIR / f"trade_recap_{safe_date}_{run_ts}.log"
        log_fp = open(log_path, "w", encoding="utf-8")
        log_rel = f"reporting/trade_recap/{log_path.name}"
        log_fp.write(f"[{datetime.now().isoformat()}] Trade recap regen start: strategy={strategy} trade_date={trade_date}\n")
        log_fp.flush()

        cmd: list[str] = [
            sys.executable,
            "-m",
            "flagship.ops.reporting.daily_trade_recap",
            "--trade-date",
            str(trade_date),
            "--output-dir",
            str(APP_DATA_DIR),
            "--log-dir",
            str(PROJECT_ROOT / "logs"),
            "--strategy",
            str(strategy),
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )

        _update_trade_recap_regen_status(
            "running",
            "Trade recap regeneration running...",
            0.1,
            pid=proc.pid,
            log_file=log_rel,
            trade_date=trade_date,
        )

        rc = proc.wait()
        if rc == 0:
            _update_trade_recap_regen_status("completed", "Trade recap regeneration completed.", 1.0, log_file=log_rel, trade_date=trade_date)
        else:
            _update_trade_recap_regen_status(
                "failed",
                f"Trade recap regeneration failed (code {rc}). See {log_rel}",
                1.0,
                log_file=log_rel,
                trade_date=trade_date,
            )
    except Exception as e:
        _update_trade_recap_regen_status("failed", str(e), 1.0, log_file=log_rel, trade_date=trade_date)
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
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "flagship.market_data.import_polygon_s3_data",
                        "--start",
                        data_start_date,
                        "--end",
                        end_date,
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            else:
                _update_status("running", f"Syncing market data via REST API (Slow)...", 0.1, log_file=log_path.name)
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "flagship.market_data.update_lab_data_incremental",
                        "--end-date",
                        end_date,
                        "--start-date",
                        data_start_date,
                        "--interval",
                        "daily",
                        "--universe",
                        "ref_tickers_cs",
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            
            subprocess.run(
                [
                    "python",
                    "-m",
                    "flagship.trading.orchestration.update_market_indices",
                    "--lookback",
                    "1200",
                ],
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
                "-m",
                "flagship.universe.build_daily_selection",
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

@app.post("/api/daily_ops/run")
async def trigger_daily_ops(req: DailyOpsRequest, background_tasks: BackgroundTasks, admin = Depends(check_admin)):
    strategy = (req.strategy or "v7").strip()
    if strategy not in ("v5", "v7"):
        raise HTTPException(status_code=400, detail="Invalid strategy (must be v5 or v7)")

    trading_date = (req.trading_date or "").strip() or None
    if trading_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", trading_date):
        raise HTTPException(status_code=400, detail="Invalid trading_date format (expected YYYY-MM-DD)")

    # Concurrency guard
    if DAILY_OPS_STATUS_FILE.exists():
        try:
            with open(DAILY_OPS_STATUS_FILE, "r") as f:
                data = json.load(f)
            if data.get("status") == "running":
                pid = data.get("pid")
                if pid is not None:
                    try:
                        if _is_pid_running(int(pid)):
                            return {"status": "error", "message": "Daily ops already running"}
                    except Exception:
                        return {"status": "error", "message": "Daily ops already running"}
                # stale running (no pid or pid not running) -> allow new trigger
        except Exception:
            pass

    background_tasks.add_task(run_daily_ops_task, strategy, trading_date)
    return {"status": "success", "message": "Daily ops started."}

@app.post("/api/model/train")
async def trigger_model_train(req: ModelTrainRequest, background_tasks: BackgroundTasks, admin = Depends(check_admin)):
    strategy = (req.strategy or "v7").strip()
    if strategy not in ("v5", "v7"):
        raise HTTPException(status_code=400, detail="Invalid strategy (must be v5 or v7)")

    trading_date = (req.trading_date or "").strip() or None
    if trading_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", trading_date):
        raise HTTPException(status_code=400, detail="Invalid trading_date format (expected YYYY-MM-DD)")

    # Concurrency guard
    if MODEL_TRAIN_STATUS_FILE.exists():
        try:
            with open(MODEL_TRAIN_STATUS_FILE, "r") as f:
                data = json.load(f)
            if data.get("status") == "running":
                pid = data.get("pid")
                if pid is not None:
                    try:
                        if _is_pid_running(int(pid)):
                            return {"status": "error", "message": "Model training already running"}
                    except Exception:
                        return {"status": "error", "message": "Model training already running"}
        except Exception:
            pass

    background_tasks.add_task(run_model_train_task, strategy, trading_date)
    return {"status": "success", "message": "Model training started."}

@app.post("/api/reports/trade_recap/regenerate")
async def regenerate_trade_recap(req: TradeRecapRegenerateRequest, background_tasks: BackgroundTasks, admin = Depends(check_admin)):
    strategy = (req.strategy or "v7").strip()
    if strategy not in ("v5", "v7"):
        raise HTTPException(status_code=400, detail="Invalid strategy (must be v5 or v7)")

    trade_date = str(req.trade_date or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
        raise HTTPException(status_code=400, detail="Invalid trade_date format (expected YYYY-MM-DD)")

    # Concurrency guard
    if TRADE_RECAP_REGEN_STATUS_FILE.exists():
        try:
            with open(TRADE_RECAP_REGEN_STATUS_FILE, "r") as f:
                data = json.load(f)
            if data.get("status") == "running":
                pid = data.get("pid")
                if pid is not None:
                    try:
                        if _is_pid_running(int(pid)):
                            return {"status": "error", "message": "Trade recap regeneration already running"}
                    except Exception:
                        return {"status": "error", "message": "Trade recap regeneration already running"}
        except Exception:
            pass

    background_tasks.add_task(run_trade_recap_regen_task, strategy=strategy, trade_date=trade_date)
    return {"status": "success", "message": "Trade recap regeneration started."}

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

@app.get("/api/daily_ops/status")
async def get_daily_ops_status(user = Depends(get_current_user)):
    if not DAILY_OPS_STATUS_FILE.exists():
        return {"status": "idle"}
    with open(DAILY_OPS_STATUS_FILE, "r") as f:
        data = json.load(f)

    if data.get("status") == "running":
        pid = data.get("pid")
        if pid is not None:
            try:
                if not _is_pid_running(int(pid)):
                    _update_daily_ops_status("idle", "Stale status reset (process not running).", 0.0)
                    return {"status": "idle"}
            except Exception:
                pass

    return data

@app.get("/api/model/train/status")
async def get_model_train_status(admin = Depends(check_admin)):
    if not MODEL_TRAIN_STATUS_FILE.exists():
        return {"status": "idle"}
    with open(MODEL_TRAIN_STATUS_FILE, "r") as f:
        data = json.load(f)

    if data.get("status") == "running":
        pid = data.get("pid")
        if pid is not None:
            try:
                if not _is_pid_running(int(pid)):
                    _update_model_train_status("idle", "Stale status reset (process not running).", 0.0)
                    return {"status": "idle"}
            except Exception:
                pass

    return data

@app.get("/api/reports/trade_recap/status")
async def get_trade_recap_regen_status(admin = Depends(check_admin)):
    if not TRADE_RECAP_REGEN_STATUS_FILE.exists():
        return {"status": "idle"}
    with open(TRADE_RECAP_REGEN_STATUS_FILE, "r") as f:
        data = json.load(f)

    if data.get("status") == "running":
        pid = data.get("pid")
        if pid is not None:
            try:
                if not _is_pid_running(int(pid)):
                    _update_trade_recap_regen_status("idle", "Stale status reset (process not running).", 0.0)
                    return {"status": "idle"}
            except Exception:
                pass

    return data

# Static assets (local-friendly):
# - /data/* serves logs/app snapshots (same as Caddy in prod)
# - / serves built React site + login.html
if APP_DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(APP_DATA_DIR), html=False), name="data")
if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
