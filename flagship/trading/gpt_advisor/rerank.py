from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.logger import logger

from flagship.trading.gpt_advisor.openai_client import chat_completions, get_openai_api_key
from flagship.trading.gpt_advisor.resample import DEFAULT_RESAMPLES, ResampleSpec, load_minute_bars, pack_bars_for_llm, resample_ohlcv


SYSTEM_PROMPT = """你是资深美股交易员与量化研究员。你会收到 50 只股票的：\n- 模型因子/特征快照（数值）\n- 30m/2h/4h 的 K 线摘要（最近固定根数）\n\n你的任务：\n1) 对这 50 只股票做二次排序，输出 gpt_rank=1..50（1最好）。\n2) 给出每只股票的 action: long / skip / short_text_only（系统实盘 long-only，short 只能作为文字提示）。\n3) 对 action=long 的股票给出：entry_limit, take_profit, stop_loss（美元价格）。\n4) 给出 confidence(0..1) 与 1-3 条原因 reasons（简短、可执行）。\n\n严格要求：\n- 只输出 JSON，不要输出解释文字。\n- JSON 必须符合给定 schema。缺字段视为失败。\n- 价格必须是正数；若无法判断，请 action=skip。\n"""


@dataclass(frozen=True)
class AdvisorConfig:
    model: str = "gpt-5.2"
    timeout_seconds: int = 60
    max_completion_tokens: int = 2000
    temperature: float = 0.1
    lookback_days: int = 14
    top_k: int = 50
    resamples: tuple[ResampleSpec, ...] = DEFAULT_RESAMPLES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in rows:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").strip().upper()
        if not sym or sym in seen:
            continue
        gpt_rank = int(it.get("gpt_rank") or 0)
        action = str(it.get("action") or "").strip()
        confidence = float(it.get("confidence") or 0.0)
        if gpt_rank <= 0:
            continue
        if action not in ("long", "skip", "short_text_only"):
            continue
        if not (0.0 <= confidence <= 1.0) or not math.isfinite(confidence):
            continue
        entry = it.get("entry_limit")
        tp = it.get("take_profit")
        sl = it.get("stop_loss")
        if action == "long":
            for v in (entry, tp, sl):
                if v is None:
                    raise ValueError(f"missing price field for long: {sym}")
                fv = float(v)
                if not (fv > 0 and math.isfinite(fv)):
                    raise ValueError(f"invalid price for long: {sym}")
        reasons = it.get("reasons")
        if reasons is None:
            reasons = []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        out.append(
            {
                "symbol": sym,
                "gpt_rank": gpt_rank,
                "action": action,
                "entry_limit": float(entry) if entry is not None else None,
                "take_profit": float(tp) if tp is not None else None,
                "stop_loss": float(sl) if sl is not None else None,
                "confidence": confidence,
                "reasons": [str(x) for x in reasons[:3]],
            }
        )
        seen.add(sym)
    out.sort(key=lambda r: int(r["gpt_rank"]))
    return out


def generate_gpt_advisor(
    *,
    project_root: Path,
    lab_path: Path,
    signal_path: Path,
    output_dir: Path,
    signal_date: date,
    cfg: AdvisorConfig | None = None,
) -> dict[str, Any] | None:
    """
    Generate GPT advisor output JSON + (later) optional signal override.
    Returns payload dict on success, else None (caller should fallback).
    """
    cfg = cfg or AdvisorConfig()

    api_key = get_openai_api_key(project_root=project_root)
    if not api_key:
        logger.warning("[gpt_advisor] OpenAI key missing; skip GPT advisor.")
        return None

    if not signal_path.exists():
        logger.warning(f"[gpt_advisor] signal not found: {signal_path}")
        return None

    signal_df = pl.read_parquet(signal_path)
    if signal_df.is_empty() or "vt_symbol" not in signal_df.columns:
        logger.warning("[gpt_advisor] signal parquet empty or missing vt_symbol")
        return None

    # Pick Top-K (exclude indices)
    df = signal_df
    if "signal" in df.columns:
        df = df.sort("signal", descending=True)
    df = df.filter(~pl.col("vt_symbol").is_in(["SPY.NASDAQ", "VIX.CBOE", "VIX3M.CBOE"]))
    df = df.head(int(cfg.top_k))
    vt_symbols = [str(v) for v in df["vt_symbol"].to_list()]
    if not vt_symbols:
        logger.warning("[gpt_advisor] no vt_symbols in Top-K")
        return None

    roots = [v.split(".")[0] for v in vt_symbols]

    lab = AlphaLab(str(lab_path))
    minute_df = load_minute_bars(lab=lab, vt_symbols=vt_symbols, end_date=signal_date, lookback_days=int(cfg.lookback_days))
    # Build resampled bars per spec
    bars_by_tf: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for spec in cfg.resamples:
        rs = resample_ohlcv(minute_df, spec=spec)
        packed: dict[str, list[dict[str, Any]]] = {}
        if not rs.is_empty():
            for sym, g in rs.group_by("vt_symbol", maintain_order=True):
                packed[str(sym)] = pack_bars_for_llm(g)
        bars_by_tf[spec.name] = packed

    # Build per-symbol feature snapshot (all columns except datetime/vt_symbol)
    feature_cols = [c for c in df.columns if c not in ("datetime", "vt_symbol")]
    features_by_symbol: dict[str, dict[str, Any]] = {}
    for row in df.select(["vt_symbol", *feature_cols]).iter_rows(named=True):
        vt = str(row["vt_symbol"])
        root = vt.split(".")[0]
        feat: dict[str, Any] = {}
        for c in feature_cols:
            v = row.get(c)
            # Keep numbers small and JSONable
            if v is None:
                continue
            try:
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    feat[c] = float(v)
                else:
                    # keep compact string form
                    s = str(v)
                    if len(s) <= 64:
                        feat[c] = s
            except Exception:
                continue
        features_by_symbol[root] = feat

    # Prompt payload (compact)
    payload_in = {
        "signal_date": signal_date.isoformat(),
        "universe": [
            {
                "symbol": r,
                "features": features_by_symbol.get(r, {}),
                "bars": {tf: bars_by_tf.get(tf, {}).get(vt, []) for tf, vt in [(s.name, f"{r}.NASDAQ") for s in cfg.resamples]},
            }
            for r in roots
        ],
        "schema": {
            "rows": [
                {
                    "symbol": "AAPL",
                    "gpt_rank": 1,
                    "action": "long",
                    "entry_limit": 0.0,
                    "take_profit": 0.0,
                    "stop_loss": 0.0,
                    "confidence": 0.0,
                    "reasons": ["..."],
                }
            ]
        },
    }

    user_prompt = json.dumps(payload_in, ensure_ascii=False)
    out = chat_completions(
        api_key=api_key,
        model=str(cfg.model),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        timeout_seconds=int(cfg.timeout_seconds),
        max_completion_tokens=int(cfg.max_completion_tokens),
        temperature=float(cfg.temperature),
    )
    if not out:
        logger.warning("[gpt_advisor] empty response from OpenAI")
        return None

    try:
        data = json.loads(out)
    except Exception as exc:
        logger.warning(f"[gpt_advisor] invalid JSON response: {exc}")
        return None

    rows = _validate_rows(data.get("rows"))

    payload_out = {
        "generated_at": _now_iso(),
        "model": str(cfg.model),
        "signal_date": signal_date.isoformat(),
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "gpt_advisor_latest.json"
    hist_dir = output_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist = hist_dir / f"gpt_advisor_{signal_date.strftime('%Y%m%d')}.json"
    latest.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    hist.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        latest.chmod(0o644)
        hist.chmod(0o644)
    except Exception:
        pass

    logger.info(f"[gpt_advisor] wrote {latest} and {hist} rows={len(rows)}")
    return payload_out

