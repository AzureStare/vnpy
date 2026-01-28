from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vnpy.trader.logger import logger


def get_openai_api_key(*, project_root: Path) -> str | None:
    """
    Align with existing repo conventions:
    - env: OPENAI_API_KEY
    - vt_setting.json: open-ai.api_key
    - fallback: vnpy.trader.setting.SETTINGS
    """
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()

    try:
        vt_path = project_root / "vt_setting.json"
        if vt_path.exists():
            data = json.loads(vt_path.read_text(encoding="utf-8"))
            k = data.get("open-ai.api_key")
            if isinstance(k, str) and k.strip():
                return k.strip()
    except Exception:
        pass

    try:
        from vnpy.trader.setting import SETTINGS
    except Exception:
        return None

    k2 = SETTINGS.get("open-ai.api_key")
    if isinstance(k2, str) and k2.strip():
        return k2.strip()
    return None


def chat_completions(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    max_completion_tokens: int,
    temperature: float = 0.2,
) -> str:
    """
    Stdlib OpenAI Chat Completions call (no extra deps).
    Matches repo usage (gpt-5.x uses max_completion_tokens).
    """
    import urllib.request

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_completion_tokens": int(max_completion_tokens),
    }

    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        try:
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.warning("[gpt_advisor] unexpected OpenAI response shape")
            return ""

