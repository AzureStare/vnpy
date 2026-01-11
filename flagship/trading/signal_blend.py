"""
Signal blending helpers.

We use a soft blend between:
- LightGBM LambdaRank raw score (lgb_signal)
- Logistic Regression meta-probability p_up = P(excess_return > 0)

The LR adjustment is only applied to the Top-M by lgb_signal, to keep the
ranking behavior close to the ranker while still gaining robustness.
"""

from __future__ import annotations

import numpy as np


def blend_lgb_with_lr(
    lgb_signal: np.ndarray,
    p_up: np.ndarray,
    *,
    top_n: int,
    top_m_multiplier: int = 5,
    alpha: float = 0.5,
    std_floor: float = 1e-6,
) -> np.ndarray:
    """
    Soft blend:
        final = lgb + alpha * std(lgb) * (2*p_up - 1)   (only on Top-M of lgb)

    Parameters
    ----------
    lgb_signal:
        Raw LightGBM scores for the current cross-section.
    p_up:
        LR predicted probability of positive excess return for each row.
    top_n:
        Strategy target holdings count (used to determine M=top_m_multiplier*top_n).
    top_m_multiplier:
        Multiplier to set M for two-stage constraint.
    alpha:
        Adjustment strength.
    std_floor:
        Lower bound for std(lgb) to avoid degenerate scaling.
    """
    lgb = np.asarray(lgb_signal, dtype=float)
    p = np.asarray(p_up, dtype=float)

    if lgb.size == 0:
        return lgb
    if p.size != lgb.size:
        return lgb

    top_m = max(1, int(top_m_multiplier) * int(top_n))
    top_m = min(int(top_m), int(lgb.size))

    order = np.argsort(-lgb)
    mask = np.zeros(int(lgb.size), dtype=bool)
    mask[order[:top_m]] = True

    std_lgb = float(np.std(lgb))
    std_lgb = max(float(std_floor), std_lgb)

    adj = float(alpha) * std_lgb * (2.0 * p - 1.0)
    out = lgb + np.where(mask, adj, 0.0)
    return out

