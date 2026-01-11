import numpy as np

from flagship.trading.signal_blend import blend_lgb_with_lr


def test_blend_only_adjusts_top_m() -> None:
    # lgb ranks: 0 > 1 > 2 > 3 > 4
    lgb = np.array([3.0, 2.0, 1.0, 0.0, -1.0], dtype=float)
    p_up = np.array([0.9, 0.1, 0.6, 0.4, 0.5], dtype=float)

    # top_n=1 -> top_m=2: only indices 0 and 1 should be adjusted
    out = blend_lgb_with_lr(lgb, p_up, top_n=1, top_m_multiplier=2, alpha=0.5, std_floor=1e-6)

    std = max(1e-6, float(np.std(lgb)))
    adj = 0.5 * std * (2.0 * p_up - 1.0)

    expected = lgb.copy()
    expected[0] = lgb[0] + adj[0]
    expected[1] = lgb[1] + adj[1]

    assert np.allclose(out, expected)
    assert np.allclose(out[2:], lgb[2:])


def test_blend_length_mismatch_returns_lgb() -> None:
    lgb = np.array([1.0, 2.0], dtype=float)
    p_up = np.array([0.6], dtype=float)
    out = blend_lgb_with_lr(lgb, p_up, top_n=1)
    assert np.allclose(out, lgb)

