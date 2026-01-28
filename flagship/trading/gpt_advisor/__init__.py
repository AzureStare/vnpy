"""
GPT Advisor (intra-day technical re-rank + trading advice).

Design goals:
- Deterministic inputs (use AlphaLab minute -> resampled bars).
- Strict JSON outputs from LLM with validation.
- Produce artifacts for both execution and Ops Console UI.
"""

