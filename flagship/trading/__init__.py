"""
Trading domain package.

Split by business domains:
- execution: broker adapter + open rebalance + executor daemon
- intraday: intraday runner + daemon
- orchestration: post-market daily cycle pipeline
- realtime: Polygon WS helpers/caches
"""


