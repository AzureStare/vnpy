from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vnpy.trader.logger import logger

from flagship.trading.execution.broker_base import AccountInfo, BrokerAdapter, OrderSide


class IbkrImportError(RuntimeError):
    pass


def _require_ib_insync():
    try:
        from ib_insync import IB  # type: ignore
        from ib_insync import Stock  # type: ignore
        from ib_insync import MarketOrder, LimitOrder  # type: ignore
        from ib_insync import Trade  # type: ignore

        return IB, Stock, MarketOrder, LimitOrder, Trade
    except Exception as exc:  # pragma: no cover
        raise IbkrImportError(
            "IBKR adapter requires 'ib_insync'. Install optional deps: pip install '.[ibkr]' (or pip install ib_insync)."
        ) from exc


@dataclass(frozen=True)
class IbkrConnection:
    host: str
    port: int
    client_id: int
    account_id: str
    display_name: str
    paper: bool = True


class IbkrAdapter(BrokerAdapter):
    """
    IBKR adapter (IB Gateway/TWS) using ib_insync.

    Assumptions:
    - One adapter corresponds to one IB login (one account_id).
    - Market data may be delayed; get_last_trade_price is best-effort.
    """

    def __init__(self, conn: IbkrConnection) -> None:
        IB, _Stock, _MarketOrder, _LimitOrder, _Trade = _require_ib_insync()
        self._IB = IB
        self._Stock = _Stock
        self._MarketOrder = _MarketOrder
        self._LimitOrder = _LimitOrder
        self._conn = conn

        self.ib = IB()

        # IB enforces unique clientId per connected API session. In practice, restarts and parallel
        # services (snapshotter/executor) can cause "client id already in use" conflicts.
        # To make the system more robust, we auto-bump the clientId on error 326.
        requested_client_id = int(conn.client_id)
        effective_client_id = requested_client_id
        max_client_id_bumps = 10

        last_connect_saw_client_id_conflict = False

        def _on_error(reqId, errorCode, errorString, contract) -> None:  # type: ignore[no-untyped-def]
            nonlocal last_connect_saw_client_id_conflict
            try:
                if int(errorCode) == 326:
                    last_connect_saw_client_id_conflict = True
            except Exception:
                return

        # Subscribe before connecting so we can detect the 326 error during connect.
        try:
            self.ib.errorEvent += _on_error  # type: ignore[attr-defined]
        except Exception:
            # If ib_insync changes, we fall back to string matching later.
            pass

        connected = False
        last_exc: Exception | None = None
        for _ in range(max_client_id_bumps + 1):
            last_connect_saw_client_id_conflict = False
            try:
                self.ib.connect(str(conn.host), int(conn.port), clientId=int(effective_client_id))
                connected = bool(self.ib.isConnected())
                if connected:
                    break
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                is_client_id_conflict = (
                    last_connect_saw_client_id_conflict
                    or ("client id" in msg.lower() and "already" in msg.lower() and "use" in msg.lower())
                )
                if is_client_id_conflict:
                    try:
                        self.ib.disconnect()
                    except Exception:
                        pass
                    effective_client_id += 1
                    continue
                raise

        if not connected:
            raise RuntimeError(
                f"IBKR connect failed. account_id={conn.account_id} host={conn.host} port={conn.port} "
                f"client_id_start={requested_client_id} bumps={max_client_id_bumps} last_error={last_exc!r}"
            )

        logger.info(
            f"[IBKR] Connected. account_id={conn.account_id} host={conn.host} port={conn.port} "
            f"client_id={effective_client_id} (requested={requested_client_id})"
        )

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass

    # --- identity ---
    def get_account_id(self) -> str:
        return self._conn.account_id

    def get_display_name(self) -> str:
        return self._conn.display_name

    def get_broker_name(self) -> str:
        return "ibkr"

    def get_env_name(self) -> str:
        return "paper" if self._conn.paper else "live"

    # --- account ---
    def get_account_info(self) -> AccountInfo:
        """
        Map IBKR AccountSummary tags to our AccountInfo.
        """
        acct = self._conn.account_id
        summary = self.ib.accountSummary(acct)
        by_tag: dict[str, Any] = {}
        for it in summary:
            try:
                by_tag[str(getattr(it, "tag", ""))] = it
            except Exception:
                continue

        def _val(tag: str) -> float:
            it = by_tag.get(tag)
            v = getattr(it, "value", None) if it is not None else None
            try:
                return float(v)
            except Exception:
                return 0.0

        # Common tags:
        # - NetLiquidation
        # - TotalCashValue
        # - AvailableFunds (or BuyingPower in some accounts)
        equity = _val("NetLiquidation")
        cash = _val("TotalCashValue")
        buying_power = _val("AvailableFunds")
        if buying_power <= 0:
            buying_power = _val("BuyingPower")
        if buying_power <= 0:
            buying_power = cash

        return AccountInfo(cash=float(cash), equity=float(equity), buying_power=float(buying_power))

    def get_buying_power(self) -> float:
        return float(self.get_account_info().buying_power)

    def get_cash(self) -> float:
        return float(self.get_account_info().cash)

    # --- portfolio ---
    def get_positions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        acct = self._conn.account_id
        for p in self.ib.positions(account=acct):
            try:
                sym = str(p.contract.symbol)
                pos = int(float(p.position))
                if pos != 0:
                    out[sym] = pos
            except Exception:
                continue
        return out

    # --- orders ---
    def cancel_all_open_orders(self) -> None:
        try:
            self.ib.reqGlobalCancel()
            logger.info(f"[IBKR] Requested global cancel (account_id={self._conn.account_id})")
        except Exception as exc:
            logger.warning(f"[IBKR] cancel open orders failed: {exc}")

    def get_last_trade_price(self, symbol: str) -> float:
        """
        Best-effort last price using snapshot tickers.
        Requires market data permission; may fallback to close.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return 0.0
        try:
            contract = self._Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            tickers = self.ib.reqTickers(contract)
            if not tickers:
                return 0.0
            t = tickers[0]
            # Prefer marketPrice(), then last, then close
            px = None
            try:
                px = t.marketPrice()
            except Exception:
                px = None
            for cand in [px, getattr(t, "last", None), getattr(t, "close", None)]:
                try:
                    v = float(cand)
                    if v > 0:
                        return v
                except Exception:
                    continue
            return 0.0
        except Exception as exc:
            logger.warning(f"[IBKR] get_last_trade_price failed for {symbol}: {exc}")
            return 0.0

    def place_order(self, symbol: str, qty: int, side: OrderSide) -> None:
        symbol = str(symbol or "").strip().upper()
        if not symbol or qty <= 0:
            return

        try:
            contract = self._Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            action = "BUY" if side == OrderSide.BUY else "SELL"
            order = self._MarketOrder(action, int(qty))
            trade = self.ib.placeOrder(contract, order)
            # Let ib_insync process network events briefly so orderId is assigned
            self.ib.sleep(0.2)
            oid = getattr(getattr(trade, "order", None), "orderId", None)
            logger.info(f"[IBKR] Submitted {action} {qty} {symbol} (orderId={oid}) account_id={self._conn.account_id}")
        except Exception as exc:
            logger.error(f"[IBKR] place_order failed for {symbol}: {exc}")

