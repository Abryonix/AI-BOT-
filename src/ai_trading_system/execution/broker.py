from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: str = "MARKET"
    product: str = "CNC"


class BrokerClient(Protocol):
    def place_order(self, order: Order) -> str: ...
    def positions(self) -> pd.DataFrame: ...


@dataclass
class PaperBroker:
    cash: float
    fills: list[Order] = field(default_factory=list)

    def place_order(self, order: Order) -> str:
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        self.fills.append(order)
        return f"PAPER-{len(self.fills):06d}"

    def positions(self) -> pd.DataFrame:
        rows = [{"symbol": o.symbol, "quantity": o.quantity if o.side == OrderSide.BUY else -o.quantity} for o in self.fills]
        return pd.DataFrame(rows).groupby("symbol", as_index=False).sum() if rows else pd.DataFrame(columns=["symbol", "quantity"])


@dataclass
class ZerodhaBroker:
    api_key: str
    access_token: str | None = None
    live_trading_enabled: bool = False

    def authenticate(self, request_token: str, api_secret: str) -> str:
        try:
            from kiteconnect import KiteConnect
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install kiteconnect for Zerodha integration.") from exc
        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        self.access_token = data["access_token"]
        return self.access_token

    def place_order(self, order: Order) -> str:
        if not self.live_trading_enabled:
            raise RuntimeError("Live trading is disabled. Enable explicitly after risk approval.")
        if not self.access_token:
            raise RuntimeError("Authenticate before placing live orders.")
        raise NotImplementedError("Wire KiteConnect.place_order in deployment-specific adapter.")

    def positions(self) -> pd.DataFrame:
        return pd.DataFrame()
