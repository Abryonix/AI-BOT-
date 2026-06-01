from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_daily_loss_pct: float = 0.01
    max_drawdown_pct: float = 0.08
    max_order_notional_pct: float = 0.12
    kill_switch_enabled: bool = True


class PreTradeRiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def validate_orders(self, orders: pd.DataFrame, equity: float, current_drawdown: float = 0.0) -> None:
        if self.limits.kill_switch_enabled and abs(current_drawdown) >= self.limits.max_drawdown_pct:
            raise RuntimeError("Kill switch active: drawdown limit breached")
        max_notional = equity * self.limits.max_order_notional_pct
        if "target_value" in orders and orders["target_value"].abs().max() > max_notional:
            raise RuntimeError("Order breaches max notional limit")
