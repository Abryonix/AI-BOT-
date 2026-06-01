from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    capital: float = 500_000
    max_names: int = 5
    max_gross_exposure: float = 0.75
    max_net_exposure: float = 0.50
    max_position_weight: float = 0.12
    min_cash_weight: float = 0.10
    allow_short: bool = True


class ConfidenceAllocator:
    """Learns exposure from the cross-sectional distribution instead of hardcoding thresholds."""

    def allocate(self, predictions: pd.DataFrame, prices: pd.DataFrame, constraints: PortfolioConstraints) -> pd.DataFrame:
        latest_date = predictions["date"].max()
        day = predictions[predictions["date"] == latest_date].copy()
        score_col = "ensemble_expected_return" if "ensemble_expected_return" in day else "expected_return"
        if score_col not in day:
            pred_cols = [c for c in day.columns if c.startswith("pred_target_return")]
            score_col = pred_cols[0]
        sigma = day[score_col].std(ddof=0)
        center = day[score_col].median()
        day["confidence"] = ((day[score_col] - center).abs() / (sigma if sigma else 1)).clip(0, 3) / 3
        candidates = day.sort_values(score_col, ascending=False).head(constraints.max_names).copy()
        if constraints.allow_short:
            shorts = day.sort_values(score_col).head(max(1, constraints.max_names // 2)).copy()
            shorts = shorts[shorts[score_col] < center]
            candidates = pd.concat([candidates, shorts]).drop_duplicates("symbol")
        candidates["side"] = np.where(candidates[score_col] >= center, 1, -1)
        raw = candidates["confidence"] * candidates["side"]
        if raw.abs().sum() == 0:
            candidates["target_weight"] = 0.0
        else:
            gross_budget = min(constraints.max_gross_exposure, 1 - constraints.min_cash_weight)
            candidates["target_weight"] = raw / raw.abs().sum() * gross_budget
            candidates["target_weight"] = candidates["target_weight"].clip(-constraints.max_position_weight, constraints.max_position_weight)
            net = candidates["target_weight"].sum()
            if abs(net) > constraints.max_net_exposure:
                candidates["target_weight"] *= constraints.max_net_exposure / abs(net)
        latest_prices = prices.sort_values("date").groupby("symbol").tail(1)[["symbol", "close"]]
        orders = candidates.merge(latest_prices, on="symbol", how="left")
        orders["target_value"] = orders["target_weight"] * constraints.capital
        orders["quantity"] = np.floor((orders["target_value"].abs() / orders["close"]).fillna(0)).astype(int) * np.sign(orders["target_weight"]).astype(int)
        invested = float((orders["quantity"].abs() * orders["close"]).sum())
        orders["cash_weight"] = max(0.0, 1 - invested / constraints.capital)
        return orders[["date", "symbol", score_col, "confidence", "side", "target_weight", "close", "quantity", "cash_weight"]]
