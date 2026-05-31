from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_trading_system.backtests.metrics import composite_score, equity_metrics


@dataclass(frozen=True)
class CostModel:
    brokerage_bps: float = 3
    slippage_bps: float = 5
    taxes_bps: float = 10

    @property
    def round_trip_rate(self) -> float:
        return (self.brokerage_bps + self.slippage_bps + self.taxes_bps) / 10_000


@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    metrics: dict[str, float]
    score: float
    trades: pd.DataFrame


class VectorizedBacktester:
    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel()

    def run(self, prices: pd.DataFrame, weights: pd.DataFrame, initial_capital: float = 500_000) -> BacktestResult:
        close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
        target = weights.pivot(index="date", columns="symbol", values="target_weight").reindex(close.index).fillna(0)
        asset_returns = close.pct_change().fillna(0)
        shifted = target.shift(1).fillna(0)
        turnover = target.diff().abs().sum(axis=1).fillna(target.abs().sum(axis=1))
        strategy_returns = (shifted * asset_returns).sum(axis=1) - turnover * self.cost_model.round_trip_rate
        equity = (1 + strategy_returns).cumprod() * initial_capital
        metrics = equity_metrics(equity)
        trades = target.stack().rename("target_weight").reset_index()
        return BacktestResult(equity, strategy_returns, metrics, composite_score(metrics), trades)
