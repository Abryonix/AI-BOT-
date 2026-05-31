from __future__ import annotations

import numpy as np
import pandas as pd


def equity_metrics(equity: pd.Series, benchmark: pd.Series | None = None, periods_per_year: int = 252) -> dict[str, float]:
    equity = equity.dropna().astype(float)
    returns = equity.pct_change().dropna()
    if equity.empty or returns.empty:
        return {k: 0.0 for k in ("cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor", "alpha", "beta")}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, len(equity) / periods_per_year)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol else 0.0
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = (returns.mean() * periods_per_year) / downside if downside else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd else 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and losses.sum() else 0.0
    alpha = beta = 0.0
    if benchmark is not None:
        bench = benchmark.reindex(equity.index).pct_change().dropna()
        aligned = pd.concat([returns, bench], axis=1).dropna()
        if len(aligned) > 2 and aligned.iloc[:, 1].var() > 0:
            beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
            alpha = float((aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()) * periods_per_year)
    return {
        "cagr": float(cagr), "sharpe": float(sharpe), "sortino": float(sortino),
        "max_drawdown": max_dd, "calmar": float(calmar), "win_rate": float((returns > 0).mean()),
        "profit_factor": float(profit_factor), "alpha": alpha, "beta": beta,
    }


def composite_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or {"cagr": 0.25, "sharpe": 0.25, "sortino": 0.2, "max_drawdown": 0.2, "calmar": 0.1}
    drawdown_component = 1 + metrics.get("max_drawdown", 0.0)
    return float(
        weights.get("cagr", 0) * metrics.get("cagr", 0)
        + weights.get("sharpe", 0) * metrics.get("sharpe", 0)
        + weights.get("sortino", 0) * metrics.get("sortino", 0)
        + weights.get("max_drawdown", 0) * drawdown_component
        + weights.get("calmar", 0) * metrics.get("calmar", 0)
    )
