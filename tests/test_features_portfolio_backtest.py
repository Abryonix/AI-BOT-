from __future__ import annotations

import pandas as pd

from ai_trading_system.backtests.engine import VectorizedBacktester
from ai_trading_system.data.sources import synthetic_ohlcv
from ai_trading_system.features.pipeline import build_feature_matrix, model_feature_columns
from ai_trading_system.models.ml import ensemble_predictions, train_predict_model
from ai_trading_system.portfolio.construction import ConfidenceAllocator, PortfolioConstraints


def test_feature_pipeline_generates_targets_without_leakage_columns() -> None:
    data = synthetic_ohlcv(["AAA.NS", "BBB.NS"], periods=260)
    features = build_feature_matrix(data)
    assert {"rsi_14", "macd", "atr_pct_14", "target_return_1d", "target_return_5d"}.issubset(features.columns)
    cols = model_feature_columns(features)
    assert "target_return_1d" not in cols
    assert "close" not in cols


def test_training_portfolio_and_backtest_smoke() -> None:
    prices = synthetic_ohlcv(["AAA.NS", "BBB.NS", "CCC.NS"], periods=320)
    features = build_feature_matrix(prices)
    cutoff = features["date"].quantile(0.7)
    result = train_predict_model(features, features["date"] <= cutoff, features["date"] > cutoff, kind="random_forest")
    predictions = ensemble_predictions([result])
    latest_weights = ConfidenceAllocator().allocate(
        predictions,
        prices,
        PortfolioConstraints(capital=100_000, max_names=2, max_position_weight=0.2),
    )
    assert latest_weights["quantity"].abs().sum() >= 0
    weights = latest_weights.assign(date=pd.to_datetime(latest_weights["date"]))
    backtest = VectorizedBacktester().run(prices, weights, initial_capital=100_000)
    assert "sharpe" in backtest.metrics
    assert backtest.equity.iloc[-1] > 0
