from __future__ import annotations

import argparse

from ai_trading_system.backtests.engine import VectorizedBacktester
from ai_trading_system.config import load_settings
from ai_trading_system.data.sources import synthetic_ohlcv
from ai_trading_system.features.pipeline import build_feature_matrix
from ai_trading_system.models.ml import ensemble_predictions, train_predict_model
from ai_trading_system.portfolio.construction import ConfidenceAllocator, PortfolioConstraints


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    symbols = settings.raw["universe"]["constituents"][:5]
    data = synthetic_ohlcv(symbols)
    features = build_feature_matrix(data)
    cutoff = features["date"].quantile(0.75)
    result = train_predict_model(features, features["date"] <= cutoff, features["date"] > cutoff)
    print({"model": result.name, "task": result.task, "metrics": result.metrics})


def backtest_main() -> None:
    settings = load_settings()
    symbols = settings.raw["universe"]["constituents"][:5]
    prices = synthetic_ohlcv(symbols)
    features = build_feature_matrix(prices)
    cutoff = features["date"].quantile(0.75)
    result = train_predict_model(features, features["date"] <= cutoff, features["date"] > cutoff)
    predictions = ensemble_predictions([result])
    weights = []
    allocator = ConfidenceAllocator()
    constraints = PortfolioConstraints(capital=settings.initial_capital, max_names=settings.max_names)
    for date in predictions["date"].drop_duplicates():
        weights.append(allocator.allocate(predictions[predictions["date"] <= date], prices[prices["date"] <= date], constraints))
    bt = VectorizedBacktester().run(prices, __import__("pandas").concat(weights), settings.initial_capital)
    print(bt.metrics | {"composite_score": bt.score})


def evaluate_main() -> None:
    backtest_main()


def monitor_main() -> None:
    print("Run `streamlit run src/ai_trading_system/monitoring/dashboard.py` with exported artifacts.")
