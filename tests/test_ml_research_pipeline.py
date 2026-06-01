from __future__ import annotations

import pandas as pd

from ai_trading_system.data.sources import synthetic_ohlcv
from ai_trading_system.features.pipeline import build_feature_matrix
from ai_trading_system.models.ml import (
    SUPPORTED_MODEL_KINDS,
    SUPPORTED_TASKS,
    feature_importance,
    hyperparameter_search,
    run_walk_forward_validation,
    train_predict_model,
)
from ai_trading_system.models.validation import time_series_cv_splits, walk_forward_splits


def _features(periods: int = 360) -> pd.DataFrame:
    prices = synthetic_ohlcv(["AAA.NS", "BBB.NS", "CCC.NS"], periods=periods)
    return build_feature_matrix(prices)


def test_supported_model_families_and_tasks_are_declared() -> None:
    assert set(SUPPORTED_MODEL_KINDS) == {"xgboost", "lightgbm", "catboost", "random_forest"}
    assert set(SUPPORTED_TASKS) == {"regression", "classification", "ranking"}


def test_train_predict_supports_regression_classification_and_ranking() -> None:
    features = _features()
    cutoff = features["date"].quantile(0.70)
    train_mask = features["date"] <= cutoff
    test_mask = features["date"] > cutoff

    regression = train_predict_model(features, train_mask, test_mask, kind="random_forest", task="regression")
    classification = train_predict_model(features, train_mask, test_mask, kind="random_forest", task="classification")
    ranking = train_predict_model(features, train_mask, test_mask, kind="random_forest", task="ranking")

    assert "rmse" in regression.metrics
    assert "auc" in classification.metrics or "accuracy" in classification.metrics
    assert "rank_ic" in ranking.metrics
    assert "prob_positive_1d" in classification.predictions.columns
    assert "pred_rank_target_1d" in ranking.predictions.columns


def test_time_series_cv_and_hyperparameter_search() -> None:
    features = _features()
    splits = time_series_cv_splits(features["date"], n_splits=2, min_train_days=180, test_days=40)
    assert len(splits) == 2
    result = hyperparameter_search(
        features,
        kind="random_forest",
        task="regression",
        horizon=1,
        splits=splits[:1],
        param_grid=[{"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 5}],
    )
    assert result.best_params["n_estimators"] == 10
    assert not result.trials.empty


def test_walk_forward_research_pipeline_and_feature_importance() -> None:
    features = _features(420)
    splits = walk_forward_splits(features["date"], train_days=180, validation_days=40, test_days=30)
    assert splits
    run = run_walk_forward_validation(
        features,
        model_kinds=["random_forest"],
        tasks=["regression", "classification", "ranking"],
        horizons=[1],
        splits=splits[:2],
        tune=False,
    )
    assert set(run.leaderboard["task"]) == {"regression", "classification", "ranking"}
    assert not run.predictions.empty
    importance = feature_importance(run.results[0])
    assert {"feature", "importance"}.issubset(importance.columns)
    assert importance["importance"].ge(0).all()
