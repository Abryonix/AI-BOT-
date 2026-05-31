from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_trading_system.features.pipeline import model_feature_columns


def _optional_model(kind: str, task: str, random_state: int) -> Any:
    if kind == "random_forest":
        cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
        return cls(n_estimators=120, max_depth=6, min_samples_leaf=20, random_state=random_state, n_jobs=-1)
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            return _optional_model("random_forest", task, random_state)
        cls = XGBRegressor if task == "regression" else XGBClassifier
        return cls(n_estimators=150, max_depth=3, learning_rate=0.04, subsample=0.8, random_state=random_state)
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError:
            return _optional_model("random_forest", task, random_state)
        cls = LGBMRegressor if task == "regression" else LGBMClassifier
        return cls(n_estimators=200, max_depth=4, learning_rate=0.035, random_state=random_state)
    if kind == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            return _optional_model("random_forest", task, random_state)
        cls = CatBoostRegressor if task == "regression" else CatBoostClassifier
        return cls(iterations=200, depth=4, learning_rate=0.04, verbose=False, random_seed=random_state)
    raise ValueError(f"Unknown model kind: {kind}")


@dataclass
class ModelResult:
    name: str
    task: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    estimator: Pipeline


class ModelFactory:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def build(self, kind: str, task: str) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if kind == "random_forest":
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", _optional_model(kind, task, self.random_state)))
        return Pipeline(steps)


def train_predict_model(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    kind: str = "random_forest",
    horizon: int = 1,
    task: str = "regression",
) -> ModelResult:
    feature_cols = model_feature_columns(frame)
    target = f"target_return_{horizon}d" if task == "regression" else f"target_positive_{horizon}d"
    clean = frame.dropna(subset=[target]).copy()
    train_idx = train_mask.reindex(clean.index, fill_value=False)
    test_idx = test_mask.reindex(clean.index, fill_value=False)
    estimator = ModelFactory().build(kind, task)
    estimator.fit(clean.loc[train_idx, feature_cols], clean.loc[train_idx, target])
    pred = estimator.predict(clean.loc[test_idx, feature_cols])
    predictions = clean.loc[test_idx, ["date", "symbol", target]].copy()
    score_col = f"pred_{target}"
    predictions[score_col] = pred
    metrics: dict[str, float]
    if task == "regression":
        metrics = {"rmse": float(mean_squared_error(predictions[target], pred, squared=False))}
    else:
        proba = estimator.predict_proba(clean.loc[test_idx, feature_cols])[:, 1] if hasattr(estimator[-1], "predict_proba") else pred
        predictions[f"prob_positive_{horizon}d"] = proba
        metrics = {"accuracy": float(accuracy_score(predictions[target], pred))}
        if len(np.unique(predictions[target])) > 1:
            metrics["auc"] = float(roc_auc_score(predictions[target], proba))
    predictions["rank_score"] = predictions.groupby("date")[score_col].rank(pct=True)
    return ModelResult(kind, task, metrics, predictions, estimator)


def ensemble_predictions(results: list[ModelResult]) -> pd.DataFrame:
    if not results:
        raise ValueError("At least one model result is required")
    merged: pd.DataFrame | None = None
    score_cols = []
    for idx, result in enumerate(results):
        pred_cols = [c for c in result.predictions.columns if c.startswith("pred_target_return")]
        if not pred_cols:
            continue
        col = f"score_{idx}_{result.name}"
        part = result.predictions[["date", "symbol", pred_cols[0]]].rename(columns={pred_cols[0]: col})
        score_cols.append(col)
        merged = part if merged is None else merged.merge(part, on=["date", "symbol"], how="outer")
    if merged is None:
        raise ValueError("No regression predictions available for ensembling")
    merged["ensemble_expected_return"] = merged[score_cols].mean(axis=1)
    merged["ensemble_rank"] = merged.groupby("date")["ensemble_expected_return"].rank(pct=True)
    return merged
