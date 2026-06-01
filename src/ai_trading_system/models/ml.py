from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_trading_system.features.pipeline import model_feature_columns
from ai_trading_system.models.validation import TimeSeriesSplitWindow, WalkForwardSplit, time_series_cv_splits, walk_forward_splits

ModelKind = Literal["random_forest", "xgboost", "lightgbm", "catboost"]
Task = Literal["regression", "classification", "ranking"]
SUPPORTED_MODEL_KINDS: tuple[ModelKind, ...] = ("xgboost", "lightgbm", "catboost", "random_forest")
SUPPORTED_TASKS: tuple[Task, ...] = ("regression", "classification", "ranking")


def _with_params(estimator: Any, params: dict[str, Any] | None) -> Any:
    if params:
        estimator.set_params(**params)
    return estimator


def _optional_model(kind: str, task: str, random_state: int, params: dict[str, Any] | None = None) -> Any:
    is_classifier = task == "classification"
    if kind == "random_forest":
        cls = RandomForestClassifier if is_classifier else RandomForestRegressor
        defaults: dict[str, Any] = {"n_estimators": 120, "max_depth": 6, "min_samples_leaf": 20, "random_state": random_state, "n_jobs": -1}
        return cls(**(defaults | (params or {})))
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            return _optional_model("random_forest", task, random_state, params)
        cls = XGBClassifier if is_classifier else XGBRegressor
        defaults = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.8, "random_state": random_state}
        if is_classifier:
            defaults |= {"eval_metric": "logloss"}
        return cls(**(defaults | (params or {})))
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError:
            return _optional_model("random_forest", task, random_state, params)
        cls = LGBMClassifier if is_classifier else LGBMRegressor
        defaults = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.035, "random_state": random_state, "verbosity": -1}
        return cls(**(defaults | (params or {})))
    if kind == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            return _optional_model("random_forest", task, random_state, params)
        cls = CatBoostClassifier if is_classifier else CatBoostRegressor
        defaults = {"iterations": 200, "depth": 4, "learning_rate": 0.04, "verbose": False, "random_seed": random_state}
        return cls(**(defaults | (params or {})))
    raise ValueError(f"Unknown model kind: {kind}")


@dataclass
class ModelResult:
    name: str
    task: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    estimator: Pipeline
    feature_columns: list[str] = field(default_factory=list)
    horizon: int = 1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HyperparameterSearchResult:
    best_params: dict[str, Any]
    best_score: float
    trials: pd.DataFrame


@dataclass
class ResearchRunResult:
    results: list[ModelResult]
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame


class ModelFactory:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def build(self, kind: str, task: str, params: dict[str, Any] | None = None) -> Pipeline:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if kind == "random_forest":
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", _optional_model(kind, task, self.random_state, params)))
        return Pipeline(steps)


def target_column(task: str, horizon: int) -> str:
    if task == "regression":
        return f"target_return_{horizon}d"
    if task == "classification":
        return f"target_positive_{horizon}d"
    if task == "ranking":
        return f"rank_target_{horizon}d"
    raise ValueError(f"Unsupported task: {task}")


def _prediction_column(task: str, horizon: int) -> str:
    if task == "regression":
        return f"pred_target_return_{horizon}d"
    if task == "classification":
        return f"pred_target_positive_{horizon}d"
    return f"pred_rank_target_{horizon}d"


def _rank_ic(predictions: pd.DataFrame, target: str, score_col: str) -> float:
    daily = []
    for _, day in predictions.groupby("date"):
        if day[target].nunique(dropna=True) > 1 and day[score_col].nunique(dropna=True) > 1:
            daily.append(day[target].corr(day[score_col], method="spearman"))
    values = pd.Series(daily, dtype=float).dropna()
    return float(values.mean()) if not values.empty else 0.0


def _classification_probability(estimator: Pipeline, x_test: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    model = estimator[-1]
    if hasattr(model, "predict_proba"):
        return estimator.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        raw = estimator.decision_function(x_test)
        return 1 / (1 + np.exp(-raw))
    return pred.astype(float)


def evaluate_predictions(predictions: pd.DataFrame, target: str, score_col: str, task: str) -> dict[str, float]:
    y_true = predictions[target]
    y_pred = predictions[score_col]
    if task == "classification":
        metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
        proba_col = next((c for c in predictions.columns if c.startswith("prob_positive_")), None)
        if proba_col:
            metrics["brier"] = float(brier_score_loss(y_true, predictions[proba_col]))
            if len(np.unique(y_true)) > 1:
                metrics["auc"] = float(roc_auc_score(y_true, predictions[proba_col]))
        return metrics
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rank_ic": _rank_ic(predictions, target, score_col),
    }
    if task == "ranking":
        metrics["top_decile_mean_target"] = float(predictions.loc[predictions[score_col] >= predictions.groupby("date")[score_col].transform(lambda s: s.quantile(0.9)), target].mean())
    return metrics


def train_predict_model(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    kind: str = "random_forest",
    horizon: int = 1,
    task: str = "regression",
    params: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
) -> ModelResult:
    feature_cols = feature_columns or model_feature_columns(frame)
    target = target_column(task, horizon)
    clean = frame.dropna(subset=[target]).copy()
    clean[feature_cols] = clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    train_idx = train_mask.reindex(clean.index, fill_value=False)
    test_idx = test_mask.reindex(clean.index, fill_value=False)
    if not train_idx.any() or not test_idx.any():
        raise ValueError("Train and test masks must each select at least one row")
    estimator = ModelFactory().build(kind, task, params)
    y_train = clean.loc[train_idx, target]
    if task == "classification":
        y_train = y_train.astype(int)
    estimator.fit(clean.loc[train_idx, feature_cols], y_train)
    x_test = clean.loc[test_idx, feature_cols]
    pred = estimator.predict(x_test)
    predictions = clean.loc[test_idx, ["date", "symbol", target]].copy()
    score_col = _prediction_column(task, horizon)
    predictions[score_col] = pred
    if task == "classification":
        proba = _classification_probability(estimator, x_test, pred)
        predictions[f"prob_positive_{horizon}d"] = proba
    predictions["rank_score"] = predictions.groupby("date")[score_col].rank(pct=True)
    metrics = evaluate_predictions(predictions, target, score_col, task)
    return ModelResult(kind, task, metrics, predictions, estimator, feature_cols, horizon, params or {})


def default_param_grid(kind: str, task: str) -> list[dict[str, Any]]:
    if kind == "random_forest":
        return [
            {"n_estimators": 40, "max_depth": 3, "min_samples_leaf": 10},
            {"n_estimators": 80, "max_depth": 5, "min_samples_leaf": 10},
        ]
    if kind in {"xgboost", "lightgbm"}:
        return [
            {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05},
            {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.03},
        ]
    if kind == "catboost":
        return [
            {"iterations": 50, "depth": 2, "learning_rate": 0.05},
            {"iterations": 80, "depth": 3, "learning_rate": 0.03},
        ]
    return [{}]


def expand_param_grid(grid: dict[str, list[Any]] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if grid is None:
        return [{}]
    if isinstance(grid, list):
        return grid
    keys = list(grid)
    return [dict(zip(keys, values, strict=False)) for values in product(*(grid[key] for key in keys))]


def objective_score(metrics: dict[str, float], task: str) -> float:
    if task == "classification":
        return metrics.get("auc", metrics.get("accuracy", 0.0)) - metrics.get("brier", 0.0)
    if task == "ranking":
        return metrics.get("rank_ic", 0.0)
    return -metrics.get("rmse", np.inf)


def hyperparameter_search(
    frame: pd.DataFrame,
    kind: str,
    task: str,
    horizon: int,
    splits: list[TimeSeriesSplitWindow] | None = None,
    param_grid: dict[str, list[Any]] | list[dict[str, Any]] | None = None,
) -> HyperparameterSearchResult:
    """Grid-search model parameters over chronological CV windows."""
    cv_splits = splits or time_series_cv_splits(frame["date"], n_splits=3, min_train_days=126, test_days=42)
    params_list = expand_param_grid(param_grid or default_param_grid(kind, task))
    trials: list[dict[str, Any]] = []
    best_params: dict[str, Any] = params_list[0] if params_list else {}
    best_score = -np.inf
    for params in params_list:
        scores = []
        for split in cv_splits:
            result = train_predict_model(
                frame,
                split.train_mask(frame["date"]),
                split.test_mask(frame["date"]),
                kind=kind,
                horizon=horizon,
                task=task,
                params=params,
            )
            scores.append(objective_score(result.metrics, task))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        row = {"kind": kind, "task": task, "horizon": horizon, "score": mean_score, **params}
        trials.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return HyperparameterSearchResult(best_params, best_score, pd.DataFrame(trials).sort_values("score", ascending=False).reset_index(drop=True))


def feature_importance(result: ModelResult) -> pd.DataFrame:
    """Extract normalized feature importances from fitted tree models when available."""
    model = result.estimator[-1]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        values = np.zeros(len(result.feature_columns), dtype=float)
    if values.size != len(result.feature_columns):
        values = np.resize(values, len(result.feature_columns))
    total = values.sum()
    normalized = values / total if total else values
    return pd.DataFrame({"feature": result.feature_columns, "importance": normalized}).sort_values("importance", ascending=False).reset_index(drop=True)


def run_walk_forward_validation(
    frame: pd.DataFrame,
    model_kinds: list[str] | None = None,
    tasks: list[str] | None = None,
    horizons: list[int] | None = None,
    splits: list[WalkForwardSplit] | None = None,
    tune: bool = False,
) -> ResearchRunResult:
    """Train/evaluate model families, tasks, and horizons across walk-forward windows."""
    kinds = model_kinds or list(SUPPORTED_MODEL_KINDS)
    selected_tasks = tasks or list(SUPPORTED_TASKS)
    selected_horizons = horizons or [1, 5]
    wf_splits = splits or walk_forward_splits(frame["date"], train_days=252, validation_days=63, test_days=42)
    results: list[ModelResult] = []
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for kind in kinds:
        for task in selected_tasks:
            for horizon in selected_horizons:
                params: dict[str, Any] = {}
                if tune:
                    cv = [TimeSeriesSplitWindow(s.train_start, s.train_end, s.validation_start, s.validation_end) for s in wf_splits]
                    params = hyperparameter_search(frame, kind, task, horizon, splits=cv).best_params
                fold_metrics: list[dict[str, float]] = []
                fold_predictions: list[pd.DataFrame] = []
                last_result: ModelResult | None = None
                for split_id, split in enumerate(wf_splits):
                    result = train_predict_model(
                        frame,
                        split.train_validation_mask(frame["date"]),
                        split.test_mask(frame["date"]),
                        kind=kind,
                        horizon=horizon,
                        task=task,
                        params=params,
                    )
                    pred = result.predictions.copy()
                    pred["model"] = kind
                    pred["task"] = task
                    pred["horizon"] = horizon
                    pred["split"] = split_id
                    fold_predictions.append(pred)
                    fold_metrics.append(result.metrics)
                    last_result = result
                if not fold_metrics or last_result is None:
                    continue
                aggregate = {key: float(np.nanmean([m.get(key, np.nan) for m in fold_metrics])) for key in set().union(*fold_metrics)}
                combined_predictions = pd.concat(fold_predictions, ignore_index=True)
                model_result = ModelResult(kind, task, aggregate, combined_predictions, last_result.estimator, last_result.feature_columns, horizon, params)
                results.append(model_result)
                prediction_frames.append(combined_predictions)
                rows.append({"model": kind, "task": task, "horizon": horizon, **aggregate, "objective_score": objective_score(aggregate, task)})
    leaderboard = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return ResearchRunResult(results, leaderboard, predictions)


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
