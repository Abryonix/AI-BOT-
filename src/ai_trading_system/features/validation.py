"""Feature validation and leakage-prevention checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class FeatureValidationError(ValueError):
    """Raised when a feature matrix violates schema, quality, or leakage rules."""


@dataclass(frozen=True)
class FeatureValidationReport:
    rows: int
    columns: int
    feature_columns: list[str]
    warnings: list[str]


TARGET_PREFIXES = ("target_", "rank_target_")
RAW_PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}
IDENTIFIER_COLUMNS = {"date", "symbol"}
AS_OF_COLUMNS = {"as_of", "fundamental_as_of", "macro_as_of", "options_as_of"}


def assert_asof_not_after_date(frame: pd.DataFrame, asof_columns: Iterable[str] = AS_OF_COLUMNS) -> None:
    """Ensure point-in-time source timestamps are not later than the feature date."""
    if "date" not in frame.columns:
        return
    dates = pd.to_datetime(frame["date"])
    for column in asof_columns:
        if column in frame.columns:
            asof = pd.to_datetime(frame[column])
            invalid = asof.notna() & (asof > dates)
            if invalid.any():
                raise FeatureValidationError(
                    f"Leakage detected: {column} is later than feature date for {int(invalid.sum())} rows"
                )


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric model features after excluding raw identifiers, prices, labels, and as-of metadata."""
    excluded = set(IDENTIFIER_COLUMNS) | set(RAW_PRICE_COLUMNS) | set(AS_OF_COLUMNS)
    excluded |= {col for col in frame.columns if col.startswith(TARGET_PREFIXES)}
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    columns = [col for col in numeric if col not in excluded]
    leakage_columns = [col for col in columns if col.startswith(TARGET_PREFIXES)]
    if leakage_columns:
        raise FeatureValidationError(f"Target leakage columns selected as features: {leakage_columns}")
    return columns


class FeatureValidator:
    """Validate final feature matrices before model training or backtesting."""

    def validate(self, frame: pd.DataFrame, require_targets: bool = True) -> FeatureValidationReport:
        if frame.empty:
            raise FeatureValidationError("Feature matrix is empty")
        missing = IDENTIFIER_COLUMNS.difference(frame.columns)
        if missing:
            raise FeatureValidationError(f"Feature matrix missing required columns: {sorted(missing)}")
        if frame.duplicated(["date", "symbol"]).any():
            raise FeatureValidationError("Feature matrix contains duplicate date/symbol rows")
        if require_targets and not any(col.startswith("target_return_") for col in frame.columns):
            raise FeatureValidationError("Feature matrix does not contain return targets")
        assert_asof_not_after_date(frame)
        feature_cols = model_feature_columns(frame)
        if not feature_cols:
            raise FeatureValidationError("No usable numeric feature columns remain after leakage filtering")
        numeric = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
        all_null = [col for col in feature_cols if numeric[col].isna().all()]
        warnings = []
        if all_null:
            warnings.append(f"All-null feature columns: {all_null}")
        return FeatureValidationReport(len(frame), len(frame.columns), feature_cols, warnings)
