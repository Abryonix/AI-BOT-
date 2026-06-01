"""Canonical data validation for market, fundamental, options, and macro datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ai_trading_system.data.errors import DataValidationError


@dataclass(frozen=True)
class DataQualityReport:
    dataset: str
    rows: int
    columns: list[str]
    warnings: list[str]


class DataValidator:
    """Schema and quality checks for ingested dataframes."""

    OHLCV_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}
    FUNDAMENTAL_COLUMNS = {"date", "symbol"}
    OPTION_COLUMNS = {"date", "symbol", "option_type", "strike_price", "open_interest"}
    MACRO_COLUMNS = {"date"}

    def _require_columns(self, frame: pd.DataFrame, required: Iterable[str], dataset: str) -> None:
        missing = set(required).difference(frame.columns)
        if missing:
            raise DataValidationError(f"{dataset} missing required columns: {sorted(missing)}")

    def _require_non_empty(self, frame: pd.DataFrame, dataset: str) -> None:
        if frame.empty:
            raise DataValidationError(f"{dataset} is empty")

    def validate_ohlcv(self, frame: pd.DataFrame, dataset: str = "ohlcv") -> DataQualityReport:
        self._require_non_empty(frame, dataset)
        self._require_columns(frame, self.OHLCV_COLUMNS, dataset)
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        numeric = ["open", "high", "low", "close", "volume"]
        for column in numeric:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[numeric].isna().any().any():
            raise DataValidationError(f"{dataset} contains non-numeric OHLCV values")
        if (data[["open", "high", "low", "close"]] <= 0).any().any():
            raise DataValidationError(f"{dataset} contains non-positive prices")
        if (data["volume"] < 0).any():
            raise DataValidationError(f"{dataset} contains negative volume")
        if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
            raise DataValidationError(f"{dataset} contains high below open/close/low")
        if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
            raise DataValidationError(f"{dataset} contains low above open/close/high")
        duplicates = data.duplicated(["date", "symbol"]).sum()
        if duplicates:
            raise DataValidationError(f"{dataset} contains duplicate date/symbol rows: {duplicates}")
        warnings: list[str] = []
        expected_symbols = data["symbol"].nunique()
        counts = data.groupby("date")["symbol"].nunique()
        if not counts.empty and counts.min() < expected_symbols:
            warnings.append("Some dates have incomplete symbol coverage")
        return DataQualityReport(dataset, len(data), list(data.columns), warnings)

    def validate_fundamentals(self, frame: pd.DataFrame, dataset: str = "fundamentals") -> DataQualityReport:
        self._require_non_empty(frame, dataset)
        self._require_columns(frame, self.FUNDAMENTAL_COLUMNS, dataset)
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        if data["symbol"].isna().any():
            raise DataValidationError(f"{dataset} contains missing symbols")
        duplicates = data.duplicated(["date", "symbol"]).sum()
        if duplicates:
            raise DataValidationError(f"{dataset} contains duplicate date/symbol rows: {duplicates}")
        metric_cols = [c for c in data.columns if c not in {"date", "symbol", "as_of", "ingested_at"}]
        if not metric_cols:
            raise DataValidationError(f"{dataset} contains no metric columns")
        return DataQualityReport(dataset, len(data), list(data.columns), [])

    def validate_options(self, frame: pd.DataFrame, dataset: str = "options") -> DataQualityReport:
        self._require_non_empty(frame, dataset)
        self._require_columns(frame, self.OPTION_COLUMNS, dataset)
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        if not set(data["option_type"].dropna().unique()).issubset({"CE", "PE"}):
            raise DataValidationError(f"{dataset} contains invalid option_type values")
        for column in ("strike_price", "open_interest"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[["strike_price", "open_interest"]].isna().any().any():
            raise DataValidationError(f"{dataset} contains invalid strike/open-interest values")
        if (data["strike_price"] <= 0).any() or (data["open_interest"] < 0).any():
            raise DataValidationError(f"{dataset} contains non-positive strikes or negative open interest")
        return DataQualityReport(dataset, len(data), list(data.columns), [])

    def validate_macro(self, frame: pd.DataFrame, dataset: str = "macro") -> DataQualityReport:
        self._require_non_empty(frame, dataset)
        self._require_columns(frame, self.MACRO_COLUMNS, dataset)
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        metric_cols = [c for c in data.columns if c not in {"date", "as_of", "ingested_at"}]
        if not metric_cols:
            raise DataValidationError(f"{dataset} contains no macro metric columns")
        if data.duplicated(["date"]).any():
            raise DataValidationError(f"{dataset} contains duplicate dates")
        return DataQualityReport(dataset, len(data), list(data.columns), [])
