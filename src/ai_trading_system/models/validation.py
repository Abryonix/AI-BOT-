from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.between(self.train_start, self.train_end)

    def validation_mask(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.between(self.validation_start, self.validation_end)

    def test_mask(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.between(self.test_start, self.test_end)

    def train_validation_mask(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.between(self.train_start, self.validation_end)


def walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex,
    train_days: int = 756,
    validation_days: int = 126,
    test_days: int = 63,
    step_days: int | None = None,
) -> list[WalkForwardSplit]:
    """Create anchored walk-forward train/validation/test windows from unique dates."""
    unique_dates = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
    splits: list[WalkForwardSplit] = []
    step = step_days or test_days
    total = train_days + validation_days + test_days
    for start in range(0, max(0, len(unique_dates) - total + 1), step):
        tr = unique_dates[start : start + train_days]
        va = unique_dates[start + train_days : start + train_days + validation_days]
        te = unique_dates[start + train_days + validation_days : start + total]
        if len(tr) and len(va) and len(te):
            splits.append(WalkForwardSplit(tr[0], tr[-1], va[0], va[-1], te[0], te[-1]))
    return splits


@dataclass(frozen=True)
class TimeSeriesSplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.between(self.train_start, self.train_end)

    def test_mask(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.between(self.test_start, self.test_end)


def time_series_cv_splits(
    dates: pd.Series | pd.DatetimeIndex,
    n_splits: int = 5,
    min_train_days: int = 252,
    test_days: int = 63,
    expanding: bool = True,
) -> list[TimeSeriesSplitWindow]:
    """Create chronological cross-validation windows with no future leakage."""
    unique_dates = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    splits: list[TimeSeriesSplitWindow] = []
    max_start = len(unique_dates) - test_days
    if max_start <= min_train_days:
        return splits
    candidate_starts = list(range(min_train_days, max_start + 1, test_days))[-n_splits:]
    for test_start_idx in candidate_starts:
        train_start_idx = 0 if expanding else max(0, test_start_idx - min_train_days)
        train = unique_dates[train_start_idx:test_start_idx]
        test = unique_dates[test_start_idx : test_start_idx + test_days]
        if len(train) and len(test):
            splits.append(TimeSeriesSplitWindow(train[0], train[-1], test[0], test[-1]))
    return splits
