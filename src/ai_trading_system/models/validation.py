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


def walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex,
    train_days: int = 756,
    validation_days: int = 126,
    test_days: int = 63,
) -> list[WalkForwardSplit]:
    unique_dates = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
    splits: list[WalkForwardSplit] = []
    step = test_days
    total = train_days + validation_days + test_days
    for start in range(0, max(0, len(unique_dates) - total + 1), step):
        tr = unique_dates[start : start + train_days]
        va = unique_dates[start + train_days : start + train_days + validation_days]
        te = unique_dates[start + train_days + validation_days : start + total]
        if len(tr) and len(va) and len(te):
            splits.append(WalkForwardSplit(tr[0], tr[-1], va[0], va[-1], te[0], te[-1]))
    return splits
