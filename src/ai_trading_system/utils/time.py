from __future__ import annotations

import pandas as pd


def ensure_datetime_index(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    data = frame.copy()
    if column in data.columns:
        data[column] = pd.to_datetime(data[column], utc=False)
        data = data.set_index(column)
    data.index = pd.to_datetime(data.index)
    return data.sort_index()
