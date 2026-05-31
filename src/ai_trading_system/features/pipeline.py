from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_system.features.technical import add_technical_features


def add_market_features(features: pd.DataFrame, index_ohlcv: pd.DataFrame | None = None) -> pd.DataFrame:
    data = features.copy()
    if index_ohlcv is None:
        market = data.groupby("date")["return_1d"].mean().rename("nifty_return_proxy")
        data = data.merge(market.reset_index(), on="date", how="left")
    else:
        idx = index_ohlcv.sort_values("date").copy()
        idx["nifty_return"] = idx["close"].pct_change()
        idx["nifty_momentum_20"] = idx["close"].pct_change(20)
        idx["nifty_volatility_21"] = idx["nifty_return"].rolling(21).std() * np.sqrt(252)
        data = data.merge(idx[["date", "nifty_return", "nifty_momentum_20", "nifty_volatility_21"]], on="date", how="left")
    return data


def add_exogenous_features(
    features: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    options: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = features.copy()
    if fundamentals is not None and not fundamentals.empty:
        data = data.merge(fundamentals, on=["date", "symbol"], how="left")
    else:
        for col in ("pe", "pb", "roe", "roce", "revenue_growth", "earnings_growth", "debt_to_equity"):
            data[col] = np.nan
    if macro is not None and not macro.empty:
        data = data.merge(macro, on="date", how="left")
    else:
        for col in ("repo_rate", "inflation", "gdp_growth", "inr_usd", "bond_yield_proxy"):
            data[col] = np.nan
    if options is not None and not options.empty:
        data = data.merge(options, on="date", how="left")
    else:
        for col in ("put_call_ratio", "open_interest", "change_in_open_interest", "implied_volatility"):
            data[col] = np.nan
    return data


def add_targets(features: pd.DataFrame, horizons: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    data = features.sort_values(["symbol", "date"]).copy()
    for horizon in horizons:
        data[f"target_return_{horizon}d"] = data.groupby("symbol")["close"].shift(-horizon) / data["close"] - 1
        data[f"target_positive_{horizon}d"] = (data[f"target_return_{horizon}d"] > 0).astype(float)
        data[f"rank_target_{horizon}d"] = data.groupby("date")[f"target_return_{horizon}d"].rank(pct=True)
    return data


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    index_ohlcv: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    options: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = add_technical_features(ohlcv)
    features = add_market_features(features, index_ohlcv)
    features = add_exogenous_features(features, fundamentals, macro, options)
    return add_targets(features)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "symbol", "open", "high", "low", "close", "volume"}
    excluded |= {col for col in frame.columns if col.startswith("target_") or col.startswith("rank_target_")}
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    return [col for col in numeric if col not in excluded]
