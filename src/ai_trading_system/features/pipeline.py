from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_system.features.technical import add_technical_features
from ai_trading_system.features.validation import (
    FeatureValidator,
    assert_asof_not_after_date,
    model_feature_columns as _model_feature_columns,
)


def _sort_for_asof(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    columns = ["date"] + ([by] if by else [])
    return frame.sort_values(columns).reset_index(drop=True)


def prepare_fundamental_features(fundamentals: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize fundamental snapshots into point-in-time daily features."""
    if fundamentals is None or fundamentals.empty:
        return None
    data = fundamentals.copy()
    data["date"] = pd.to_datetime(data["date"])
    if "as_of" not in data.columns:
        data["as_of"] = data["date"]
    data["fundamental_as_of"] = pd.to_datetime(data["as_of"])
    metric_cols = [c for c in data.columns if c not in {"date", "symbol", "as_of", "ingested_at", "fundamental_as_of"}]
    for col in metric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if {"pe", "pb"}.issubset(data.columns):
        data["earnings_yield"] = 1 / data["pe"].replace(0, np.nan)
        data["book_to_price"] = 1 / data["pb"].replace(0, np.nan)
    if {"roe", "debt_to_equity"}.issubset(data.columns):
        data["quality_score"] = data["roe"] - data["debt_to_equity"].fillna(0) / 100
    if {"revenue_growth", "earnings_growth"}.issubset(data.columns):
        data["growth_score"] = data[["revenue_growth", "earnings_growth"]].mean(axis=1)
    return data.drop(columns=["as_of"], errors="ignore")


def prepare_macro_features(macro: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize macro series and add rate/inflation/growth derived features."""
    if macro is None or macro.empty:
        return None
    data = macro.copy()
    data["date"] = pd.to_datetime(data["date"])
    if "as_of" not in data.columns:
        data["as_of"] = data["date"]
    data["macro_as_of"] = pd.to_datetime(data["as_of"])
    metric_cols = [c for c in data.columns if c not in {"date", "as_of", "ingested_at", "macro_as_of"}]
    for col in metric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if {"repo_rate", "inflation"}.issubset(data.columns):
        data["real_policy_rate"] = data["repo_rate"] - data["inflation"]
    if {"gdp_growth", "inflation"}.issubset(data.columns):
        data["macro_growth_inflation_mix"] = data["gdp_growth"] - data["inflation"]
    if "inr_usd" in data.columns:
        data["inr_usd_change_21"] = data["inr_usd"].pct_change(1)
    if "bond_yield_proxy" in data.columns:
        data["bond_yield_change"] = data["bond_yield_proxy"].diff()
    return data.drop(columns=["as_of"], errors="ignore")


def prepare_options_features(options: pd.DataFrame | None) -> pd.DataFrame | None:
    """Aggregate current option-chain rows into market-level option sentiment features."""
    if options is None or options.empty:
        return None
    data = options.copy()
    data["date"] = pd.to_datetime(data["date"])
    if "as_of" not in data.columns:
        data["as_of"] = data["date"]
    data["options_as_of"] = pd.to_datetime(data["as_of"])
    for col in ("open_interest", "change_in_open_interest", "implied_volatility", "strike_price"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    grouped = []
    for date, day in data.groupby("date"):
        calls = day[day["option_type"].eq("CE")]
        puts = day[day["option_type"].eq("PE")]
        call_oi = calls["open_interest"].sum() if "open_interest" in day else np.nan
        put_oi = puts["open_interest"].sum() if "open_interest" in day else np.nan
        total_oi = call_oi + put_oi
        oi_by_strike = day.groupby("strike_price")["open_interest"].sum() if "strike_price" in day else pd.Series(dtype=float)
        grouped.append(
            {
                "date": date,
                "options_as_of": day["options_as_of"].max(),
                "put_call_ratio": put_oi / call_oi if call_oi else np.nan,
                "call_open_interest": call_oi,
                "put_open_interest": put_oi,
                "open_interest": total_oi,
                "change_in_open_interest": day.get("change_in_open_interest", pd.Series(dtype=float)).sum(),
                "implied_volatility": day.get("implied_volatility", pd.Series(dtype=float)).mean(),
                "call_implied_volatility": calls.get("implied_volatility", pd.Series(dtype=float)).mean(),
                "put_implied_volatility": puts.get("implied_volatility", pd.Series(dtype=float)).mean(),
                "iv_skew_put_call": puts.get("implied_volatility", pd.Series(dtype=float)).mean()
                - calls.get("implied_volatility", pd.Series(dtype=float)).mean(),
                "max_oi_strike": oi_by_strike.idxmax() if not oi_by_strike.empty else np.nan,
                "option_oi_concentration": oi_by_strike.max() / total_oi if total_oi and not oi_by_strike.empty else np.nan,
            }
        )
    return pd.DataFrame(grouped).sort_values("date").reset_index(drop=True)


def add_market_regime_features(features: pd.DataFrame, index_ohlcv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add Nifty return, trend, volatility, drawdown, and regime-state features."""
    data = features.copy()
    if index_ohlcv is None:
        market = data.groupby("date")["return_1d"].mean().rename("nifty_return_proxy").reset_index()
        market = market.sort_values("date")
        market["nifty_return"] = market["nifty_return_proxy"]
        market["nifty_momentum_20"] = market["nifty_return"].rolling(20).sum()
        market["nifty_volatility_21"] = market["nifty_return"].rolling(21).std() * np.sqrt(252)
        market["nifty_trend_50_200"] = np.nan
        market["nifty_drawdown"] = np.nan
    else:
        market = index_ohlcv.sort_values("date").copy()
        market["date"] = pd.to_datetime(market["date"])
        market["nifty_return"] = market["close"].pct_change()
        market["nifty_momentum_20"] = market["close"].pct_change(20)
        market["nifty_momentum_63"] = market["close"].pct_change(63)
        market["nifty_volatility_21"] = market["nifty_return"].rolling(21).std() * np.sqrt(252)
        market["nifty_volatility_63"] = market["nifty_return"].rolling(63).std() * np.sqrt(252)
        market["nifty_trend_50_200"] = market["close"].rolling(50).mean() / market["close"].rolling(200).mean() - 1
        market["nifty_drawdown"] = market["close"] / market["close"].cummax() - 1
        market = market[
            [
                "date",
                "nifty_return",
                "nifty_momentum_20",
                "nifty_momentum_63",
                "nifty_volatility_21",
                "nifty_volatility_63",
                "nifty_trend_50_200",
                "nifty_drawdown",
            ]
        ]
    market["market_volatility_percentile_252"] = market["nifty_volatility_21"].rolling(252, min_periods=20).rank(pct=True)
    market["risk_off_regime"] = (
        (market["nifty_drawdown"].fillna(0) < -0.08)
        | (market["market_volatility_percentile_252"].fillna(0) > 0.8)
    ).astype(float)
    market["bull_regime"] = ((market["nifty_momentum_20"].fillna(0) > 0) & (market["nifty_trend_50_200"].fillna(0) > 0)).astype(float)
    return data.merge(market, on="date", how="left")


def _merge_symbol_asof(base: pd.DataFrame, point_in_time: pd.DataFrame, asof_column: str) -> pd.DataFrame:
    pieces = []
    base = base.sort_values(["symbol", "date"]).copy()
    point_in_time = point_in_time.sort_values(["symbol", asof_column]).copy()
    for symbol, group in base.groupby("symbol", sort=False):
        right = point_in_time[point_in_time["symbol"].eq(symbol)]
        if right.empty:
            pieces.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("date"),
            right.sort_values(asof_column).drop(columns=["date"], errors="ignore"),
            left_on="date",
            right_on=asof_column,
            by="symbol",
            direction="backward",
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _merge_market_asof(base: pd.DataFrame, point_in_time: pd.DataFrame, asof_column: str) -> pd.DataFrame:
    return pd.merge_asof(
        base.sort_values("date"),
        point_in_time.sort_values(asof_column).drop(columns=["date"], errors="ignore"),
        left_on="date",
        right_on=asof_column,
        direction="backward",
    ).sort_values(["symbol", "date"]).reset_index(drop=True)


def add_exogenous_features(
    features: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    options: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = features.copy()
    fundamental_features = prepare_fundamental_features(fundamentals)
    if fundamental_features is not None:
        data = _merge_symbol_asof(data, fundamental_features, "fundamental_as_of")
    else:
        for col in (
            "pe",
            "pb",
            "roe",
            "roce",
            "revenue_growth",
            "earnings_growth",
            "debt_to_equity",
            "earnings_yield",
            "book_to_price",
            "quality_score",
            "growth_score",
        ):
            data[col] = np.nan
    macro_features = prepare_macro_features(macro)
    if macro_features is not None:
        data = _merge_market_asof(data, macro_features, "macro_as_of")
    else:
        for col in (
            "repo_rate",
            "inflation",
            "gdp_growth",
            "inr_usd",
            "bond_yield_proxy",
            "real_policy_rate",
            "macro_growth_inflation_mix",
            "inr_usd_change_21",
            "bond_yield_change",
        ):
            data[col] = np.nan
    option_features = prepare_options_features(options)
    if option_features is not None:
        data = _merge_market_asof(data, option_features, "options_as_of")
    else:
        for col in (
            "put_call_ratio",
            "call_open_interest",
            "put_open_interest",
            "open_interest",
            "change_in_open_interest",
            "implied_volatility",
            "call_implied_volatility",
            "put_implied_volatility",
            "iv_skew_put_call",
            "max_oi_strike",
            "option_oi_concentration",
        ):
            data[col] = np.nan
    assert_asof_not_after_date(data)
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
    validate: bool = True,
) -> pd.DataFrame:
    features = add_technical_features(ohlcv)
    features = add_market_regime_features(features, index_ohlcv)
    features = add_exogenous_features(features, fundamentals, macro, options)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = add_targets(features)
    if validate:
        FeatureValidator().validate(features)
    return features


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Backward-compatible wrapper for leakage-safe feature-column selection."""
    return _model_feature_columns(frame)
