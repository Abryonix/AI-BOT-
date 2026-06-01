from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading_system.data.sources import synthetic_ohlcv
from ai_trading_system.features.pipeline import build_feature_matrix, model_feature_columns, prepare_options_features
from ai_trading_system.features.validation import FeatureValidationError, FeatureValidator, assert_asof_not_after_date


def _fundamentals(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "date": pd.Timestamp("2025-01-31"),
                "as_of": pd.Timestamp("2025-01-31"),
                "symbol": symbol,
                "pe": 20.0,
                "pb": 3.0,
                "roe": 0.18,
                "roce": 0.22,
                "revenue_growth": 0.09,
                "earnings_growth": 0.07,
                "debt_to_equity": 35.0,
            }
        )
    return pd.DataFrame(rows)


def _macro() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")],
            "as_of": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")],
            "repo_rate": [6.5, 6.25],
            "inflation": [5.1, 4.8],
            "gdp_growth": [6.6, 6.7],
            "inr_usd": [83.0, 83.5],
            "bond_yield_proxy": [7.1, 7.0],
        }
    )


def _options() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-02-01")] * 4,
            "as_of": [pd.Timestamp("2025-02-01")] * 4,
            "symbol": ["NIFTY"] * 4,
            "option_type": ["CE", "CE", "PE", "PE"],
            "strike_price": [22000, 22100, 22000, 21900],
            "open_interest": [1000, 500, 1500, 800],
            "change_in_open_interest": [100, -20, 140, 10],
            "implied_volatility": [12.0, 12.5, 14.0, 14.5],
        }
    )


def test_feature_matrix_contains_required_feature_families_and_no_leakage() -> None:
    symbols = ["AAA.NS", "BBB.NS"]
    ohlcv = synthetic_ohlcv(symbols, periods=320)
    index = synthetic_ohlcv(["^NSEI"], periods=320)
    features = build_feature_matrix(
        ohlcv,
        index_ohlcv=index,
        fundamentals=_fundamentals(symbols),
        macro=_macro(),
        options=_options(),
    )

    required = {
        # technical indicators
        "rsi_14",
        "macd",
        "adx_14",
        "cci_20",
        "stoch_k_14",
        # volatility features
        "realized_volatility_20",
        "rolling_volatility_20",
        "parkinson_vol_21",
        "garman_klass_vol_21",
        # volume features
        "volume_ratio_20",
        "relative_volume_20",
        "obv",
        "chaikin_money_flow_20",
        # fundamental features
        "earnings_yield",
        "book_to_price",
        "quality_score",
        "growth_score",
        # market regime features
        "nifty_return",
        "nifty_volatility_21",
        "nifty_drawdown",
        "risk_off_regime",
        "bull_regime",
        # options-chain features
        "put_call_ratio",
        "iv_skew_put_call",
        "option_oi_concentration",
        # targets
        "target_return_1d",
        "target_return_5d",
    }
    assert required.issubset(features.columns)
    report = FeatureValidator().validate(features)
    assert "close" not in report.feature_columns
    assert "target_return_1d" not in model_feature_columns(features)
    assert_asof_not_after_date(features)


def test_options_chain_aggregation_features() -> None:
    prepared = prepare_options_features(_options())
    row = prepared.iloc[0]
    assert row["put_call_ratio"] == pytest.approx(2300 / 1500)
    assert row["open_interest"] == 3800
    assert row["iv_skew_put_call"] == pytest.approx(2.0)
    assert 0 < row["option_oi_concentration"] <= 1


def test_feature_validator_catches_future_asof_leakage() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "symbol": ["AAA.NS"],
            "fundamental_as_of": [pd.Timestamp("2026-02-01")],
            "feature": [1.0],
            "target_return_1d": [0.01],
        }
    )
    with pytest.raises(FeatureValidationError, match="Leakage detected"):
        FeatureValidator().validate(frame)


def test_model_feature_columns_excludes_targets_raw_prices_and_asof_metadata() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "symbol": ["AAA.NS"],
            "close": [100.0],
            "volume": [1000],
            "feature_alpha": [0.2],
            "macro_as_of": [pd.Timestamp("2025-12-31")],
            "target_return_1d": [0.01],
            "rank_target_1d": [1.0],
        }
    )
    assert model_feature_columns(frame) == ["feature_alpha"]


def test_feature_validator_rejects_infinite_duplicate_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            "symbol": ["AAA.NS", "AAA.NS"],
            "feature_alpha": [1.0, np.inf],
            "target_return_1d": [0.01, 0.02],
        }
    )
    with pytest.raises(FeatureValidationError, match="duplicate"):
        FeatureValidator().validate(frame)
