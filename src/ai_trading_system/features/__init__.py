"""Feature engineering and validation."""

from ai_trading_system.features.pipeline import (
    add_exogenous_features,
    add_market_regime_features,
    add_targets,
    build_feature_matrix,
    model_feature_columns,
    prepare_fundamental_features,
    prepare_macro_features,
    prepare_options_features,
)
from ai_trading_system.features.technical import add_technical_features
from ai_trading_system.features.validation import FeatureValidationError, FeatureValidator

__all__ = [
    "FeatureValidationError",
    "FeatureValidator",
    "add_exogenous_features",
    "add_market_regime_features",
    "add_targets",
    "add_technical_features",
    "build_feature_matrix",
    "model_feature_columns",
    "prepare_fundamental_features",
    "prepare_macro_features",
    "prepare_options_features",
]
