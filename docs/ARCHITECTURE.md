# Architecture

The system is split into isolated modules so research code can graduate into production without coupling model training, portfolio construction, execution, and monitoring.

## Data Layer

`data.sources` contains free adapters for Yahoo Finance OHLCV, NSE option-chain data, and an immutable parquet data lake. The NSE adapter is isolated because public endpoints can change and should be swapped for a licensed feed in production.

## Feature Layer

`features.technical` creates technical, volatility, and volume features. `features.pipeline` merges market, fundamental, macro, and options features and creates 1-day/5-day return, direction, and ranking targets. `model_feature_columns` explicitly excludes prices, identifiers, and target columns to reduce leakage risk.

## Model Layer

`models.ml` supports Random Forest natively and XGBoost/LightGBM/CatBoost when installed. Optional libraries gracefully fall back for lightweight CI. `models.validation` provides walk-forward windows for time-series validation.

## RL Layer

`rl.environment.NiftyTradingEnv` exposes six actions: hold, long, short, increase, reduce, and cash. Rewards include return, transaction-cost, and drawdown terms so cash decisions and confidence thresholds are learned rather than hardcoded.

## Portfolio Layer

`portfolio.construction.ConfidenceAllocator` turns ranked expected returns into signed weights, cash, and integer share quantities. Exposure scales with the cross-sectional dispersion of expected returns, which reduces trading in low-confidence regimes.

## Backtesting Layer

`backtests.engine.VectorizedBacktester` evaluates target weights with slippage, brokerage, and tax assumptions. `backtests.metrics` computes institutional metrics and a composite score for model selection.

## Execution Layer

`execution.broker` defines a broker protocol, a paper broker, and a Zerodha-compatible adapter that refuses live orders unless explicitly enabled. `execution.risk` applies pre-trade kill-switch and notional checks.

## Monitoring and Reporting

`monitoring.dashboard` creates performance, prediction-quality, drift-ready, and exposure snapshots. `reports.reporting` writes Markdown reports and model ranking tables.
