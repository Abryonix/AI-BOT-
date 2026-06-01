# Nifty 50 Institutional AI Trading System

Production-oriented research platform for AI-driven swing trading in Indian equities. The codebase is designed for conservative capital preservation, walk-forward experimentation, ML-vs-RL comparison, portfolio construction, backtesting, broker integration, and monitoring.

> Educational/research software only. Live trading is disabled by default and must be enabled explicitly after broker, compliance, and risk review.

## Capabilities

- Nifty 50 universe configuration with free-data adapters for Yahoo Finance, NSE public options chain, CSV/parquet data lake, and placeholders for RBI/MOSPI/World Bank macro feeds.
- Feature engineering for RSI, MACD, ATR, Bollinger Bands, ADX, CCI, stochastic oscillator, momentum, ROC, moving averages, realized volatility, Parkinson/Garman-Klass volatility, volume ratios, OBV, accumulation/distribution, market, fundamental, macro, and options features.
- Targets for next-day return, next-5-day return, positive-return probability, and cross-sectional ranking.
- ML model framework for XGBoost, LightGBM, CatBoost, and Random Forest with leakage-aware feature selection and walk-forward validation utilities.
- RL environment for long/short/hold/increase/reduce/cash actions and risk-adjusted rewards with drawdown, volatility, turnover, and transaction-cost penalties.
- AI-driven portfolio allocator producing top opportunities, signed weights, cash allocation, and real share quantities suitable for small accounts.
- Vectorized backtester with slippage/brokerage/taxes, CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, alpha, beta, and composite scoring.
- Zerodha-compatible broker abstraction with paper trading default and live order kill-switch protections.
- Monitoring snapshots and Streamlit-ready dashboard entry point.

## Repository Structure

```text
config/                       Runtime configuration
src/ai_trading_system/data/    Free data adapters and data lake
src/ai_trading_system/features/Feature pipeline and targets
src/ai_trading_system/models/  ML training, ensembling, validation
src/ai_trading_system/rl/      Gymnasium trading environment and SB3 training
src/ai_trading_system/backtests/Backtest engine and metrics
src/ai_trading_system/portfolio/Portfolio construction
src/ai_trading_system/execution/Broker and risk controls
src/ai_trading_system/monitoring/Dashboard and health snapshots
src/ai_trading_system/reports/Report generation
tests/                         Smoke and regression tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ai-trade-train --config config/config.yaml
ai-trade-backtest
```

Install optional research extras as needed:

```bash
pip install -e '.[all]'
```


## How to Run

For a complete command-by-command guide, see [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md). The shortest offline smoke run is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
ai-trade-train --config config/config.yaml
ai-trade-backtest
```

The demo uses deterministic synthetic OHLCV data, so it can validate the training/backtesting pipeline before real market-data downloads are configured.

## Data Sources

The platform only assumes free sources:

1. OHLCV and Nifty index data: Yahoo Finance via optional `yfinance`.
2. Current options chain: NSE public option-chain endpoint.
3. Fundamentals: Yahoo Finance fields or user-provided Screener/Tijori exports loaded into the data lake.
4. Macro: RBI, MOSPI, World Bank, FRED, and USD/INR free endpoints or curated CSV/parquet files.

Production deployments should cache all raw data with timestamps and use immutable feature snapshots for reproducibility.

## Research Workflow

1. Build or refresh the data lake.
2. Generate point-in-time features with `build_feature_matrix`.
3. Train models by walk-forward split; never train on rows after the prediction date.
4. Ensemble expected returns and rank stocks cross-sectionally.
5. Allocate with `ConfidenceAllocator`, which derives exposure from current cross-sectional confidence instead of fixed thresholds.
6. Backtest with explicit cost assumptions and compare composite scores.
7. Promote only models that beat Nifty 50 on risk-adjusted metrics and pass drift/risk gates.
8. Paper trade through `PaperBroker`; live Zerodha execution requires explicit `live_trading_enabled=True`.

## Development Commands

```bash
pytest
python -m ai_trading_system.cli --help  # module help depends on chosen entry point
ai-trade-train --config config/config.yaml
ai-trade-backtest
```

## Risk Controls

- Conservative max gross/net exposure and position caps.
- Minimum cash allocation.
- Dynamic confidence-based exposure reduction.
- Transaction-cost-aware backtesting.
- Kill switch for drawdown breaches.
- Paper-trading default for every broker path.

## Extension Points

- Replace free data adapters with licensed institutional feeds.
- Add Optuna search spaces per model family.
- Persist model artifacts with MLflow or a model registry.
- Add borrow/SLB constraints for Indian equity shorting.
- Extend reports with attribution, capacity, and regime diagnostics.
