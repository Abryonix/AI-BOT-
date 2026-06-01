# How to Use and Run the Nifty 50 AI Trading System

This guide shows the practical commands for installing, running the offline demo, training models, running backtests, and preparing paper/live broker workflows.

## 1. Prerequisites

Use Python 3.10 or 3.11 for the smoothest dependency support.

```bash
python --version
```

The project is packaged with `pyproject.toml`, so install it from the repository root.

## 2. Create an Environment

```bash
cd /path/to/AI-BOT-
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

### Minimal research/dev install

```bash
pip install -e '.[dev]'
```

### Full research install

Use this when you want gradient boosting, reinforcement learning, dashboards, and optional broker/data integrations:

```bash
pip install -r requirements-all.txt
```

The full install includes optional support for XGBoost, LightGBM, CatBoost, Stable-Baselines3, Streamlit, Yahoo Finance downloads, and Zerodha/Kite Connect.

## 4. Run the Built-In Offline Demo

The fastest way to verify the system is working is the synthetic-data backtest. It does not need market-data credentials or internet access after dependencies are installed.

```bash
ai-trade-backtest
```

Equivalent Python invocation:

```bash
python scripts/run_demo.py
```

Expected output is a dictionary of metrics such as CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, alpha, beta, and composite score.

## 5. Train a Baseline Model

```bash
ai-trade-train --config config/config.yaml
```

This command loads configuration, generates deterministic synthetic Nifty-style OHLCV data, builds features and targets, trains a baseline model, and prints validation metrics.

## 6. Run Tests and Static Checks

```bash
pytest
ruff check src tests
python -m compileall -q src tests
```

If `pytest` fails with missing modules, install the dev dependencies first:

```bash
pip install -e '.[dev]'
```

## 7. Configure the Trading Research Setup

Edit `config/config.yaml` before running real experiments:

- `universe.constituents`: Nifty 50 symbols in Yahoo Finance `.NS` format.
- `data.start_date`: history start date.
- `portfolio.initial_capital`: account capital in INR.
- `portfolio.max_gross_exposure`, `max_net_exposure`, and `max_position_weight`: conservative exposure constraints.
- `execution.paper_trading`: must remain `true` for paper workflows.
- `execution.live_trading_enabled`: must remain `false` unless live trading has been explicitly approved.
- `risk`: kill-switch and drawdown controls.

## 8. Use Real Free Market Data

Install optional data dependencies:

```bash
pip install yfinance pyarrow
```

Then use the Yahoo adapter from Python:

```python
from ai_trading_system.data.sources import YahooFinanceSource
from ai_trading_system.features.pipeline import build_feature_matrix

symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
ohlcv = YahooFinanceSource().fetch_ohlcv(symbols, start="2014-01-01")
features = build_feature_matrix(ohlcv)
print(features.tail())
```

For production research, cache downloaded raw data into the data lake and version each feature snapshot.

## 9. Generate Portfolio Orders

The normal flow is:

1. Build feature matrix.
2. Train one or more models.
3. Ensemble expected returns.
4. Allocate positions.
5. Run risk checks.
6. Send orders to the paper broker.

Example:

```python
from ai_trading_system.data.sources import synthetic_ohlcv
from ai_trading_system.features.pipeline import build_feature_matrix
from ai_trading_system.models.ml import train_predict_model, ensemble_predictions
from ai_trading_system.portfolio.construction import ConfidenceAllocator, PortfolioConstraints

prices = synthetic_ohlcv(["RELIANCE.NS", "TCS.NS", "INFY.NS"], periods=420)
features = build_feature_matrix(prices)
cutoff = features["date"].quantile(0.75)
result = train_predict_model(features, features["date"] <= cutoff, features["date"] > cutoff)
predictions = ensemble_predictions([result])
orders = ConfidenceAllocator().allocate(predictions, prices, PortfolioConstraints(capital=500_000))
print(orders)
```

## 10. Paper Trading and Zerodha Safety

Paper trading is the safe default:

```python
from ai_trading_system.execution.broker import Order, OrderSide, PaperBroker

broker = PaperBroker(cash=500_000)
order_id = broker.place_order(Order(symbol="RELIANCE.NS", side=OrderSide.BUY, quantity=1))
print(order_id)
print(broker.positions())
```

The Zerodha adapter refuses live orders unless `live_trading_enabled=True` is explicitly set in code after approval. Do not enable live trading until data reconciliation, risk checks, exchange-session checks, and paper-trading validation are complete.

## 11. Monitoring

After exporting equity and positions CSVs, launch a Streamlit dashboard entry point or call `monitoring_snapshot` from Python.

```bash
ai-trade-monitor
```

Programmatic snapshot:

```python
from ai_trading_system.monitoring.dashboard import monitoring_snapshot

snapshot = monitoring_snapshot(equity, predictions, positions)
print(snapshot)
```

## 12. Common Troubleshooting

### `ModuleNotFoundError: pandas`

Install dependencies:

```bash
pip install -e '.[dev]'
```

### Package index or proxy blocks installation

Use a network that can reach PyPI, configure your corporate package mirror, or build a wheelhouse on a connected machine:

```bash
pip download -r requirements-all.txt -d wheelhouse
pip install --no-index --find-links wheelhouse -r requirements-all.txt
```

### NSE option-chain failures

NSE public endpoints can change and sometimes block non-browser clients. Treat the NSE adapter as a research convenience and replace it with a licensed feed for production.

## 13. Recommended First Run

For a new machine, run these commands in order:

```bash
cd /path/to/AI-BOT-
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'
python -m compileall -q src tests
ruff check src tests
pytest
ai-trade-train --config config/config.yaml
ai-trade-backtest
```
