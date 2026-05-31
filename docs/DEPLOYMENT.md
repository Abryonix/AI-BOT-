# Deployment Preparation

## Environment

Use Python 3.10+ with pinned dependencies in production. The default install is intentionally lightweight; install `requirements-all.txt` on research machines that need gradient boosting, RL, dashboards, Yahoo downloads, Zerodha, and parquet support.

## Artifact Flow

1. Ingest raw data into an immutable data lake partitioned by source and timestamp.
2. Generate feature snapshots and labels for a specific as-of date.
3. Train ML and RL candidates with walk-forward splits.
4. Store model artifacts, feature schema, validation report, and code commit hash.
5. Run backtests and paper trading.
6. Promote a model only after risk and drift gates pass.

## Live Trading Safety

- `paper_trading` is true by default.
- `live_trading_enabled` is false by default.
- `ZerodhaBroker.place_order` raises unless live trading is explicitly enabled.
- Pre-trade risk checks validate drawdown and notional limits.
- Add exchange holiday calendars, borrow availability, and market-session checks before production deployment.

## Monitoring

Export daily equity, positions, predictions, realized returns, and broker holdings. Feed these into `monitoring_snapshot` or a Streamlit/Prometheus/Grafana deployment. Alert on drawdown, prediction drift, missing data, stale models, excessive turnover, and broker reconciliation breaks.
