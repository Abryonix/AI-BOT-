from __future__ import annotations

import pandas as pd

from ai_trading_system.backtests.metrics import equity_metrics


def monitoring_snapshot(equity: pd.Series, predictions: pd.DataFrame, positions: pd.DataFrame) -> dict[str, object]:
    metrics = equity_metrics(equity)
    latest_positions = positions.tail(20).to_dict(orient="records")
    prediction_quality = {
        "mean_expected_return": float(predictions.filter(like="expected_return").mean(numeric_only=True).mean()) if not predictions.empty else 0.0,
        "coverage": int(len(predictions)),
    }
    exposure = float(positions.get("target_weight", pd.Series(dtype=float)).abs().sum()) if not positions.empty else 0.0
    return {"metrics": metrics, "prediction_quality": prediction_quality, "gross_exposure": exposure, "positions": latest_positions}


def run_streamlit_dashboard(equity_path: str, positions_path: str) -> None:  # pragma: no cover
    import streamlit as st
    equity = pd.read_csv(equity_path, parse_dates=["date"]).set_index("date")["equity"]
    positions = pd.read_csv(positions_path, parse_dates=["date"])
    st.title("Nifty 50 AI Trading Monitor")
    st.line_chart(equity)
    st.json(monitoring_snapshot(equity, pd.DataFrame(), positions))
