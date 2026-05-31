from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(group: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = group["close"].shift(1)
    tr = pd.concat([
        group["high"] - group["low"],
        (group["high"] - prev_close).abs(),
        (group["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _adx(group: pd.DataFrame, window: int = 14) -> pd.Series:
    up_move = group["high"].diff()
    down_move = -group["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr(group, window).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    data = frame.sort_values(["symbol", "date"]).copy()
    out = []
    for _, group in data.groupby("symbol", sort=False):
        g = group.copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["volume"]
        returns = close.pct_change()
        g["return_1d"] = returns
        g["log_return_1d"] = np.log(close).diff()
        for window in (5, 10, 14, 20, 50, 100, 200):
            g[f"sma_{window}"] = close.rolling(window).mean()
            g[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
            g[f"momentum_{window}"] = close / close.shift(window) - 1
            g[f"roc_{window}"] = close.pct_change(window)
            g[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252)
            g[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean()
        g["rsi_14"] = _rsi(close)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]
        g["atr_14"] = _atr(g)
        g["atr_pct_14"] = g["atr_14"] / close
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        g["bb_upper_20"] = mid + 2 * std
        g["bb_lower_20"] = mid - 2 * std
        g["bb_width_20"] = (g["bb_upper_20"] - g["bb_lower_20"]) / mid
        typical = (high + low + close) / 3
        g["cci_20"] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std())
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        g["stoch_k_14"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        g["stoch_d_3"] = g["stoch_k_14"].rolling(3).mean()
        g["adx_14"] = _adx(g)
        g["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        money_flow_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        g["accumulation_distribution"] = (money_flow_mult.fillna(0) * volume).cumsum()
        g["parkinson_vol_21"] = np.sqrt((1 / (4 * np.log(2))) * (np.log(high / low) ** 2).rolling(21).mean() * 252)
        g["garman_klass_vol_21"] = np.sqrt((0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / g["open"]) ** 2).rolling(21).mean().clip(lower=0) * 252)
        g["sma_20_50_cross"] = g["sma_20"] / g["sma_50"] - 1
        out.append(g)
    return pd.concat(out, ignore_index=True)
