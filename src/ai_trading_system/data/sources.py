from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests


class MarketDataSource(Protocol):
    def fetch_ohlcv(self, symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame: ...


@dataclass
class YahooFinanceSource:
    """Free OHLCV/fundamental source using yfinance when installed."""

    auto_adjust: bool = True

    def fetch_ohlcv(self, symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Install yfinance to download free Yahoo Finance data.") from exc
        raw = yf.download(symbols, start=start, end=end, auto_adjust=self.auto_adjust, group_by="ticker")
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            frame = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
            frame = frame.rename(columns=str.lower).assign(symbol=symbol)
            frames.append(frame.reset_index().rename(columns={"Date": "date", "index": "date"}))
        return pd.concat(frames, ignore_index=True)


@dataclass
class NSEOptionsChainSource:
    """Current NSE public option-chain adapter.

    NSE endpoints can change and require browser-like headers; this adapter is intentionally isolated so
    production deployments can swap in a licensed market-data feed without changing downstream code.
    """

    base_url: str = "https://www.nseindia.com"
    timeout: int = 20

    def fetch_index_options(self, symbol: str = "NIFTY") -> pd.DataFrame:
        session = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        session.get(self.base_url, headers=headers, timeout=self.timeout)
        url = f"{self.base_url}/api/option-chain-indices?symbol={symbol}"
        payload = session.get(url, headers=headers, timeout=self.timeout).json()
        rows = []
        for item in payload.get("records", {}).get("data", []):
            for side in ("CE", "PE"):
                if side in item:
                    rows.append(item[side] | {"option_type": side})
        return pd.DataFrame(rows)


@dataclass
class CSVDataLake:
    root: Path

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / f"{name}.parquet")

    def write(self, name: str, frame: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.root / f"{name}.parquet", index=False)


def synthetic_ohlcv(symbols: list[str], periods: int = 420, seed: int = 7) -> pd.DataFrame:
    """Deterministic sample data for tests, demos, and CI without network access."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    frames = []
    for i, symbol in enumerate(symbols):
        returns = rng.normal(0.0004 + i * 0.00002, 0.018, periods)
        close = 1000 * np.exp(np.cumsum(returns))
        high = close * (1 + rng.uniform(0.001, 0.025, periods))
        low = close * (1 - rng.uniform(0.001, 0.025, periods))
        open_ = close * (1 + rng.normal(0, 0.006, periods))
        volume = rng.integers(100_000, 4_000_000, periods)
        frames.append(pd.DataFrame({
            "date": dates, "symbol": symbol, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        }))
    return pd.concat(frames, ignore_index=True)
