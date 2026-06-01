from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import requests

from ai_trading_system.data.errors import DataSourceError
from ai_trading_system.data.retry import retry

LOGGER = logging.getLogger(__name__)


class MarketDataSource(Protocol):
    def fetch_ohlcv(self, symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame: ...


class FundamentalDataSource(Protocol):
    def fetch_fundamentals(self, symbols: list[str], as_of: str | None = None) -> pd.DataFrame: ...


class OptionsDataSource(Protocol):
    def fetch_index_options(self, symbol: str = "NIFTY") -> pd.DataFrame: ...


class MacroDataSource(Protocol):
    def fetch_macro(self, start: str | None = None, end: str | None = None) -> pd.DataFrame: ...


def _optional_module(name: str) -> Any:
    if importlib.util.find_spec(name) is None:
        raise DataSourceError(f"Install optional dependency `{name}` to use this data source.")
    return importlib.import_module(name)


@dataclass
class YahooFinanceSource:
    """Free Yahoo Finance adapter for OHLCV and point-in-time snapshot fundamentals."""

    auto_adjust: bool = True

    @retry(attempts=3, initial_delay=0.5, backoff=2.0, exceptions=(Exception,))
    def fetch_ohlcv(self, symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        if not symbols:
            raise DataSourceError("At least one symbol is required for OHLCV download")
        yf = _optional_module("yfinance")
        LOGGER.info("Fetching Yahoo OHLCV", extra={"symbols": symbols, "start": start, "end": end})
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise DataSourceError("Yahoo Finance returned no OHLCV data")
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    LOGGER.warning("Yahoo response missing symbol", extra={"symbol": symbol})
                    continue
                frame = raw[symbol]
            else:
                frame = raw
            normalized = frame.rename(columns=str.lower).reset_index()
            normalized = normalized.rename(columns={"Date": "date", "Datetime": "date", "index": "date"})
            normalized["symbol"] = symbol
            frames.append(normalized[["date", "symbol", "open", "high", "low", "close", "volume"]])
        if not frames:
            raise DataSourceError("No Yahoo OHLCV frames could be normalized")
        return pd.concat(frames, ignore_index=True)

    @retry(attempts=3, initial_delay=0.5, backoff=2.0, exceptions=(Exception,))
    def fetch_fundamentals(self, symbols: list[str], as_of: str | None = None) -> pd.DataFrame:
        if not symbols:
            raise DataSourceError("At least one symbol is required for fundamentals download")
        yf = _optional_module("yfinance")
        rows: list[dict[str, Any]] = []
        snapshot_date = pd.Timestamp(as_of or datetime.now(timezone.utc).date()).normalize()
        fields = {
            "trailingPE": "pe",
            "priceToBook": "pb",
            "returnOnEquity": "roe",
            "returnOnAssets": "roa",
            "revenueGrowth": "revenue_growth",
            "earningsGrowth": "earnings_growth",
            "debtToEquity": "debt_to_equity",
            "marketCap": "market_cap",
        }
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.get_info() if hasattr(ticker, "get_info") else ticker.info
            row: dict[str, Any] = {"date": snapshot_date, "symbol": symbol, "as_of": snapshot_date}
            for source_field, target_field in fields.items():
                row[target_field] = info.get(source_field)
            rows.append(row)
        if not rows:
            raise DataSourceError("Yahoo Finance returned no fundamental rows")
        return pd.DataFrame(rows)


@dataclass
class CSVFundamentalSource:
    """Load user-curated fundamentals from CSV files such as Screener exports."""

    path: Path

    def fetch_fundamentals(self, symbols: list[str], as_of: str | None = None) -> pd.DataFrame:
        data = pd.read_csv(self.path, parse_dates=["date"])
        if symbols:
            data = data[data["symbol"].isin(symbols)]
        if as_of and "as_of" in data.columns:
            data = data[pd.to_datetime(data["as_of"]) <= pd.Timestamp(as_of)]
        return data.reset_index(drop=True)


@dataclass
class NSEOptionsChainSource:
    """Current NSE public option-chain adapter.

    NSE endpoints can change and require browser-like headers; this adapter is intentionally isolated so
    production deployments can swap in a licensed market-data feed without changing downstream code.
    """

    base_url: str = "https://www.nseindia.com"
    timeout: int = 20

    @retry(attempts=3, initial_delay=0.5, backoff=2.0, exceptions=(requests.RequestException, ValueError, KeyError))
    def fetch_index_options(self, symbol: str = "NIFTY") -> pd.DataFrame:
        session = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        LOGGER.info("Fetching NSE option chain", extra={"symbol": symbol})
        session.get(self.base_url, headers=headers, timeout=self.timeout)
        url = f"{self.base_url}/api/option-chain-indices?symbol={symbol}"
        payload = session.get(url, headers=headers, timeout=self.timeout).json()
        timestamp = payload.get("records", {}).get("timestamp")
        snapshot_date = pd.Timestamp(timestamp).normalize() if timestamp else pd.Timestamp.now(tz="UTC").normalize()
        rows = []
        for item in payload.get("records", {}).get("data", []):
            strike = item.get("strikePrice")
            expiry = item.get("expiryDate")
            for side in ("CE", "PE"):
                if side in item:
                    option = item[side]
                    rows.append(
                        {
                            "date": snapshot_date,
                            "symbol": symbol,
                            "option_type": side,
                            "strike_price": option.get("strikePrice", strike),
                            "expiry_date": expiry,
                            "open_interest": option.get("openInterest", 0),
                            "change_in_open_interest": option.get("changeinOpenInterest"),
                            "implied_volatility": option.get("impliedVolatility"),
                            "last_price": option.get("lastPrice"),
                            "underlying_value": option.get("underlyingValue"),
                            "ingested_at": pd.Timestamp.now(tz="UTC"),
                        }
                    )
        if not rows:
            raise DataSourceError("NSE returned no option-chain rows")
        return pd.DataFrame(rows)


@dataclass
class CSVOptionsChainSource:
    path: Path

    def fetch_index_options(self, symbol: str = "NIFTY") -> pd.DataFrame:
        data = pd.read_csv(self.path, parse_dates=["date"])
        return data[data["symbol"].eq(symbol)].reset_index(drop=True) if "symbol" in data.columns else data


@dataclass
class WorldBankMacroSource:
    """Free World Bank macro adapter for India annual macro indicators."""

    country: str = "IND"
    timeout: int = 20
    indicators: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.indicators is None:
            self.indicators = {
                "inflation": "FP.CPI.TOTL.ZG",
                "gdp_growth": "NY.GDP.MKTP.KD.ZG",
            }

    @retry(attempts=3, initial_delay=0.5, backoff=2.0, exceptions=(requests.RequestException, ValueError, KeyError))
    def fetch_macro(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_year = pd.Timestamp(start).year if start else 1960
        end_year = pd.Timestamp(end).year if end else pd.Timestamp.now(tz="UTC").year
        for name, indicator in (self.indicators or {}).items():
            url = f"https://api.worldbank.org/v2/country/{self.country}/indicator/{indicator}"
            params = {"format": "json", "per_page": 20000, "date": f"{start_year}:{end_year}"}
            LOGGER.info("Fetching World Bank macro series", extra={"indicator": indicator, "name": name})
            payload = requests.get(url, params=params, timeout=self.timeout).json()
            records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            rows = [
                {"date": pd.Timestamp(year=int(item["date"]), month=12, day=31), name: item.get("value")}
                for item in records
                if item.get("value") is not None
            ]
            if rows:
                frames.append(pd.DataFrame(rows))
        if not frames:
            raise DataSourceError("World Bank returned no macro data")
        macro = frames[0]
        for frame in frames[1:]:
            macro = macro.merge(frame, on="date", how="outer")
        macro = macro.sort_values("date").reset_index(drop=True)
        macro["as_of"] = macro["date"]
        return macro


@dataclass
class CSVMacroSource:
    path: Path

    def fetch_macro(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        data = pd.read_csv(self.path, parse_dates=["date"])
        if start:
            data = data[data["date"] >= pd.Timestamp(start)]
        if end:
            data = data[data["date"] <= pd.Timestamp(end)]
        return data.reset_index(drop=True)


@dataclass
class CSVDataLake:
    root: Path

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / f"{name}.csv", parse_dates=["date"])

    def write(self, name: str, frame: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.root / f"{name}.csv", index=False)


def synthetic_ohlcv(symbols: list[str], periods: int = 420, seed: int = 7) -> pd.DataFrame:
    """Deterministic sample data for tests, demos, and CI without network access."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    frames = []
    for i, symbol in enumerate(symbols):
        returns = rng.normal(0.0004 + i * 0.00002, 0.018, periods)
        close = 1000 * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.006, periods))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.025, periods))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.025, periods))
        volume = rng.integers(100_000, 4_000_000, periods)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
