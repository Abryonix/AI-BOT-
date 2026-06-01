from __future__ import annotations

import pandas as pd
import pytest

from ai_trading_system.data.cache import DataCache
from ai_trading_system.data.errors import DataValidationError
from ai_trading_system.data.ingestion import DataIngestionService
from ai_trading_system.data.sources import synthetic_ohlcv
from ai_trading_system.data.validation import DataValidator


class FakeMarketSource:
    calls = 0

    def fetch_ohlcv(self, symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        self.calls += 1
        return synthetic_ohlcv(symbols, periods=40)


class FakeFundamentalSource:
    def fetch_fundamentals(self, symbols: list[str], as_of: str | None = None) -> pd.DataFrame:
        date = pd.Timestamp(as_of or "2026-01-01")
        return pd.DataFrame(
            {
                "date": [date] * len(symbols),
                "symbol": symbols,
                "pe": [20.0] * len(symbols),
                "pb": [3.0] * len(symbols),
                "roe": [0.18] * len(symbols),
                "roce": [0.21] * len(symbols),
                "revenue_growth": [0.08] * len(symbols),
                "earnings_growth": [0.07] * len(symbols),
                "debt_to_equity": [0.4] * len(symbols),
                "as_of": [date] * len(symbols),
            }
        )


class FakeOptionsSource:
    def fetch_index_options(self, symbol: str = "NIFTY") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
                "symbol": [symbol, symbol],
                "option_type": ["CE", "PE"],
                "strike_price": [24000, 24000],
                "expiry_date": ["2026-01-29", "2026-01-29"],
                "open_interest": [1000, 1500],
                "change_in_open_interest": [100, -50],
                "implied_volatility": [13.2, 14.1],
            }
        )


class FakeMacroSource:
    def fetch_macro(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-12-31")],
                "repo_rate": [6.5, 6.25],
                "inflation": [5.2, 4.8],
                "gdp_growth": [6.7, 6.5],
                "inr_usd": [83.0, 84.0],
                "bond_yield_proxy": [7.1, 6.9],
            }
        )


def test_data_cache_round_trip(tmp_path) -> None:
    cache = DataCache(tmp_path)
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "value": [1.0]})
    record = cache.write("macro", frame, params={"source": "test"})
    assert record.rows == 1
    assert cache.exists("macro", {"source": "test"})
    loaded = cache.read("macro", {"source": "test"})
    assert loaded.loc[0, "value"] == 1.0
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_validator_rejects_invalid_ohlcv() -> None:
    frame = synthetic_ohlcv(["AAA.NS"], periods=5)
    frame.loc[0, "high"] = 0
    with pytest.raises(DataValidationError):
        DataValidator().validate_ohlcv(frame)


def test_ingests_historical_ohlcv_with_index_and_cache(tmp_path) -> None:
    source = FakeMarketSource()
    service = DataIngestionService(tmp_path)
    result = service.ingest_historical_ohlcv(
        source,
        symbols=["AAA.NS", "BBB.NS"],
        start="2024-01-01",
        index_symbol="^NSEI",
    )
    assert result.quality.rows == 120
    assert set(result.frame["symbol"].unique()) == {"AAA.NS", "BBB.NS", "^NSEI"}
    cached = service.ingest_historical_ohlcv(
        source,
        symbols=["AAA.NS", "BBB.NS"],
        start="2024-01-01",
        index_symbol="^NSEI",
    )
    assert cached.from_cache is True
    assert source.calls == 1


def test_ingests_required_exogenous_datasets(tmp_path) -> None:
    service = DataIngestionService(tmp_path)
    fundamentals = service.ingest_fundamentals(
        FakeFundamentalSource(), ["AAA.NS", "BBB.NS"], as_of="2026-01-01"
    )
    options = service.ingest_current_options_chain(FakeOptionsSource(), symbol="NIFTY")
    macro = service.ingest_macro(FakeMacroSource(), start="2025-01-01", end="2026-12-31")

    assert fundamentals.quality.rows == 2
    assert {"pe", "pb", "roe", "debt_to_equity"}.issubset(fundamentals.frame.columns)
    assert options.quality.rows == 2
    assert set(options.frame["option_type"]) == {"CE", "PE"}
    assert macro.quality.rows == 2
    assert {"repo_rate", "inflation", "gdp_growth", "inr_usd", "bond_yield_proxy"}.issubset(
        macro.frame.columns
    )
