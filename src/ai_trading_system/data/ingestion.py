"""Orchestrated data ingestion with validation, retries, logging, and caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading_system.data.cache import CacheRecord, DataCache
from ai_trading_system.data.sources import (
    FundamentalDataSource,
    MacroDataSource,
    MarketDataSource,
    OptionsDataSource,
)
from ai_trading_system.data.validation import DataQualityReport, DataValidator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    dataset: str
    frame: pd.DataFrame
    quality: DataQualityReport
    cache_record: CacheRecord | None
    from_cache: bool = False


@dataclass
class DataIngestionService:
    """High-level ingestion service for required free-data domains."""

    cache_dir: Path
    validator: DataValidator = field(default_factory=DataValidator)
    use_cache: bool = True

    def __post_init__(self) -> None:
        self.cache = DataCache(Path(self.cache_dir))

    def _stamp(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if "ingested_at" not in data.columns:
            data["ingested_at"] = pd.Timestamp(datetime.now(timezone.utc))
        return data

    def _cache_or_return(
        self,
        dataset: str,
        frame: pd.DataFrame,
        params: dict[str, Any],
        quality: DataQualityReport,
    ) -> IngestionResult:
        cache_record = None
        if self.use_cache:
            cache_record = self.cache.write(
                dataset,
                frame,
                params=params,
                metadata={"quality_warnings": quality.warnings},
            )
        return IngestionResult(dataset, frame, quality, cache_record, from_cache=False)

    def ingest_historical_ohlcv(
        self,
        source: MarketDataSource,
        symbols: list[str],
        start: str,
        end: str | None = None,
        index_symbol: str | None = "^NSEI",
        force_refresh: bool = False,
    ) -> IngestionResult:
        """Ingest historical Nifty 50 constituents plus optional Nifty index OHLCV."""
        all_symbols = list(dict.fromkeys(symbols + ([index_symbol] if index_symbol else [])))
        params = {"symbols": all_symbols, "start": start, "end": end, "dataset_type": "historical_ohlcv"}
        dataset = "raw_ohlcv"
        if self.use_cache and not force_refresh and self.cache.exists(dataset, params):
            frame = self.cache.read(dataset, params)
            quality = self.validator.validate_ohlcv(frame, dataset)
            return IngestionResult(dataset, frame, quality, None, from_cache=True)
        LOGGER.info("Starting historical OHLCV ingestion", extra=params)
        frame = self._stamp(source.fetch_ohlcv(all_symbols, start=start, end=end))
        quality = self.validator.validate_ohlcv(frame, dataset)
        return self._cache_or_return(dataset, frame, params, quality)

    def ingest_fundamentals(
        self,
        source: FundamentalDataSource,
        symbols: list[str],
        as_of: str | None = None,
        force_refresh: bool = False,
    ) -> IngestionResult:
        """Ingest point-in-time fundamental snapshots for configured symbols."""
        params = {"symbols": symbols, "as_of": as_of, "dataset_type": "fundamentals"}
        dataset = "raw_fundamentals"
        if self.use_cache and not force_refresh and self.cache.exists(dataset, params):
            frame = self.cache.read(dataset, params)
            quality = self.validator.validate_fundamentals(frame, dataset)
            return IngestionResult(dataset, frame, quality, None, from_cache=True)
        LOGGER.info("Starting fundamentals ingestion", extra=params)
        frame = self._stamp(source.fetch_fundamentals(symbols, as_of=as_of))
        quality = self.validator.validate_fundamentals(frame, dataset)
        return self._cache_or_return(dataset, frame, params, quality)

    def ingest_current_options_chain(
        self,
        source: OptionsDataSource,
        symbol: str = "NIFTY",
        force_refresh: bool = True,
    ) -> IngestionResult:
        """Ingest current Nifty option-chain snapshot.

        Options are force-refreshed by default because current option chains are time-sensitive.
        """
        today = pd.Timestamp.now(tz="UTC").date().isoformat()
        params = {"symbol": symbol, "snapshot_date": today, "dataset_type": "options_chain"}
        dataset = "raw_options_chain"
        if self.use_cache and not force_refresh and self.cache.exists(dataset, params):
            frame = self.cache.read(dataset, params)
            quality = self.validator.validate_options(frame, dataset)
            return IngestionResult(dataset, frame, quality, None, from_cache=True)
        LOGGER.info("Starting options-chain ingestion", extra=params)
        frame = self._stamp(source.fetch_index_options(symbol=symbol))
        quality = self.validator.validate_options(frame, dataset)
        return self._cache_or_return(dataset, frame, params, quality)

    def ingest_macro(
        self,
        source: MacroDataSource,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
    ) -> IngestionResult:
        """Ingest macroeconomic series for India from a free source or curated CSV."""
        params = {"start": start, "end": end, "dataset_type": "macro"}
        dataset = "raw_macro"
        if self.use_cache and not force_refresh and self.cache.exists(dataset, params):
            frame = self.cache.read(dataset, params)
            quality = self.validator.validate_macro(frame, dataset)
            return IngestionResult(dataset, frame, quality, None, from_cache=True)
        LOGGER.info("Starting macro ingestion", extra=params)
        frame = self._stamp(source.fetch_macro(start=start, end=end))
        quality = self.validator.validate_macro(frame, dataset)
        return self._cache_or_return(dataset, frame, params, quality)
