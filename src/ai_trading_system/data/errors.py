"""Data-layer exception types."""

from __future__ import annotations


class DataLayerError(RuntimeError):
    """Base exception for data-layer failures."""


class DataSourceError(DataLayerError):
    """Raised when a remote or local data source cannot provide valid data."""


class DataValidationError(DataLayerError):
    """Raised when ingested data violates a canonical schema or quality rule."""


class DataCacheError(DataLayerError):
    """Raised when cache read/write operations fail."""
