"""CSV-backed data cache with metadata sidecars.

The cache deliberately uses CSV rather than parquet so the minimal project dependencies are enough for
local development and CI. Production deployments can replace this class with a parquet/object-store
implementation behind the same interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading_system.data.errors import DataCacheError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheRecord:
    dataset: str
    path: Path
    metadata_path: Path
    rows: int
    metadata: dict[str, Any]


@dataclass
class DataCache:
    """Small file cache for raw/interim data artifacts."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def key(self, dataset: str, params: dict[str, Any] | None = None) -> str:
        payload = json.dumps(params or {}, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        safe_dataset = dataset.replace("/", "_").replace(" ", "_")
        return f"{safe_dataset}_{digest}"

    def _paths(self, dataset: str, params: dict[str, Any] | None = None) -> tuple[Path, Path]:
        key = self.key(dataset, params)
        return self.root / f"{key}.csv", self.root / f"{key}.meta.json"

    def exists(self, dataset: str, params: dict[str, Any] | None = None) -> bool:
        path, metadata_path = self._paths(dataset, params)
        return path.exists() and metadata_path.exists()

    def read(self, dataset: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        path, _ = self._paths(dataset, params)
        if not path.exists():
            raise DataCacheError(f"Cache entry does not exist: {path}")
        LOGGER.info("Reading cached dataset", extra={"dataset": dataset, "path": str(path)})
        data = pd.read_csv(path)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        if "as_of" in data.columns:
            data["as_of"] = pd.to_datetime(data["as_of"])
        if "ingested_at" in data.columns:
            data["ingested_at"] = pd.to_datetime(data["ingested_at"])
        return data

    def write(
        self,
        dataset: str,
        frame: pd.DataFrame,
        params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheRecord:
        if frame.empty:
            raise DataCacheError(f"Refusing to cache empty dataset: {dataset}")
        self.root.mkdir(parents=True, exist_ok=True)
        path, metadata_path = self._paths(dataset, params)
        data = frame.copy()
        data.to_csv(path, index=False)
        meta = {
            "dataset": dataset,
            "params": params or {},
            "rows": int(len(data)),
            "columns": list(data.columns),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            meta.update(metadata)
        metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
        LOGGER.info(
            "Cached dataset",
            extra={"dataset": dataset, "path": str(path), "metadata_path": str(metadata_path), "rows": len(data)},
        )
        return CacheRecord(dataset, path, metadata_path, len(data), meta)
