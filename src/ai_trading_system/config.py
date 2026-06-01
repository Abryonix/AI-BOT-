from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    @property
    def initial_capital(self) -> float:
        return float(self.raw["portfolio"]["initial_capital"])

    @property
    def max_names(self) -> int:
        return int(self.raw["portfolio"].get("max_names", 5))

    @property
    def cost_bps(self) -> float:
        execution = self.raw.get("execution", {})
        return float(execution.get("brokerage_bps", 0) + execution.get("slippage_bps", 0) + execution.get("stt_bps", 0))


def load_settings(path: str | Path = "config/config.yaml") -> Settings:
    with Path(path).open("r", encoding="utf-8") as handle:
        return Settings(yaml.safe_load(handle))
