from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / self.raw["database"]["path"]

    @property
    def model_path(self) -> Path:
        return self.root / self.raw["model"]["path"]

    @property
    def log_path(self) -> Path:
        return self.root / self.raw["logging"]["file"]


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    load_dotenv()
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Settings(raw=raw, root=path.parent)

