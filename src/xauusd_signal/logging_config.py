from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(file_path),
            logging.StreamHandler(),
        ],
    )

