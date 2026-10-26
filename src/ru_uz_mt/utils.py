"""Shared helpers: logging, device selection, JSONL IO."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once (level from arg or ``$LOG_LEVEL``)."""
    logging.basicConfig(
        level=(level or os.environ.get("LOG_LEVEL", "INFO")).upper(),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device: explicit preference, else cuda > mps > cpu."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, creating parent dirs. Returns the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
