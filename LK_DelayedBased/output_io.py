"""
Helpers for writing benchmark outputs under outputs/<category>/... with run_id + JSON sidecars.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def outputs_root(base: Path | None = None) -> Path:
    """Default: Mar24GitUpdates/LK_DelayedBased/outputs"""
    if base is None:
        base = Path(__file__).resolve().parent
    return base / "outputs"


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def ensure_run_dir(base: Path, *parts: str) -> Path:
    d = outputs_root(base)
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
