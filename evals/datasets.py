"""Dataset manifest checks. Pending reviews cannot be downloaded."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast


def load_manifest(path: Path) -> list[dict[str, object]]:
    parsed = cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
    raw = parsed.get("dataset")
    if not isinstance(raw, list):
        raise ValueError("dataset manifest is not a list")
    rows: list[dict[str, object]] = []
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            rows.append(cast(dict[str, object], item))
    return rows


def assert_download_allowed(path: Path) -> None:
    """Raise until every row is reviewed and pinned. Smoke runs must not call this."""

    for row in load_manifest(path):
        review = row.get("review")
        revision = row.get("revision")
        if review != "accepted" or not isinstance(revision, str) or revision in {"", "unpinned"}:
            identifier = row.get("id", "unknown")
            raise RuntimeError(f"dataset {identifier} is not cleared for download")


__all__ = ["assert_download_allowed", "load_manifest"]
