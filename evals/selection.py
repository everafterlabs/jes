"""Selection gate. One backend cannot rank, and burned or reserved ids cannot be reused."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from evals.slices import development_cohort, source_id

LOCK_PATH = Path(__file__).with_name("split_lock.json")
_CLOSED = frozenset({"selection", "calibration", "audit", "holdout"})


def assert_can_rank(provider_profiles: Iterable[str], example_ids: Iterable[str]) -> None:
    """Refuse a ranking until two finalists are named and every id is still unused."""

    profiles = set(provider_profiles)
    if len(profiles) < 2:
        raise RuntimeError("selection needs two finalists")
    assert_ids_allowed("selection", example_ids)


def assert_ids_allowed(phase: str, example_ids: Iterable[str]) -> None:
    """Development may use burned ids. Reserved ids stay unsent. Audit is closed."""

    if phase in {"audit", "holdout"}:
        raise RuntimeError("audit is not opened")
    burned, reserved = _locked_ids()
    for example_id in example_ids:
        document = source_id(example_id)
        if phase == "development" and document in reserved:
            raise RuntimeError(f"{document} is reserved")
        if phase in _CLOSED and document in burned:
            raise RuntimeError(f"{document} was burned in development")
        if phase in {"selection", "calibration"} and document in reserved:
            raise RuntimeError(f"{document} is reserved")


def lock_digest() -> str:
    payload = json.dumps(_lock_body(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _locked_ids() -> tuple[set[str], set[str]]:
    body = _lock_body()
    burned: set[str] = set()
    reserved: set[str] = set()
    raw_sets = body["sets"]
    if not isinstance(raw_sets, dict):
        raise RuntimeError("split lock is missing sets")
    sets = cast(dict[object, object], raw_sets)
    for group in sets.values():
        if not isinstance(group, dict):
            continue
        mapping = cast(dict[str, object], group)
        burned.update(_strings(mapping.get("burned")))
        for key in ("reserved_holdout", "reserved_test"):
            reserved.update(_strings(mapping.get(key)))
    return burned, reserved


def _lock_body() -> dict[str, object]:
    parsed = cast(object, json.loads(LOCK_PATH.read_text(encoding="utf-8")))
    if not isinstance(parsed, dict):
        raise RuntimeError("split lock is not an object")
    raw = cast(dict[object, object], parsed)
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _strings(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in cast(list[object], value) if isinstance(item, str)}


def holdout_matches_cohort(example_ids: Mapping[str, str]) -> bool:
    """True when every recorded train/XSTest id sits in the cohort the file claims."""

    return all(
        (kind == "burned") is development_cohort(example_id)
        for example_id, kind in example_ids.items()
    )
