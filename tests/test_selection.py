"""Burned development ids and the two-finalist ranking gate."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from evals.selection import (
    LOCK_PATH,
    assert_can_rank,
    assert_ids_allowed,
    holdout_matches_cohort,
    lock_digest,
)

ROOT = Path(__file__).parents[1]


def test_lock_matches_the_development_rule_and_manifest() -> None:
    body = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    pairs: dict[str, str] = {}
    sets = body["sets"]
    assert body["audit_opened"] is False
    for group in sets.values():
        for example_id in group["burned"]:
            pairs[example_id] = "burned"
        for example_id in group["reserved_holdout"]:
            pairs[example_id] = "reserved"
    assert holdout_matches_cohort(pairs)
    assert len(sets["xstest-prompts"]["burned"]) == 353
    assert len(sets["xstest-prompts"]["reserved_holdout"]) == 97
    assert len(sets["bipia-table"]["reserved_test"]) == 100
    assert len(sets["bipia-code"]["reserved_test"]) == 50
    manifest = tomllib.loads((ROOT / "evals" / "datasets.toml").read_text(encoding="utf-8"))
    assert manifest["lock"]["id_sha256"] == lock_digest()
    assert manifest["lock"]["audit_opened"] is False


def test_selection_refuses_one_finalist_and_closed_ids() -> None:
    body = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    burned = body["sets"]["xstest-prompts"]["burned"][0]
    reserved = body["sets"]["bipia-table"]["reserved_holdout"][0]
    held = body["sets"]["xstest-prompts"]["reserved_holdout"][0]
    test_id = body["sets"]["bipia-table"]["reserved_test"][0]
    with pytest.raises(RuntimeError, match="two finalists"):
        assert_can_rank(["typesafe.jev.v1"], ["dolly:unassigned:0"])
    with pytest.raises(RuntimeError, match="burned"):
        assert_ids_allowed("selection", [burned])
    with pytest.raises(RuntimeError, match="reserved"):
        assert_ids_allowed("selection", [f"{reserved}:clean"])
    with pytest.raises(RuntimeError, match="reserved"):
        assert_ids_allowed("calibration", [test_id])
    with pytest.raises(RuntimeError, match="audit is not opened"):
        assert_ids_allowed("audit", [test_id])
    with pytest.raises(RuntimeError, match="reserved"):
        assert_ids_allowed("development", [held])
    assert_can_rank(["typesafe.jev.v1", "ollama.chat.v1"], ["dolly:unassigned:0"])
