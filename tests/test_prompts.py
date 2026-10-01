"""Frozen question text."""

from __future__ import annotations

import pytest

from jes.policies.prompts import PROMPT_HASHES, PROMPTS, prompt_hash, verify_prompt_bytes


def test_frozen_prompt_bytes_match_their_recorded_hashes() -> None:
    verify_prompt_bytes()
    assert set(PROMPTS) == set(PROMPT_HASHES)
    for prompt_id in PROMPTS:
        assert prompt_hash(prompt_id) == PROMPT_HASHES[prompt_id]


def test_changed_prompt_bytes_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(PROMPTS, "injection.v1", PROMPTS["injection.v1"] + " ")
    with pytest.raises(RuntimeError, match=r"injection\.v1"):
        verify_prompt_bytes()

    monkeypatch.setitem(PROMPTS, "extra.v1", "An unrecorded prompt.")
    with pytest.raises(RuntimeError, match="diverged"):
        verify_prompt_bytes()
