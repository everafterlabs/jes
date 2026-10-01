"""Lesson 10, both variants: the backend is down on purpose (nothing on port 9)."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_failure_modes(variant: str) -> None:
    lesson = importlib.import_module(f"examples.10_failures.{variant}")
    raised, blocked, opened, capped = lesson.main()
    assert raised == "raised"
    assert blocked.decision == "block"
    assert blocked.complete is False
    assert opened.decision == "allow"
    assert opened.complete is False
    assert opened.ok is False
    assert opened.onward == "Blocked: backend_error."
    assert "hello" not in opened.onward
    assert capped.decision == "block"
    assert capped.complete is False
    assert capped.onward == "Blocked: input_too_long."
