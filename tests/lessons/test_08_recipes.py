"""Lesson 08, both variants, offline: fake scores replace the decision model."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_recipes(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.08_recipes.{variant}")
    # Every other question, including each URL, scores 0 (safe).
    monkeypatch.setattr(
        lesson,
        "MODEL",
        mock_jev({"sentiment.violation": 0.88, "factual_consistency.violation": 0.9}),
    )
    hostile, redacted, linked, inconsistent = lesson.main()
    assert hostile.onward == "Blocked: sentiment."
    assert "Acme" not in redacted.onward
    assert linked.ok
    assert inconsistent.onward == "Blocked: factual_consistency."
