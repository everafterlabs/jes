"""Lesson 06, both variants, offline: fake scores replace the decision model."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_topics_and_toxicity_block(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.06_topics_toxicity.{variant}")
    # Keys are question ids, "policy.question"; topics numbers each denied topic.
    monkeypatch.setattr(lesson, "MODEL", mock_jev({"topics.topic_0": 0.8, "toxicity.insult": 0.9}))
    medical, insult = lesson.main()
    assert medical.decision == "block"
    assert insult.decision == "block"
    assert any(finding.label == "insult" for finding in insult.findings)
