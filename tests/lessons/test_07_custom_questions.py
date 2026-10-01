"""Lesson 07, both variants, offline: fake scores replace the decision model."""

from __future__ import annotations

import importlib

import pytest

from jes.questions import ChoiceAnswer, ScoreAnswer
from tests.lesson_fakes import mock_jev

# One answer per question type. "probability" says the numbers are
# probabilities (each answer's values sum to 1).
SCORES = {
    "refund.violation": 0.9,
    "route.violation": ChoiceAnswer({"billing": 0.8, "other": 0.2}, "probability"),
    "severity.violation": ScoreAnswer((0.1, 0.1, 0.8), "probability"),
}


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_custom_questions_block(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.07_custom_questions.{variant}")
    monkeypatch.setattr(lesson, "MODEL", mock_jev(SCORES))
    refund, route, severity = lesson.main()
    assert refund.decision == "block"
    assert route.decision == "block"
    assert severity.decision == "block"
