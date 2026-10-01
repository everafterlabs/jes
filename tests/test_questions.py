"""Questions, answers, thresholds, and violation scores."""

from __future__ import annotations

import math

import pytest

from jes.errors import BackendError, PolicyError
from jes.questions import (
    Choice,
    ChoiceAnswer,
    Score,
    ScoreAnswer,
    Threshold,
    YesNo,
    YesNoAnswer,
    violation_score,
)


def test_questions_validate_their_shape() -> None:
    with pytest.raises(PolicyError, match="instructions"):
        YesNo("  ")
    with pytest.raises(PolicyError, match="two options"):
        Choice("Pick one.", {"only": None})
    with pytest.raises(PolicyError, match="choice label"):
        Choice("Pick one.", {"ok": None, "not ok": None})
    with pytest.raises(PolicyError, match="between 2 and 10"):
        Score("Rate it.", ("only",))
    with pytest.raises(PolicyError, match="unique"):
        Score("Rate it.", ("low", "low"))
    with pytest.raises(PolicyError, match="unique and non-empty"):
        Score("Rate it.", ("low", " "))


def test_choice_options_are_frozen() -> None:
    options = {"billing": "Money problems", "other": None}
    question = Choice("Which team?", options)
    options["billing"] = "changed"
    assert question.options["billing"] == "Money problems"
    with pytest.raises(TypeError):
        question.options["other"] = "x"  # type: ignore[index]


def test_answers_must_be_probabilities() -> None:
    for bad in (-0.1, 1.1, math.nan, math.inf, True):
        with pytest.raises(BackendError):
            YesNoAnswer(bad)  # type: ignore[arg-type]
    with pytest.raises(BackendError, match="confidence"):
        YesNoAnswer(0.5, confidence=2.0)
    with pytest.raises(BackendError, match="sum to one"):
        ChoiceAnswer({"a": 0.5, "b": 0.2})
    with pytest.raises(BackendError, match="numbers"):
        ChoiceAnswer({})
    with pytest.raises(BackendError, match="sum to one"):
        ScoreAnswer((0.1, 0.1))
    with pytest.raises(BackendError, match="numbers"):
        ScoreAnswer(())
    assert ChoiceAnswer({"a": 0.5004, "b": 0.5}).scores["a"] == 0.5004


def test_threshold_bounds_and_coercion() -> None:
    assert Threshold.coerce(0.5) == Threshold(block_at=0.5)
    existing = Threshold(0.8, flag_at=0.4)
    assert Threshold.coerce(existing) is existing
    assert isinstance(Threshold(1).block_at, float)
    for bad in (True, -0.1, 1.5, math.nan):
        with pytest.raises(PolicyError, match="block_at"):
            Threshold.coerce(bad)  # type: ignore[arg-type]
    with pytest.raises(PolicyError, match="flag_at must be"):
        Threshold(0.5, flag_at=2.0)
    with pytest.raises(PolicyError, match="not be above"):
        Threshold(0.5, flag_at=0.6)


def test_threshold_action() -> None:
    threshold = Threshold(0.8, flag_at=0.5)
    assert threshold.action(0.9) == "block"
    assert threshold.action(0.8) == "block"
    assert threshold.action(0.5) == "flag"
    assert threshold.action(0.49) is None
    assert Threshold(0.8).action(0.79) is None


def test_violation_score_for_each_question_type() -> None:
    assert violation_score(YesNo("Bad?"), YesNoAnswer(0.7)) == 0.7

    choice = Choice("Which team?", {"billing": None, "refunds": None, "other": None})
    answer = ChoiceAnswer({"billing": 0.5, "refunds": 0.3, "other": 0.2})
    assert violation_score(choice, answer, violating=["billing", "refunds"]) == pytest.approx(0.8)
    assert violation_score(choice, answer, violating=["billing", "billing"]) == 0.5

    score = Score("How severe?", ("low", "mid", "high"))
    levels = ScoreAnswer((0.2, 0.3, 0.5))
    assert violation_score(score, levels, violation_level=1) == pytest.approx(0.8)
    assert violation_score(score, levels, violation_level=0) == pytest.approx(1.0)


def test_violation_score_never_exceeds_one() -> None:
    choice = Choice("Pick.", {"a": None, "b": None})
    answer = ChoiceAnswer({"a": 0.6, "b": 0.4009})
    assert violation_score(choice, answer, violating=["a", "b"]) == 1.0


def test_violation_score_rejects_mismatches() -> None:
    choice = Choice("Pick.", {"a": None, "b": None})
    with pytest.raises(PolicyError, match="non-empty subset"):
        violation_score(choice, ChoiceAnswer({"a": 0.5, "b": 0.5}))
    with pytest.raises(PolicyError, match="non-empty subset"):
        violation_score(choice, ChoiceAnswer({"a": 0.5, "b": 0.5}), violating=["c"])
    with pytest.raises(BackendError, match="labels do not match"):
        violation_score(choice, ChoiceAnswer({"a": 0.5, "c": 0.5}), violating=["a"])

    score = Score("Rate.", ("low", "high"))
    with pytest.raises(PolicyError, match="violation_level"):
        violation_score(score, ScoreAnswer((0.5, 0.5)))
    with pytest.raises(PolicyError, match="violation_level"):
        violation_score(score, ScoreAnswer((0.5, 0.5)), violation_level=2)
    with pytest.raises(PolicyError, match="violation_level"):
        violation_score(score, ScoreAnswer((0.5, 0.5)), violation_level=True)
    with pytest.raises(BackendError, match="length"):
        violation_score(score, ScoreAnswer((0.2, 0.3, 0.5)), violation_level=1)

    with pytest.raises(BackendError, match="answer type"):
        violation_score(YesNo("Bad?"), ScoreAnswer((0.5, 0.5)))
