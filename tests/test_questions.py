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
    validate_answer,
    violation_score,
)


def test_yesno_rejects_empty_instructions() -> None:
    with pytest.raises(PolicyError):
        YesNo("   ")


def test_choice_requires_two_options_and_valid_labels() -> None:
    with pytest.raises(PolicyError):
        Choice("pick", {"only": None})
    with pytest.raises(PolicyError):
        Choice("pick", {"1bad": None, "ok": None})
    question = Choice("pick", {"safe": None, "bad": "harmful"})
    assert question.top if False else set(question.options) == {"safe", "bad"}


def test_score_level_bounds() -> None:
    with pytest.raises(PolicyError):
        Score("rate", ("a",))
    with pytest.raises(PolicyError):
        Score("rate", tuple(str(index) for index in range(11)))
    with pytest.raises(PolicyError):
        Score("rate", ("low", "low"))


def test_answers_reject_non_finite_and_boolean() -> None:
    with pytest.raises(BackendError):
        YesNoAnswer(True, "probability")  # type: ignore[arg-type]
    with pytest.raises(BackendError):
        YesNoAnswer(math.nan, "probability")
    with pytest.raises(BackendError):
        ChoiceAnswer({"a": 0.5, "b": 0.4}, "probability")
    with pytest.raises(BackendError):
        ScoreAnswer((0.5, 0.6), "probability")


def test_choice_top_and_score_expected_level() -> None:
    choice = ChoiceAnswer({"safe": 0.2, "bad": 0.8}, "probability")
    assert choice.top == "bad"
    score = ScoreAnswer((0.0, 0.5, 0.5), "probability")
    assert score.expected_level == 1.5


def test_validate_answer_and_violation_score() -> None:
    yes = YesNo("bad?")
    choice = Choice("pick", {"safe": None, "bad": None})
    score = Score("rate", ("low", "high"))
    validate_answer(yes, YesNoAnswer(0.9, "probability"))
    with pytest.raises(BackendError):
        validate_answer(yes, ChoiceAnswer({"safe": 0.5, "bad": 0.5}, "probability"))
    with pytest.raises(BackendError):
        validate_answer(choice, ChoiceAnswer({"other": 1.0}, "probability"))
    with pytest.raises(BackendError):
        validate_answer(score, ScoreAnswer((1.0,), "probability"))
    assert violation_score(yes, YesNoAnswer(0.4, "probability")) == 0.4
    assert (
        violation_score(
            choice,
            ChoiceAnswer({"safe": 0.55, "bad": 0.45}, "probability"),
            violating={"bad"},
        )
        == 0.45
    )
    assert (
        violation_score(
            score,
            ScoreAnswer((0.2, 0.8), "probability"),
            violation_level=1,
        )
        == 0.8
    )
    with pytest.raises(PolicyError):
        violation_score(choice, ChoiceAnswer({"safe": 0.5, "bad": 0.5}, "probability"))


def test_threshold_validation() -> None:
    with pytest.raises(PolicyError):
        Threshold(block_at=True)  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        Threshold(block_at=1.2)
    with pytest.raises(PolicyError):
        Threshold(block_at=0.2, flag_at=0.5)
    assert Threshold.coerce(0.8).block_at == 0.8
    assert Threshold.coerce(Threshold(0.3)).block_at == 0.3
