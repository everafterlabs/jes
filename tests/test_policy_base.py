"""judge(), Judgment.evaluate, and the transform building blocks."""

from __future__ import annotations

import time

import pytest

from jes.errors import DeadlineExceeded, PolicyError
from jes.policies import (
    Item,
    TransformContext,
    TransformOutcome,
    judge,
)
from jes.policies.base import SensitiveHit, freeze_stages, merge_spans
from jes.questions import (
    Choice,
    ChoiceAnswer,
    Score,
    ScoreAnswer,
    Threshold,
    YesNo,
    YesNoAnswer,
)
from jes.testing import FakeBackend
from jes.text.textmap import Edit
from jes.types import Span


def test_a_single_question_is_called_violation() -> None:
    judgment = judge("refund", YesNo("The text asks for money back."), threshold=0.7)
    assert list(judgment.questions) == ["violation"]
    assert judgment.threshold == Threshold(0.7)
    assert judgment.stages == {"input", "output"}
    assert judgment.backend is None
    assert judgment.evaluate({"violation": YesNoAnswer(0.9)}) == [("violation", 0.9, "block")]
    assert judgment.evaluate({"violation": YesNoAnswer(0.1)}) == [("violation", 0.1, None)]


def test_choice_and_score_questions_need_their_violation_settings() -> None:
    choice = Choice("Which team?", {"billing": None, "other": None})
    route = judge("route", choice, threshold=0.5, violating=["billing"])
    answer = ChoiceAnswer({"billing": 0.6, "other": 0.4})
    assert route.evaluate({"violation": answer}) == [("violation", 0.6, "block")]

    severity = judge(
        "severity",
        {"level": Score("How bad?", ("low", "mid", "high"))},
        threshold=Threshold(0.9, flag_at=0.5),
        violation_level=1,
    )
    levels = ScoreAnswer((0.4, 0.3, 0.3))
    assert severity.evaluate({"level": levels}) == [("level", pytest.approx(0.6), "flag")]


def test_several_choices_take_a_mapping() -> None:
    questions = {
        "a": Choice("A?", {"x": None, "y": None}),
        "b": Choice("B?", {"x": None, "y": None}),
        "c": Score("C?", ("low", "high")),
        "d": Score("D?", ("low", "high")),
    }
    judgment = judge(
        "multi",
        questions,
        threshold=0.5,
        violating={"a": ["x"], "b": ["y"]},
        violation_level={"c": 1, "d": 0},
    )
    assert judgment.violating == {"a": {"x"}, "b": {"y"}}
    assert judgment.violation_levels == {"c": 1, "d": 0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "bad name"}, "policy name"),
        ({"questions": {}}, "at least one question"),
        ({"questions": {"bad id": YesNo("q")}}, "question id"),
        ({"stages": ()}, "at least one stage"),
        ({"stages": ("input", "sideways")}, "unknown stage"),
        ({"context": "maybe"}, "context must be"),
        ({"context": "required"}, "output only"),
        ({"on_overflow": "ignore"}, "on_overflow"),
        ({"max_items": 3}, "max_items needs items"),
        ({"items": lambda text: (), "whole_text": True}, "cannot be whole_text"),
        ({"items": lambda text: (), "max_items": 0}, "max_items must be positive"),
        ({"violating": ["x"]}, "only to choice"),
        ({"violation_level": 1}, "only to score"),
        ({"model": object()}, "model must be"),
    ],
)
def test_judge_rejects_bad_settings(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "name": "check",
        "questions": YesNo("Bad?"),
        "threshold": 0.5,
        **kwargs,
    }
    with pytest.raises(PolicyError, match=message):
        judge(**arguments)  # type: ignore[arg-type]


def test_choice_and_score_settings_are_validated() -> None:
    choice = Choice("Pick.", {"x": None, "y": None})
    score = Score("Rate.", ("low", "high"))
    with pytest.raises(PolicyError, match="need violating"):
        judge("c", choice, threshold=0.5)
    with pytest.raises(PolicyError, match="several choices"):
        judge("c", {"a": choice, "b": choice}, threshold=0.5, violating=["x"])
    with pytest.raises(PolicyError, match="several choices"):
        judge("c", choice, threshold=0.5, violating="x")
    with pytest.raises(PolicyError, match="every choice"):
        judge("c", {"a": choice, "b": choice}, threshold=0.5, violating={"a": ["x"]})
    with pytest.raises(PolicyError, match="non-empty options"):
        judge("c", choice, threshold=0.5, violating=["z"])
    with pytest.raises(PolicyError, match="need a violation_level"):
        judge("s", score, threshold=0.5)
    with pytest.raises(PolicyError, match="several scores"):
        judge("s", {"a": score, "b": score}, threshold=0.5, violation_level=1)
    with pytest.raises(PolicyError, match="every score"):
        judge("s", {"a": score, "b": score}, threshold=0.5, violation_level={"a": 1})
    with pytest.raises(PolicyError, match="one of the levels"):
        judge("s", score, threshold=0.5, violation_level=2)
    with pytest.raises(PolicyError, match="one of the levels"):
        judge("s", score, threshold=0.5, violation_level=True)


def test_a_judgment_can_bring_its_own_model() -> None:
    fake = FakeBackend()
    assert judge("x", YesNo("q"), threshold=0.5, model=fake).backend is fake


def test_stage_and_span_helpers() -> None:
    assert freeze_stages(["input", "input"]) == {"input"}
    assert merge_spans([Span(5, 7), Span(0, 2), Span(1, 3), Span(3, 4)]) == (
        Span(0, 4),
        Span(5, 7),
    )
    assert merge_spans([]) == ()


def test_transform_building_blocks_hide_text() -> None:
    outcome = TransformOutcome(edits=[Edit(0, 1, "secret")], findings=[])  # type: ignore[arg-type]
    assert outcome.edits == (Edit(0, 1, "secret"),)
    assert repr(outcome) == "TransformOutcome(edits=1, findings=0)"
    assert "secret" not in repr(Item("secret url", Span(0, 10)))
    hit = SensitiveHit(Span(0, 4), "EMAIL_ADDRESS", "token", "redact")
    assert "EMAIL_ADDRESS" in repr(hit)


def test_transform_context_deadline() -> None:
    TransformContext(stage="input", origin="input", target="text").check_deadline()
    past = TransformContext(
        stage="input", origin="input", target="text", deadline=time.monotonic() - 1
    )
    with pytest.raises(DeadlineExceeded):
        past.check_deadline()
