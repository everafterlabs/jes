"""The backend protocol, the TypeSafe backend, and FakeBackend."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_typesafe import (
    Choice as TSChoice,
    ChoiceAnswer as TSChoiceAnswer,
    Noul,
    NoulAnswer,
    Score as TSScore,
    ScoreAnswer as TSScoreAnswer,
)

from jes.backend import (
    AsyncBackend,
    Reply,
    Request,
    SyncBackend,
    TypeSafe,
    question_bytes,
    render_state,
    resolve_model,
)
from jes.errors import BackendError, PolicyError
from jes.questions import Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import Message, State

QUESTIONS = {
    "safety.violation": YesNo("The text is unsafe.", true="unsafe", false="safe"),
    "route.team": Choice("Which team?", {"billing": "Money", "other": None}),
    "severity.level": Score("How severe?", ("low", "high")),
}


def _response(
    *,
    noul: float = 0.9,
    choice: dict[str, float] | None = None,
    levels: dict[int, float] | None = None,
) -> SimpleNamespace:
    choice = {"billing": 0.25, "other": 0.75} if choice is None else choice
    levels = {0: 0.4, 1: 0.6} if levels is None else levels
    return SimpleNamespace(
        answers={
            "safety.violation": NoulAnswer(type="noul", noul=noul),
            "route.team": TSChoiceAnswer(
                type="choice", choice="other", probabilities=choice, confidence=0.8
            ),
            "severity.level": TSScoreAnswer(
                type="score",
                score=0.6,
                legend={0: "low", 1: "high"},
                probabilities=levels,
                confidence=0.5,
            ),
        },
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )


class FakeClassifier:
    model = "jev-test"

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.inputs: list[Any] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.inputs.append(input)
        if self.error is not None:
            raise self.error
        return self.response

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.invoke(input, config, **kwargs)


def _request(text: str = "hello", **state: Any) -> Request:
    return Request(State(stage="input", text=text, **state), QUESTIONS)


def test_requests_never_show_text() -> None:
    request = Request(State(stage="input", text="secret text"), QUESTIONS, timeout=1.0)
    assert "secret" not in repr(request)
    assert "route.team" in repr(request)


def test_answers_of_the_wrong_type_are_malformed() -> None:
    noul = NoulAnswer(type="noul", noul=0.5)
    for question in (Choice("Which?", {"a": None, "b": None}), Score("How?", ("low", "high"))):
        response = SimpleNamespace(answers={"q.id": noul})
        with pytest.raises(BackendError, match="malformed_answer"):
            TypeSafe(FakeClassifier(response)).decide(
                Request(State(stage="input", text="x"), {"q.id": question})
            )
    response = SimpleNamespace(answers={"q.id": _response().answers["route.team"]})
    with pytest.raises(BackendError, match="malformed_answer"):
        TypeSafe(FakeClassifier(response)).decide(
            Request(State(stage="input", text="x"), {"q.id": YesNo("Bad?")})
        )


def test_render_state_is_the_bare_text_without_context() -> None:
    assert render_state(State(stage="input", text="hi")) == "hi"
    rendered = render_state(
        State(
            stage="tool_call",
            text='{"cmd":"ls"}',
            prompt="list files",
            history=(Message("user", "earlier"),),
            tool="bash",
        )
    )
    assert rendered == (
        '{"history":[{"role":"user","text":"earlier"}],"prompt":"list files",'
        '"question":null,"sources":[],"text":"{\\"cmd\\":\\"ls\\"}","tool":"bash"}'
    )


def test_typesafe_maps_every_question_type() -> None:
    classifier = FakeClassifier(_response())
    backend = TypeSafe(classifier)
    assert backend.model == "jev-test"
    reply = backend.decide(_request(prompt="p"))
    assert reply.answers["safety.violation"] == YesNoAnswer(0.9)
    assert reply.answers["route.team"] == ChoiceAnswer({"billing": 0.25, "other": 0.75}, 0.8)
    assert reply.answers["severity.level"] == ScoreAnswer((0.4, 0.6), 0.5)
    assert (reply.input_tokens, reply.output_tokens) == (12, 3)

    sent = classifier.inputs[0]
    assert sent["state"].startswith('{"history":[]')
    questions = sent["questions"]
    assert isinstance(questions["safety.violation"], Noul)
    assert questions["safety.violation"].criteria.true == "unsafe"
    assert isinstance(questions["route.team"], TSChoice)
    assert questions["route.team"].criteria == {"billing": "Money", "other": None}
    assert isinstance(questions["severity.level"], TSScore)
    assert questions["severity.level"].criteria == ["low", "high"]


def test_typesafe_async_matches_sync() -> None:
    backend = TypeSafe(FakeClassifier(_response()))
    reply = asyncio.run(backend.adecide(_request()))
    assert reply == backend.decide(_request())


def test_probabilities_near_one_are_rescaled_and_others_rejected() -> None:
    reply = TypeSafe(FakeClassifier(_response(choice={"billing": 0.5, "other": 0.505}))).decide(
        _request()
    )
    assert sum(reply.answers["route.team"].scores.values()) == pytest.approx(1.0)  # type: ignore[union-attr]
    for response in (
        _response(choice={"billing": 0.2, "other": 0.3}),
        _response(choice={"billing": 1.0}),
        _response(levels={0: 0.5, 1: True}),  # type: ignore[dict-item]
        SimpleNamespace(answers=None),
        SimpleNamespace(answers={}),
    ):
        with pytest.raises(BackendError, match="malformed_answer"):
            TypeSafe(FakeClassifier(response)).decide(_request())


def _error(name: str, status: int | None = None) -> Exception:
    error = type(name, (Exception,), {})()
    if status is not None:
        error.status_code = status  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize(
    ("error", "reason", "status"),
    [
        (_error("TypeSafeAPITimeoutError"), "timeout", None),
        (_error("TypeSafeAPIResponseValidationError"), "malformed_response", None),
        (_error("TypeSafeAuthenticationError", 401), "unauthorized", 401),
        (_error("TypeSafeRateLimitError", 429), "rate_limited", 429),
        (_error("TypeSafeInternalServerError", 503), "upstream_error", 503),
        (_error("TypeSafeNotFoundError", 404), "rejected", 404),
        (_error("TypeSafeAPIConnectionError"), "connection_error", None),
        (RuntimeError("boom"), "unexpected_error", None),
    ],
)
def test_classifier_errors_become_backend_errors(
    error: Exception, reason: str, status: int | None
) -> None:
    with pytest.raises(BackendError) as raised:
        TypeSafe(FakeClassifier(error=error)).decide(_request())
    assert raised.value.reason == reason
    assert raised.value.status_code == status
    assert "boom" not in str(raised.value)
    with pytest.raises(BackendError, match=reason):
        asyncio.run(TypeSafe(FakeClassifier(error=error)).adecide(_request()))


@pytest.mark.parametrize("key", [None, "", "  "])
def test_a_missing_api_key_is_named(monkeypatch: pytest.MonkeyPatch, key: str | None) -> None:
    if key is None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", key)
    with pytest.raises(BackendError, match="missing_api_key"):
        TypeSafe("jev-test").decide(_request())
    with pytest.raises(BackendError, match="missing_api_key"):
        asyncio.run(TypeSafe("jev-test").adecide(_request()))


def test_other_client_setup_failures_hide_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_kwargs: object) -> None:
        raise ValueError("secret detail")

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr("langchain_typesafe.TypeSafeClassifier", broken)
    with pytest.raises(BackendError, match="client_setup_failed") as raised:
        TypeSafe("jev-test").decide(_request())
    assert "secret detail" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_a_model_id_builds_one_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    backend = TypeSafe("jev-test", timeout=5)
    first = backend._client()
    assert backend._client() is first
    assert first.model == "jev-test"
    assert repr(backend) == "TypeSafe(model='jev-test')"


def test_typesafe_settings_are_checked() -> None:
    with pytest.raises(PolicyError, match="must not be empty"):
        TypeSafe("  ")
    with pytest.raises(PolicyError, match="timeout"):
        TypeSafe("jev", timeout=0)


def test_headroom_shrinks_with_the_text_and_questions() -> None:
    backend = TypeSafe("jev", max_request_bytes=1_000)
    short = backend.headroom(State(stage="input", text="a"), QUESTIONS)
    long = backend.headroom(State(stage="input", text="a" * 100), QUESTIONS)
    assert short - long == 99
    assert question_bytes(QUESTIONS) > question_bytes({"q": YesNo("Bad?")}) > 0


def test_resolve_model() -> None:
    assert resolve_model(None) is None
    resolved = resolve_model("jev-1.13.0")
    assert isinstance(resolved, TypeSafe) and resolved.model == "jev-1.13.0"
    wrapped = resolve_model(FakeClassifier())
    assert isinstance(wrapped, TypeSafe) and wrapped.model == "jev-test"
    fake = FakeBackend()
    assert resolve_model(fake) is fake
    assert isinstance(fake, SyncBackend) and isinstance(fake, AsyncBackend)
    with pytest.raises(PolicyError, match="model must be"):
        resolve_model(object())  # type: ignore[arg-type]


def test_fake_backend_answers_and_records() -> None:
    fake = FakeBackend(
        {"safety.violation": YesNoAnswer(0.9), "violation": YesNoAnswer(0.1)},
        default=YesNoAnswer(0.0),
        rule=lambda request, question_id: (
            YesNoAnswer(1.0)
            if "attack" in request.state.text and question_id == "x.violation"
            else None
        ),
    )
    questions = {
        "safety.violation": YesNo("a?"),
        "other.violation": YesNo("b?"),
        "z.q": YesNo("c?"),
    }
    reply = fake.decide(Request(State(stage="input", text="hello"), questions))
    assert reply.answers == {
        "safety.violation": YesNoAnswer(0.9),
        "other.violation": YesNoAnswer(0.1),
        "z.q": YesNoAnswer(0.0),
    }
    ruled = fake.decide(Request(State(stage="input", text="attack"), {"x.violation": YesNo("d?")}))
    assert ruled.answers["x.violation"] == YesNoAnswer(1.0)
    assert len(fake.requests) == 2
    fake.answer("late", YesNoAnswer(0.5))
    assert fake.decide(Request(State(stage="input", text="x"), {"p.late": YesNo("e?")})).answers[
        "p.late"
    ] == YesNoAnswer(0.5)

    with pytest.raises(BackendError, match="no_answer"):
        FakeBackend().decide(Request(State(stage="input", text="x"), {"q": YesNo("f?")}))


def test_fake_backend_headroom_and_delay() -> None:
    assert FakeBackend().headroom(State(stage="input", text="abc"), {}) is None
    limited = FakeBackend(max_request_bytes=10)
    assert limited.headroom(State(stage="input", text="abc"), {"q": YesNo("g?")}) == 6
    slow = FakeBackend(default=YesNoAnswer(0.0), delay_s=0.05)
    request = Request(State(stage="input", text="x"), {"q": YesNo("h?")})
    started = time.perf_counter()
    assert isinstance(slow.decide(request), Reply)
    assert isinstance(asyncio.run(slow.adecide(request)), Reply)
    assert isinstance(asyncio.run(FakeBackend(default=YesNoAnswer(0.0)).adecide(request)), Reply)
    assert time.perf_counter() - started >= 0.1
