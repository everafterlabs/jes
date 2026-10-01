"""Backends answer questions about a text. jes ships one for TypeSafe decision models.

A custom backend is any object with ``name``, ``model``, ``headroom``, and ``decide``,
``adecide``, or both. ``Guard`` calls ``decide`` and ``AsyncGuard`` prefers ``adecide``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from jes.errors import BackendError, PolicyError
from jes.questions import (
    Answer,
    Choice,
    ChoiceAnswer,
    Question,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
)
from jes.types import State


@dataclass(frozen=True, slots=True, repr=False)
class Request:
    """One backend call: the state to judge, the questions, and the seconds left."""

    state: State
    questions: Mapping[str, Question]
    timeout: float | None = None

    def __repr__(self) -> str:
        return f"Request(state={self.state!r}, questions={sorted(self.questions)})"


@dataclass(frozen=True, slots=True)
class Reply:
    """A backend's answers, keyed by question id, and the tokens it reports."""

    answers: Mapping[str, Answer]
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(self.answers)))


@runtime_checkable
class Backend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        """Bytes left in a request for this state and these questions. None means no limit."""
        ...


@runtime_checkable
class SyncBackend(Backend, Protocol):
    def decide(self, request: Request) -> Reply: ...


@runtime_checkable
class AsyncBackend(Backend, Protocol):
    async def adecide(self, request: Request) -> Reply: ...


class DecisionClassifier(Protocol):
    """LangChain's ``TypeSafeClassifier``, or anything with the same calls."""

    model: str

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


ModelSpec: TypeAlias = str | DecisionClassifier | SyncBackend | AsyncBackend


def render_state(state: State) -> str:
    """The text TypeSafe judges: the bare text, or JSON when there is context."""

    if (
        state.prompt is None
        and state.question is None
        and state.tool is None
        and not state.sources
        and not state.history
    ):
        return state.text
    payload: dict[str, object] = {
        "history": [{"role": message.role, "text": message.text} for message in state.history],
        "prompt": state.prompt,
        "question": state.question,
        "sources": list(state.sources),
        "text": state.text,
    }
    if state.tool is not None:
        payload["tool"] = state.tool
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_bytes(questions: Mapping[str, Question]) -> int:
    """A generous estimate of the bytes questions add to a request."""

    total = 0
    for question_id, question in questions.items():
        total += 64 + len(question_id) + len(question.instructions.encode("utf-8"))
        if isinstance(question, YesNo):
            total += len((question.true or "").encode()) + len((question.false or "").encode())
        elif isinstance(question, Choice):
            total += sum(
                16 + len(label) + len((text or "").encode())
                for label, text in question.options.items()
            )
        else:
            total += sum(16 + len(level.encode()) for level in question.levels)
    return total


class TypeSafe:
    """Judge with a TypeSafe decision model through LangChain's ``TypeSafeClassifier``.

    ``model`` is a model id such as ``"jev-1.13.0"``, or a classifier you configured
    yourself. A model id reads ``TYPESAFE_API_KEY`` and ``TYPESAFE_BASE_URL``, and each
    request times out after ``timeout`` seconds.
    """

    name = "typesafe"

    def __init__(
        self,
        model: str | DecisionClassifier = "jev-latest",
        *,
        timeout: float = 30.0,
        max_request_bytes: int = 1_048_576,
    ) -> None:
        if isinstance(model, str):
            if not model.strip():
                raise PolicyError("the TypeSafe model id must not be empty")
            self.model = model.strip()
            self._classifier: DecisionClassifier | None = None
        else:
            self.model = model.model
            self._classifier = model
        if not timeout > 0:
            raise PolicyError("timeout must be positive")
        self._timeout = timeout
        self._max_request_bytes = max_request_bytes
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"TypeSafe(model={self.model!r})"

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int:
        used = len(render_state(state).encode("utf-8")) + question_bytes(questions)
        return self._max_request_bytes - used

    def decide(self, request: Request) -> Reply:
        payload = self._payload(request)
        try:
            response = self._client().invoke(payload)
        except Exception as error:
            raise _backend_error(self.name, error, request) from None
        return _reply(self.name, response, request.questions)

    async def adecide(self, request: Request) -> Reply:
        payload = self._payload(request)
        try:
            response = await self._client().ainvoke(payload)
        except Exception as error:
            raise _backend_error(self.name, error, request) from None
        return _reply(self.name, response, request.questions)

    def _client(self) -> DecisionClassifier:
        with self._lock:
            if self._classifier is None:
                from langchain_typesafe import TypeSafeClassifier

                try:
                    self._classifier = cast(
                        "DecisionClassifier",
                        TypeSafeClassifier(model=self.model, timeout=self._timeout),
                    )
                except Exception:
                    # Most often TYPESAFE_API_KEY is not set. The message is not repeated:
                    # jes errors carry metadata only.
                    raise BackendError(self.name, "client_setup_failed") from None
            return self._classifier

    def _payload(self, request: Request) -> dict[str, object]:
        return {
            "state": render_state(request.state),
            "questions": typesafe_questions(request.questions),
        }


def typesafe_questions(questions: Mapping[str, Question]) -> dict[str, object]:
    """jes questions as TypeSafe ``Noul``, ``Choice``, and ``Score`` primitives."""

    from langchain_typesafe import Choice as TSChoice, Noul, NoulCriteria, Score as TSScore

    payload: dict[str, object] = {}
    for question_id, question in questions.items():
        if isinstance(question, YesNo):
            criteria = None
            if question.true is not None or question.false is not None:
                criteria = NoulCriteria(true=question.true, false=question.false)
            payload[question_id] = Noul(instructions=question.instructions, criteria=criteria)
        elif isinstance(question, Choice):
            payload[question_id] = TSChoice(
                instructions=question.instructions,
                criteria=dict(question.options),
            )
        else:
            payload[question_id] = TSScore(
                instructions=question.instructions,
                criteria=list(question.levels),
            )
    return payload


def _reply(backend: str, response: object, questions: Mapping[str, Question]) -> Reply:
    raw_answers = getattr(response, "answers", None)
    if not isinstance(raw_answers, Mapping):
        raise BackendError(backend, "malformed_answer", question_ids=questions)
    answers: dict[str, Answer] = {}
    for question_id, question in questions.items():
        raw = cast(Mapping[str, object], raw_answers).get(question_id)
        try:
            answers[question_id] = _answer(question, raw)
        except (AttributeError, LookupError, TypeError, ValueError, BackendError):
            raise BackendError(backend, "malformed_answer", question_ids=(question_id,)) from None
    usage = getattr(response, "usage", None)
    return Reply(
        answers=answers,
        input_tokens=_count(getattr(usage, "input_tokens", None)),
        output_tokens=_count(getattr(usage, "output_tokens", None)),
    )


def _answer(question: Question, raw: object) -> Answer:
    from langchain_typesafe import ChoiceAnswer as TSChoice, NoulAnswer, ScoreAnswer as TSScore

    if isinstance(question, YesNo):
        if not isinstance(raw, NoulAnswer):
            raise TypeError("expected a noul answer")
        return YesNoAnswer(float(raw.noul))
    if isinstance(question, Choice):
        if not isinstance(raw, TSChoice):
            raise TypeError("expected a choice answer")
        probabilities = cast(Mapping[str, float], raw.probabilities)
        values = _normalized([probabilities[label] for label in question.options])
        return ChoiceAnswer(dict(zip(question.options, values, strict=True)), float(raw.confidence))
    if not isinstance(raw, TSScore):
        raise TypeError("expected a score answer")
    levels = cast(Mapping[int, float], raw.probabilities)
    values = _normalized([levels[index] for index in range(len(question.levels))])
    return ScoreAnswer(tuple(values), float(raw.confidence))


def _normalized(values: list[float]) -> list[float]:
    """Probabilities rescaled to sum to one, if they were within 1% of it."""

    total = sum(values)
    if not 0.99 <= total <= 1.01:
        raise ValueError("probabilities do not sum to one")
    return [value / total for value in values]


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _backend_error(backend: str, error: Exception, request: Request) -> BackendError:
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    if name == "TypeSafeAPITimeoutError":
        reason = "timeout"
    elif name == "TypeSafeAPIResponseValidationError":
        reason = "malformed_response"
    elif status in (401, 403):
        reason = "unauthorized"
    elif status == 429:
        reason = "rate_limited"
    elif isinstance(status, int) and status >= 500:
        reason = "upstream_error"
    elif isinstance(status, int) and status >= 400:
        reason = "rejected"
    elif name == "TypeSafeAPIConnectionError":
        reason = "connection_error"
    elif isinstance(error, BackendError):
        return error
    else:
        reason = "unexpected_error"
    code = status if isinstance(status, int) else None
    return BackendError(backend, reason, status_code=code, question_ids=request.questions)


def resolve_model(spec: ModelSpec | None) -> SyncBackend | AsyncBackend | None:
    """A backend from a TypeSafe model id, a ``TypeSafeClassifier``, or a backend."""

    if spec is None:
        return None
    if isinstance(spec, str):
        return TypeSafe(spec)
    if isinstance(spec, (SyncBackend, AsyncBackend)):
        return spec
    if callable(getattr(spec, "invoke", None)) and callable(getattr(spec, "ainvoke", None)):
        return TypeSafe(spec)
    raise PolicyError("model must be a TypeSafe model id, a TypeSafeClassifier, or a backend")


__all__ = [
    "AsyncBackend",
    "Backend",
    "DecisionClassifier",
    "ModelSpec",
    "Reply",
    "Request",
    "SyncBackend",
    "TypeSafe",
    "render_state",
    "resolve_model",
    "typesafe_questions",
]
