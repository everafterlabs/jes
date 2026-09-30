"""LangChain TypeSafe judge and the request metadata the engine still records."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import langchain_typesafe
from langchain_typesafe import (
    Choice as TypeSafeChoice,
    ChoiceAnswer as TypeSafeChoiceAnswer,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score as TypeSafeScore,
    ScoreAnswer as TypeSafeScoreAnswer,
    TypeSafeClassifier,
)
from langchain_typesafe.client import TypeSafeError
from langchain_typesafe.types import ClassifierRequest

from jes.errors import BackendError, DeadlineExceeded, PolicyError
from jes.questions import (
    Answer,
    Choice,
    ChoiceAnswer,
    Question,
    ScoreAnswer,
    ScoreKind,
    YesNo,
    YesNoAnswer,
    validate_answer,
)
from jes.types import Stage, State

_K = TypeVar("_K")
_V = TypeVar("_V")

_SCORE_KINDS: Mapping[str, frozenset[ScoreKind]] = {"*": frozenset({"probability"})}


def _freeze_mapping(value: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    tasks: frozenset[str] | None
    max_options: int | None
    max_attempts: int
    score_kinds: Mapping[str, frozenset[ScoreKind]]
    budget_fidelity: Literal["exact", "conservative"]

    def __post_init__(self) -> None:
        if self.max_options is not None and self.max_options < 2:
            raise PolicyError("max_options must be at least two")
        if self.max_attempts < 1:
            raise PolicyError("max_attempts must be at least one")
        frozen = {task: frozenset(kinds) for task, kinds in self.score_kinds.items()}
        if not frozen or any(not kinds for kinds in frozen.values()):
            raise PolicyError("backend score kinds must not be empty")
        object.__setattr__(self, "score_kinds", _freeze_mapping(frozen))

    def supports_task(self, task: str) -> bool:
        return self.tasks is None or task in self.tasks

    def supported_kinds(self, task: str) -> frozenset[ScoreKind]:
        return self.score_kinds.get(task, self.score_kinds.get("*", frozenset()))


@dataclass(frozen=True, slots=True)
class BackendProfile:
    adapter: str
    adapter_version: str
    scorer_version: str
    parser_version: str
    provider: str
    provider_profile: str
    model: str
    revision: str | None
    artifact_digest: str | None
    tokenizer_revision: str | None
    template_revision: str | None
    mode: str
    generation_settings: Mapping[str, str | int | float | bool | None]
    context_window_tokens: int | None
    max_request_bytes: int | None
    output_reserve: int
    budget_attestation: str | None
    dependency_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.context_window_tokens is None and self.max_request_bytes is None:
            raise PolicyError("backend profile requires a token or byte budget")
        if self.context_window_tokens is not None and self.context_window_tokens < 1:
            raise PolicyError("context window must be positive")
        if self.max_request_bytes is not None and self.max_request_bytes < 1:
            raise PolicyError("request byte budget must be positive")
        if self.output_reserve < 0:
            raise PolicyError("output reserve must be non-negative")
        object.__setattr__(self, "generation_settings", _freeze_mapping(self.generation_settings))
        object.__setattr__(self, "dependency_versions", _freeze_mapping(self.dependency_versions))

    @property
    def model_identity(self) -> str:
        suffix = self.artifact_digest or self.revision
        return self.model if suffix is None else f"{self.model}@{suffix}"


@dataclass(frozen=True, slots=True)
class RequestProfile:
    backend_fingerprint: str
    attestation_digest: str | None
    transform_digest: str
    engine_dependency_digest: str
    transform_asset_digest: str
    renderer_version: str
    planner_version: str
    stage: Stage
    context_mode: Literal["none", "optional", "required"]
    sources: bool
    subject_mode: Literal["text", "items"]
    question_partition_digest: str
    chunk_config_digest: str
    budget_digest: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DecisionProfile:
    request_profile: str
    stage: Stage
    policy_kind: str
    policy_version: str
    subset_digest: str
    interpretation_version: str
    merge_version: str
    score_kinds_digest: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RequestPermit:
    permit_id: str
    deadline: float | None


class RequestBudget(Protocol):
    def acquire(
        self,
        deadline: float | None,
        *,
        logical_index: int,
        attempt: int,
    ) -> AbstractContextManager[RequestPermit]: ...

    def aacquire(
        self,
        deadline: float | None,
        *,
        logical_index: int,
        attempt: int,
    ) -> AbstractAsyncContextManager[RequestPermit]: ...


@dataclass(frozen=True, slots=True)
class RequestContext:
    logical_index: int
    deadline: float | None
    budget: RequestBudget
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class BackendUsage:
    permit_id: str
    attempt: int
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class BackendResult:
    answers: Mapping[str, Answer]
    usage: tuple[BackendUsage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", _freeze_mapping(self.answers))
        object.__setattr__(self, "usage", tuple(self.usage))


@runtime_checkable
class Backend(Protocol):
    name: str
    capabilities: BackendCapabilities
    profile: BackendProfile

    def count_units(self, text: str) -> int: ...

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]: ...

    def headroom(
        self,
        state: State,
        questions: Mapping[str, Question],
    ) -> int | None: ...


@runtime_checkable
class SyncBackend(Backend, Protocol):
    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult: ...


@runtime_checkable
class AsyncBackend(Backend, Protocol):
    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult: ...


class DecisionClassifier(Protocol):
    """LangChain TypeSafe classifier, or a test double with the same calls."""

    model: str

    def invoke(self, input: ClassifierRequest, config: Any = None, **kwargs: Any) -> Any: ...

    async def ainvoke(
        self,
        input: ClassifierRequest,
        config: Any = None,
        **kwargs: Any,
    ) -> Any: ...


ModelSpec: TypeAlias = str | DecisionClassifier | Backend


def attestation_digest_for(backend: Backend) -> str | None:
    """Decision-model calls are not attested deployment artifacts."""

    del backend
    return None


def render_state(state: State) -> str:
    """Render the text being judged. Questions are sent separately."""

    if (
        state.prompt is None
        and state.question is None
        and state.tool is None
        and not state.sources
        and not state.history
    ):
        return state.text
    import json

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


def typesafe_questions(
    questions: Mapping[str, Question],
) -> dict[str, Noul | TypeSafeChoice | TypeSafeScore]:
    """Map jes questions onto TypeSafe primitives."""

    payload: dict[str, Noul | TypeSafeChoice | TypeSafeScore] = {}
    for question_id, question in questions.items():
        if isinstance(question, YesNo):
            criteria = None
            if question.true is not None or question.false is not None:
                criteria = NoulCriteria(true=question.true, false=question.false)
            payload[question_id] = Noul(instructions=question.instructions, criteria=criteria)
        elif isinstance(question, Choice):
            payload[question_id] = TypeSafeChoice(
                instructions=question.instructions,
                criteria=dict(question.options),
            )
        else:
            payload[question_id] = TypeSafeScore(
                instructions=question.instructions,
                criteria=list(question.levels),
            )
    return payload


def answers_from_response(
    response: Any,
    questions: Mapping[str, Question],
) -> dict[str, Answer]:
    """Read a TypeSafe classifier response into jes answers."""

    raw_answers = getattr(response, "answers", None)
    if not isinstance(raw_answers, Mapping):
        raise BackendError("typesafe", "malformed_answer", question_ids=questions)
    body = cast(Mapping[str, object], raw_answers)
    answers: dict[str, Answer] = {}
    for question_id, question in questions.items():
        raw = body.get(question_id)
        if isinstance(question, YesNo):
            if not isinstance(raw, NoulAnswer):
                raise BackendError("typesafe", "malformed_answer", question_ids=(question_id,))
            answer: Answer = YesNoAnswer(float(raw.noul), "probability")
        elif isinstance(question, Choice):
            if not isinstance(raw, TypeSafeChoiceAnswer):
                raise BackendError("typesafe", "malformed_answer", question_ids=(question_id,))
            answer = ChoiceAnswer(
                _choice_scores(raw.probabilities, tuple(question.options)),
                "probability",
                float(raw.confidence),
            )
        else:
            if not isinstance(raw, TypeSafeScoreAnswer):
                raise BackendError("typesafe", "malformed_answer", question_ids=(question_id,))
            answer = ScoreAnswer(
                _level_scores(raw.probabilities, len(question.levels)),
                "probability",
                float(raw.confidence),
            )
        validate_answer(question, answer)
        answers[question_id] = answer
    return answers


class Judge:
    """Ask a LangChain TypeSafe classifier for each planned question batch."""

    name = "typesafe"

    def __init__(
        self,
        model: str | DecisionClassifier = "jev-latest",
        *,
        max_request_bytes: int = 1_048_576,
    ) -> None:
        if isinstance(model, str):
            identity = model.strip()
            if not identity:
                raise PolicyError("TypeSafe model must not be empty")
            self._model_name = identity
            self._classifier: DecisionClassifier | None = None
        else:
            self._model_name = model.model
            self._classifier = model
        self.capabilities = BackendCapabilities(
            tasks=None,
            max_options=None,
            max_attempts=1,
            score_kinds=_SCORE_KINDS,
            budget_fidelity="conservative",
        )
        self.profile = BackendProfile(
            adapter="langchain_typesafe",
            adapter_version="1",
            scorer_version="1",
            parser_version="1",
            provider="typesafe",
            provider_profile="typesafe.systemone.v1",
            model=self._model_name,
            revision=None,
            artifact_digest=None,
            tokenizer_revision=None,
            template_revision=None,
            mode="decision",
            generation_settings={},
            context_window_tokens=None,
            max_request_bytes=max_request_bytes,
            output_reserve=16,
            budget_attestation=None,
            dependency_versions={"langchain-typesafe": langchain_typesafe.__version__},
        )

    def count_units(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]:
        return (dict(questions),)

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        del questions
        budget = self.profile.max_request_bytes
        if budget is None:
            return None
        return budget - self.count_units(render_state(state)) - self.profile.output_reserve

    def render(self, state: State, questions: Mapping[str, Question]) -> str:
        del questions
        return render_state(state)

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        with request.budget.acquire(
            request.deadline,
            logical_index=request.logical_index,
            attempt=0,
        ) as permit:
            self._check_deadline(request, questions)
            response = self._complete(state, questions)
            return self._result(response, questions, permit)

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        async with request.budget.aacquire(
            request.deadline,
            logical_index=request.logical_index,
            attempt=0,
        ) as permit:
            self._check_deadline(request, questions)
            response = await self._acomplete(state, questions)
            return self._result(response, questions, permit)

    def _complete(self, state: State, questions: Mapping[str, Question]) -> Any:
        try:
            return self._ensure().invoke(self._request(state, questions))
        except TypeSafeError as error:
            raise BackendError(self.name, _reason(error), question_ids=questions) from None

    async def _acomplete(self, state: State, questions: Mapping[str, Question]) -> Any:
        try:
            return await self._ensure().ainvoke(self._request(state, questions))
        except TypeSafeError as error:
            raise BackendError(self.name, _reason(error), question_ids=questions) from None

    def _request(self, state: State, questions: Mapping[str, Question]) -> ClassifierRequest:
        return {"state": render_state(state), "questions": typesafe_questions(questions)}

    def _ensure(self) -> DecisionClassifier:
        classifier = self._classifier
        if classifier is None:
            classifier = TypeSafeClassifier(model=self._model_name)
            self._classifier = classifier
        return classifier

    def _result(
        self,
        response: Any,
        questions: Mapping[str, Question],
        permit: RequestPermit,
    ) -> BackendResult:
        try:
            answers = answers_from_response(response, questions)
        except BackendError:
            raise
        except (TypeError, ValueError, AttributeError):
            raise BackendError(self.name, "malformed_answer", question_ids=questions) from None
        usage = getattr(response, "usage", None)
        return BackendResult(
            answers=answers,
            usage=(
                BackendUsage(
                    permit_id=permit.permit_id,
                    attempt=0,
                    input_tokens=_token(getattr(usage, "input_tokens", None)),
                    output_tokens=_token(getattr(usage, "output_tokens", None)),
                ),
            ),
        )

    def _check_deadline(self, request: RequestContext, questions: Mapping[str, Question]) -> None:
        if request.deadline is not None and time.monotonic() >= request.deadline:
            raise DeadlineExceeded(self.name, question_ids=questions)


def resolve_judge(model: ModelSpec | None) -> Backend | None:
    """Wrap a TypeSafe model name or classifier. An existing backend is kept."""

    if model is None:
        return None
    if isinstance(model, str):
        return Judge(model)
    if isinstance(model, Judge):
        return model
    if hasattr(model, "decide") and hasattr(model, "capabilities") and hasattr(model, "profile"):
        return cast(Backend, model)
    if hasattr(model, "invoke") and hasattr(model, "ainvoke"):
        return Judge(cast(DecisionClassifier, model))
    raise PolicyError("model must be a TypeSafe model name or TypeSafeClassifier")


def _choice_scores(
    probabilities: Mapping[str, float],
    options: tuple[str, ...],
) -> dict[str, float]:
    values: list[float] = []
    for option in options:
        raw = probabilities.get(option)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise BackendError("typesafe", "malformed_answer")
        values.append(float(raw))
    return dict(zip(options, _normalize(values), strict=True))


def _level_scores(probabilities: Mapping[int, float], width: int) -> tuple[float, ...]:
    values: list[float] = []
    for index in range(width):
        raw = probabilities.get(index)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise BackendError("typesafe", "malformed_answer")
        values.append(float(raw))
    return tuple(_normalize(values))


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise BackendError("typesafe", "malformed_answer")
    if abs(total - 1.0) > 1e-3:
        return [value / total for value in values]
    return values


def _token(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _reason(error: Exception) -> str:
    if type(error).__name__ == "TypeSafeAPITimeoutError":
        return "timeout"
    status = getattr(error, "status_code", None)
    if status == 429:
        return "rate_limited"
    if isinstance(status, int) and status >= 500:
        return "upstream_error"
    if isinstance(status, int) and status >= 400:
        return "rejected"
    return "connection_error"


__all__ = [
    "AsyncBackend",
    "Backend",
    "BackendCapabilities",
    "BackendProfile",
    "BackendResult",
    "BackendUsage",
    "DecisionClassifier",
    "DecisionProfile",
    "Judge",
    "ModelSpec",
    "RequestBudget",
    "RequestContext",
    "RequestPermit",
    "RequestProfile",
    "SyncBackend",
    "answers_from_response",
    "attestation_digest_for",
    "render_state",
    "resolve_judge",
    "typesafe_questions",
]
