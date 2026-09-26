"""Backend protocols and immutable backend metadata."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

from jes.errors import PolicyError
from jes.questions import Answer, Question, ScoreKind
from jes.types import Stage, State

_K = TypeVar("_K")
_V = TypeVar("_V")


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
        object.__setattr__(
            self,
            "generation_settings",
            _freeze_mapping(self.generation_settings),
        )
        object.__setattr__(
            self,
            "dependency_versions",
            _freeze_mapping(self.dependency_versions),
        )

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


@dataclass(frozen=True, slots=True, repr=False)
class _ProfileAttestation:
    profile_fingerprint: str
    source: Literal["installed_artifact", "provider_registry", "signed_deployment"]
    evidence_digest: str
    signature: str


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


if TYPE_CHECKING:
    from jes.backends.litellm_judge import LiteLLMJudge
    from jes.backends.llama_guard import LlamaGuard4
    from jes.backends.prompt_guard import PromptGuard2
    from jes.backends.system_one import SystemOne


def __getattr__(name: str) -> object:
    if name == "LiteLLMJudge":
        from jes.backends.litellm_judge import LiteLLMJudge as _LiteLLMJudge

        return _LiteLLMJudge
    if name == "LlamaGuard4":
        from jes.backends.llama_guard import LlamaGuard4 as _LlamaGuard4

        return _LlamaGuard4
    if name == "PromptGuard2":
        from jes.backends.prompt_guard import PromptGuard2 as _PromptGuard2

        return _PromptGuard2
    if name == "SystemOne":
        from jes.backends.system_one import SystemOne as _SystemOne

        return _SystemOne
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncBackend",
    "Backend",
    "BackendCapabilities",
    "BackendProfile",
    "BackendResult",
    "BackendUsage",
    "DecisionProfile",
    "LiteLLMJudge",
    "LlamaGuard4",
    "PromptGuard2",
    "RequestBudget",
    "RequestContext",
    "RequestPermit",
    "RequestProfile",
    "SyncBackend",
    "SystemOne",
]
