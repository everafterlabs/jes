"""Prompt Guard 2 adapter. Classifies explicit instruction overrides only."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import Any, Never, Self, SupportsIndex, cast

import httpx

from jes.backends import (
    BackendCapabilities,
    BackendProfile,
    BackendResult,
    BackendUsage,
    RequestContext,
)
from jes.backends._codec import dumps
from jes.backends._http import (
    arequest_capped,
    http_timeout,
    log_http,
    request_capped,
    status_reason,
)
from jes.backends._profiles import (
    ADAPTER_VERSION,
    PARSER_VERSION,
    SCORER_VERSION,
    require_known_provider_profile,
)
from jes.backends._tokens import utf8_bytes
from jes.errors import BackendError, PolicyError
from jes.questions import Question, YesNo, YesNoAnswer
from jes.types import State

PROMPT_GUARD_WINDOW = 512
PROMPT_GUARD_SPECIAL_TOKENS = 2
PROMPT_GUARD_OUTPUT_RESERVE = 1
_MALICIOUS_LABEL = "LABEL_1"
_MALICIOUS_INDEX = 1

ClassifyFn = Callable[[str], object]


def _softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    scaled = [math.exp(value - peak) for value in logits]
    total = sum(scaled)
    if total <= 0:
        raise BackendError("prompt_guard", "malformed_answer")
    return [value / total for value in scaled]


def _probability(raw: object) -> float:
    if isinstance(raw, dict):
        payload = cast(dict[str, object], raw)
        scores = payload.get("scores")
        if isinstance(scores, dict):
            value = cast(dict[str, object], scores).get(_MALICIOUS_LABEL)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BackendError("prompt_guard", "malformed_answer")
            return float(value)
        if "logits" in payload:
            return _probability(payload.get("logits"))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        logits: list[float] = []
        for item in cast(Sequence[object], raw):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise BackendError("prompt_guard", "malformed_answer")
            logits.append(float(item))
        if len(logits) <= _MALICIOUS_INDEX:
            raise BackendError("prompt_guard", "malformed_answer")
        return _softmax(logits)[_MALICIOUS_INDEX]
    raise BackendError("prompt_guard", "malformed_answer")


def _versions() -> Mapping[str, str]:
    versions = {
        "httpx": httpx.__version__,
        "torch": "not-installed",
        "transformers": "not-installed",
    }
    for name in ("transformers", "torch"):
        if importlib.util.find_spec(name) is None:
            continue
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        versions[name] = version if isinstance(version, str) else "unknown"
    return MappingProxyType(versions)


class PromptGuard2:
    """Binary injection classifier. It does not answer indirect_injection."""

    name = "prompt_guard"

    def __init__(
        self,
        *,
        model: str,
        provider_profile: str,
        revision: str,
        local: bool,
        base_url: str | None,
        api_key: str | None,
        tokenizer_revision: str | None,
        artifact_digest: str | None,
        context_window_tokens: int,
        timeout_s: float,
        transport: httpx.BaseTransport | None,
        async_transport: httpx.AsyncBaseTransport | None,
        classify_fn: ClassifyFn | None,
    ) -> None:
        if not model.strip() or not revision.strip():
            raise PolicyError("model and revision are required")
        spec = require_known_provider_profile(provider_profile, "prompt_guard")
        if bool(spec.get("local")) != local:
            raise PolicyError("provider profile does not match Prompt Guard mode")
        if local and classify_fn is None:
            self._require_extra()
        attempts = spec.get("max_attempts", 1)
        max_attempts = attempts if isinstance(attempts, int) and attempts >= 1 else 1
        self.capabilities = BackendCapabilities(
            tasks=frozenset({"injection"}),
            max_options=2,
            max_attempts=max_attempts,
            score_kinds={"injection": frozenset({"probability"})},
            budget_fidelity="conservative",
        )
        self.profile = BackendProfile(
            adapter="prompt_guard",
            adapter_version=ADAPTER_VERSION,
            scorer_version=SCORER_VERSION,
            parser_version=PARSER_VERSION,
            provider=provider_profile.split(".", 1)[0],
            provider_profile=provider_profile,
            model=model,
            revision=revision,
            artifact_digest=artifact_digest,
            tokenizer_revision=tokenizer_revision,
            template_revision=None,
            mode="probability",
            generation_settings={
                "special_tokens": PROMPT_GUARD_SPECIAL_TOKENS,
                "temperature": 0,
            },
            context_window_tokens=context_window_tokens,
            max_request_bytes=None,
            output_reserve=PROMPT_GUARD_OUTPUT_RESERVE,
            budget_attestation=None,
            dependency_versions=_versions(),
        )
        self._base_url = None if base_url is None else base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._transport = transport
        self._async_transport = async_transport
        self._classify_fn = classify_fn
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self.calls: list[str] = []

    @staticmethod
    def _require_extra() -> None:
        missing_extra = (
            importlib.util.find_spec("transformers") is None
            or importlib.util.find_spec("torch") is None
        )
        if missing_extra:
            raise PolicyError("PromptGuard2.local requires the jes[prompt-guard] extra")

    @classmethod
    def endpoint(
        cls,
        base_url: str,
        *,
        model: str,
        provider_profile: str,
        revision: str,
        tokenizer_revision: str | None = None,
        api_key: str | None = None,
        context_window_tokens: int = PROMPT_GUARD_WINDOW,
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> PromptGuard2:
        return cls(
            model=model,
            provider_profile=provider_profile,
            revision=revision,
            local=False,
            base_url=base_url,
            api_key=api_key,
            tokenizer_revision=tokenizer_revision,
            artifact_digest=None,
            context_window_tokens=context_window_tokens,
            timeout_s=timeout_s,
            transport=transport,
            async_transport=async_transport,
            classify_fn=None,
        )

    @classmethod
    def local(
        cls,
        model: str = "meta-llama/Llama-Prompt-Guard-2-86M",
        *,
        provider_profile: str = "transformers.sequence_classification.v1",
        revision: str,
        artifact_digest: str | None = None,
        context_window_tokens: int = PROMPT_GUARD_WINDOW,
        classify_fn: ClassifyFn | None = None,
    ) -> PromptGuard2:
        return cls(
            model=model,
            provider_profile=provider_profile,
            revision=revision,
            local=True,
            base_url=None,
            api_key=None,
            tokenizer_revision=None,
            artifact_digest=artifact_digest,
            context_window_tokens=context_window_tokens,
            timeout_s=5.0,
            transport=None,
            async_transport=None,
            classify_fn=classify_fn,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        self.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def __repr__(self) -> str:
        return (
            f"PromptGuard2(model={self.profile.model!r}, "
            f"provider_profile={self.profile.provider_profile!r})"
        )

    def __copy__(self) -> PromptGuard2:
        raise TypeError("PromptGuard2 cannot be copied")

    def __deepcopy__(self, memo: object) -> PromptGuard2:
        del memo
        raise TypeError("PromptGuard2 cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("PromptGuard2 cannot be pickled")

    def render(self, state: State, questions: Mapping[str, Question]) -> str:
        del questions
        return state.text

    def count_units(self, text: str) -> int:
        return utf8_bytes(text)

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]:
        _validate_questions(questions)
        return (dict(questions),)

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        window = self.profile.context_window_tokens
        if window is None:
            return None
        used = self.count_units(self.render(state, questions))
        return window - PROMPT_GUARD_SPECIAL_TOKENS - self.profile.output_reserve - used

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        _validate_questions(questions)
        with request.budget.acquire(
            request.deadline,
            logical_index=request.logical_index,
            attempt=0,
        ) as permit:
            rendered = self.render(state, questions)
            self.calls.append(rendered)
            score = self._score(rendered, request)
            return _result(questions, score, permit.permit_id, self.count_units(rendered))

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        _validate_questions(questions)
        async with request.budget.aacquire(
            request.deadline,
            logical_index=request.logical_index,
            attempt=0,
        ) as permit:
            rendered = self.render(state, questions)
            self.calls.append(rendered)
            score = await self._ascore(rendered, request)
            return _result(questions, score, permit.permit_id, self.count_units(rendered))

    def _score(self, text: str, request: RequestContext) -> float:
        if self._classify_fn is not None:
            return _probability(self._classify_fn(text))
        status, body = self._post(text, request)
        return _read_body(status, body)

    async def _ascore(self, text: str, request: RequestContext) -> float:
        if self._classify_fn is not None:
            return _probability(self._classify_fn(text))
        status, body = await self._post_async(text, request)
        return _read_body(status, body)

    def _post(self, text: str, request: RequestContext) -> tuple[int, bytes]:
        if self._base_url is None:
            raise BackendError(self.name, "connection_error")
        client = self._sync_client()
        return request_capped(
            client,
            "POST",
            f"{self._base_url}/classify",
            backend=self.name,
            content=dumps({"input": text, "model": self.profile.model}),
            headers=_headers(self._api_key),
            timeout=http_timeout(request.deadline, self._timeout_s),
            max_bytes=request.max_response_bytes,
        )

    async def _post_async(self, text: str, request: RequestContext) -> tuple[int, bytes]:
        if self._base_url is None:
            raise BackendError(self.name, "connection_error")
        return await arequest_capped(
            self._async_client_obj(),
            "POST",
            f"{self._base_url}/classify",
            backend=self.name,
            content=dumps({"input": text, "model": self.profile.model}),
            headers=_headers(self._api_key),
            timeout=http_timeout(request.deadline, self._timeout_s),
            max_bytes=request.max_response_bytes,
        )

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.Client(**kwargs)
        return self._client

    def _async_client_obj(self) -> httpx.AsyncClient:
        if self._async_client is None:
            kwargs: dict[str, Any] = {}
            if self._async_transport is not None:
                kwargs["transport"] = self._async_transport
            elif self._transport is not None:
                kwargs["transport"] = self._transport
            self._async_client = httpx.AsyncClient(**kwargs)
        return self._async_client


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json; charset=utf-8"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _read_body(status: int, body: bytes) -> float:
    log_http("response", backend="prompt_guard", status_code=status)
    if status != 200:
        raise BackendError("prompt_guard", status_reason(status), status_code=status)
    try:
        parsed: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BackendError("prompt_guard", "malformed_answer") from None
    return _probability(parsed)


def _validate_questions(questions: Mapping[str, Question]) -> None:
    for question in questions.values():
        if not isinstance(question, YesNo) or question.task != "injection":
            raise PolicyError("prompt_guard answers injection yes/no questions only")


def _result(
    questions: Mapping[str, Question],
    score: float,
    permit_id: str,
    input_tokens: int,
) -> BackendResult:
    answers = {
        key: YesNoAnswer(score, "probability")
        for key in questions
    }
    return BackendResult(
        answers=answers,
        usage=(
            BackendUsage(
                permit_id=permit_id,
                attempt=0,
                input_tokens=input_tokens,
                output_tokens=1,
            ),
        ),
    )


__all__ = ["PromptGuard2"]
