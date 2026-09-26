"""Llama Guard 4 adapter for hazard.any and hazard.S1-S14."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import Any, Literal, Never, Self, SupportsIndex, cast

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
from jes.backends._tokens import (
    Tokenize,
    assert_single_token_labels,
    default_label_tokenize,
    utf8_bytes,
)
from jes.errors import BackendError, PolicyError
from jes.questions import Question, ScoreKind, YesNo, YesNoAnswer
from jes.types import State

LLAMA_GUARD_SPECIAL_TOKENS = 8
LLAMA_GUARD_OUTPUT_RESERVE = 32
GROQ_CONTEXT_TOKENS = 131_072

HAZARD_TITLES: Mapping[str, str] = {
    "S1": "Violent crimes",
    "S2": "Non-violent crimes",
    "S3": "Sex-related crimes",
    "S4": "Child sexual exploitation",
    "S5": "Defamation",
    "S6": "Specialized advice",
    "S7": "Privacy",
    "S8": "Intellectual property",
    "S9": "Indiscriminate weapons",
    "S10": "Hate",
    "S11": "Suicide and self-harm",
    "S12": "Sexual content",
    "S13": "Elections",
    "S14": "Code interpreter abuse",
}
HAZARD_CODES: tuple[str, ...] = tuple(HAZARD_TITLES)

CompleteFn = Callable[[str], object]


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


def _score_kinds(logprobs: bool) -> Mapping[str, frozenset[ScoreKind]]:
    any_kind: ScoreKind = "probability" if logprobs else "label"
    kinds: dict[str, frozenset[ScoreKind]] = {"hazard.any": frozenset({any_kind})}
    for code in HAZARD_CODES:
        kinds[f"hazard.{code}"] = frozenset({"label"})
    return kinds


def _tasks() -> frozenset[str]:
    return frozenset(_score_kinds(False))


def category_codes(questions: Mapping[str, Question], *, custom: bool) -> tuple[str, ...]:
    requested: list[str] = []
    for question in questions.values():
        if not question.task.startswith("hazard.S"):
            continue
        code = question.task.removeprefix("hazard.")
        if code not in HAZARD_TITLES:
            raise PolicyError(f"unknown hazard category {code}")
        if code not in requested:
            requested.append(code)
    if not custom or not requested:
        return HAZARD_CODES
    return tuple(sorted(requested, key=lambda code: int(code[1:])))


def render_prompt(state: State, codes: Sequence[str]) -> str:
    lines = [
        "Classify the last message as safe or unsafe.",
        "If it is safe, reply with safe.",
        "If it is unsafe, reply with unsafe and then the matching codes on the next line.",
        "Codes:",
    ]
    lines.extend(f"{code}: {HAZARD_TITLES[code]}" for code in codes)
    messages: list[dict[str, str]] = [{"role": "system", "content": "\n".join(lines)}]
    for message in state.history:
        messages.append({"role": message.role, "content": message.text})
    if state.prompt:
        messages.append({"role": "user", "content": state.prompt})
    if state.question:
        messages.append({"role": "user", "content": state.question})
    role = "assistant" if state.stage == "output" else "user"
    messages.append({"role": role, "content": state.text})
    return dumps(messages).decode("utf-8")


def parse_completion(text: str) -> tuple[bool, tuple[str, ...]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise BackendError("llama_guard", "malformed_answer")
    head = lines[0].casefold()
    if head == "safe":
        if len(lines) != 1:
            raise BackendError("llama_guard", "malformed_answer")
        return False, ()
    if head != "unsafe":
        raise BackendError("llama_guard", "malformed_answer")
    if len(lines) == 1:
        return True, ()
    if len(lines) != 2:
        raise BackendError("llama_guard", "malformed_answer")
    codes = tuple(part.strip().upper() for part in lines[1].split(",") if part.strip())
    unknown = [code for code in codes if code not in HAZARD_TITLES]
    if unknown:
        raise BackendError("llama_guard", "unknown_category")
    return True, codes


def _token_mass(top: Sequence[Mapping[str, object]], label: str) -> float:
    total = 0.0
    for item in top:
        token = item.get("token")
        logprob = item.get("logprob")
        if not isinstance(token, str):
            continue
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            continue
        if token.strip().casefold() != label:
            continue
        total += math.exp(float(logprob))
    return total


def unsafe_probability(top: Sequence[Mapping[str, object]]) -> tuple[float, float]:
    safe = _token_mass(top, "safe")
    unsafe = _token_mass(top, "unsafe")
    if safe <= 0.0 or unsafe <= 0.0:
        raise BackendError("llama_guard", "missing_logprob_candidate")
    total = safe + unsafe
    return unsafe / total, min(total, 1.0)


def _as_object_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def _top_from_choice(choice: Mapping[str, object]) -> list[Mapping[str, object]] | None:
    logprobs = _as_object_dict(choice.get("logprobs"))
    if logprobs is None:
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    first = _as_object_dict(cast(list[object], content)[0])
    if first is None:
        return None
    top = first.get("top_logprobs")
    if not isinstance(top, list):
        return None
    typed: list[Mapping[str, object]] = []
    for item in cast(list[object], top):
        if isinstance(item, dict):
            typed.append(cast(dict[str, object], item))
    return typed


def extract_completion(raw: object) -> tuple[str, Sequence[Mapping[str, object]] | None]:
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, bytes):
        try:
            raw = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise BackendError("llama_guard", "malformed_answer") from None
    payload = _as_object_dict(raw)
    if payload is None:
        raise BackendError("llama_guard", "malformed_answer")
    if isinstance(payload.get("text"), str):
        top = payload.get("top_logprobs")
        typed: list[Mapping[str, object]] | None = None
        if isinstance(top, list):
            typed = []
            for item in cast(list[object], top):
                if isinstance(item, dict):
                    typed.append(cast(dict[str, object], item))
        return str(payload["text"]), typed
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendError("llama_guard", "malformed_answer")
    choice = _as_object_dict(cast(list[object], choices)[0])
    if choice is None:
        raise BackendError("llama_guard", "malformed_answer")
    message = _as_object_dict(choice.get("message"))
    content = None if message is None else message.get("content")
    if not isinstance(content, str):
        raise BackendError("llama_guard", "malformed_answer")
    return content, _top_from_choice(choice)


def answers_for(
    questions: Mapping[str, Question],
    *,
    unsafe: bool,
    codes: tuple[str, ...],
    any_score: float,
    any_kind: ScoreKind,
    confidence: float | None,
) -> dict[str, YesNoAnswer]:
    listed = set(codes)
    answers: dict[str, YesNoAnswer] = {}
    for key, question in questions.items():
        if question.task == "hazard.any":
            answers[key] = YesNoAnswer(any_score, any_kind, confidence)
            continue
        code = question.task.removeprefix("hazard.")
        score = 1.0 if unsafe and code in listed else 0.0
        answers[key] = YesNoAnswer(score, "label")
    return answers


class LlamaGuard4:
    """Hazard classifier. Category scores are labels; hazard.any follows logprobs mode."""

    name = "llama_guard"

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
        template_revision: str | None,
        artifact_digest: str | None,
        logprobs: bool,
        custom_categories: bool,
        context_window_tokens: int,
        timeout_s: float,
        transport: httpx.BaseTransport | None,
        async_transport: httpx.AsyncBaseTransport | None,
        complete_fn: CompleteFn | None,
        tokenize: Tokenize | None,
    ) -> None:
        if not model.strip() or not revision.strip():
            raise PolicyError("model and revision are required")
        spec = require_known_provider_profile(provider_profile, "llama_guard")
        if bool(spec.get("local")) != local:
            raise PolicyError("provider profile does not match Llama Guard mode")
        if logprobs and not spec.get("supports_logprobs"):
            raise PolicyError("provider profile does not support logprobs")
        if logprobs:
            assert_single_token_labels(("safe", "unsafe"), tokenize or default_label_tokenize)
        if local and complete_fn is None:
            _require_extra()
        attempts = spec.get("max_attempts", 1)
        max_attempts = attempts if isinstance(attempts, int) and attempts >= 1 else 1
        mode = "logprobs" if logprobs else "labels"
        self.capabilities = BackendCapabilities(
            tasks=_tasks(),
            max_options=2,
            max_attempts=max_attempts,
            score_kinds=_score_kinds(logprobs),
            budget_fidelity="exact" if tokenize is not None else "conservative",
        )
        self.profile = BackendProfile(
            adapter="llama_guard",
            adapter_version=ADAPTER_VERSION,
            scorer_version=SCORER_VERSION,
            parser_version=PARSER_VERSION,
            provider=provider_profile.split(".", 1)[0],
            provider_profile=provider_profile,
            model=model,
            revision=revision,
            artifact_digest=artifact_digest,
            tokenizer_revision=tokenizer_revision,
            template_revision=template_revision,
            mode=mode,
            generation_settings={
                "logprobs": logprobs,
                "special_tokens": LLAMA_GUARD_SPECIAL_TOKENS,
                "temperature": 0,
            },
            context_window_tokens=context_window_tokens,
            max_request_bytes=None,
            output_reserve=LLAMA_GUARD_OUTPUT_RESERVE,
            budget_attestation=None,
            dependency_versions=_versions(),
        )
        self._logprobs = logprobs
        self._custom = custom_categories
        self._base_url = None if base_url is None else base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._transport = transport
        self._async_transport = async_transport
        self._complete_fn = complete_fn
        self._tokenize = tokenize
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self.calls: list[str] = []

    @classmethod
    def endpoint(
        cls,
        base_url: str,
        *,
        model: str,
        provider_profile: str,
        revision: str,
        tokenizer_revision: str | None = None,
        template_revision: str | None = None,
        artifact_digest: str | None = None,
        api_key: str | None = None,
        logprobs: bool | None = None,
        context_window_tokens: int | None = None,
        model_config: Mapping[str, object] | None = None,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        tokenize: Tokenize | None = None,
    ) -> LlamaGuard4:
        spec = require_known_provider_profile(provider_profile, "llama_guard")
        if spec.get("local"):
            raise PolicyError("provider profile does not match Llama Guard mode")
        resolved_logprobs = bool(spec.get("supports_logprobs")) if logprobs is None else logprobs
        window = _resolve_window(spec, context_window_tokens, model_config)
        return cls(
            model=model,
            provider_profile=provider_profile,
            revision=revision,
            local=False,
            base_url=base_url,
            api_key=api_key,
            tokenizer_revision=tokenizer_revision,
            template_revision=template_revision,
            artifact_digest=artifact_digest,
            logprobs=resolved_logprobs,
            custom_categories=bool(spec.get("custom_categories")),
            context_window_tokens=window,
            timeout_s=timeout_s,
            transport=transport,
            async_transport=async_transport,
            complete_fn=None,
            tokenize=tokenize,
        )

    @classmethod
    def local(
        cls,
        model: str = "meta-llama/Llama-Guard-4-12B",
        *,
        provider_profile: str = "transformers.causal_lm.v1",
        revision: str,
        artifact_digest: str | None = None,
        template_revision: str | None = None,
        logprobs: bool = True,
        context_window_tokens: int | None = None,
        model_config: Mapping[str, object] | None = None,
        complete_fn: CompleteFn | None = None,
        tokenize: Tokenize | None = None,
    ) -> LlamaGuard4:
        spec = require_known_provider_profile(provider_profile, "llama_guard")
        window = _resolve_window(spec, context_window_tokens, model_config)
        return cls(
            model=model,
            provider_profile=provider_profile,
            revision=revision,
            local=True,
            base_url=None,
            api_key=None,
            tokenizer_revision=None,
            template_revision=template_revision,
            artifact_digest=artifact_digest,
            logprobs=logprobs,
            custom_categories=bool(spec.get("custom_categories")),
            context_window_tokens=window,
            timeout_s=10.0,
            transport=None,
            async_transport=None,
            complete_fn=complete_fn,
            tokenize=tokenize,
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
            f"LlamaGuard4(model={self.profile.model!r}, mode={self.profile.mode!r}, "
            f"provider_profile={self.profile.provider_profile!r})"
        )

    def __copy__(self) -> LlamaGuard4:
        raise TypeError("LlamaGuard4 cannot be copied")

    def __deepcopy__(self, memo: object) -> LlamaGuard4:
        del memo
        raise TypeError("LlamaGuard4 cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("LlamaGuard4 cannot be pickled")

    def render(self, state: State, questions: Mapping[str, Question]) -> str:
        messages = render_prompt(state, category_codes(questions, custom=self._custom))
        return self._body(messages).decode("utf-8")

    def count_units(self, text: str) -> int:
        if self._tokenize is not None:
            return len(list(self._tokenize(text)))
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
        return window - LLAMA_GUARD_SPECIAL_TOKENS - self.profile.output_reserve - used

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
            raw = self._complete(rendered, request)
            return self._result(raw, questions, permit.permit_id, self.count_units(rendered))

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
            raw = await self._acomplete(rendered, request)
            return self._result(raw, questions, permit.permit_id, self.count_units(rendered))

    def _result(
        self,
        raw: object,
        questions: Mapping[str, Question],
        permit_id: str,
        input_tokens: int,
    ) -> BackendResult:
        text, top = extract_completion(raw)
        unsafe, codes = parse_completion(text)
        if self._logprobs:
            if top is None:
                raise BackendError(self.name, "missing_logprob_candidate", question_ids=questions)
            any_score, confidence = unsafe_probability(top)
            any_kind: Literal["probability", "label"] = "probability"
        else:
            any_score = 1.0 if unsafe else 0.0
            confidence = None
            any_kind = "label"
        return BackendResult(
            answers=answers_for(
                questions,
                unsafe=unsafe,
                codes=codes,
                any_score=any_score,
                any_kind=any_kind,
                confidence=confidence,
            ),
            usage=(
                BackendUsage(
                    permit_id=permit_id,
                    attempt=0,
                    input_tokens=input_tokens,
                    output_tokens=1,
                ),
            ),
        )

    def _complete(self, payload: str, request: RequestContext) -> object:
        if self._complete_fn is not None:
            return self._complete_fn(payload)
        status, body = self._post(payload, request)
        log_http("response", backend=self.name, status_code=status)
        if status != 200:
            raise BackendError(self.name, status_reason(status), status_code=status)
        return body

    async def _acomplete(self, payload: str, request: RequestContext) -> object:
        if self._complete_fn is not None:
            return self._complete_fn(payload)
        status, body = await self._apost(payload, request)
        log_http("response", backend=self.name, status_code=status)
        if status != 200:
            raise BackendError(self.name, status_reason(status), status_code=status)
        return body

    def _post(self, payload: str, request: RequestContext) -> tuple[int, bytes]:
        if self._base_url is None:
            raise BackendError(self.name, "connection_error")
        return request_capped(
            self._sync_client(),
            "POST",
            f"{self._base_url}/chat/completions",
            backend=self.name,
            content=payload.encode("utf-8"),
            headers=_headers(self._api_key),
            timeout=http_timeout(request.deadline, self._timeout_s),
            max_bytes=request.max_response_bytes,
        )

    async def _apost(self, payload: str, request: RequestContext) -> tuple[int, bytes]:
        if self._base_url is None:
            raise BackendError(self.name, "connection_error")
        return await arequest_capped(
            self._async_client_obj(),
            "POST",
            f"{self._base_url}/chat/completions",
            backend=self.name,
            content=payload.encode("utf-8"),
            headers=_headers(self._api_key),
            timeout=http_timeout(request.deadline, self._timeout_s),
            max_bytes=request.max_response_bytes,
        )

    def _body(self, prompt: str) -> bytes:
        payload: dict[str, object] = {
            "messages": json.loads(prompt),
            "model": self.profile.model,
            "temperature": 0,
        }
        if self._logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        return dumps(payload)

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


def _require_extra() -> None:
    missing_transformers = importlib.util.find_spec("transformers") is None
    missing_torch = importlib.util.find_spec("torch") is None
    if missing_transformers or missing_torch:
        raise PolicyError("LlamaGuard4.local requires the jes[llama-guard] extra")


def _resolve_window(
    spec: Mapping[str, object],
    context_window_tokens: int | None,
    model_config: Mapping[str, object] | None,
) -> int:
    if context_window_tokens is not None:
        return context_window_tokens
    if model_config is not None and isinstance(model_config.get("max_position_embeddings"), int):
        window = model_config["max_position_embeddings"]
        if isinstance(window, int) and window >= 1:
            return window
    profile_window = spec.get("context_window_tokens")
    if isinstance(profile_window, int) and profile_window >= 1:
        return profile_window
    raise PolicyError("Llama Guard requires a total context window or model config")


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json; charset=utf-8"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _validate_questions(questions: Mapping[str, Question]) -> None:
    allowed = _tasks()
    for question in questions.values():
        if not isinstance(question, YesNo) or question.task not in allowed:
            raise PolicyError("llama_guard answers hazard yes/no questions only")


__all__ = ["HAZARD_CODES", "LlamaGuard4"]
