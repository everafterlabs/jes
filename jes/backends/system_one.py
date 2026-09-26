"""System One adapter for Laya, Jev, and compatible endpoints."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import random
import time
from collections.abc import Mapping
from types import MappingProxyType, TracebackType
from typing import Any, Literal, Never, Self, SupportsIndex, cast

import httpx

from jes._env import load_project_env
from jes.backends import (
    BackendCapabilities,
    BackendProfile,
    BackendResult,
    BackendUsage,
    RequestContext,
)
from jes.errors import BackendError, DeadlineExceeded, PolicyError
from jes.questions import Question
from jes.types import State

from ._codec import encode_question, parse_system_one_response, serialized_system_one_request
from ._http import (
    arequest_capped,
    http_timeout,
    log_http,
    remaining_timeout,
    request_capped,
    retryable_status,
    status_reason,
)
from ._profiles import (
    ADAPTER_VERSION,
    JEV_MODELS,
    JEV_REQUEST_TOKENS,
    JEV_STATE_TOKENS,
    PARSER_VERSION,
    SCORER_VERSION,
    canonical_manifest_bytes,
    conservative_byte_budget,
    make_attestation,
    output_reserve_for,
    require_known_provider_profile,
    verify_manifest_signature,
)
from ._tokens import load_tokenizer, utf8_bytes

Mode = Literal["local", "hosted", "in_process"]


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json; charset=utf-8"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _dependency_versions() -> Mapping[str, str]:
    return MappingProxyType({"httpx": httpx.__version__})


class SystemOne:
    """HTTP and in-process System One backend."""

    name = "system_one"

    def __init__(
        self,
        *,
        mode: Mode,
        base_url: str | None,
        model: str,
        provider_profile: str,
        revision: str | None,
        artifact_digest: str | None,
        tokenizer_revision: str | None,
        template_revision: str | None,
        api_key: str | None,
        context_window_tokens: int | None,
        max_request_bytes: int | None,
        timeout_s: float,
        transport: httpx.BaseTransport | None,
        async_transport: httpx.AsyncBaseTransport | None,
        query_fn: Any | None,
    ) -> None:
        if context_window_tokens is not None and max_request_bytes is not None:
            raise PolicyError("supply exactly one of context_window_tokens or max_request_bytes")
        family = "jev" if model in JEV_MODELS else "laya"
        spec = require_known_provider_profile(provider_profile, family)
        del spec
        tokenizer = load_tokenizer(tokenizer_revision)
        exact = tokenizer is not None and context_window_tokens is not None
        if exact:
            budget_fidelity: Literal["exact", "conservative"] = "exact"
            token_budget = context_window_tokens
            byte_budget = None
        else:
            if context_window_tokens is not None and tokenizer_revision is not None:
                raise PolicyError("exact token budgets require the tokenizers extra")
            if context_window_tokens is not None:
                raise PolicyError("token budgets require a pinned tokenizer")
            budget_fidelity = "conservative"
            token_budget = None
            byte_budget = conservative_byte_budget(model, max_request_bytes=max_request_bytes)
        self.capabilities = BackendCapabilities(
            tasks=None,
            max_options=None,
            max_attempts=3,
            score_kinds={"*": frozenset({"probability"})},
            budget_fidelity=budget_fidelity,
        )
        self.profile = BackendProfile(
            adapter="system_one",
            adapter_version=ADAPTER_VERSION,
            scorer_version=SCORER_VERSION,
            parser_version=PARSER_VERSION,
            provider="typesafe" if family == "jev" else "laya",
            provider_profile=provider_profile,
            model=model,
            revision=revision,
            artifact_digest=artifact_digest,
            tokenizer_revision=tokenizer_revision,
            template_revision=template_revision,
            mode=mode,
            generation_settings={"temperature": 0},
            context_window_tokens=token_budget,
            max_request_bytes=byte_budget,
            output_reserve=output_reserve_for(model),
            budget_attestation=None,
            dependency_versions=_dependency_versions(),
        )
        self._model = model
        self._family = family
        self._base_url = None if base_url is None else base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._tokenize = tokenizer
        self._query_fn = query_fn
        self._transport = transport
        self._async_transport = async_transport
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._sleep = time.sleep
        self._asleep = asyncio.sleep
        self._rng = random.Random(0)

    @classmethod
    def local(
        cls,
        base_url: str = "http://127.0.0.1:8000",
        *,
        model: str = "english",
        provider_profile: str = "laya-serve.v1",
        revision: str | None = None,
        artifact_digest: str | None = None,
        deployment_manifest: Mapping[str, object] | None = None,
        trusted_manifest_keys: Mapping[str, bytes] | None = None,
        api_key: str | None = None,
        context_window_tokens: int | None = None,
        max_request_bytes: int | None = None,
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> SystemOne:
        keys = dict(trusted_manifest_keys or {})
        manifest_body = _mapping_body(deployment_manifest)
        backend = cls(
            mode="local",
            base_url=base_url,
            model=model,
            provider_profile=provider_profile,
            revision=revision,
            artifact_digest=artifact_digest,
            tokenizer_revision=_optional_str(manifest_body, "tokenizer_revision"),
            template_revision=_optional_str(manifest_body, "template_revision"),
            api_key=api_key,
            context_window_tokens=context_window_tokens,
            max_request_bytes=max_request_bytes,
            timeout_s=timeout_s,
            transport=transport,
            async_transport=async_transport,
            query_fn=None,
        )
        if deployment_manifest is not None:
            backend._attest_manifest(deployment_manifest, keys)
        return backend

    @classmethod
    def hosted(
        cls,
        model: str = "jev-1.13.0",
        *,
        base_url: str = "https://api.typesafe.ai",
        api_key: str | None = None,
        provider_profile: str = "typesafe.jev.v1",
        revision: str | None = None,
        artifact_digest: str | None = None,
        context_window_tokens: int | None = None,
        max_request_bytes: int | None = None,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> SystemOne:
        if api_key is None:
            load_project_env()
            key = os.environ.get("TYPESAFE_API_KEY")
        else:
            key = api_key
        return cls(
            mode="hosted",
            base_url=base_url,
            model=model,
            provider_profile=provider_profile,
            revision=revision,
            artifact_digest=artifact_digest,
            tokenizer_revision=None,
            template_revision=None,
            api_key=key,
            context_window_tokens=context_window_tokens,
            max_request_bytes=max_request_bytes,
            timeout_s=timeout_s,
            transport=transport,
            async_transport=async_transport,
            query_fn=None,
        )

    @classmethod
    def in_process(
        cls,
        checkpoint: str = "english",
        *,
        provider_profile: str = "laya.in_process.v1",
        revision: str | None = None,
        artifact_digest: str | None = None,
        context_window_tokens: int | None = None,
        max_request_bytes: int | None = None,
        query_fn: Any | None = None,
    ) -> SystemOne:
        fn = query_fn
        if fn is None:
            if importlib.util.find_spec("laya") is None:
                raise PolicyError("SystemOne.in_process requires the jes[laya] extra")
            laya_module: Any = importlib.import_module("laya")
            fn = getattr(laya_module, "query", None) or getattr(laya_module, "decide", None)
            if fn is None:
                raise PolicyError("laya package does not expose query or decide")
        backend = cls(
            mode="in_process",
            base_url=None,
            model=checkpoint,
            provider_profile=provider_profile,
            revision=revision,
            artifact_digest=artifact_digest,
            tokenizer_revision=None,
            template_revision=None,
            api_key=None,
            context_window_tokens=context_window_tokens,
            max_request_bytes=max_request_bytes,
            timeout_s=5.0,
            transport=None,
            async_transport=None,
            query_fn=fn,
        )
        evidence = json.dumps(
            {"checkpoint": checkpoint, "revision": revision, "digest": artifact_digest},
            sort_keys=True,
        ).encode()
        if artifact_digest:
            make_attestation(
                backend.profile,
                source="installed_artifact",
                evidence=evidence,
                signature=artifact_digest,
            )
        return backend

    def _attest_manifest(
        self,
        manifest: Mapping[str, object],
        trusted_keys: Mapping[str, bytes],
    ) -> None:
        if not trusted_keys:
            raise PolicyError("caller labels without a trusted manifest key never qualify")
        body = _mapping_body(manifest)
        if body is None:
            raise PolicyError("deployment manifest body is required")
        key_id = manifest.get("key_id")
        signature = manifest.get("signature")
        if not isinstance(key_id, str) or not isinstance(signature, str):
            raise PolicyError("deployment manifest signature is incomplete")
        verify_manifest_signature(body, signature, trusted_keys, key_id)
        handshake = self._handshake()
        expected_window = body.get("context_window_tokens")
        expected_truncation = body.get("truncation", "fail")
        if handshake.get("model") != self._model:
            raise PolicyError("deployment handshake does not match the signed manifest")
        if handshake.get("context_window_tokens") != expected_window:
            raise PolicyError("deployment handshake does not match the signed manifest")
        if handshake.get("truncation") != expected_truncation:
            raise PolicyError("deployment handshake does not match the signed manifest")
        if body.get("model") != self._model:
            raise PolicyError("deployment manifest model does not match the backend")
        make_attestation(
            self.profile,
            source="signed_deployment",
            evidence=canonical_manifest_bytes(body),
            signature=signature,
        )

    def _handshake(self) -> dict[str, object]:
        if self._base_url is None:
            raise PolicyError("handshake requires an HTTP System One backend")
        status, body = request_capped(
            self._sync_client(),
            "GET",
            f"{self._base_url}/v1/health",
            backend=self.name,
            timeout=http_timeout(None, self._timeout_s),
            max_bytes=4_096,
        )
        if status != 200:
            raise PolicyError("deployment handshake failed")
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise PolicyError("deployment handshake is not JSON") from None
        if not isinstance(parsed, dict):
            raise PolicyError("deployment handshake is not JSON")
        return cast(dict[str, object], parsed)

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
            f"SystemOne(mode={self.profile.mode!r}, model={self.profile.model!r}, "
            f"provider_profile={self.profile.provider_profile!r})"
        )

    def __copy__(self) -> SystemOne:
        raise TypeError("SystemOne cannot be copied")

    def __deepcopy__(self, memo: object) -> SystemOne:
        del memo
        raise TypeError("SystemOne cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("SystemOne cannot be pickled")

    def render(self, state: State, questions: Mapping[str, Question]) -> str:
        return serialized_system_one_request(self._model, state, questions).decode("utf-8")

    def count_units(self, text: str) -> int:
        if self._tokenize is not None:
            return len(list(self._tokenize(text)))
        return utf8_bytes(text)

    def _budget(self) -> int | None:
        if self.profile.context_window_tokens is not None:
            return self.profile.context_window_tokens
        return self.profile.max_request_bytes

    def _partition_limit(self) -> int:
        if self._family == "jev":
            return JEV_REQUEST_TOKENS if self._tokenize is not None else (
                self.profile.max_request_bytes or JEV_REQUEST_TOKENS
            )
        return self._budget() or 1_048_576

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]:
        if self._family != "jev" or len(questions) < 2:
            return (dict(questions),)
        groups: dict[str, dict[str, Question]] = {}
        for key, question in questions.items():
            namespace = key.split(".", 1)[0]
            groups.setdefault(namespace, {})[key] = question
        empty = State(stage="input", text="")
        packed: list[dict[str, Question]] = []
        current: dict[str, Question] = {}
        limit = self._partition_limit()
        for namespace in sorted(groups):
            group = groups[namespace]
            proposed = {**current, **group}
            used = self.count_units(self.render(empty, proposed))
            if current and used > limit:
                packed.append(current)
                current = dict(group)
                continue
            current = proposed
        if current:
            packed.append(current)
        return tuple(packed) if packed else (dict(questions),)

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        budget = self._budget()
        if budget is None:
            return None
        used = self.count_units(self.render(state, questions))
        if self._family == "jev" and self._tokenize is not None:
            state_units = self.count_units(self.render(state, {}))
            if state_units > JEV_STATE_TOKENS:
                return JEV_STATE_TOKENS - state_units
        return budget - used - self.profile.output_reserve

    def _backoff(self, attempt: int, deadline: float | None) -> float:
        remaining = remaining_timeout(deadline, self._timeout_s)
        base = min(0.05 * (2**attempt), remaining / 2 if remaining > 0 else 0.0)
        return min(base * (0.5 + self._rng.random() * 0.5), remaining)

    def _in_process(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
        attempt: int,
    ) -> BackendResult:
        if request.deadline is not None and time.monotonic() >= request.deadline:
            raise DeadlineExceeded(self.name, question_ids=questions)
        query = self._query_fn
        if query is None:
            raise BackendError(self.name, "connection_error", question_ids=questions)
        raw = query(
            model=self._model,
            state=self.render(state, questions),
            questions={key: encode_question(value) for key, value in questions.items()},
        )
        if isinstance(raw, bytes):
            body = raw
        else:
            encoded = raw if isinstance(raw, str) else json.dumps(raw)
            body = encoded.encode("utf-8")
        if len(body) > request.max_response_bytes:
            raise BackendError(self.name, "response_too_large")
        answers, input_tokens, output_tokens = parse_system_one_response(
            body,
            questions,
            backend=self.name,
        )
        return BackendResult(
            answers=answers,
            usage=(
                BackendUsage(
                    permit_id="",
                    attempt=attempt,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            ),
        )

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        usages: list[BackendUsage] = []
        last_error: BackendError | None = None
        for attempt in range(self.capabilities.max_attempts):
            with request.budget.acquire(
                request.deadline,
                logical_index=request.logical_index,
                attempt=attempt,
            ) as permit:
                if self._query_fn is not None:
                    result = self._in_process(state, questions, request, attempt)
                    usage = result.usage[0]
                    return BackendResult(
                        answers=result.answers,
                        usage=(
                            BackendUsage(
                                permit_id=permit.permit_id,
                                attempt=attempt,
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                            ),
                        ),
                    )
                status, body = self._send(state, questions, request)
                log_http("response", backend=self.name, status_code=status)
                usages.append(
                    BackendUsage(
                        permit_id=permit.permit_id,
                        attempt=attempt,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
                if status == 200:
                    answers, input_tokens, output_tokens = parse_system_one_response(
                        body,
                        questions,
                        backend=self.name,
                    )
                    usages[-1] = BackendUsage(
                        permit_id=permit.permit_id,
                        attempt=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    return BackendResult(answers=answers, usage=tuple(usages))
                last_error = BackendError(
                    self.name,
                    status_reason(status),
                    status_code=status,
                    question_ids=questions,
                )
                if not retryable_status(status) or attempt + 1 >= self.capabilities.max_attempts:
                    raise last_error
            self._sleep(self._backoff(attempt, request.deadline))
        raise last_error or BackendError(self.name, "http_error", question_ids=questions)

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        usages: list[BackendUsage] = []
        last_error: BackendError | None = None
        for attempt in range(self.capabilities.max_attempts):
            async with request.budget.aacquire(
                request.deadline,
                logical_index=request.logical_index,
                attempt=attempt,
            ) as permit:
                if self._query_fn is not None:
                    result = self._in_process(state, questions, request, attempt)
                    usage = result.usage[0]
                    return BackendResult(
                        answers=result.answers,
                        usage=(
                            BackendUsage(
                                permit_id=permit.permit_id,
                                attempt=attempt,
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                            ),
                        ),
                    )
                status, body = await self._asend(state, questions, request)
                log_http("response", backend=self.name, status_code=status)
                usages.append(
                    BackendUsage(
                        permit_id=permit.permit_id,
                        attempt=attempt,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
                if status == 200:
                    answers, input_tokens, output_tokens = parse_system_one_response(
                        body,
                        questions,
                        backend=self.name,
                    )
                    usages[-1] = BackendUsage(
                        permit_id=permit.permit_id,
                        attempt=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    return BackendResult(answers=answers, usage=tuple(usages))
                last_error = BackendError(
                    self.name,
                    status_reason(status),
                    status_code=status,
                    question_ids=questions,
                )
                if not retryable_status(status) or attempt + 1 >= self.capabilities.max_attempts:
                    raise last_error
            await self._asleep(self._backoff(attempt, request.deadline))
        raise last_error or BackendError(self.name, "http_error", question_ids=questions)

    def _decide_url(self, questions: Mapping[str, Question]) -> str:
        if self._base_url is None:
            raise BackendError(self.name, "connection_error", question_ids=questions)
        route = "systemone" if self.profile.mode == "hosted" else "decide"
        return f"{self._base_url}/v1/{route}"

    def _send(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> tuple[int, bytes]:
        return request_capped(
            self._sync_client(),
            "POST",
            self._decide_url(questions),
            backend=self.name,
            content=serialized_system_one_request(self._model, state, questions),
            headers=_headers(self._api_key),
            timeout=http_timeout(request.deadline, self._timeout_s),
            max_bytes=request.max_response_bytes,
        )

    async def _asend(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> tuple[int, bytes]:
        return await arequest_capped(
            self._async_client_obj(),
            "POST",
            self._decide_url(questions),
            backend=self.name,
            content=serialized_system_one_request(self._model, state, questions),
            headers=_headers(self._api_key),
            timeout=http_timeout(request.deadline, self._timeout_s),
            max_bytes=request.max_response_bytes,
        )


def _mapping_body(manifest: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if manifest is None:
        return None
    body = manifest.get("body")
    if isinstance(body, Mapping):
        return cast(Mapping[str, object], body)
    return None


def _optional_str(body: Mapping[str, object] | None, key: str) -> str | None:
    if body is None:
        return None
    value = body.get(key)
    return value if isinstance(value, str) and value else None


__all__ = ["SystemOne"]
