"""LiteLLM judge adapter with jes-scheduled parse retry and capped transport."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Callable, Mapping
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
from jes.errors import BackendError, PolicyError
from jes.questions import Question
from jes.types import State

from ._codec import (
    dumps,
    litellm_messages,
    message_content,
    parse_logprob_choice,
    parse_verbalized_content,
    question_candidates,
    usage_from_mapping,
)
from ._http import arequest_capped, http_timeout, log_http, request_capped
from ._profiles import (
    ADAPTER_VERSION,
    LITELLM_PORTABLE_TOP_LOGPROBS,
    PARSER_VERSION,
    SCORER_VERSION,
    conservative_byte_budget,
    require_known_provider_profile,
)
from ._tokens import (
    Tokenize,
    assert_single_token_labels,
    default_label_tokenize,
    load_tokenizer,
    utf8_bytes,
)

CompletionFn = Callable[..., Any]
Mode = Literal["logprobs", "verbalized"]

_PAYLOAD_LOGGERS = ("LiteLLM", "litellm", "LiteLLM Router")


def _install_payload_filters() -> None:
    filt = _DropPayloadFilter()
    for name in ("jes", *_PAYLOAD_LOGGERS):
        logging.getLogger(name).addFilter(filt)


class _DropPayloadFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        blocked = ("messages", "prompt", "content", "state", "canary")
        return not any(token in message.lower() for token in blocked)


class LiteLLMJudge:
    """Capability-qualified LiteLLM backend."""

    name = "litellm"

    def __init__(
        self,
        model: str,
        *,
        mode: Mode,
        provider: str,
        provider_profile: str,
        revision: str,
        tokenizer_revision: str | None = None,
        template_revision: str | None = None,
        context_window_tokens: int | None = None,
        max_request_bytes: int | None = None,
        top_logprobs: int = LITELLM_PORTABLE_TOP_LOGPROBS,
        timeout_s: float = 10.0,
        num_retries: int = 0,
        fallbacks: object = None,
        cache: bool = False,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        completion_fn: CompletionFn | None = None,
        tokenize: Tokenize | None = None,
        chat_url: str = "https://llm.invalid/v1/chat/completions",
    ) -> None:
        if mode not in {"logprobs", "verbalized"}:
            raise PolicyError("LiteLLMJudge mode must be logprobs or verbalized")
        if num_retries != 0:
            raise PolicyError("LiteLLM internal retries must be disabled")
        if fallbacks is not None:
            raise PolicyError("LiteLLM fallbacks must be disabled")
        if cache:
            raise PolicyError("LiteLLM caching must be disabled")
        if context_window_tokens is not None and max_request_bytes is not None:
            raise PolicyError("supply exactly one of context_window_tokens or max_request_bytes")
        spec = require_known_provider_profile(provider_profile, "litellm")
        if spec.get("verbalized_only") and mode != "verbalized":
            raise PolicyError("provider profile does not support logprobs")
        raw_allowed = spec.get("top_logprobs_max", LITELLM_PORTABLE_TOP_LOGPROBS)
        allowed = raw_allowed if isinstance(raw_allowed, int) else 0
        if mode == "logprobs" and (top_logprobs < 2 or (allowed and top_logprobs > allowed)):
            raise PolicyError("top_logprobs is outside the provider profile range")
        tokenizer = tokenize or load_tokenizer(tokenizer_revision) or default_label_tokenize
        kind: Literal["probability", "verbalized"] = (
            "probability" if mode == "logprobs" else "verbalized"
        )
        exact = context_window_tokens is not None
        self.capabilities = BackendCapabilities(
            tasks=None,
            max_options=top_logprobs if mode == "logprobs" else None,
            max_attempts=2,
            score_kinds={"*": frozenset({kind})},
            budget_fidelity="exact" if exact else "conservative",
        )
        self.profile = BackendProfile(
            adapter="litellm",
            adapter_version=ADAPTER_VERSION,
            scorer_version=SCORER_VERSION,
            parser_version=PARSER_VERSION,
            provider=provider,
            provider_profile=provider_profile,
            model=model,
            revision=revision,
            artifact_digest=None,
            tokenizer_revision=tokenizer_revision,
            template_revision=template_revision,
            mode=mode,
            generation_settings={
                "temperature": 0,
                "num_retries": 0,
                "cache": False,
                "top_logprobs": top_logprobs if mode == "logprobs" else None,
            },
            context_window_tokens=context_window_tokens,
            max_request_bytes=(
                None
                if exact
                else conservative_byte_budget(
                    "unknown",
                    max_request_bytes=max_request_bytes or 8_192,
                )
            ),
            output_reserve=16,
            budget_attestation=None,
            dependency_versions=MappingProxyType({"httpx": httpx.__version__}),
        )
        self._model = model
        self._mode: Mode = mode
        self._top_logprobs = top_logprobs
        self._timeout_s = timeout_s
        self._completion_fn = completion_fn
        self._http_client = http_client
        self._async_http_client = async_http_client
        self._owns_client = http_client is None and completion_fn is None
        self._chat_url = chat_url
        self._tokenize = tokenizer
        if completion_fn is None and http_client is None and async_http_client is None:
            self._require_litellm()

    def _require_litellm(self) -> None:
        if importlib.util.find_spec("litellm") is None:
            raise PolicyError("LiteLLMJudge requires the jes[litellm] extra")
        litellm: Any = importlib.import_module("litellm")
        if not getattr(litellm, "turn_off_message_logging", None) and not getattr(
            litellm, "suppress_debug_info", None
        ):
            raise PolicyError("supported LiteLLM versions must suppress payload logs")
        litellm.drop_params = True
        if hasattr(litellm, "turn_off_message_logging"):
            litellm.turn_off_message_logging = True
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        _install_payload_filters()

    def close(self) -> None:
        if self._owns_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    async def aclose(self) -> None:
        self.close()
        if self._async_http_client is not None:
            await self._async_http_client.aclose()
            self._async_http_client = None

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
            f"LiteLLMJudge(model={self.profile.model!r}, mode={self.profile.mode!r}, "
            f"provider_profile={self.profile.provider_profile!r})"
        )

    def __copy__(self) -> LiteLLMJudge:
        raise TypeError("LiteLLMJudge cannot be copied")

    def __deepcopy__(self, memo: object) -> LiteLLMJudge:
        del memo
        raise TypeError("LiteLLMJudge cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("LiteLLMJudge cannot be pickled")

    def render(self, state: State, questions: Mapping[str, Question]) -> str:
        messages, delimiter = litellm_messages(self._model, state, questions, mode=self._mode)
        return dumps({"delimiter": delimiter, "messages": messages}).decode("utf-8")

    def count_units(self, text: str) -> int:
        if self.capabilities.budget_fidelity == "exact":
            return len(list(self._tokenize(text)))
        return utf8_bytes(text)

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]:
        self._validate_questions(questions)
        return (dict(questions),)

    def _validate_questions(self, questions: Mapping[str, Question]) -> None:
        labels = [
            label
            for question in questions.values()
            for label in question_candidates(question)
        ]
        if self._mode == "logprobs":
            assert_single_token_labels(labels, self._tokenize)
            if self.capabilities.max_options is not None:
                widest = max(
                    (len(question_candidates(question)) for question in questions.values()),
                    default=0,
                )
                if widest > self.capabilities.max_options:
                    raise PolicyError("candidate set exceeds provider top_logprobs")

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        self._validate_questions(questions)
        budget = self.profile.context_window_tokens or self.profile.max_request_bytes
        if budget is None:
            return None
        used = self.count_units(self.render(state, questions))
        return budget - used - self.profile.output_reserve

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        self._validate_questions(questions)
        usages: list[BackendUsage] = []
        last_error: BackendError | None = None
        for attempt in range(self.capabilities.max_attempts):
            with request.budget.acquire(
                request.deadline,
                logical_index=request.logical_index,
                attempt=attempt,
            ) as permit:
                raw, delimiter = self._complete(state, questions, request)
                log_http("completion", backend=self.name)
                answers, input_tokens, output_tokens, error = self._parse(
                    raw,
                    questions,
                    delimiter=delimiter,
                )
                usages.append(
                    BackendUsage(
                        permit_id=permit.permit_id,
                        attempt=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
                if error is None and answers is not None:
                    return BackendResult(answers=answers, usage=tuple(usages))
                last_error = error
                if attempt + 1 >= self.capabilities.max_attempts:
                    raise last_error or BackendError(self.name, "malformed_answer")
        raise last_error or BackendError(self.name, "malformed_answer")

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        self._validate_questions(questions)
        usages: list[BackendUsage] = []
        last_error: BackendError | None = None
        for attempt in range(self.capabilities.max_attempts):
            async with request.budget.aacquire(
                request.deadline,
                logical_index=request.logical_index,
                attempt=attempt,
            ) as permit:
                raw, delimiter = await self._acomplete(state, questions, request)
                log_http("completion", backend=self.name)
                answers, input_tokens, output_tokens, error = self._parse(
                    raw,
                    questions,
                    delimiter=delimiter,
                )
                usages.append(
                    BackendUsage(
                        permit_id=permit.permit_id,
                        attempt=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
                if error is None and answers is not None:
                    return BackendResult(answers=answers, usage=tuple(usages))
                last_error = error
                if attempt + 1 >= self.capabilities.max_attempts:
                    raise last_error or BackendError(self.name, "malformed_answer")
        raise last_error or BackendError(self.name, "malformed_answer")

    def _parse(
        self,
        raw: object,
        questions: Mapping[str, Question],
        *,
        delimiter: str,
    ) -> tuple[dict[str, Any] | None, int | None, int | None, BackendError | None]:
        try:
            if self._mode == "logprobs":
                answers, input_tokens, output_tokens = parse_logprob_choice(
                    raw,
                    questions,
                    backend=self.name,
                )
                return answers, input_tokens, output_tokens, None
            answers = parse_verbalized_content(
                message_content(raw),
                questions,
                backend=self.name,
                delimiter=delimiter,
            )
            usage = cast(dict[str, object], raw).get("usage") if isinstance(raw, dict) else None
            input_tokens, output_tokens = usage_from_mapping(usage)
            return answers, input_tokens, output_tokens, None
        except BackendError as error:
            return None, None, None, BackendError(self.name, error.reason, question_ids=questions)

    def _complete(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> tuple[object, str]:
        messages, delimiter = litellm_messages(self._model, state, questions, mode=self._mode)
        if self._completion_fn is not None:
            return (
                self._completion_fn(
                    model=self._model,
                    messages=messages,
                    temperature=0,
                    logprobs=self._mode == "logprobs",
                    top_logprobs=self._top_logprobs if self._mode == "logprobs" else None,
                    num_retries=0,
                    fallbacks=None,
                    cache=False,
                    delimiter=delimiter,
                ),
                delimiter,
            )
        body = self._chat_body(messages)
        client = self._http_client or httpx.Client()
        owned = self._http_client is None
        try:
            status, content = request_capped(
                client,
                "POST",
                self._chat_url,
                backend=self.name,
                content=body,
                headers={"content-type": "application/json"},
                timeout=http_timeout(request.deadline, self._timeout_s),
                max_bytes=request.max_response_bytes,
            )
        finally:
            if owned:
                client.close()
        if status != 200:
            from ._http import status_reason

            raise BackendError(self.name, status_reason(status), status_code=status)
        return _decode_json(content, backend=self.name), delimiter

    async def _acomplete(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> tuple[object, str]:
        if self._completion_fn is not None:
            return self._complete(state, questions, request)
        messages, delimiter = litellm_messages(self._model, state, questions, mode=self._mode)
        body = self._chat_body(messages)
        client = self._async_http_client or httpx.AsyncClient()
        owned = self._async_http_client is None
        try:
            status, content = await arequest_capped(
                client,
                "POST",
                self._chat_url,
                backend=self.name,
                content=body,
                headers={"content-type": "application/json"},
                timeout=http_timeout(request.deadline, self._timeout_s),
                max_bytes=request.max_response_bytes,
            )
        finally:
            if owned:
                await client.aclose()
        if status != 200:
            from ._http import status_reason

            raise BackendError(self.name, status_reason(status), status_code=status)
        return _decode_json(content, backend=self.name), delimiter

    def _chat_body(self, messages: list[dict[str, str]]) -> bytes:
        payload: dict[str, Any] = {
            "messages": messages,
            "model": self._model,
            "temperature": 0,
        }
        if self._mode == "logprobs":
            payload["logprobs"] = True
            payload["top_logprobs"] = self._top_logprobs
        return dumps(payload)


def _decode_json(body: bytes, *, backend: str) -> object:
    import json

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BackendError(backend, "malformed_answer") from None


__all__ = ["LiteLLMJudge"]
