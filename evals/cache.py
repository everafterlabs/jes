"""Development-only answer cache. Values are validated answers, not inputs."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping

from jes.judge import (
    AsyncBackend,
    Backend,
    BackendResult,
    BackendUsage,
    RequestContext,
    SyncBackend,
)
from jes.questions import Question
from jes.types import State


def cache_key(state: State, questions: Mapping[str, Question], backend: Backend) -> str:
    schema = {
        key: {
            "instructions": question.instructions,
            "task": question.task,
            "type": type(question).__name__,
        }
        for key, question in questions.items()
    }
    payload = {
        "generation": dict(backend.profile.generation_settings),
        "history": [(message.role, message.text) for message in state.history],
        "model": backend.profile.model_identity,
        "prompt": state.prompt,
        "question": state.question,
        "questions": schema,
        "sources": list(state.sources),
        "stage": state.stage,
        "text": state.text,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CachingBackend:
    """Delegate to ``inner`` and, when enabled, reuse validated answers."""

    def __init__(self, inner: Backend, *, enabled: bool) -> None:
        self.inner = inner
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.name = inner.name
        self.capabilities = inner.capabilities
        self.profile = inner.profile
        self._answers: dict[str, BackendResult] = {}
        self._lock = threading.Lock()

    def count_units(self, text: str) -> int:
        return self.inner.count_units(text)

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]:
        return self.inner.partition_questions(questions)

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        return self.inner.headroom(state, questions)

    def _lookup(self, key: str) -> BackendResult | None:
        if not self.enabled:
            return None
        with self._lock:
            return self._answers.get(key)

    def _hit(self, found: BackendResult, permit_id: str) -> BackendResult:
        self.hits += 1
        return BackendResult(
            answers=dict(found.answers),
            usage=(
                BackendUsage(
                    permit_id=permit_id,
                    attempt=0,
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        )

    def _store(self, key: str, result: BackendResult) -> None:
        if not self.enabled:
            return
        stored = BackendResult(answers=dict(result.answers), usage=())
        with self._lock:
            self._answers[key] = stored

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        if not isinstance(self.inner, SyncBackend):
            raise TypeError("caching wrapper requires a sync backend")
        key = cache_key(state, questions, self.inner)
        found = self._lookup(key)
        if found is not None:
            with request.budget.acquire(
                request.deadline,
                logical_index=request.logical_index,
                attempt=0,
            ) as permit:
                return self._hit(found, permit.permit_id)
        self.misses += 1
        result = self.inner.decide(state, questions, request)
        self._store(key, result)
        return result

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        if not isinstance(self.inner, AsyncBackend):
            raise TypeError("caching wrapper requires an async backend")
        key = cache_key(state, questions, self.inner)
        found = self._lookup(key)
        if found is not None:
            async with request.budget.aacquire(
                request.deadline,
                logical_index=request.logical_index,
                attempt=0,
            ) as permit:
                return self._hit(found, permit.permit_id)
        self.misses += 1
        result = await self.inner.adecide(state, questions, request)
        self._store(key, result)
        return result

    def stored_texts(self) -> str:
        """Return a debug view that must not contain checked text."""

        return str(len(self._answers))


__all__ = ["CachingBackend", "cache_key"]
