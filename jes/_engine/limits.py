"""Resource limits and deterministic request admission."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, fields
from typing import TypeVar

from jes.backends import RequestBudget, RequestPermit
from jes.errors import DeadlineExceeded, PolicyError

_T = TypeVar("_T")


class ResourceLimit(Exception):
    """Internal control flow for a strict resource block."""

    def __init__(self, label: str) -> None:
        self.label = label
        super().__init__(label)


def utf8_size(text: str, *, limit: int) -> int:
    """Count UTF-8 bytes without allocating an encoded copy."""

    total = 0
    for codepoint in map(ord, text):
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ResourceLimit("invalid_unicode")
        if codepoint <= 0x7F:
            total += 1
        elif codepoint <= 0x7FF:
            total += 2
        elif codepoint <= 0xFFFF:
            total += 3
        else:
            total += 4
        if total > limit:
            raise ResourceLimit("input_too_long")
    return total


@dataclass(frozen=True, slots=True)
class GuardLimits:
    max_input_bytes: int = 1_048_576
    max_context_bytes: int = 2_097_152
    max_context_items: int = 256
    max_normalized_bytes: int = 2_097_152
    max_normalized_context_bytes: int = 2_097_152
    max_chunks: int = 32
    max_items: int = 100
    max_requests: int = 128
    max_findings: int = 1_000
    max_locations: int = 4_096
    max_authorities: int = 2_048
    max_metadata_bytes: int = 65_536
    max_call_redactions: int = 1_000
    max_call_redaction_bytes: int = 8_388_608
    max_restorations: int = 1_000
    max_restored_output_bytes: int = 2_097_152
    max_response_bytes: int = 1_048_576
    max_active_checks: int = 32
    max_concurrency: int = 8
    max_worker_threads: int = 8
    max_queued_work: int = 32

    def __post_init__(self) -> None:
        for item in fields(self):
            field_name = item.name
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PolicyError(f"{field_name} must be a positive integer")


class SharedAdmission:
    """Guard-wide active-check, backend-call, and worker bounds."""

    def __init__(self, limits: GuardLimits) -> None:
        self._checks = threading.BoundedSemaphore(limits.max_active_checks)
        self.backend_calls = threading.BoundedSemaphore(limits.max_concurrency)
        self._work = threading.BoundedSemaphore(
            limits.max_worker_threads + limits.max_queued_work
        )
        self._executor = ThreadPoolExecutor(
            max_workers=limits.max_worker_threads,
            thread_name_prefix="jes",
        )

    @contextmanager
    def check(self) -> Generator[None, None, None]:
        if not self._checks.acquire(blocking=False):
            raise ResourceLimit("guard_busy")
        try:
            yield
        finally:
            self._checks.release()

    @asynccontextmanager
    async def acheck(self) -> AsyncGenerator[None, None]:
        if not self._checks.acquire(blocking=False):
            raise ResourceLimit("guard_busy")
        try:
            yield
        finally:
            self._checks.release()

    async def run_sync(self, function: Callable[..., _T], *args: object) -> _T:
        if not self._work.acquire(blocking=False):
            raise ResourceLimit("guard_busy")
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, lambda: function(*args))
        finally:
            self._work.release()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class CheckRequestBudget(RequestBudget):
    """A check-local deterministic attempt-slot budget."""

    def __init__(
        self,
        *,
        slots: Iterable[tuple[int, int]],
        max_requests: int,
        backend_semaphore: threading.BoundedSemaphore,
    ) -> None:
        reserved = frozenset(slots)
        if len(reserved) > max_requests:
            raise ResourceLimit("too_many_requests")
        self._slots = reserved
        self._backend = backend_semaphore
        self._active: set[tuple[int, int]] = set()
        self._used: set[tuple[int, int]] = set()
        self._lock = threading.Lock()
        self._nonce = uuid.uuid4().hex

    def _reserve(self, logical_index: int, attempt: int) -> None:
        slot = (logical_index, attempt)
        with self._lock:
            if slot not in self._slots or slot in self._used or slot in self._active:
                raise ResourceLimit("too_many_requests")
            self._active.add(slot)

    def _finish(self, logical_index: int, attempt: int, *, used: bool) -> None:
        slot = (logical_index, attempt)
        with self._lock:
            self._active.discard(slot)
            if used:
                self._used.add(slot)

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not math.isfinite(remaining):
            raise DeadlineExceeded("request_budget")
        return remaining

    @contextmanager
    def acquire(
        self,
        deadline: float | None,
        *,
        logical_index: int,
        attempt: int,
    ) -> Generator[RequestPermit, None, None]:
        self._reserve(logical_index, attempt)
        acquired = False
        try:
            remaining = self._remaining(deadline)
            acquired = self._backend.acquire(timeout=remaining)
            if not acquired:
                raise DeadlineExceeded("request_budget")
            permit = RequestPermit(
                permit_id=f"{self._nonce}:{logical_index}:{attempt}",
                deadline=deadline,
            )
            yield permit
        finally:
            if acquired:
                self._backend.release()
            self._finish(logical_index, attempt, used=acquired)

    @asynccontextmanager
    async def aacquire(
        self,
        deadline: float | None,
        *,
        logical_index: int,
        attempt: int,
    ) -> AsyncGenerator[RequestPermit, None]:
        self._reserve(logical_index, attempt)
        acquired = False
        try:
            while not acquired:
                self._remaining(deadline)
                acquired = self._backend.acquire(blocking=False)
                if not acquired:
                    await asyncio.sleep(0.001)
            permit = RequestPermit(
                permit_id=f"{self._nonce}:{logical_index}:{attempt}",
                deadline=deadline,
            )
            yield permit
        finally:
            if acquired:
                self._backend.release()
            self._finish(logical_index, attempt, used=acquired)

