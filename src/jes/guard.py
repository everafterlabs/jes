"""Guard and AsyncGuard: check each step of an agent against your policies."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import weakref
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import cast

from jes.backend import AsyncBackend, ModelSpec, Reply, Request, SyncBackend, resolve_model
from jes.engine.pipeline import (
    Check,
    OnBackendError,
    Outcome,
    Pipeline,
    Plan,
    Planned,
    Prepared,
    resolve_store,
)
from jes.errors import BackendError, DeadlineExceeded, PolicyError
from jes.limits import LimitExceeded, Limits
from jes.policies.base import Policy
from jes.policies.transforms import valid_tool_name
from jes.redactions import Redactions
from jes.result import Result
from jes.types import Message

History = Result | Message


class _Base:
    def __init__(
        self,
        policies: Iterable[Policy],
        *,
        model: ModelSpec | None,
        limits: Limits | None,
        deadline_s: float | None,
        on_backend_error: OnBackendError,
        fail_fast: bool,
        sync_only: bool,
    ) -> None:
        if deadline_s is not None and (
            isinstance(deadline_s, bool) or not math.isfinite(deadline_s) or deadline_s <= 0
        ):
            raise PolicyError("deadline_s must be a positive number of seconds, or None")
        self._limits = Limits() if limits is None else limits
        self._deadline_s = deadline_s
        self._pipeline = Pipeline(
            list(policies),
            backend=resolve_model(model),
            limits=self._limits,
            on_backend_error=on_backend_error,
            fail_fast=fail_fast,
            sync_only=sync_only,
        )

    def _start(self, check: Check) -> tuple[Check, float, float | None]:
        # Resolve the store up front, so even a check that fails early returns it.
        store = resolve_store(check)
        deadline = None if self._deadline_s is None else time.monotonic() + self._deadline_s
        return replace(check, redactions=store), time.perf_counter(), deadline

    def _prepare(self, check: Check, deadline: float | None) -> tuple[Prepared, Plan]:
        prepared = self._pipeline.prepare(check, deadline)
        return prepared, self._pipeline.plan(prepared)

    def _failed(self, check: Check, error: Exception, started: float) -> Result:
        store = cast(Redactions, check.redactions)
        if isinstance(error, LimitExceeded):
            return self._pipeline.failed(check, store, error.label, "block", started)
        if self._pipeline.on_backend_error == "raise":
            raise error
        action = "block" if self._pipeline.on_backend_error == "block" else "flag"
        return self._pipeline.failed(check, store, "deadline_exceeded", action, started)


def _input(text: str, redactions: Redactions | None, history: Sequence[History]) -> Check:
    return Check("input", _text(text), history=tuple(history), redactions=redactions)


def _untrusted(text: str, question: str | Result | None, redactions: Redactions | None) -> Check:
    return Check("untrusted", _text(text), question=question, redactions=redactions)


def _tool_call(
    name: str,
    arguments: str | Mapping[str, object],
    prompt: str | Result,
    redactions: Redactions | None,
) -> Check:
    return Check(
        "tool_call",
        freeze_arguments(arguments),
        tool=_tool(name),
        prompt=prompt,
        redactions=redactions,
    )


def _tool_result(
    text: str, name: str, prompt: str | Result | None, redactions: Redactions | None
) -> Check:
    return Check("tool_result", _text(text), tool=_tool(name), prompt=prompt, redactions=redactions)


def _output(
    text: str,
    prompt: str | Result,
    sources: Sequence[str | Result],
    history: Sequence[History],
    redactions: Redactions | None,
) -> Check:
    return Check(
        "output",
        _text(text),
        prompt=prompt,
        sources=tuple(sources),
        history=tuple(history),
        redactions=redactions,
    )


class Guard(_Base):
    """Checks text with your policies. Backend requests run in parallel threads.

    - ``model``: a TypeSafe model id such as ``"jev-1.13.0"``, a ``TypeSafeClassifier``,
      or your own backend. Judgments use it unless they bring their own.
    - ``deadline_s``: seconds a check may take. Requests still running then count as
      backend errors.
    - ``on_backend_error``: ``"raise"`` (the default), ``"block"``, or ``"allow"``. With
      ``"block"`` and ``"allow"``, the result is incomplete, so ``ok`` is false either way.
    - ``fail_fast``: skip judgments once a transform has blocked.

    Close the guard, or use it as a context manager, to stop its threads.
    """

    def __init__(
        self,
        policies: Iterable[Policy],
        *,
        model: ModelSpec | None = None,
        limits: Limits | None = None,
        deadline_s: float | None = 30.0,
        on_backend_error: OnBackendError = "raise",
        fail_fast: bool = False,
    ) -> None:
        super().__init__(
            policies,
            model=model,
            limits=limits,
            deadline_s=deadline_s,
            on_backend_error=on_backend_error,
            fail_fast=fail_fast,
            sync_only=True,
        )
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None

    def __enter__(self) -> Guard:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def check_input(
        self,
        text: str,
        *,
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> Result:
        """User input, before the model sees it. The result carries the conversation store."""

        return self._run(_input(text, redactions, history))

    def check_untrusted(
        self,
        text: str,
        *,
        question: str | Result | None = None,
        redactions: Redactions | None = None,
    ) -> Result:
        """Retrieved text, such as a web page or a file, before it enters the prompt."""

        return self._run(_untrusted(text, question, redactions))

    def check_tool_call(
        self,
        name: str,
        arguments: str | Mapping[str, object],
        *,
        prompt: str | Result,
        redactions: Redactions | None = None,
    ) -> Result:
        """A tool call, before it runs. The arguments pass on unchanged when it is allowed."""

        return self._run(_tool_call(name, arguments, prompt, redactions))

    def check_tool_result(
        self,
        text: str,
        *,
        name: str,
        prompt: str | Result | None = None,
        redactions: Redactions | None = None,
    ) -> Result:
        """What a tool returned, before the model reads it."""

        return self._run(_tool_result(text, name, prompt, redactions))

    def check_output(
        self,
        text: str,
        *,
        prompt: str | Result,
        sources: Sequence[str | Result] = (),
        history: Sequence[History] = (),
        redactions: Redactions | None = None,
    ) -> Result:
        """The model's complete reply, before the user sees it. Placeholders are restored."""

        return self._run(_output(text, prompt, sources, history, redactions))

    def _run(self, check: Check) -> Result:
        check, started, deadline = self._start(check)
        try:
            prepared, plan = self._prepare(check, deadline)
        except (LimitExceeded, DeadlineExceeded) as error:
            return self._failed(check, error, started)
        outcomes = self._execute(plan.requests, deadline)
        return self._pipeline.finish(prepared, plan, outcomes)

    def _execute(self, requests: list[Planned], deadline: float | None) -> list[Outcome]:
        if not requests:
            return []
        pool = self._executor()
        futures: list[Future[Outcome]] = [
            pool.submit(_decide, planned, deadline) for planned in requests
        ]
        wait(futures, timeout=_remaining(deadline))
        outcomes: list[Outcome] = []
        for planned, future in zip(requests, futures, strict=True):
            if future.done():
                outcomes.append(future.result())
            else:
                future.cancel()
                outcomes.append(_late(planned))
        return outcomes

    def _executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._limits.max_concurrency, thread_name_prefix="jes"
                )
            return self._pool


class AsyncGuard(_Base):
    """``Guard`` for asyncio. Each check's backend requests run concurrently.

    A backend with ``adecide`` is awaited. One with only ``decide`` runs in a thread.
    """

    def __init__(
        self,
        policies: Iterable[Policy],
        *,
        model: ModelSpec | None = None,
        limits: Limits | None = None,
        deadline_s: float | None = 30.0,
        on_backend_error: OnBackendError = "raise",
        fail_fast: bool = False,
    ) -> None:
        super().__init__(
            policies,
            model=model,
            limits=limits,
            deadline_s=deadline_s,
            on_backend_error=on_backend_error,
            fail_fast=fail_fast,
            sync_only=False,
        )
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()

    async def __aenter__(self) -> AsyncGuard:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """AsyncGuard holds no threads; this exists so both guards close the same way."""

    async def check_input(
        self,
        text: str,
        *,
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> Result:
        return await self._run(_input(text, redactions, history))

    async def check_untrusted(
        self,
        text: str,
        *,
        question: str | Result | None = None,
        redactions: Redactions | None = None,
    ) -> Result:
        return await self._run(_untrusted(text, question, redactions))

    async def check_tool_call(
        self,
        name: str,
        arguments: str | Mapping[str, object],
        *,
        prompt: str | Result,
        redactions: Redactions | None = None,
    ) -> Result:
        return await self._run(_tool_call(name, arguments, prompt, redactions))

    async def check_tool_result(
        self,
        text: str,
        *,
        name: str,
        prompt: str | Result | None = None,
        redactions: Redactions | None = None,
    ) -> Result:
        return await self._run(_tool_result(text, name, prompt, redactions))

    async def check_output(
        self,
        text: str,
        *,
        prompt: str | Result,
        sources: Sequence[str | Result] = (),
        history: Sequence[History] = (),
        redactions: Redactions | None = None,
    ) -> Result:
        return await self._run(_output(text, prompt, sources, history, redactions))

    async def _run(self, check: Check) -> Result:
        check, started, deadline = self._start(check)
        try:
            # Transforms and planning are CPU work, so they leave the event loop.
            prepared, plan = await asyncio.to_thread(self._prepare, check, deadline)
        except (LimitExceeded, DeadlineExceeded) as error:
            return self._failed(check, error, started)
        outcomes = await self._execute(plan.requests, deadline)
        return self._pipeline.finish(prepared, plan, outcomes)

    async def _execute(self, requests: list[Planned], deadline: float | None) -> list[Outcome]:
        if not requests:
            return []
        semaphore = self._semaphore()
        tasks = [
            asyncio.ensure_future(self._call(planned, deadline, semaphore)) for planned in requests
        ]
        try:
            await asyncio.wait(tasks, timeout=_remaining(deadline))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
        return [
            task.result() if task.done() and not task.cancelled() else _late(planned)
            for planned, task in zip(requests, tasks, strict=True)
        ]

    async def _call(
        self, planned: Planned, deadline: float | None, semaphore: asyncio.Semaphore
    ) -> Outcome:
        async with semaphore:
            backend = planned.backend
            if isinstance(backend, AsyncBackend):
                request = Request(planned.state, planned.questions, _remaining(deadline))
                try:
                    reply = await backend.adecide(request)
                except BackendError as error:
                    return error
                except Exception:
                    return BackendError(
                        backend.name, "unexpected_error", question_ids=planned.questions
                    )
                return _checked(planned, reply)
            return await asyncio.to_thread(_decide, planned, deadline)

    def _semaphore(self) -> asyncio.Semaphore:
        # A semaphore belongs to one event loop, and a guard may outlive several.
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._limits.max_concurrency)
            self._semaphores[loop] = semaphore
        return semaphore


def _decide(planned: Planned, deadline: float | None) -> Outcome:
    """Call a sync backend. Never raises: every failure becomes a BackendError outcome."""

    backend = cast(SyncBackend, planned.backend)
    request = Request(planned.state, planned.questions, _remaining(deadline))
    try:
        reply = backend.decide(request)
    except BackendError as error:
        return error
    except Exception:
        return BackendError(backend.name, "unexpected_error", question_ids=planned.questions)
    return _checked(planned, reply)


def _checked(planned: Planned, reply: object) -> Outcome:
    # B1: a backend that returned None used to count as an allow.
    if not isinstance(reply, Reply):
        return BackendError(planned.backend.name, "malformed_reply", question_ids=planned.questions)
    return reply


def _late(planned: Planned) -> DeadlineExceeded:
    return DeadlineExceeded(planned.backend.name, question_ids=planned.questions)


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _text(text: str) -> str:
    if not isinstance(text, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("text must be a str")
    return text


def _tool(name: str) -> str:
    if not valid_tool_name(name):
        raise PolicyError("a tool name is one line of 1 to 256 characters")
    return name


def freeze_arguments(arguments: str | Mapping[str, object]) -> str:
    """Tool arguments as one string: as given, or a mapping as canonical JSON."""

    if isinstance(arguments, str):
        return arguments
    return json.dumps(
        _plain(arguments),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _plain(value: object) -> object:
    """``value`` as plain JSON types, or PolicyError if it is not JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PolicyError("tool arguments must be JSON values")
        return value
    if isinstance(value, Mapping):
        plain: dict[str, object] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise PolicyError("tool argument keys must be strings")
            plain[key] = _plain(item)
        return plain
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in cast(Sequence[object], value)]
    raise PolicyError("tool arguments must be JSON values")


__all__ = ["AsyncGuard", "Guard", "History", "freeze_arguments"]
