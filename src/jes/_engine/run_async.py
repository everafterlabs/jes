"""Asynchronous Guard implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from functools import partial

from jes.errors import BackendError, DeadlineExceeded, PolicyError, PolicyExecutionError
from jes.judge import (
    AsyncBackend,
    ModelSpec,
    RequestBudget,
    RequestContext,
    SyncBackend,
    resolve_judge,
)
from jes.payload import freeze_arguments, require_tool_name
from jes.policies import Policy
from jes.redactions import Redactions
from jes.types import Finding, History, InputResult, ScanResult, Stage

from .core import (
    BaseGuard,
    CheckPlan,
    ContextValue,
    OnBackendError,
    RequestExecution,
)
from .limits import GuardLimits, ResourceLimit
from .plan import PlannedRequest


class AsyncGuard(BaseGuard):
    """Run transforms in bounded workers and backend requests concurrently."""

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        model: ModelSpec | None = None,
        fail_fast: bool = False,
        on_backend_error: OnBackendError = "raise",
        max_input_bytes: int = 1_048_576,
        max_context_bytes: int = 2_097_152,
        max_context_items: int = 256,
        max_normalized_bytes: int = 2_097_152,
        max_normalized_context_bytes: int = 2_097_152,
        max_chunks: int = 32,
        max_items: int = 100,
        max_requests: int = 128,
        max_findings: int = 1_000,
        max_locations: int = 4_096,
        max_authorities: int = 2_048,
        max_metadata_bytes: int = 65_536,
        max_call_redactions: int = 1_000,
        max_call_redaction_bytes: int = 8_388_608,
        max_restorations: int = 1_000,
        max_restored_output_bytes: int = 2_097_152,
        max_response_bytes: int = 1_048_576,
        max_active_checks: int = 32,
        max_concurrency: int = 8,
        max_worker_threads: int = 8,
        max_queued_work: int = 32,
        deadline_s: float | None = 30.0,
        trace: bool = False,
    ) -> None:
        limits = GuardLimits(
            max_input_bytes=max_input_bytes,
            max_context_bytes=max_context_bytes,
            max_context_items=max_context_items,
            max_normalized_bytes=max_normalized_bytes,
            max_normalized_context_bytes=max_normalized_context_bytes,
            max_chunks=max_chunks,
            max_items=max_items,
            max_requests=max_requests,
            max_findings=max_findings,
            max_locations=max_locations,
            max_authorities=max_authorities,
            max_metadata_bytes=max_metadata_bytes,
            max_call_redactions=max_call_redactions,
            max_call_redaction_bytes=max_call_redaction_bytes,
            max_restorations=max_restorations,
            max_restored_output_bytes=max_restored_output_bytes,
            max_response_bytes=max_response_bytes,
            max_active_checks=max_active_checks,
            max_concurrency=max_concurrency,
            max_worker_threads=max_worker_threads,
            max_queued_work=max_queued_work,
        )
        super().__init__(
            policies,
            backend=resolve_judge(model),
            fail_fast=fail_fast,
            on_backend_error=on_backend_error,
            limits=limits,
            deadline_s=deadline_s,
            trace=trace,
        )
        for compiled in self._compiled.judgments:
            if not isinstance(compiled.backend, (AsyncBackend, SyncBackend)):
                raise PolicyError(f"backend for {compiled.policy.name} is not executable")

    async def __aenter__(self) -> AsyncGuard:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args
        self.close()

    async def check_input(
        self,
        text: str,
        *,
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> InputResult:
        result = await self._check(
            stage="input",
            text=text,
            redactions=redactions,
            history=history,
        )
        if not isinstance(result, InputResult):
            raise AssertionError("input check did not return InputResult")
        return result

    async def check_untrusted(
        self,
        text: str,
        *,
        question: str | InputResult | None = None,
        redactions: Redactions | None = None,
    ) -> ScanResult:
        return await self._check(
            stage="untrusted",
            text=text,
            redactions=redactions,
            question=question,
        )

    async def check_output(
        self,
        text: str,
        *,
        prompt: str | InputResult,
        sources: Sequence[str | ScanResult] = (),
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> ScanResult:
        return await self._check(
            stage="output",
            text=text,
            redactions=redactions,
            prompt=prompt,
            sources=sources,
            history=history,
        )

    async def check_tool_call(
        self,
        name: str,
        arguments: str | Mapping[str, object],
        *,
        prompt: str | InputResult,
        redactions: Redactions | None = None,
    ) -> ScanResult:
        return await self._check(
            stage="tool_call",
            text=freeze_arguments(arguments),
            redactions=redactions,
            prompt=prompt,
            tool=require_tool_name(name),
        )

    async def check_tool_result(
        self,
        text: str,
        *,
        name: str,
        prompt: str | InputResult | None = None,
        redactions: Redactions | None = None,
    ) -> ScanResult:
        return await self._check(
            stage="tool_result",
            text=text,
            redactions=redactions,
            question=prompt,
            tool=require_tool_name(name),
        )

    async def _execute_one(
        self,
        request: PlannedRequest,
        *,
        budget: RequestBudget,
        deadline: float | None,
    ) -> RequestExecution:
        context = RequestContext(
            logical_index=request.logical_index,
            deadline=deadline,
            budget=budget,
            max_response_bytes=self._limits.max_response_bytes,
        )
        backend = request.backend
        failure: PolicyExecutionError | None = None
        result = None
        try:
            if isinstance(backend, AsyncBackend):
                result = await backend.adecide(request.state, request.questions, context)
            elif isinstance(backend, SyncBackend):
                result = await self.admission.run_sync(
                    backend.decide,
                    request.state,
                    request.questions,
                    context,
                )
            else:
                raise PolicyError("backend implements neither decide nor adecide")
        except BackendError as error:
            return RequestExecution(request=request, error=error)
        except Exception:
            failure = PolicyExecutionError("custom backend raised")
        if failure is not None:
            raise failure
        if result is None:
            raise PolicyError("backend implements neither decide nor adecide")
        if deadline is not None and time.monotonic() >= deadline:
            return RequestExecution(
                request=request,
                error=DeadlineExceeded(backend.name, question_ids=request.questions),
            )
        return RequestExecution(request=request, result=result)

    async def _check(
        self,
        *,
        stage: Stage,
        text: str,
        redactions: Redactions | None,
        prompt: str | InputResult | None = None,
        question: str | InputResult | None = None,
        sources: Sequence[str | ScanResult] = (),
        history: Sequence[History] = (),
        tool: str | None = None,
    ) -> ScanResult:
        deadline = self._deadline()
        primary: ContextValue | None = prompt if prompt is not None else question
        store = self._resolve_redactions(stage, redactions, primary)
        prepared = None
        try:
            async with self.admission.acheck():
                prepare_call = partial(
                    self.prepare,
                    stage=stage,
                    text=text,
                    redactions=store,
                    prompt=prompt,
                    question=question,
                    sources=sources,
                    history=history,
                    tool=tool,
                    deadline=deadline,
                )
                prepared = await self.admission.run_sync(prepare_call)
                try:
                    try:
                        plan = self.plan(prepared)
                    except ResourceLimit as error:
                        prepared.findings.append(
                            Finding(policy="jes", label=error.label, action="block")
                        )
                        plan = CheckPlan(
                            requests=(),
                            findings=prepared.findings,
                            complete=False,
                        )
                    budget = self.request_budget(plan.requests)
                    started = time.perf_counter()
                    executions = await asyncio.gather(
                        *(
                            self._execute_one(
                                request,
                                budget=budget,
                                deadline=deadline,
                            )
                            for request in plan.requests
                        )
                    )
                    interpretation = self.interpret(executions)
                    judgment_ms = (time.perf_counter() - started) * 1_000
                    return self.result(
                        prepared=prepared,
                        plan=plan,
                        interpretation=interpretation,
                        judgment_ms=judgment_ms,
                    )
                except BaseException:
                    prepared.transaction.close()
                    raise
        except asyncio.CancelledError:
            if prepared is not None:
                prepared.transaction.close()
            raise
        except ResourceLimit as error:
            return self.failure_result(
                stage=stage,
                redactions=store,
                label=error.label,
            )
        except DeadlineExceeded:
            if self._on_backend_error == "raise":
                raise
            return self.failure_result(
                stage=stage,
                redactions=store,
                label="deadline_exceeded",
                action="block" if self._on_backend_error == "block" else "flag",
            )

