"""Synchronous Guard implementation."""

from __future__ import annotations

import time
from collections.abc import Sequence

from jes.backends import RequestContext, SyncBackend
from jes.errors import BackendError, DeadlineExceeded, PolicyError, PolicyExecutionError
from jes.policies import Policy
from jes.redactions import Redactions
from jes.types import Finding, History, InputResult, ScanResult, Stage

from .core import BaseGuard, CheckPlan, ContextValue, OnBackendError, RequestExecution
from .limits import GuardLimits, ResourceLimit


class Guard(BaseGuard):
    """Run transforms and backend requests synchronously."""

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        backend: SyncBackend | None = None,
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
            backend=backend,
            fail_fast=fail_fast,
            on_backend_error=on_backend_error,
            limits=limits,
            deadline_s=deadline_s,
            trace=trace,
        )
        for compiled in self._compiled.judgments:
            if not isinstance(compiled.backend, SyncBackend):
                raise PolicyError(f"backend for {compiled.policy.name} is async-only")

    def __enter__(self) -> Guard:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    def check_input(
        self,
        text: str,
        *,
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> InputResult:
        result = self._check(
            stage="input",
            text=text,
            redactions=redactions,
            history=history,
        )
        if not isinstance(result, InputResult):
            raise AssertionError("input check did not return InputResult")
        return result

    def check_untrusted(
        self,
        text: str,
        *,
        question: str | InputResult | None = None,
        redactions: Redactions | None = None,
    ) -> ScanResult:
        return self._check(
            stage="untrusted",
            text=text,
            redactions=redactions,
            question=question,
        )

    def check_output(
        self,
        text: str,
        *,
        prompt: str | InputResult,
        sources: Sequence[str | ScanResult] = (),
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> ScanResult:
        return self._check(
            stage="output",
            text=text,
            redactions=redactions,
            prompt=prompt,
            sources=sources,
            history=history,
        )

    def _check(
        self,
        *,
        stage: Stage,
        text: str,
        redactions: Redactions | None,
        prompt: str | InputResult | None = None,
        question: str | InputResult | None = None,
        sources: Sequence[str | ScanResult] = (),
        history: Sequence[History] = (),
    ) -> ScanResult:
        deadline = self._deadline()
        primary: ContextValue | None = prompt if prompt is not None else question
        store = self._resolve_redactions(stage, redactions, primary)
        try:
            with self.admission.check():
                prepared = self.prepare(
                    stage=stage,
                    text=text,
                    redactions=store,
                    prompt=prompt,
                    question=question,
                    sources=sources,
                    history=history,
                    deadline=deadline,
                )
                try:
                    try:
                        plan = self.plan(prepared)
                    except ResourceLimit as error:
                        prepared.findings.append(
                            self._overflow_finding_for_engine(error.label)
                        )
                        plan = self._empty_plan(prepared.findings)

                    budget = self.request_budget(plan.requests)
                    started = time.perf_counter()
                    executions: list[RequestExecution] = []
                    for request in plan.requests:
                        backend = request.backend
                        if not isinstance(backend, SyncBackend):
                            raise PolicyError("Guard cannot execute an async-only backend")
                        context = RequestContext(
                            logical_index=request.logical_index,
                            deadline=deadline,
                            budget=budget,
                            max_response_bytes=self._limits.max_response_bytes,
                        )
                        failure: PolicyExecutionError | None = None
                        response = None
                        try:
                            response = backend.decide(request.state, request.questions, context)
                        except BackendError as error:
                            executions.append(RequestExecution(request=request, error=error))
                            continue
                        except Exception:
                            failure = PolicyExecutionError("custom backend raised")
                        if failure is not None:
                            raise failure
                        if response is None:
                            continue
                        if deadline is not None and time.monotonic() >= deadline:
                            executions.append(
                                RequestExecution(
                                    request=request,
                                    error=DeadlineExceeded(
                                        backend.name,
                                        question_ids=request.questions,
                                    ),
                                )
                            )
                        else:
                            executions.append(
                                RequestExecution(request=request, result=response)
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

    @staticmethod
    def _overflow_finding_for_engine(label: str) -> Finding:
        return Finding(policy="jes", label=label, action="block")

    @staticmethod
    def _empty_plan(findings: list[Finding]) -> CheckPlan:
        return CheckPlan(requests=(), findings=findings, complete=False)

