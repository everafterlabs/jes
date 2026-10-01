"""Exceptions raised by jes.

Exception messages intentionally contain metadata only. Checked text and provider
response bodies must never be interpolated into these classes.
"""

from __future__ import annotations

from collections.abc import Iterable


class JesError(Exception):
    """Base class for jes errors."""


class PolicyError(JesError):
    """A policy or guard configuration is invalid."""


class PolicyExecutionError(JesError):
    """A custom policy violated its runtime contract."""


class RedactionError(JesError):
    """A redaction store, scope, or authority check failed."""


class BackendError(JesError):
    """A backend request or response violated its contract."""

    def __init__(
        self,
        backend: str,
        reason: str,
        *,
        status_code: int | None = None,
        question_ids: Iterable[str] = (),
    ) -> None:
        self.backend = backend
        self.reason = reason
        self.status_code = status_code
        self.question_ids = tuple(question_ids)

        fields = [f"backend={backend}", f"reason={reason}"]
        if status_code is not None:
            fields.append(f"status={status_code}")
        if self.question_ids:
            fields.append(f"questions={','.join(self.question_ids)}")
        super().__init__("backend error (" + "; ".join(fields) + ")")


class DeadlineExceeded(BackendError):
    """A check reached its absolute monotonic deadline."""

    def __init__(self, backend: str, *, question_ids: Iterable[str] = ()) -> None:
        super().__init__(backend, "deadline_exceeded", question_ids=question_ids)
