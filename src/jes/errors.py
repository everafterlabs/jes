"""Exceptions raised by jes.

Messages carry metadata only. Checked text and provider response bodies are never put
into an exception, so an error is always safe to log.
"""

from __future__ import annotations

from collections.abc import Iterable


class JesError(Exception):
    """Base class for jes errors."""


class ConfigError(JesError):
    """A hook payload, config file, or environment setting is invalid."""


class PolicyError(JesError):
    """A policy or guard configuration is invalid."""


class PolicyExecutionError(JesError):
    """Custom policy code broke its contract while a check ran."""


class RedactionError(JesError):
    """A redaction store, scope, or saved blob failed a check."""


class BackendError(JesError):
    """A backend call failed or returned something jes cannot use."""

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
        super().__init__(f"backend error ({'; '.join(fields)})")


class DeadlineExceeded(BackendError):
    """A check ran out of time before its backend calls finished."""

    def __init__(self, backend: str, *, question_ids: Iterable[str] = ()) -> None:
        super().__init__(backend, "deadline_exceeded", question_ids=question_ids)


__all__ = [
    "BackendError",
    "ConfigError",
    "DeadlineExceeded",
    "JesError",
    "PolicyError",
    "PolicyExecutionError",
    "RedactionError",
]
