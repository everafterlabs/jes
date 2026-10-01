"""jes: guardrails for AI agents."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from jes.errors import (
    BackendError,
    ConfigError,
    DeadlineExceeded,
    JesError,
    PolicyError,
    PolicyExecutionError,
    RedactionError,
)
from jes.limits import Limits
from jes.questions import Choice, Score, Threshold, YesNo
from jes.types import Finding, Message, ScoreResult, Span, Stage, Usage

try:
    __version__ = version("jes")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0+unknown"

__all__ = [
    "BackendError",
    "Choice",
    "ConfigError",
    "DeadlineExceeded",
    "Finding",
    "JesError",
    "Limits",
    "Message",
    "PolicyError",
    "PolicyExecutionError",
    "RedactionError",
    "Score",
    "ScoreResult",
    "Span",
    "Stage",
    "Threshold",
    "Usage",
    "YesNo",
    "__version__",
]
