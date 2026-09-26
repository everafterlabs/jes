"""Evaluated decision-profile thresholds.

The table stays empty until a sealed audit qualifies a decision profile.
Tests may register ephemeral entries through the private hook. There is no
recommended bundle.
"""

from __future__ import annotations

from jes.questions import Threshold

# fingerprint -> (threshold, evaluation_run)
THRESHOLDS: dict[str, tuple[Threshold, str]] = {}


def lookup(fingerprint: str) -> tuple[Threshold, str] | None:
    """Return the evaluated threshold for a decision-profile fingerprint."""

    return THRESHOLDS.get(fingerprint)


def _register(fingerprint: str, threshold: Threshold, evaluation_run: str) -> None:  # pyright: ignore[reportUnusedFunction]
    THRESHOLDS[fingerprint] = (threshold, evaluation_run)


def _clear() -> None:  # pyright: ignore[reportUnusedFunction]
    THRESHOLDS.clear()


__all__ = ["THRESHOLDS", "_clear", "_register", "lookup"]
