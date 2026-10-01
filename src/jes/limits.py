"""Bounds on the work one check can cause."""

from __future__ import annotations

from dataclasses import dataclass, fields

from jes.errors import PolicyError


@dataclass(frozen=True, slots=True)
class Limits:
    """Per-guard limits. A check that crosses one blocks with an incomplete result.

    - ``max_input_bytes``: the checked text, in UTF-8.
    - ``max_context_bytes``: prompt, question, sources, and history together.
    - ``max_chunks``: chunks per judgment when the text does not fit one request.
    - ``max_items``: items, such as URLs, extracted across all item policies.
    - ``max_requests``: backend requests per check.
    - ``max_concurrency``: backend requests in flight across the whole guard.
    - ``max_redactions``: sensitive values replaced in one check.
    """

    max_input_bytes: int = 1_048_576
    max_context_bytes: int = 2_097_152
    max_chunks: int = 32
    max_items: int = 100
    max_requests: int = 128
    max_concurrency: int = 8
    max_redactions: int = 1_000

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PolicyError(f"{item.name} must be a positive integer")


class LimitExceeded(Exception):
    """A check crossed a limit. The engine turns this into a blocking finding with ``label``."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def utf8_size(text: str) -> int:
    """The UTF-8 length of ``text``. A lone surrogate cannot be encoded, so it blocks."""

    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise LimitExceeded("invalid_unicode") from None


def require_size(text: str, limit: int, label: str) -> int:
    """``utf8_size(text)``, or ``LimitExceeded(label)`` when it is above ``limit``."""

    size = utf8_size(text)
    if size > limit:
        raise LimitExceeded(label)
    return size


__all__ = ["LimitExceeded", "Limits", "require_size", "utf8_size"]
