"""Replacing the values that pii, secrets, and canary find.

Every standalone occurrence of a found value is replaced, not just the one the policy
pointed at, so a value that appears twice never reaches a backend. An occurrence inside
a longer word or number is a different value and stays.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from jes.engine.restore import LocalMarkers
from jes.errors import PolicyError
from jes.limits import LimitExceeded
from jes.policies.base import SensitiveHit, SensitiveMode
from jes.redactions import Redactions
from jes.text.textmap import Edit
from jes.types import Span

_WORD = re.compile(r"\w")


@dataclass(slots=True)
class Redactor:
    """Replacements for one check: new tokens to commit, markers to restore, and a count."""

    store: Redactions
    markers: LocalMarkers
    max_redactions: int
    staged: dict[str, str] = field(default_factory=dict[str, str])
    count: int = 0

    def edits(
        self,
        text: str,
        hits: Sequence[SensitiveHit],
        *,
        tokens: bool,
        local: bool,
        key: bytes | None,
    ) -> list[Edit]:
        """Edits that replace every standalone occurrence of each hit's value.

        ``tokens`` allows conversation tokens and ``local`` allows reply markers. Where
        neither is allowed, the value is removed for good.
        """

        spans = _all_occurrences(text, hits)
        self.count += len(spans)
        if self.count > self.max_redactions:
            raise LimitExceeded("too_many_redactions")
        edits: list[Edit] = []
        for span, hit in spans:
            value = text[span.start : span.end]
            edits.append(Edit(span.start, span.end, self._replace(value, hit, tokens, local, key)))
        return edits

    def _replace(
        self, value: str, hit: SensitiveHit, tokens: bool, local: bool, key: bytes | None
    ) -> str:
        mode: SensitiveMode = hit.mode
        if mode == "token" and tokens:
            token = self.store._token(value)
            if self.store._value(token) is None:
                self.staged[token] = value
            return token
        if mode == "local" and local:
            return self.markers.marker(value)
        if mode == "mask":
            return "******"
        if mode == "partial":
            # Short values show nothing: four characters of eight would be half the secret.
            return f"{value[:2]}****{value[-2:]}" if len(value) > 8 else "*" * len(value)
        if mode == "hmac":
            if key is None:
                raise PolicyError("hmac replacement needs a key")
            return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"[REDACTED_{hit.entity.upper()}]"


def _all_occurrences(text: str, hits: Sequence[SensitiveHit]) -> list[tuple[Span, SensitiveHit]]:
    """The hits plus every other standalone occurrence of their values, merged."""

    found: list[tuple[Span, SensitiveHit]] = [(hit.span, hit) for hit in hits]
    seen: dict[str, int] = {}
    first: dict[str, SensitiveHit] = {}
    for hit in hits:
        value = text[hit.span.start : hit.span.end]
        seen[value] = seen.get(value, 0) + 1
        first.setdefault(value, hit)
    for value, detected in seen.items():
        # str.count is cheap, so only values that appear again pay for a regex scan.
        if value and text.count(value) > detected:
            found.extend((span, first[value]) for span in _standalone(text, value))
    merged: list[tuple[Span, SensitiveHit]] = []
    for span, hit in sorted(found, key=lambda item: (item[0].start, -item[0].end)):
        if merged and span.start < merged[-1][0].end:
            last_span, last_hit = merged[-1]
            merged[-1] = (Span(last_span.start, max(last_span.end, span.end)), last_hit)
        else:
            merged.append((span, hit))
    return merged


def _standalone(text: str, value: str) -> list[Span]:
    before = r"(?<!\w)" if _WORD.match(value[0]) else ""
    after = r"(?!\w)" if _WORD.match(value[-1]) else ""
    pattern = re.compile(before + re.escape(value) + after)
    return [Span(match.start(), match.end()) for match in pattern.finditer(text)]


__all__ = ["Redactor"]
