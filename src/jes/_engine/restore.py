"""Complete-reply restoration of context tokens and output-local edits."""

from __future__ import annotations

import re
from collections.abc import Mapping

from jes._engine.sensitive import FinalizerMap, UriPolicy
from jes._engine.uri import canonical_origin, containing_candidates, uri_candidates
from jes.types import Finding, Span

_TOKEN_RE = re.compile(r"\[JES_v[0-9]+_[^\]\r\n]{1,256}\]")


def finalize_output(
    *,
    original: str,
    sanitized: str,
    store_values: Mapping[str, str],
    generation_tokens: frozenset[str],
    finalizers: FinalizerMap,
    uri_policy: UriPolicy | None,
    restore_context: bool,
    max_restorations: int,
    max_restored_output_bytes: int,
) -> tuple[str | None, tuple[Finding, ...]]:
    """Return restored text, or None when a restoration cap is crossed."""

    findings: list[Finding] = []
    restorations = 0
    text = sanitized
    if restore_context:
        authorized = {
            token
            for token in generation_tokens
            if token in store_values and token in original and token in sanitized
        }
        candidates = uri_candidates(text)
        edits: list[tuple[int, int, str]] = []
        for match in _TOKEN_RE.finditer(text):
            token = match.group()
            if token not in authorized:
                continue
            span = Span(match.start(), match.end())
            containers = containing_candidates(span, candidates)
            if containers and not _all_origins_allowed(containers, text, uri_policy):
                action = "flag"
                if uri_policy is None or uri_policy.on_placeholder_in_url == "block":
                    action = "block"
                findings.append(Finding(policy="jes", label="placeholder_in_url", action=action))
                continue
            edits.append((match.start(), match.end(), store_values[token]))
            restorations += 1
        text = _apply(text, edits)
        finalizers.compose(tuple((start, end, replacement) for start, end, replacement in edits))
    restored = finalizers.restore(text)
    restorations += _count_finalizer_restores(finalizers, text)
    encoded = 0
    for char in restored:
        code = ord(char)
        encoded += 1 if code <= 0x7F else 2 if code <= 0x7FF else 3 if code <= 0xFFFF else 4
        if encoded > max_restored_output_bytes:
            return None, (Finding(policy="jes", label="restored_output_too_large", action="block"),)
    if restorations > max_restorations:
        return None, (Finding(policy="jes", label="restored_output_too_large", action="block"),)
    return restored, tuple(findings)


def _all_origins_allowed(
    containers: tuple[Span, ...],
    text: str,
    uri_policy: UriPolicy | None,
) -> bool:
    allowed: frozenset[tuple[str, str, int]] = (
        frozenset() if uri_policy is None else uri_policy.origins
    )
    for container in containers:
        origin = canonical_origin(text[container.start : container.end])
        if origin is None or origin not in allowed:
            return False
    return True


def _apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    if not edits:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in edits:
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _count_finalizer_restores(finalizers: FinalizerMap, sanitized: str) -> int:
    return sum(
        1
        for entry in finalizers._entries
        if not entry.tombstoned and sanitized[entry.start : entry.end] == entry.marker
    )


__all__ = ["finalize_output"]
