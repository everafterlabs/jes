"""Engine-owned sensitive-edit modes and output-local finalizer maps.

Milestone 1 supplies the private channel and test fakes. Built-in PII, secrets,
and canary policies arrive in Milestone 3.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from jes.errors import PolicyExecutionError
from jes.policies._protocols import CallContext, TransformFinding, TransformPolicy
from jes.redactions import RedactionTransaction
from jes.types import Action, Span

_SensitiveMode = Literal[
    "conversation_token",
    "irreversible",
    "mask_all",
    "mask_partial",
    "hmac",
    "output_local",
]


@dataclass(frozen=True, slots=True, repr=False)
class _SensitiveEdit:
    span: Span
    entity: str
    mode: _SensitiveMode
    action: Action

    def __repr__(self) -> str:
        return (
            f"_SensitiveEdit(span={self.span!r}, entity={self.entity!r}, "
            f"mode={self.mode!r}, action={self.action!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class _SensitiveOutcome:
    edits: tuple[_SensitiveEdit, ...]
    findings: tuple[TransformFinding, ...] = ()

    def __repr__(self) -> str:
        return f"_SensitiveOutcome(edits={len(self.edits)}, findings={len(self.findings)})"


class _SensitiveTransform(Protocol):
    def _apply_sensitive(
        self,
        text: str,
        call: CallContext,
        transaction: RedactionTransaction,
    ) -> _SensitiveOutcome: ...


_ = _SensitiveTransform


@dataclass(frozen=True, slots=True)
class UriPolicy:
    origins: frozenset[tuple[str, str, int]]
    on_placeholder_in_url: Literal["block", "allow"] = "block"


@dataclass(frozen=True, slots=True)
class _SensitiveRegistration:
    hmac_key: bytes | None = None
    uri_policy: UriPolicy | None = None
    restore_context: bool = True


_REGISTRY: dict[int, tuple[TransformPolicy, _SensitiveRegistration]] = {}


def register_sensitive(
    policy: TransformPolicy,
    *,
    hmac_key: bytes | None = None,
    uri_policy: UriPolicy | None = None,
    restore_context: bool = True,
) -> TransformPolicy:
    """Add a jes-owned or test fake policy to the closed sensitive table."""

    _REGISTRY[id(policy)] = (
        policy,
        _SensitiveRegistration(
            hmac_key=hmac_key,
            uri_policy=uri_policy,
            restore_context=restore_context,
        ),
    )
    return policy


def is_sensitive(policy: TransformPolicy) -> bool:
    return id(policy) in _REGISTRY


def sensitive_key(policy: TransformPolicy) -> bytes | None:
    registered = _REGISTRY.get(id(policy))
    return None if registered is None else registered[1].hmac_key


def sensitive_settings(policy: TransformPolicy) -> _SensitiveRegistration | None:
    registered = _REGISTRY.get(id(policy))
    return None if registered is None else registered[1]


@dataclass(slots=True)
class _FinalizerEntry:
    start: int
    end: int
    marker: str
    value: str
    tombstoned: bool = False


class FinalizerMap:
    """Call-local output restorations that reject copy and overlap."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[_FinalizerEntry] = []

    def add(self, start: int, end: int, marker: str, value: str) -> None:
        self._entries.append(_FinalizerEntry(start, end, marker, value))

    def compose(self, edits: Sequence[tuple[int, int, str]]) -> None:
        for entry in self._entries:
            if entry.tombstoned:
                continue
            if any(
                not (end <= entry.start or start >= entry.end) for start, end, _replacement in edits
            ):
                entry.tombstoned = True
                continue
            delta = 0
            for start, end, replacement in edits:
                if end <= entry.start:
                    delta += len(replacement) - (end - start)
            entry.start += delta
            entry.end += delta

    def restore(self, sanitized: str) -> str:
        live = [
            entry
            for entry in self._entries
            if not entry.tombstoned and sanitized[entry.start : entry.end] == entry.marker
        ]
        live.sort(key=lambda item: item.start)
        parts: list[str] = []
        cursor = 0
        for entry in live:
            parts.append(sanitized[cursor : entry.start])
            parts.append(entry.value)
            cursor = entry.end
        parts.append(sanitized[cursor:])
        return "".join(parts)

    def __repr__(self) -> str:
        return f"FinalizerMap(entries={len(self._entries)})"

    def __copy__(self) -> FinalizerMap:
        raise TypeError("finalizer maps cannot be copied")

    def __deepcopy__(self, memo: object) -> FinalizerMap:
        del memo
        raise TypeError("finalizer maps cannot be copied")


def replacement_for(
    *,
    mode: _SensitiveMode,
    entity: str,
    value: str,
    transaction: RedactionTransaction | None,
    hmac_key: bytes | None,
    allow_conversation_token: bool,
) -> str:
    if mode == "conversation_token" and allow_conversation_token and transaction is not None:
        return transaction.token_for(entity, value)
    if mode == "mask_all":
        return "******"
    if mode == "mask_partial":
        if len(value) > 4:
            masked = f"{value[:2]}{value[-2:]}"
            if value not in masked:
                return masked
        return "*" * max(len(value), 1)
    if mode == "hmac":
        if hmac_key is None:
            raise PolicyExecutionError("hmac sensitive mode requires a configured key")
        digest = hmac.new(hmac_key, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return digest
    return f"[REDACTED_{entity.upper()}]"


def apply_sensitive_edits(
    text: str,
    edits: Sequence[_SensitiveEdit],
    *,
    transaction: RedactionTransaction | None,
    hmac_key: bytes | None,
    allow_conversation_token: bool,
    finalizers: FinalizerMap,
) -> tuple[str, tuple[tuple[int, int, str], ...], frozenset[str]]:
    """Apply engine-owned replacements and return public-style edits."""

    ordered = sorted(edits, key=lambda item: (item.span.start, item.span.end))
    last = 0
    public: list[tuple[int, int, str]] = []
    issued: set[str] = set()
    for edit in ordered:
        if edit.span.start < last or edit.span.end < edit.span.start or edit.span.end > len(text):
            raise PolicyExecutionError("sensitive edits must be sorted and non-overlapping")
        value = text[edit.span.start : edit.span.end]
        if not value:
            raise PolicyExecutionError("sensitive edit must cover a complete value")
        replacement = replacement_for(
            mode=edit.mode,
            entity=edit.entity,
            value=value,
            transaction=transaction,
            hmac_key=hmac_key,
            allow_conversation_token=allow_conversation_token,
        )
        if value in replacement:
            raise PolicyExecutionError("sensitive replacement still contains the detected value")
        public.append((edit.span.start, edit.span.end, replacement))
        if edit.mode == "conversation_token" and replacement.startswith("[JES_"):
            issued.add(replacement)
        last = edit.span.end

    parts: list[str] = []
    cursor = 0
    shift_map: list[tuple[int, int, str]] = []
    for start, end, replacement in public:
        parts.append(text[cursor:start])
        out_start = sum(len(part) for part in parts)
        parts.append(replacement)
        if any(edit.mode == "output_local" and edit.span.start == start for edit in ordered):
            finalizers.add(out_start, out_start + len(replacement), replacement, text[start:end])
        shift_map.append((start, end, replacement))
        cursor = end
    parts.append(text[cursor:])
    rewritten = "".join(parts)
    for start, end, _replacement in public:
        value = text[start:end]
        if value and value in rewritten:
            raise PolicyExecutionError("sensitive value remained after replacement")
    return rewritten, tuple(shift_map), frozenset(issued)


def iter_sensitive(policies: Iterable[TransformPolicy]) -> tuple[TransformPolicy, ...]:
    return tuple(policy for policy in policies if is_sensitive(policy))
