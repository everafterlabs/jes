"""detect-secrets plus documented high-value token patterns."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

from jes._engine.sensitive import _SensitiveEdit, _SensitiveOutcome, register_sensitive
from jes.errors import PolicyError
from jes.policies._fold import machine_identifier
from jes.policies._protocols import (
    CallContext,
    Phase,
    TransformFinding,
    TransformOutcome,
    TransformPolicy,
)
from jes.questions import validate_identifier
from jes.redactions import RedactionTransaction
from jes.types import Span, Stage

SecretRedact = Literal["all", "partial", "hmac"]
_ALL_STAGES: tuple[Stage, ...] = ("input", "untrusted", "output")
_LOCK = threading.Lock()
_configured: str | None = None

_HIGH_VALUE = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,48}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def _load_detect_secrets() -> Any:
    try:
        import detect_secrets
    except ImportError:
        raise PolicyError("secrets() requires the jes[secrets] extra") from None
    return detect_secrets


def _configure(signature: str) -> None:
    global _configured
    with _LOCK:
        if _configured is None:
            _configured = signature
            return
        if signature != _configured:
            raise PolicyError("detect-secrets is already configured with different settings")


def _plugin_hits(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    try:
        from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class
    except Exception:
        return spans
    with _LOCK:
        for plugin_cls in get_mapping_from_secret_type_to_class().values():
            try:
                plugin = plugin_cls()
            except Exception:
                continue
            try:
                found = plugin.analyze_line(filename="memory", line=text, line_number=1)
            except Exception:
                continue
            for secret in found:
                secret_value = getattr(secret, "secret_value", None)
                if not secret_value:
                    continue
                start = 0
                while True:
                    index = text.find(secret_value, start)
                    if index < 0:
                        break
                    spans.append((index, index + len(secret_value)))
                    start = index + 1
    return spans


def _pattern_hits(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _HIGH_VALUE:
        for match in pattern.finditer(text):
            spans.append(match.span())
    return spans


def _merge(spans: list[tuple[int, int]]) -> tuple[Span, ...]:
    if not spans:
        return ()
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], end))
        else:
            merged.append((start, end))
    return tuple(Span(start, end) for start, end in merged)


@dataclass(frozen=True, slots=True)
class _Secrets:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    mode: Literal["mask_all", "mask_partial", "hmac"]

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        del call
        return TransformOutcome(text=text)

    def _apply_sensitive(
        self,
        text: str,
        call: CallContext,
        transaction: RedactionTransaction,
    ) -> _SensitiveOutcome:
        del call, transaction
        mapped = machine_identifier(text)
        hits = [*_pattern_hits(mapped.text), *_plugin_hits(mapped.text)]
        mapped_hits = [
            (mapped.origin(start, end).start, mapped.origin(start, end).end) for start, end in hits
        ]
        spans = _merge(mapped_hits)
        if not spans:
            return _SensitiveOutcome(())
        edits = tuple(_SensitiveEdit(span, "secret", self.mode, "redact") for span in spans)
        return _SensitiveOutcome(edits, (TransformFinding("secret", "redact", spans),))


def secrets(
    redact: SecretRedact = "all",
    *,
    key: bytes | None = None,
    name: str = "secrets",
    stages: Iterable[Stage] = _ALL_STAGES,
) -> _Secrets:
    """Redact secrets in memory with detect-secrets and high-value patterns."""

    if redact not in {"all", "partial", "hmac"}:
        raise PolicyError("secrets redact must be all, partial, or hmac")
    if redact == "hmac" and (key is None or len(key) < 32):
        raise PolicyError("hmac secrets mode requires a 32-byte key")
    validate_identifier(name, field="policy name")
    _load_detect_secrets()
    signature = hashlib.sha256(b"detect-secrets-default").hexdigest()
    _configure(signature)
    mode: Literal["mask_all", "mask_partial", "hmac"]
    if redact == "all":
        mode = "mask_all"
    elif redact == "partial":
        mode = "mask_partial"
    else:
        mode = "hmac"
    frozen_stages = frozenset(stages)
    if not frozen_stages:
        raise PolicyError("policy stages must not be empty")
    policy = _Secrets(
        name=name,
        labels=frozenset({"secret"}),
        stages=frozen_stages,
        phase="detect",
        fingerprint=hashlib.sha256(
            "\0".join(["secrets", name, redact, *sorted(frozen_stages)]).encode()
        ).hexdigest(),
        mode=mode,
    )
    register_sensitive(cast(TransformPolicy, policy), hmac_key=key)
    return policy


def _reset_secrets_config() -> None:  # pyright: ignore[reportUnusedFunction]
    global _configured
    with _LOCK:
        _configured = None


__all__ = ["secrets"]
