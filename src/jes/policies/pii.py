"""Presidio-backed PII detection with NFKC and machine-identifier views."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

from jes._engine.sensitive import (
    UriPolicy,
    _SensitiveEdit,
    _SensitiveMode,
    _SensitiveOutcome,
    register_sensitive,
)
from jes._engine.uri import parse_restore_origin
from jes.errors import PolicyError
from jes.policies._fold import machine_identifier, nfkc
from jes.policies._protocols import (
    CallContext,
    Phase,
    TransformFinding,
    TransformOutcome,
    TransformPolicy,
)
from jes.questions import validate_identifier
from jes.redactions import RedactionTransaction
from jes.types import Action, Span, Stage

PiiInputMode = Literal["redact", "mask", "block"]
PiiUntrustedMode = Literal["mask", "redact", "block"]
PiiOutputMode = Literal["flag", "redact", "block"]
UrlMode = Literal["block", "allow"]

_ALL_STAGES: tuple[Stage, ...] = ("input", "untrusted", "tool_call", "tool_result", "output")
_DEFAULT_ENTITIES = (
    "CREDIT_CARD",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "PERSON",
    "PHONE_NUMBER",
    "US_SSN",
    "US_BANK_NUMBER",
    "UUID",
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_BANK = re.compile(r"\b\d{9}\b")
_CRYPTO = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL_ADDRESS", _EMAIL),
    ("UUID", _UUID),
    ("IP_ADDRESS", _IPV4),
    ("US_SSN", _SSN),
    ("IBAN_CODE", _IBAN),
    ("PHONE_NUMBER", _PHONE),
    ("US_BANK_NUMBER", _BANK),
    ("CRYPTO", _CRYPTO),
    ("CREDIT_CARD", _CARD),
)


def _luhn(number: str) -> bool:
    digits = [int(char) for char in number if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = digit * 2 if index % 2 == parity else digit
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def _load_presidio() -> None:
    import importlib.util

    if importlib.util.find_spec("presidio_analyzer") is None:
        raise PolicyError("pii() requires the jes[pii] extra")


def _merge_hits(hits: list[tuple[Span, str]]) -> tuple[tuple[Span, str], ...]:
    if not hits:
        return ()
    ordered = sorted(hits, key=lambda item: (item[0].start, -item[0].end, item[1]))
    kept: list[tuple[Span, str]] = []
    for span, entity in ordered:
        if any(span.start < existing.end and span.end > existing.start for existing, _ in kept):
            continue
        kept.append((span, entity))
    return tuple(kept)


def _pattern_hits(text: str, entities: frozenset[str]) -> list[tuple[Span, str]]:
    mapped = machine_identifier(text)
    hits: list[tuple[Span, str]] = []
    for entity, pattern in _PATTERNS:
        if entity not in entities:
            continue
        for match in pattern.finditer(mapped.text):
            if entity == "CREDIT_CARD" and not _luhn(match.group()):
                continue
            origin = mapped.origin(match.start(), match.end())
            if origin.end > origin.start:
                hits.append((origin, entity))
    return hits


_analyzer_engine: Any = None
_analyzer_ready = False


def _analyzer() -> Any:
    global _analyzer_engine, _analyzer_ready
    if _analyzer_ready:
        return _analyzer_engine
    _analyzer_ready = True
    try:
        import spacy
        from presidio_analyzer import AnalyzerEngine

        if not any(spacy.util.is_package(name) for name in ("en_core_web_lg", "en_core_web_sm")):
            return None
        _analyzer_engine = AnalyzerEngine()
    except Exception:
        _analyzer_engine = None
    return _analyzer_engine


def _presidio_hits(text: str, entities: frozenset[str], language: str) -> list[tuple[Span, str]]:
    analyzer = _analyzer()
    if analyzer is None:
        return []
    view = nfkc(text)
    try:
        results = analyzer.analyze(text=view.text, language=language, entities=list(entities))
    except Exception:
        return []
    hits: list[tuple[Span, str]] = []
    for result in results:
        origin = view.origin(result.start, result.end)
        entity = getattr(result, "entity_type", "")
        if entity in entities and origin.end > origin.start:
            hits.append((origin, entity))
    return hits


def _mode_for(
    *,
    stage: Stage,
    input_mode: PiiInputMode,
    untrusted_mode: PiiUntrustedMode,
    output_mode: PiiOutputMode,
    restore: bool,
) -> tuple[_SensitiveMode, Action]:
    if stage == "tool_result":
        stage = "untrusted"
    if stage == "output":
        if output_mode == "flag":
            return "output_local", "flag"
        if output_mode == "block":
            return "irreversible", "block"
        return "irreversible", "redact"
    if stage == "untrusted":
        if untrusted_mode == "block":
            return "mask_partial", "block"
        if untrusted_mode == "redact":
            return "irreversible", "redact"
        return "mask_partial", "redact"
    if input_mode == "mask":
        return "mask_partial", "redact"
    if input_mode == "block":
        return ("conversation_token" if restore else "irreversible"), "block"
    if restore:
        return "conversation_token", "redact"
    return "irreversible", "redact"


@dataclass(frozen=True, slots=True)
class _Pii:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    entities: frozenset[str]
    language: str
    input_mode: PiiInputMode
    untrusted_mode: PiiUntrustedMode
    output_mode: PiiOutputMode
    restore: bool

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        del call
        return TransformOutcome(text=text)

    def _apply_sensitive(
        self,
        text: str,
        call: CallContext,
        transaction: RedactionTransaction,
    ) -> _SensitiveOutcome:
        del transaction
        hits = _pattern_hits(text, self.entities)
        if "PERSON" in self.entities:
            hits.extend(
                hit
                for hit in _presidio_hits(text, frozenset({"PERSON"}), self.language)
                if hit[1] == "PERSON"
            )
        merged = _merge_hits(hits)
        if not merged:
            return _SensitiveOutcome(())
        if call.origin_stage == "tool_call":
            return _SensitiveOutcome(
                (),
                tuple(TransformFinding(entity, "block", (span,)) for span, entity in merged),
            )
        mode, action = _mode_for(
            stage=call.origin_stage,
            input_mode=self.input_mode,
            untrusted_mode=self.untrusted_mode,
            output_mode=self.output_mode,
            restore=self.restore,
        )
        edits = tuple(_SensitiveEdit(span, entity, mode, action) for span, entity in merged)
        findings = tuple(TransformFinding(entity, action, (span,)) for span, entity in merged)
        return _SensitiveOutcome(edits, findings)


def pii(
    entities: Iterable[str] | None = None,
    *,
    language: str = "en",
    ner: str = "spacy",
    input_mode: PiiInputMode = "redact",
    untrusted_mode: PiiUntrustedMode = "mask",
    output_mode: PiiOutputMode = "flag",
    restore: bool = True,
    restore_origins: Iterable[str] = (),
    on_placeholder_in_url: UrlMode = "block",
    name: str = "pii",
    stages: Iterable[Stage] = _ALL_STAGES,
) -> _Pii:
    """Detect PII and ask the engine to apply jes-owned sensitive replacements."""

    if language != "en":
        raise PolicyError("pii supports language='en' only in v1")
    if input_mode not in {"redact", "mask", "block"}:
        raise PolicyError("invalid pii input_mode")
    if untrusted_mode not in {"mask", "redact", "block"}:
        raise PolicyError("invalid pii untrusted_mode")
    if output_mode not in {"flag", "redact", "block"}:
        raise PolicyError("invalid pii output_mode")
    if on_placeholder_in_url not in {"block", "allow"}:
        raise PolicyError("on_placeholder_in_url must be block or allow")
    if ner != "spacy":
        import importlib.util

        if importlib.util.find_spec("transformers") is None:
            raise PolicyError("pii(ner=<model>) requires the jes[pii-ner] extra")
    _load_presidio()
    selected = tuple(_DEFAULT_ENTITIES if entities is None else entities)
    if not selected:
        raise PolicyError("pii requires at least one entity")
    for entity in selected:
        validate_identifier(entity, field="finding label")
        if entity not in _DEFAULT_ENTITIES and entity != "PERSON":
            validate_identifier(entity, field="finding label")
    validate_identifier(name, field="policy name")
    origins = frozenset(parse_restore_origin(item) for item in restore_origins)
    frozen_stages = frozenset(stages)
    if not frozen_stages:
        raise PolicyError("policy stages must not be empty")
    policy = _Pii(
        name=name,
        labels=frozenset(selected),
        stages=frozen_stages,
        phase="detect",
        fingerprint=hashlib.sha256(
            "\0".join(
                [
                    "pii",
                    name,
                    language,
                    ner,
                    input_mode,
                    untrusted_mode,
                    output_mode,
                    str(restore),
                    *sorted(selected),
                    *sorted(frozen_stages),
                ]
            ).encode()
        ).hexdigest(),
        entities=frozenset(selected),
        language=language,
        input_mode=input_mode,
        untrusted_mode=untrusted_mode,
        output_mode=output_mode,
        restore=restore,
    )
    register_sensitive(
        cast(TransformPolicy, policy),
        uri_policy=UriPolicy(origins=origins, on_placeholder_in_url=on_placeholder_in_url),
        restore_context=restore,
    )
    return policy


__all__ = ["pii"]
