"""pii, secrets, and canary: policies that find values the engine then replaces."""

from __future__ import annotations

import importlib.util
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, TypeAlias

from jes.engine.restore import Origin, UrlMode, parse_origin
from jes.errors import PolicyError
from jes.policies.base import (
    SensitiveHit,
    SensitiveMode,
    SensitivePolicy,
    TransformContext,
    freeze_stages,
    merge_spans,
)
from jes.text import fold, machine_identifier, nfkc
from jes.types import STAGES, Action, Span, Stage, validate_identifier

PiiInputMode: TypeAlias = Literal["redact", "mask", "block"]
PiiUntrustedMode: TypeAlias = Literal["mask", "redact", "block"]
PiiOutputMode: TypeAlias = Literal["flag", "redact", "block"]
ToolCallMode: TypeAlias = Literal["block", "flag"]
SecretRedact: TypeAlias = Literal["all", "partial", "hmac"]

# Checked in this order. Where two matches overlap, the earlier and longer one wins.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL_ADDRESS", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("UUID", re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
    ),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    # North American numbers with separators. A bare run of ten digits is too often
    # something else, such as a Unix timestamp.
    (
        "PHONE_NUMBER",
        re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\w)"),
    ),
    ("US_BANK_NUMBER", re.compile(r"\b\d{9}\b")),
    ("CRYPTO", re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
)
PII_ENTITIES: tuple[str, ...] = (*(entity for entity, _pattern in _PII_PATTERNS), "PERSON")
DEFAULT_PII_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "CRYPTO",
    "PERSON",
)
_SPACY_MODELS = ("en_core_web_lg", "en_core_web_md", "en_core_web_sm")

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{19,}"),  # OpenAI, including sk-proj-; Anthropic
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),  # GitHub fine-grained tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key ids
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),  # Google API keys
    re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}"),  # Stripe
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.DOTALL
    ),
)
# Public IP addresses are not credentials. IP_ADDRESS is a pii entity instead.
_SKIPPED_PLUGINS = frozenset({"IPPublicDetector"})

ALL_STAGES: tuple[Stage, ...] = STAGES


def _non_overlapping(hits: Iterable[tuple[Span, str]]) -> list[tuple[Span, str]]:
    kept: list[tuple[Span, str]] = []
    for span, entity in sorted(hits, key=lambda item: (item[0].start, -item[0].end)):
        if kept and span.start < kept[-1][0].end:
            continue
        kept.append((span, entity))
    return kept


def _luhn(number: str) -> bool:
    digits = [int(char) for char in number if char.isdigit()]
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = digit * 2 if index % 2 == parity else digit
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def _installed_spacy_model() -> str | None:
    if importlib.util.find_spec("spacy") is None:
        return None
    import spacy.util

    return next((name for name in _SPACY_MODELS if spacy.util.is_package(name)), None)


class _People:
    """Presidio's analyzer with the installed spaCy model, built once for every pii policy."""

    _lock = threading.Lock()
    _analyzer: Any = None

    @classmethod
    def analyzer(cls) -> Any:
        with cls._lock:
            if cls._analyzer is None:
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import NlpEngineProvider

                model = _installed_spacy_model()
                if model is None:
                    raise PolicyError("PERSON needs a spaCy English model")
                configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": model}],
                }
                engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
                cls._analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])
            return cls._analyzer


def _people(text: str) -> list[tuple[Span, str]]:
    view = nfkc(text)
    results = _People.analyzer().analyze(text=view.text, language="en", entities=["PERSON"])
    hits: list[tuple[Span, str]] = []
    for result in results:
        span = view.origin(result.start, result.end)
        if result.entity_type == "PERSON" and span.end > span.start:
            hits.append((span, "PERSON"))
    return hits


@dataclass(frozen=True, slots=True, eq=False)
class Pii(SensitivePolicy):
    name: str
    stages: frozenset[Stage]
    entities: frozenset[str]
    input_mode: PiiInputMode
    untrusted_mode: PiiUntrustedMode
    output_mode: PiiOutputMode
    tool_call_mode: ToolCallMode
    restore: bool
    restore_origins: frozenset[Origin]
    on_placeholder_in_url: UrlMode

    def detect(self, text: str, context: TransformContext) -> list[SensitiveHit]:
        context.check_deadline()
        view = machine_identifier(text)
        hits: list[tuple[Span, str]] = []
        for entity, pattern in _PII_PATTERNS:
            if entity not in self.entities:
                continue
            for match in pattern.finditer(view.text):
                if entity == "CREDIT_CARD" and not _luhn(match.group()):
                    continue
                hits.append((view.origin(match.start(), match.end()), entity))
        if "PERSON" in self.entities:
            hits.extend(_people(text))
        mode, action = self._replacement(context.origin)
        return [SensitiveHit(span, entity, mode, action) for span, entity in _non_overlapping(hits)]

    def _replacement(self, origin: Stage) -> tuple[SensitiveMode, Action]:
        """How to replace a value, by where the text came from."""

        if origin == "tool_call":
            return "remove", self.tool_call_mode
        if origin == "output":
            if self.output_mode == "flag":
                return "local", "flag"
            return "remove", self.output_mode
        if origin in ("untrusted", "tool_result"):
            if self.untrusted_mode == "redact":
                return "remove", "redact"
            return "partial", ("block" if self.untrusted_mode == "block" else "redact")
        if self.input_mode == "mask":
            return "partial", "redact"
        if self.input_mode == "block":
            return "remove", "block"
        return ("token" if self.restore else "remove"), "redact"


def pii(
    entities: Iterable[str] | None = None,
    *,
    input_mode: PiiInputMode = "redact",
    untrusted_mode: PiiUntrustedMode = "mask",
    output_mode: PiiOutputMode = "flag",
    tool_call_mode: ToolCallMode = "block",
    restore: bool = True,
    restore_origins: Iterable[str] = (),
    on_placeholder_in_url: UrlMode = "block",
    name: str = "pii",
    stages: Iterable[Stage] = ALL_STAGES,
) -> Pii:
    """Find personal data and keep it from the model, then give it back in the reply.

    In user input a value becomes a conversation token, such as ``[JES_PII_...]``, that
    ``check_output`` restores. Retrieved text and tool results are masked. A reply's own
    personal data is flagged and hidden from judges only. A tool call that carries one
    is blocked, because its arguments cannot be rewritten.

    The default entities are email, phone, credit card, US SSN, IBAN, crypto address,
    and person. ``UUID``, ``IP_ADDRESS``, and ``US_BANK_NUMBER`` are opt-in. ``PERSON``
    uses Presidio with a spaCy English model, such as ``en_core_web_sm``.

    A token inside a URL restores only for the http or https origins in
    ``restore_origins``. Otherwise ``on_placeholder_in_url`` blocks the reply, or with
    ``"allow"`` lets it through with the token left in place.
    """

    _choice(input_mode, ("redact", "mask", "block"), "input_mode")
    _choice(untrusted_mode, ("mask", "redact", "block"), "untrusted_mode")
    _choice(output_mode, ("flag", "redact", "block"), "output_mode")
    _choice(tool_call_mode, ("block", "flag"), "tool_call_mode")
    _choice(on_placeholder_in_url, ("block", "allow"), "on_placeholder_in_url")
    selected = frozenset(DEFAULT_PII_ENTITIES if entities is None else entities)
    if not selected:
        raise PolicyError("pii needs at least one entity")
    unknown = selected - set(PII_ENTITIES)
    if unknown:
        raise PolicyError(f"unknown pii entity: {sorted(unknown)[0]}")
    if "PERSON" in selected:
        if importlib.util.find_spec("presidio_analyzer") is None:
            raise PolicyError("PERSON requires the jes[pii] extra")
        if _installed_spacy_model() is None:
            raise PolicyError(
                "PERSON needs a spaCy English model: run `python -m spacy download "
                "en_core_web_sm`, or pass entities without PERSON"
            )
    return Pii(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        entities=selected,
        input_mode=input_mode,
        untrusted_mode=untrusted_mode,
        output_mode=output_mode,
        tool_call_mode=tool_call_mode,
        restore=restore,
        restore_origins=frozenset(parse_origin(origin) for origin in restore_origins),
        on_placeholder_in_url=on_placeholder_in_url,
    )


def _choice(value: str, options: tuple[str, ...], field: str) -> None:
    if value not in options:
        raise PolicyError(f"{field} must be one of {', '.join(options)}")


@cache
def _secret_plugins() -> tuple[Any, ...]:
    from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class

    classes = get_mapping_from_secret_type_to_class().values()
    return tuple(plugin() for plugin in classes if plugin.__name__ not in _SKIPPED_PLUGINS)


def _secret_spans(text: str) -> list[Span]:
    spans = [
        Span(*match.span()) for pattern in _SECRET_PATTERNS for match in pattern.finditer(text)
    ]
    offset = 0
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        for plugin in _secret_plugins():
            for secret in plugin.analyze_line(filename="jes", line=line, line_number=number):
                value = getattr(secret, "secret_value", None)
                if not value:
                    continue
                start = line.find(value)
                while start >= 0:
                    spans.append(Span(offset + start, offset + start + len(value)))
                    start = line.find(value, start + 1)
        offset += len(line)
    return list(merge_spans(spans))


@dataclass(frozen=True, slots=True, eq=False)
class Secrets(SensitivePolicy):
    name: str
    stages: frozenset[Stage]
    mode: Literal["mask", "partial", "hmac"]
    key: bytes | None

    @property
    def hmac_key(self) -> bytes | None:
        return self.key

    def detect(self, text: str, context: TransformContext) -> list[SensitiveHit]:
        context.check_deadline()
        view = machine_identifier(text)
        action: Action = "block" if context.origin == "tool_call" else "redact"
        hits = [view.origin(span.start, span.end) for span in _secret_spans(view.text)]
        return [SensitiveHit(span, "secret", self.mode, action) for span in merge_spans(hits)]

    def __repr__(self) -> str:
        return f"Secrets(name={self.name!r}, mode={self.mode!r})"


def secrets(
    redact: SecretRedact = "all",
    *,
    key: bytes | None = None,
    name: str = "secrets",
    stages: Iterable[Stage] = ALL_STAGES,
) -> Secrets:
    """Find API keys, tokens, and passwords with detect-secrets and known key formats.

    ``"all"`` replaces a secret with ``******``, ``"partial"`` keeps its first and last two
    characters, and ``"hmac"`` replaces it with a keyed hash so the same secret matches
    across conversations. A tool call that carries a secret is blocked.
    """

    _choice(redact, ("all", "partial", "hmac"), "redact")
    if redact == "hmac":
        if key is None or len(key) < 32:
            raise PolicyError("hmac mode needs a key of at least 32 bytes")
    elif key is not None:
        raise PolicyError("key is only for hmac mode")
    if importlib.util.find_spec("detect_secrets") is None:
        raise PolicyError("secrets() requires the jes[secrets] extra")
    modes: dict[str, Literal["mask", "partial", "hmac"]] = {
        "all": "mask",
        "partial": "partial",
        "hmac": "hmac",
    }
    return Secrets(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        mode=modes[redact],
        key=key,
    )


@dataclass(frozen=True, slots=True, eq=False)
class Canary(SensitivePolicy):
    name: str
    stages: frozenset[Stage]
    token: str
    folded: str

    def detect(self, text: str, context: TransformContext) -> list[SensitiveHit]:
        context.check_deadline()
        spans = _occurrences(text, self.token)
        view = fold(text)
        spans.extend(
            view.origin(span.start, span.end) for span in _occurrences(view.text, self.folded)
        )
        return [SensitiveHit(span, "canary", "remove", "block") for span in merge_spans(spans)]

    def __repr__(self) -> str:
        return f"Canary(name={self.name!r})"


def _occurrences(text: str, needle: str) -> list[Span]:
    spans: list[Span] = []
    start = text.find(needle)
    while start >= 0:
        spans.append(Span(start, start + len(needle)))
        start = text.find(needle, start + 1)
    return spans


def canary(
    token: str,
    *,
    name: str = "canary",
    stages: Iterable[Stage] = ("output", "tool_call"),
) -> Canary:
    """Block, and remove, a marker that must never leave: in a reply or a tool call.

    Put the marker in your system prompt. Matching also catches lookalike, cased, and
    invisible-character spellings of it.
    """

    if not token or not token.strip():
        raise PolicyError("the canary token must not be empty")
    folded = fold(token).text
    if not folded:
        raise PolicyError("the canary token folds to nothing")
    return Canary(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        token=token,
        folded=folded,
    )


__all__ = [
    "DEFAULT_PII_ENTITIES",
    "PII_ENTITIES",
    "Canary",
    "Pii",
    "PiiInputMode",
    "PiiOutputMode",
    "PiiUntrustedMode",
    "SecretRedact",
    "Secrets",
    "ToolCallMode",
    "canary",
    "pii",
    "secrets",
]
