"""Core text transforms: invisible_text, regex, substrings, and token_limit."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from jes._engine.sensitive import _SensitiveEdit, _SensitiveOutcome, register_sensitive
from jes.errors import DeadlineExceeded, PolicyError
from jes.policies._exploit_terms import EXPLOIT_TERMS
from jes.policies._fold import MappedText, fold
from jes.policies._protocols import (
    CallContext,
    Phase,
    TransformEdit,
    TransformFinding,
    TransformOutcome,
    TransformPolicy,
    apply_transform_edits,
)
from jes.policies._unicode import (
    is_bidi_control,
    is_c0_c1,
    is_default_ignorable,
    is_extra_format,
    is_joiner,
    is_kept_control,
    is_registered_variation,
    is_variation_selector,
)
from jes.questions import validate_identifier
from jes.redactions import RedactionTransaction
from jes.types import Action, Span, Stage

_REDACTED = "[REDACTED]"
_ALL_STAGES: tuple[Stage, ...] = ("input", "untrusted", "output")
RegexMatch = Literal["search", "fullmatch"]
InvisibleMode = Literal["targeted", "all"]
TokenMode = Literal["block", "truncate"]
RegexAction = Literal["block", "redact"]


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _freeze_stages(stages: Iterable[Stage]) -> frozenset[Stage]:
    values = frozenset(stages)
    if not values:
        raise PolicyError("policy stages must not be empty")
    return values


def _check_deadline(call: CallContext) -> None:
    if call.deadline is not None and time.monotonic() >= call.deadline:
        raise DeadlineExceeded("transform")


def _merge_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
    if not spans:
        return ()
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start <= previous.end:
            merged[-1] = Span(previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _is_whole_word(text: str, start: int, end: int) -> bool:
    if start > 0 and _is_word_char(text[start - 1]):
        return False
    return not (end < len(text) and _is_word_char(text[end]))


def _origin_span(mapped: MappedText | None, start: int, end: int) -> Span:
    if mapped is None:
        return Span(start, end)
    return mapped.origin(start, end)


def _redact_outcome(
    text: str,
    spans: Sequence[Span],
    *,
    label: str,
    replacement: str = _REDACTED,
) -> TransformOutcome:
    merged = _merge_spans(spans)
    edits = tuple(TransformEdit(span.start, span.end, replacement) for span in merged)
    return TransformOutcome(
        text=apply_transform_edits(text, edits),
        findings=(TransformFinding(label, "redact", merged),),
        edits=edits,
    )


def _finding_outcome(
    text: str,
    spans: Sequence[Span],
    *,
    label: str,
    action: Action,
) -> TransformOutcome:
    return TransformOutcome(
        text=text,
        findings=(TransformFinding(label, action, _merge_spans(spans)),),
    )


@dataclass(frozen=True, slots=True)
class _InvisibleText:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    mode: InvisibleMode
    block: bool

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        _check_deadline(call)
        kept: list[str] = []
        removed: list[Span] = []
        remove_start: int | None = None
        for index, char in enumerate(text):
            if _remove_invisible(char, self.mode, kept[-1] if kept else None):
                if remove_start is None:
                    remove_start = index
                continue
            if remove_start is not None:
                removed.append(Span(remove_start, index))
                remove_start = None
            kept.append(char)
        if remove_start is not None:
            removed.append(Span(remove_start, len(text)))
        if not removed:
            return TransformOutcome(text=text)
        if self.block:
            return _finding_outcome(text, removed, label="invisible_text", action="block")
        return _redact_outcome(text, removed, label="invisible_text", replacement="")


def _remove_invisible(char: str, mode: InvisibleMode, previous: str | None) -> bool:
    code = ord(char)
    if is_kept_control(code):
        return False
    if is_c0_c1(code) or is_bidi_control(code):
        return True
    if is_variation_selector(code):
        return previous is None or not is_registered_variation(previous + char)
    if is_joiner(code):
        return mode == "all"
    if is_default_ignorable(code):
        return True
    return mode == "all" and is_extra_format(char)


@dataclass(frozen=True, slots=True)
class _Regex:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    action: RegexAction
    match: RegexMatch
    require: bool
    fold: bool
    timeout_ms: int
    _patterns: tuple[Any, ...]

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        _check_deadline(call)
        mapped = fold(text) if self.fold else None
        subject = mapped.text if mapped is not None else text
        timeout = _regex_timeout(call, self.timeout_ms)
        spans: list[Span] = []
        try:
            for compiled in self._patterns:
                if self.match == "fullmatch":
                    found = compiled.fullmatch(subject, timeout=timeout)
                    matches = () if found is None else (found,)
                else:
                    matches = tuple(compiled.finditer(subject, timeout=timeout))
                for match in matches:
                    start, end = match.span()
                    if start == end:
                        continue
                    spans.append(_origin_span(mapped, start, end))
        except TimeoutError:
            return _finding_outcome(text, (), label="regex", action="block")
        if not spans:
            if self.require:
                return _finding_outcome(text, (), label="regex", action="block")
            return TransformOutcome(text=text)
        if self.require:
            return TransformOutcome(text=text)
        if self.action == "block":
            return _finding_outcome(text, spans, label="regex", action="block")
        return _redact_outcome(text, spans, label="regex")


@dataclass(frozen=True, slots=True)
class _Substrings:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    action: RegexAction
    whole_words: bool
    fold: bool
    _needles: tuple[tuple[str, str], ...]

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        _check_deadline(call)
        mapped = fold(text) if self.fold else None
        subject = mapped.text if mapped is not None else text
        spans: list[Span] = []
        for _original, needle in self._needles:
            start = 0
            while needle:
                found = subject.find(needle, start)
                if found < 0:
                    break
                end = found + len(needle)
                if not self.whole_words or _is_whole_word(subject, found, end):
                    spans.append(_origin_span(mapped, found, end))
                start = found + 1
        if not spans:
            return TransformOutcome(text=text)
        if self.action == "block":
            return _finding_outcome(text, spans, label="substrings", action="block")
        return _redact_outcome(text, spans, label="substrings")


@dataclass(frozen=True, slots=True)
class _TokenLimit:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    limit: int
    mode: TokenMode
    _encoding: Any

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        _check_deadline(call)
        tokens = self._encoding.encode(text)
        if len(tokens) <= self.limit:
            return TransformOutcome(text=text)
        if self.mode == "block":
            return _finding_outcome(
                text,
                (Span(0, len(text)),),
                label="token_limit",
                action="block",
            )
        kept = self._encoding.decode(tokens[: self.limit])
        if text.startswith(kept):
            edits = (TransformEdit(len(kept), len(text), ""),)
            rewritten = kept
        else:
            edits = (TransformEdit(0, len(text), kept),)
            rewritten = kept
        return TransformOutcome(
            text=rewritten,
            findings=(TransformFinding("token_limit", "flag", (Span(0, len(text)),)),),
            edits=edits,
        )


def _regex_timeout(call: CallContext, timeout_ms: int) -> float:
    requested = timeout_ms / 1000.0
    if call.deadline is None:
        return requested
    remaining = call.deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("transform")
    return min(requested, remaining)


def _load_regex() -> Any:
    try:
        import regex as module
    except ImportError:
        raise PolicyError("regex() requires the jes[regex] extra") from None
    return module


def _load_tiktoken() -> Any:
    try:
        import tiktoken as module
    except ImportError:
        raise PolicyError("token_limit() requires the jes[tokens] extra") from None
    return module


def invisible_text(
    mode: InvisibleMode = "targeted",
    *,
    block: bool = False,
    name: str = "invisible_text",
    stages: Iterable[Stage] = _ALL_STAGES,
) -> _InvisibleText:
    """Remove invisible and control characters used to smuggle text."""

    if mode not in {"targeted", "all"}:
        raise PolicyError("invisible_text mode must be targeted or all")
    validate_identifier(name, field="policy name")
    frozen_stages = _freeze_stages(stages)
    return _InvisibleText(
        name=name,
        labels=frozenset({"invisible_text"}),
        stages=frozen_stages,
        phase="normalize",
        fingerprint=_fingerprint("invisible_text", name, mode, str(block), *sorted(frozen_stages)),
        mode=mode,
        block=block,
    )


def regex(
    patterns: Iterable[str],
    *,
    action: RegexAction = "block",
    match: RegexMatch = "search",
    require: bool = False,
    fold: bool = False,
    timeout_ms: int = 50,
    name: str = "regex",
    stages: Iterable[Stage] = _ALL_STAGES,
) -> _Regex:
    """Match caller patterns with a timeout-bounded regex engine."""

    if action not in {"block", "redact"}:
        raise PolicyError("regex action must be block or redact")
    if match not in {"search", "fullmatch"}:
        raise PolicyError("regex match must be search or fullmatch")
    if isinstance(timeout_ms, bool) or timeout_ms < 1:
        raise PolicyError("timeout_ms must be a positive integer")
    compiled_patterns = tuple(patterns)
    if not compiled_patterns:
        raise PolicyError("regex requires at least one pattern")
    validate_identifier(name, field="policy name")
    module = _load_regex()
    try:
        loaded = tuple(module.compile(pattern) for pattern in compiled_patterns)
    except Exception:
        raise PolicyError("invalid regex pattern") from None
    frozen_stages = _freeze_stages(stages)
    return _Regex(
        name=name,
        labels=frozenset({"regex"}),
        stages=frozen_stages,
        phase="detect",
        fingerprint=_fingerprint(
            "regex",
            name,
            action,
            match,
            str(require),
            str(fold),
            str(timeout_ms),
            *compiled_patterns,
            *sorted(frozen_stages),
        ),
        action=action,
        match=match,
        require=require,
        fold=fold,
        timeout_ms=timeout_ms,
        _patterns=loaded,
    )


def substrings(
    terms: Iterable[str],
    *,
    action: RegexAction = "block",
    whole_words: bool = False,
    fold: bool = True,
    name: str = "substrings",
    stages: Iterable[Stage] = _ALL_STAGES,
) -> _Substrings:
    """Ban or redact exact and lookalike substrings."""

    if action not in {"block", "redact"}:
        raise PolicyError("substrings action must be block or redact")
    values = tuple(terms)
    if not values or any(not term for term in values):
        raise PolicyError("substrings requires non-empty terms")
    validate_identifier(name, field="policy name")
    needles: list[tuple[str, str]] = []
    for term in values:
        needle = fold_text(term) if fold else term
        if not needle:
            raise PolicyError("substrings term folded to empty")
        needles.append((term, needle))
    frozen_stages = _freeze_stages(stages)
    return _Substrings(
        name=name,
        labels=frozenset({"substrings"}),
        stages=frozen_stages,
        phase="detect",
        fingerprint=_fingerprint(
            "substrings",
            name,
            action,
            str(whole_words),
            str(fold),
            *values,
            *sorted(frozen_stages),
        ),
        action=action,
        whole_words=whole_words,
        fold=fold,
        _needles=tuple(needles),
    )


def fold_text(text: str) -> str:
    return fold(text).text


def token_limit(
    limit: int,
    *,
    encoding: str = "cl100k_base",
    mode: TokenMode = "block",
    name: str = "token_limit",
    stages: Iterable[Stage] = ("input",),
) -> _TokenLimit:
    """Count tokens and either block or truncate past the limit."""

    if isinstance(limit, bool) or limit < 1:
        raise PolicyError("token_limit requires a positive limit")
    if mode not in {"block", "truncate"}:
        raise PolicyError("token_limit mode must be block or truncate")
    validate_identifier(name, field="policy name")
    module = _load_tiktoken()
    try:
        loaded = module.get_encoding(encoding)
    except Exception:
        raise PolicyError("unknown token encoding") from None
    frozen_stages = _freeze_stages(stages)
    return _TokenLimit(
        name=name,
        labels=frozenset({"token_limit"}),
        stages=frozen_stages,
        phase="limit",
        fingerprint=_fingerprint(
            "token_limit",
            name,
            str(limit),
            encoding,
            mode,
            *sorted(frozen_stages),
        ),
        limit=limit,
        mode=mode,
        _encoding=loaded,
    )


@dataclass(frozen=True, slots=True)
class _Canary:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str
    token: str

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
        edits: list[_SensitiveEdit] = []
        spans: list[Span] = []
        start = 0
        while self.token:
            found = text.find(self.token, start)
            if found < 0:
                break
            span = Span(found, found + len(self.token))
            spans.append(span)
            edits.append(_SensitiveEdit(span, "canary", "irreversible", "block"))
            start = span.end
        if not edits:
            return _SensitiveOutcome(())
        return _SensitiveOutcome(
            tuple(edits),
            (TransformFinding("canary", "block", tuple(spans)),),
        )


def canary(
    token: str,
    *,
    name: str = "canary",
    stages: Iterable[Stage] = ("output",),
) -> _Canary:
    """Irreversibly remove and block a leaked application canary token."""

    if not token:
        raise PolicyError("canary token must be non-empty")
    validate_identifier(name, field="policy name")
    frozen_stages = _freeze_stages(stages)
    policy = _Canary(
        name=name,
        labels=frozenset({"canary"}),
        stages=frozen_stages,
        phase="detect",
        fingerprint=_fingerprint("canary", name, token, *sorted(frozen_stages)),
        token=token,
    )
    register_sensitive(cast(TransformPolicy, policy))
    return policy


__all__ = [
    "EXPLOIT_TERMS",
    "canary",
    "invisible_text",
    "regex",
    "substrings",
    "token_limit",
]
