"""Local transforms: invisible characters, patterns, banned terms, token limits, and tool names."""

from __future__ import annotations

import importlib
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from jes.errors import PolicyError
from jes.policies.base import (
    Phase,
    TransformContext,
    TransformFinding,
    TransformOutcome,
    freeze_stages,
    merge_spans,
)
from jes.text import fold as fold_text
from jes.text._ucd_data import BIDI_CONTROL, DEFAULT_IGNORABLE
from jes.text.folding import Folded
from jes.text.textmap import Edit
from jes.text.unicode import (
    char_class,
    is_bidi_control,
    is_c0_c1,
    is_default_ignorable,
    is_extra_format,
    is_joiner,
    is_kept_control,
    is_registered_variation,
    is_variation_selector,
)
from jes.types import Span, Stage, validate_identifier

InvisibleMode: TypeAlias = Literal["targeted", "all"]
MatchMode: TypeAlias = Literal["search", "fullmatch"]
TermAction: TypeAlias = Literal["block", "redact"]
TokenMode: TypeAlias = Literal["block", "truncate"]

# Transforms that edit text skip tool calls by default: the app runs the arguments as given.
TEXT_STAGES: tuple[Stage, ...] = ("input", "untrusted", "tool_result", "output")
_REDACTED = "[REDACTED]"

# Opt-in exploit phrases, from LLM Guard's BanSubstrings list.
EXPLOIT_TERMS: tuple[str, ...] = (
    "arbitrary code execution",
    "a backdoor",
    "code injection",
    "cross-site scripting",
    "email spoofing",
    "a keylogger",
    "privilege escalation",
    "a rootkit",
    "a bootkit",
    "screen scraping",
    "corrupting a disk partition",
    "cryptojacking",
    "a botnet",
    "adware",
    "ransomware",
    r"Windows\System32\Config\SAM",
    "etc/shadow",
    "escalate privilege",
    "read the encrypted system password file",
)

# Every character the targeted mode could remove. Text without any skips the scan.
_TARGETED = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    + char_class(BIDI_CONTROL)
    + char_class(DEFAULT_IGNORABLE)
    + "]"
)


def _redacted(
    spans: Iterable[Span], label: str, *, replacement: str = _REDACTED
) -> TransformOutcome:
    merged = merge_spans(spans)
    return TransformOutcome(
        edits=tuple(Edit(span.start, span.end, replacement) for span in merged),
        findings=(TransformFinding(label, "redact", merged),),
    )


def _found(spans: Iterable[Span], label: str, action: Literal["block", "flag"]) -> TransformOutcome:
    return TransformOutcome(findings=(TransformFinding(label, action, merge_spans(spans)),))


@dataclass(frozen=True, slots=True)
class InvisibleText:
    name: str
    stages: frozenset[Stage]
    mode: InvisibleMode
    block: bool
    phase: Phase = "normalize"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        context.check_deadline()
        if self.mode == "targeted" and not _TARGETED.search(text):
            return TransformOutcome()
        removed: list[Span] = []
        previous: str | None = None
        for index, char in enumerate(text):
            if _removed(char, self.mode, previous):
                if removed and removed[-1].end == index:
                    removed[-1] = Span(removed[-1].start, index + 1)
                else:
                    removed.append(Span(index, index + 1))
            else:
                previous = char
        if not removed:
            return TransformOutcome()
        if self.block:
            return _found(removed, "invisible_text", "block")
        return _redacted(removed, "invisible_text", replacement="")


def _removed(char: str, mode: InvisibleMode, previous: str | None) -> bool:
    code = ord(char)
    if is_kept_control(code):
        return False
    if is_c0_c1(code) or is_bidi_control(code):
        return True
    if is_variation_selector(code):
        # A variation selector stays only where it forms a registered sequence.
        return previous is None or not is_registered_variation(previous + char)
    if is_joiner(code):
        # Joiners build emoji and some scripts, so only "all" removes them.
        return mode == "all"
    if is_default_ignorable(code):
        return True
    return mode == "all" and is_extra_format(char)


def invisible_text(
    mode: InvisibleMode = "targeted",
    *,
    block: bool = False,
    name: str = "invisible_text",
    stages: Iterable[Stage] = TEXT_STAGES,
) -> InvisibleText:
    """Remove invisible and control characters that smuggle instructions, or block on them.

    ``"targeted"`` removes controls, bidi overrides, and default-ignorable characters but
    keeps joiners and registered variation sequences. ``"all"`` also removes joiners,
    format, private-use, and unassigned characters.
    """

    if mode not in ("targeted", "all"):
        raise PolicyError("invisible_text mode must be targeted or all")
    return InvisibleText(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        mode=mode,
        block=block,
    )


def _folded(text: str, enabled: bool) -> Folded | None:
    return fold_text(text) if enabled else None


def _origin(folded: Folded | None, start: int, end: int) -> Span:
    return Span(start, end) if folded is None else folded.origin(start, end)


@dataclass(frozen=True, slots=True)
class Regex:
    name: str
    stages: frozenset[Stage]
    action: TermAction
    match: MatchMode
    require: bool
    fold: bool
    timeout_ms: int
    patterns: tuple[Any, ...]
    phase: Phase = "detect"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        context.check_deadline()
        folded = _folded(text, self.fold)
        subject = text if folded is None else folded.text
        timeout = _timeout(context, self.timeout_ms)
        spans: list[Span] = []
        try:
            for pattern in self.patterns:
                if self.match == "fullmatch":
                    found = pattern.fullmatch(subject, timeout=timeout)
                    matches = () if found is None else (found,)
                else:
                    matches = pattern.finditer(subject, timeout=timeout)
                for item in matches:
                    start, end = item.span()
                    if end > start:
                        spans.append(_origin(folded, start, end))
        except TimeoutError:
            return _found((), "regex_timeout", "block")
        if self.require:
            return TransformOutcome() if spans else _found((), "regex", "block")
        if not spans:
            return TransformOutcome()
        if self.action == "block":
            return _found(spans, "regex", "block")
        return _redacted(spans, "regex")


def _timeout(context: TransformContext, timeout_ms: int) -> float:
    timeout = timeout_ms / 1_000
    if context.deadline is None:
        return timeout
    # A spent deadline gives a zero timeout, so the pattern stops at once and blocks.
    return min(timeout, max(0.0, context.deadline - time.monotonic()))


def _optional_module(module: str, extra: str, factory: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        raise PolicyError(f"{factory}() requires the jes[{extra}] extra") from None


def regex(
    patterns: Iterable[str],
    *,
    action: TermAction = "block",
    match: MatchMode = "search",
    require: bool = False,
    fold: bool = False,
    timeout_ms: int = 50,
    name: str = "regex",
    stages: Iterable[Stage] = TEXT_STAGES,
) -> Regex:
    """Block or redact what your patterns match, with a per-check timeout.

    ``require=True`` inverts it: the text must match, or the check blocks. ``fold=True``
    matches lookalike and differently cased spellings. Patterns use the ``regex`` module.
    """

    if action not in ("block", "redact"):
        raise PolicyError("regex action must be block or redact")
    if match not in ("search", "fullmatch"):
        raise PolicyError("regex match must be search or fullmatch")
    if type(timeout_ms) is not int or timeout_ms < 1:
        raise PolicyError("timeout_ms must be a positive integer")
    sources = tuple(patterns)
    if not sources:
        raise PolicyError("regex needs at least one pattern")
    module = _optional_module("regex", "regex", "regex")
    try:
        compiled = tuple(module.compile(source) for source in sources)
    except Exception:
        raise PolicyError("invalid regex pattern") from None
    return Regex(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        action=action,
        match=match,
        require=require,
        fold=fold,
        timeout_ms=timeout_ms,
        patterns=compiled,
    )


@dataclass(frozen=True, slots=True)
class Substrings:
    name: str
    stages: frozenset[Stage]
    action: TermAction
    whole_words: bool
    fold: bool
    needles: tuple[str, ...]
    phase: Phase = "detect"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        context.check_deadline()
        folded = _folded(text, self.fold)
        subject = text if folded is None else folded.text
        spans: list[Span] = []
        for needle in self.needles:
            start = subject.find(needle)
            while start >= 0:
                end = start + len(needle)
                if not self.whole_words or _whole_word(subject, start, end):
                    spans.append(_origin(folded, start, end))
                start = subject.find(needle, start + 1)
        if not spans:
            return TransformOutcome()
        if self.action == "block":
            return _found(spans, "substrings", "block")
        return _redacted(spans, "substrings")


def _whole_word(text: str, start: int, end: int) -> bool:
    before = start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_")
    after = end < len(text) and (text[end].isalnum() or text[end] == "_")
    return not before and not after


def substrings(
    terms: Iterable[str],
    *,
    action: TermAction = "block",
    whole_words: bool = False,
    fold: bool = True,
    name: str = "substrings",
    stages: Iterable[Stage] = TEXT_STAGES,
) -> Substrings:
    """Block or redact banned terms. Folding also catches lookalike and cased spellings."""

    if action not in ("block", "redact"):
        raise PolicyError("substrings action must be block or redact")
    values = tuple(terms)
    if not values or not all(values):
        raise PolicyError("substrings needs non-empty terms")
    needles = tuple(fold_text(term).text if fold else term for term in values)
    if not all(needles):
        raise PolicyError("a substrings term folds to nothing")
    return Substrings(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        action=action,
        whole_words=whole_words,
        fold=fold,
        needles=needles,
    )


@dataclass(frozen=True, slots=True)
class TokenLimit:
    name: str
    stages: frozenset[Stage]
    limit: int
    mode: TokenMode
    encoding: Any
    phase: Phase = "limit"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        context.check_deadline()
        tokens = self.encoding.encode(text, disallowed_special=())
        if len(tokens) <= self.limit:
            return TransformOutcome()
        whole = (Span(0, len(text)),)
        if self.mode == "block":
            return _found(whole, "token_limit", "block")
        kept: str = self.encoding.decode(tokens[: self.limit])
        # Decoding a cut token list can change the last characters, so fall back to a rewrite.
        edit = Edit(len(kept), len(text), "") if text.startswith(kept) else Edit(0, len(text), kept)
        return TransformOutcome(
            edits=(edit,), findings=(TransformFinding("token_limit", "flag", whole),)
        )


def token_limit(
    limit: int,
    *,
    encoding: str = "cl100k_base",
    mode: TokenMode = "block",
    name: str = "token_limit",
    stages: Iterable[Stage] = ("input",),
) -> TokenLimit:
    """Block, or truncate, text longer than ``limit`` tokens of a tiktoken encoding."""

    if type(limit) is not int or limit < 1:
        raise PolicyError("token_limit needs a positive limit")
    if mode not in ("block", "truncate"):
        raise PolicyError("token_limit mode must be block or truncate")
    module = _optional_module("tiktoken", "tokens", "token_limit")
    try:
        loaded = module.get_encoding(encoding)
    except Exception:
        raise PolicyError("unknown token encoding") from None
    return TokenLimit(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(stages),
        limit=limit,
        mode=mode,
        encoding=loaded,
    )


@dataclass(frozen=True, slots=True)
class AllowedTools:
    name: str
    names: frozenset[str]
    stages: frozenset[Stage] = frozenset({"tool_call"})
    phase: Phase = "detect"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        if context.tool in self.names:
            return TransformOutcome()
        return _found((), "tool_name", "block")


def valid_tool_name(name: object) -> bool:
    """A tool name is one line of 1 to 256 characters without surrounding spaces."""

    return (
        isinstance(name, str)
        and 0 < len(name) <= 256
        and name.strip() == name
        and not any(ord(char) < 32 for char in name)
    )


def allowed_tools(names: Iterable[str], *, name: str = "allowed_tools") -> AllowedTools:
    """Block a tool call whose tool is not in ``names``. Arguments are not changed."""

    values = frozenset(names)
    if not values or not all(valid_tool_name(item) for item in values):
        raise PolicyError("allowed_tools needs non-empty, single-line tool names")
    return AllowedTools(name=validate_identifier(name, field="policy name"), names=values)


__all__ = [
    "EXPLOIT_TERMS",
    "TEXT_STAGES",
    "AllowedTools",
    "InvisibleMode",
    "InvisibleText",
    "MatchMode",
    "Regex",
    "Substrings",
    "TermAction",
    "TokenLimit",
    "TokenMode",
    "allowed_tools",
    "invisible_text",
    "regex",
    "substrings",
    "token_limit",
    "valid_tool_name",
]
