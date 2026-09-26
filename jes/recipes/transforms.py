"""Unevaluated transform recipes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import time
from collections.abc import Iterable
from typing import Any, cast

from jes.errors import DeadlineExceeded, PolicyError
from jes.policies import TransformPolicy, substrings
from jes.policies._protocols import (
    CallContext,
    Phase,
    TransformEdit,
    TransformFinding,
    TransformOutcome,
)
from jes.questions import validate_identifier
from jes.recipes.prompts import REFUSAL_PHRASES_V1
from jes.types import Span, Stage

_WORDS = re.compile(r"\S+")
_WORDS_PER_MINUTE = 200


def competitors(names: Iterable[str]) -> TransformPolicy:
    """Redact competitor names. This recipe has no threshold."""

    values = tuple(names)
    if not values:
        raise PolicyError("competitors requires at least one name")
    return cast(TransformPolicy, substrings(values, action="redact", name="competitors"))


def refusal_phrases(phrases: Iterable[str] | None = None) -> TransformPolicy:
    """Block on a fixed refusal-phrase list. This recipe has no threshold."""

    values = tuple(REFUSAL_PHRASES_V1 if phrases is None else phrases)
    return cast(
        TransformPolicy,
        substrings(values, action="block", whole_words=False, name="refusal_phrases"),
    )


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _deadline(call: CallContext) -> None:
    if call.deadline is not None and time.monotonic() >= call.deadline:
        raise DeadlineExceeded("transform")


class _ReadingTime:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str | None

    def __init__(
        self,
        *,
        name: str,
        stages: frozenset[Stage],
        fingerprint: str,
        max_words: int,
        mode: str,
    ) -> None:
        self.name = name
        self.labels = frozenset({"reading_time"})
        self.stages = stages
        self.phase = "limit"
        self.fingerprint = fingerprint
        self.max_words = max_words
        self.mode = mode

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        _deadline(call)
        matches = tuple(_WORDS.finditer(text))
        if len(matches) <= self.max_words:
            return TransformOutcome(text=text)
        if self.mode == "block":
            return TransformOutcome(
                text=text,
                findings=(TransformFinding("reading_time", "block", (Span(0, len(text)),)),),
            )
        end = matches[self.max_words - 1].end()
        edit = TransformEdit(end, len(text), "")
        shortened = text[:end]
        return TransformOutcome(
            text=shortened,
            findings=(TransformFinding("reading_time", "redact", (Span(end, len(text)),)),),
            edits=(edit,),
        )


def reading_time(
    max_minutes: float,
    *,
    mode: str = "block",
    name: str = "reading_time",
) -> _ReadingTime:
    """Block or truncate past 200 words per minute."""

    if isinstance(max_minutes, bool) or max_minutes <= 0:
        raise PolicyError("reading_time requires a positive max_minutes")
    if mode not in {"block", "truncate"}:
        raise PolicyError("reading_time mode must be block or truncate")
    max_words = int(max_minutes * _WORDS_PER_MINUTE)
    if max_words < 1:
        raise PolicyError("reading_time allows fewer than one word")
    validate_identifier(name, field="policy name")
    stages: frozenset[Stage] = frozenset(("input", "output"))
    return _ReadingTime(
        name=name,
        stages=stages,
        fingerprint=_fingerprint("reading_time", name, mode, str(max_words)),
        max_words=max_words,
        mode=mode,
    )


def _match_json(text: str, start: int) -> int | None:
    pairs = {"{": "}", "[": "]"}
    if text[start] not in pairs:
        return None
    stack = [text[start]]
    in_string = False
    escape = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(char)
        elif char in "}]":
            if not stack or pairs[stack[-1]] != char:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def _first_json(text: str) -> tuple[int, int, object] | None:
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        end = _match_json(text, index)
        if end is None:
            continue
        snippet = text[index:end]
        try:
            return index, end, json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def _element_count(value: object) -> int:
    if isinstance(value, dict):
        return len(cast(dict[str, object], value))
    if isinstance(value, list):
        return len(cast(list[object], value))
    return 0


class _JsonCheck:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str | None

    def __init__(
        self,
        *,
        name: str,
        fingerprint: str,
        required_elements: int,
        repair: bool,
    ) -> None:
        self.name = name
        self.labels = frozenset({"json_check"})
        self.stages = frozenset({"output"})
        self.phase = "detect"
        self.fingerprint = fingerprint
        self.required_elements = required_elements
        self.repair = repair

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        _deadline(call)
        found = _first_json(text)
        if found is None and self.repair:
            found = _repair(text)
        if found is None:
            return TransformOutcome(
                text=text,
                findings=(TransformFinding("json_check", "block", (Span(0, len(text)),)),),
            )
        start, end, value = found
        if _element_count(value) < self.required_elements:
            return TransformOutcome(
                text=text,
                findings=(TransformFinding("json_check", "block", (Span(start, end),)),),
            )
        if not self.repair:
            return TransformOutcome(text=text)
        repaired = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if text[start:end] == repaired:
            return TransformOutcome(text=text)
        edit = TransformEdit(start, end, repaired)
        return TransformOutcome(
            text=text[:start] + repaired + text[end:],
            findings=(TransformFinding("json_check", "redact", (Span(start, end),)),),
            edits=(edit,),
        )


def _load_json_repair() -> Any:
    if importlib.util.find_spec("json_repair") is None:
        raise PolicyError("json_check(repair=True) requires the jes[json] extra")
    return importlib.import_module("json_repair")


def _repair(text: str) -> tuple[int, int, object] | None:
    module = _load_json_repair()
    start = next((index for index, char in enumerate(text) if char in "{["), None)
    if start is None:
        return None
    try:
        value: object = module.loads(text[start:])
    except Exception:
        return None
    if not isinstance(value, (dict, list)):
        return None
    return start, len(text), cast(object, value)


def json_check(
    required_elements: int = 0,
    repair: bool = False,
    *,
    name: str = "json_check",
) -> _JsonCheck:
    """Find one JSON value and block when it is missing or too small."""

    if isinstance(required_elements, bool) or required_elements < 0:
        raise PolicyError("required_elements must be a non-negative integer")
    if repair:
        _load_json_repair()
    validate_identifier(name, field="policy name")
    return _JsonCheck(
        name=name,
        fingerprint=_fingerprint("json_check", name, str(required_elements), str(repair)),
        required_elements=required_elements,
        repair=repair,
    )
