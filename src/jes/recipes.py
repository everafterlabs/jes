"""Ready-made policies beyond the core set. None has an evaluated threshold: tune your own."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from jes.backend import ModelSpec
from jes.errors import PolicyError
from jes.policies.base import (
    Item,
    Judgment,
    Phase,
    TransformContext,
    TransformFinding,
    TransformOutcome,
    freeze_stages,
    judge,
)
from jes.policies.transforms import Substrings, substrings
from jes.questions import Choice, Question, Threshold, YesNo
from jes.text.textmap import Edit
from jes.types import Span, Stage, validate_identifier

EMOTIONS: tuple[str, ...] = (
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
)
CODE_LANGUAGES: tuple[str, ...] = ("python", "javascript", "sql", "shell", "html", "other")
REFUSAL_PHRASES: tuple[str, ...] = (
    "I cannot help with that",
    "I can't assist with that",
    "I will not provide that",
)

_URL = re.compile(r"https?://[^\s<>\"']+")
_WORD = re.compile(r"\S+")
_WORDS_PER_MINUTE = 200
# json_check tries at most this many "{" or "[" positions before it gives up.
_MAX_JSON_STARTS = 100


def _yes_no(
    name: str,
    instructions: str,
    *,
    threshold: Threshold | float,
    model: ModelSpec | None,
    stages: tuple[Stage, ...] = ("input", "output"),
    required_context: bool = False,
    sources: bool = False,
) -> Judgment:
    return judge(
        name,
        YesNo(instructions),
        threshold=threshold,
        stages=("output",) if required_context else stages,
        context="required" if required_context else "none",
        sources=sources,
        whole_text=required_context,
        model=model,
    )


def sentiment(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """Hostile or strongly negative text."""

    return _yes_no(
        "sentiment", "The text is hostile or strongly negative.", threshold=threshold, model=model
    )


def gibberish(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """Text that is not meaningful language."""

    return _yes_no(
        "gibberish", "The text is not meaningful language.", threshold=threshold, model=model
    )


def bias(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """A reply that demeans or stereotypes a group."""

    return _yes_no(
        "bias",
        "The text demeans or stereotypes a group.",
        threshold=threshold,
        model=model,
        stages=("output",),
    )


def refusal(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """A reply that refuses the request."""

    return _yes_no(
        "refusal",
        "The text refuses the request.",
        threshold=threshold,
        model=model,
        stages=("output",),
    )


def relevance(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """A reply that does not address the prompt. Needs ``prompt=``."""

    return _yes_no(
        "relevance",
        "The text does not address the prompt.",
        threshold=threshold,
        model=model,
        required_context=True,
    )


def factual_consistency(
    *, threshold: Threshold | float, model: ModelSpec | None = None
) -> Judgment:
    """Claims in a reply that the sources, or else the prompt, contradict or do not support."""

    return _yes_no(
        "factual_consistency",
        "The text makes claims that the sources, or the prompt when no sources are passed, "
        "contradict or do not support.",
        threshold=threshold,
        model=model,
        required_context=True,
        sources=True,
    )


def language_same(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """A reply in a different language than the prompt. Needs ``prompt=``."""

    return _yes_no(
        "language_same",
        "The text is in a different language than the prompt.",
        threshold=threshold,
        model=model,
        required_context=True,
    )


def emotions(
    blocked: Iterable[str] = EMOTIONS,
    *,
    threshold: Threshold | float,
    model: ModelSpec | None = None,
) -> Judgment:
    """Text that expresses any of the ``blocked`` emotions. A finding names the emotion."""

    selected = tuple(blocked)
    if not selected:
        raise PolicyError("emotions needs at least one emotion")
    questions: dict[str, Question] = {
        validate_identifier(emotion, field="emotion"): YesNo(f"The text expresses {emotion}.")
        for emotion in selected
    }
    return judge("emotions", questions, threshold=threshold, model=model)


def language(
    allowed: Iterable[str],
    *,
    threshold: Threshold | float,
    model: ModelSpec | None = None,
) -> Judgment:
    """Text in a language outside ``allowed``, given as codes such as ``"en"``."""

    codes = tuple(allowed)
    if not codes or "other" in codes or len(set(codes)) != len(codes):
        raise PolicyError("language needs unique codes, not including other")
    options: dict[str, str | None] = {code: code for code in codes}
    options["other"] = "another language"
    return judge(
        "language",
        Choice("Which language is the text written in?", options),
        threshold=threshold,
        violating=("other",),
        model=model,
    )


def code(
    mode: Literal["ban", "allow"],
    languages: Iterable[str],
    *,
    threshold: Threshold | float,
    model: ModelSpec | None = None,
) -> Judgment:
    """Code in a banned language, or with ``mode="allow"``, in any language not listed."""

    if mode not in ("ban", "allow"):
        raise PolicyError("code mode must be ban or allow")
    selected = tuple(languages)
    if not selected or len(set(selected)) != len(selected):
        raise PolicyError("code needs unique languages")
    unknown = [item for item in selected if item not in CODE_LANGUAGES]
    if unknown:
        raise PolicyError(f"unknown code language: {unknown[0]}")
    violating = selected if mode == "ban" else tuple(set(CODE_LANGUAGES) - set(selected))
    if not violating:
        raise PolicyError("allowing every language leaves nothing to block")
    options: dict[str, str | None] = {item: None for item in CODE_LANGUAGES}
    options["not_code"] = "The text is not code."
    return judge(
        "code",
        Choice("Which programming language is the text, if it is code?", options),
        threshold=threshold,
        violating=violating,
        model=model,
    )


def _urls(text: str) -> Iterator[Item]:
    for match in _URL.finditer(text):
        yield Item(match.group(), Span(match.start(), match.end()))


def malicious_urls(
    *,
    threshold: Threshold | float,
    max_urls: int = 20,
    model: ModelSpec | None = None,
) -> Judgment:
    """Judge each http or https URL on its own. More than ``max_urls`` blocks the check."""

    return judge(
        "malicious_urls",
        YesNo("The URL string itself is malicious."),
        threshold=threshold,
        stages=("input", "untrusted", "tool_result", "output"),
        items=_urls,
        max_items=max_urls,
        model=model,
    )


def competitors(names: Iterable[str]) -> Substrings:
    """Redact competitor names, including lookalike spellings."""

    values = tuple(names)
    if not values:
        raise PolicyError("competitors needs at least one name")
    return substrings(values, action="redact", name="competitors")


def refusal_phrases(phrases: Iterable[str] = REFUSAL_PHRASES) -> Substrings:
    """Block replies that contain a fixed refusal phrase."""

    return substrings(phrases, action="block", name="refusal_phrases", stages=("output",))


@dataclass(frozen=True, slots=True)
class ReadingTime:
    name: str
    stages: frozenset[Stage]
    max_words: int
    mode: Literal["block", "truncate"]
    phase: Phase = "limit"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        context.check_deadline()
        words = list(_WORD.finditer(text))
        if len(words) <= self.max_words:
            return TransformOutcome()
        if self.mode == "block":
            finding = TransformFinding("reading_time", "block", (Span(0, len(text)),))
            return TransformOutcome(findings=(finding,))
        end = words[self.max_words - 1].end()
        finding = TransformFinding("reading_time", "redact", (Span(end, len(text)),))
        return TransformOutcome(edits=(Edit(end, len(text), ""),), findings=(finding,))


def reading_time(
    max_minutes: float,
    *,
    mode: Literal["block", "truncate"] = "block",
    name: str = "reading_time",
) -> ReadingTime:
    """Block, or cut off, text that takes longer than ``max_minutes`` to read at 200 wpm."""

    if isinstance(max_minutes, bool) or not max_minutes > 0:
        raise PolicyError("reading_time needs a positive max_minutes")
    if mode not in ("block", "truncate"):
        raise PolicyError("reading_time mode must be block or truncate")
    max_words = int(max_minutes * _WORDS_PER_MINUTE)
    if max_words < 1:
        raise PolicyError("reading_time allows fewer than one word")
    return ReadingTime(
        name=validate_identifier(name, field="policy name"),
        stages=freeze_stages(("input", "output")),
        max_words=max_words,
        mode=mode,
    )


@dataclass(frozen=True, slots=True)
class JsonCheck:
    name: str
    required_elements: int
    repair: Any
    stages: frozenset[Stage] = frozenset({"output"})
    phase: Phase = "detect"

    def apply(self, text: str, context: TransformContext) -> TransformOutcome:
        context.check_deadline()
        if self.repair is None:
            found = _first_json(text, _MAX_JSON_STARTS)
        else:
            # Repair starts at the first bracket, so a broken object is never skipped in
            # favour of a valid array nested inside it.
            found = _first_json(text, 1) or _repaired(text, self.repair)
        if found is None:
            return _blocked(Span(0, len(text)))
        start, end, value = found
        if _size(value) < self.required_elements:
            return _blocked(Span(start, end))
        if self.repair is None:
            return TransformOutcome()
        canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if text[start:end] == canonical:
            return TransformOutcome()
        finding = TransformFinding("json_check", "redact", (Span(start, end),))
        return TransformOutcome(edits=(Edit(start, end, canonical),), findings=(finding,))


def _blocked(span: Span) -> TransformOutcome:
    return TransformOutcome(findings=(TransformFinding("json_check", "block", (span,)),))


def _first_json(text: str, attempts: int) -> tuple[int, int, object] | None:
    """The first JSON object or array, trying at most ``attempts`` bracket positions."""

    decoder = json.JSONDecoder()
    starts = (match.start() for match in re.finditer(r"[{\[]", text))
    for _attempt, start in zip(range(attempts), starts, strict=False):
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            continue
        return start, end, value
    return None


def _repaired(text: str, module: Any) -> tuple[int, int, object] | None:
    match = re.search(r"[{\[]", text)
    if match is None:
        return None
    try:
        value: object = module.loads(text[match.start() :])
    except Exception:
        return None
    if not isinstance(value, (dict, list)):
        return None
    return match.start(), len(text), cast(object, value)


def _size(value: object) -> int:
    return len(value) if isinstance(value, (dict, list)) else 0  # pyright: ignore[reportUnknownArgumentType]


def json_check(
    required_elements: int = 0,
    *,
    repair: bool = False,
    name: str = "json_check",
) -> JsonCheck:
    """Block a reply without a JSON object or array of at least ``required_elements`` items.

    ``repair=True`` (the jes[json] extra) fixes broken JSON and rewrites it in place,
    replacing everything from the first bracket to the end of the reply.
    """

    if isinstance(required_elements, bool) or required_elements < 0:
        raise PolicyError("required_elements must be zero or more")
    module = None
    if repair:
        try:
            module = importlib.import_module("json_repair")
        except ImportError:
            raise PolicyError("json_check(repair=True) requires the jes[json] extra") from None
    return JsonCheck(
        name=validate_identifier(name, field="policy name"),
        required_elements=required_elements,
        repair=module,
    )


__all__ = [
    "CODE_LANGUAGES",
    "EMOTIONS",
    "REFUSAL_PHRASES",
    "JsonCheck",
    "ReadingTime",
    "bias",
    "code",
    "competitors",
    "emotions",
    "factual_consistency",
    "gibberish",
    "json_check",
    "language",
    "language_same",
    "malicious_urls",
    "reading_time",
    "refusal",
    "refusal_phrases",
    "relevance",
    "sentiment",
]
