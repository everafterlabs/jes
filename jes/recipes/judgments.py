"""Unevaluated judgment recipes built with ``judge``."""

from __future__ import annotations

import re
from collections.abc import Iterable

from jes.backends import Backend
from jes.errors import PolicyError
from jes.policies import Item, judge
from jes.policies._protocols import ContextMode
from jes.policies.judgments import _JudgePolicy
from jes.questions import Choice, Threshold, YesNo, validate_identifier
from jes.recipes.prompts import (
    BIAS_V1,
    CODE_LANGUAGES_V1,
    CODE_V1,
    EMOTION_V1,
    EMOTIONS_V1,
    FACTUAL_CONSISTENCY_V1,
    GIBBERISH_V1,
    LANGUAGE_SAME_V1,
    LANGUAGE_V1,
    MALICIOUS_URL_V1,
    REFUSAL_V1,
    RELEVANCE_V1,
    SENTIMENT_V1,
)
from jes.types import Span, Stage

_URL = re.compile(r"https?://[^\s<>\"']+")
_MAX_URLS = 20


def _version(version: str, canonical: str) -> None:
    if version not in {"v1", canonical}:
        raise PolicyError(f"unknown recipe version {version}")


def _yesno(
    name: str,
    instructions: str,
    *,
    threshold: float | Threshold,
    version: str,
    stages: tuple[Stage, ...] = ("input", "output"),
    context: ContextMode = "none",
    sources: bool = False,
    whole_text: bool = False,
    backend: Backend | None = None,
) -> _JudgePolicy:
    return judge(
        name,
        YesNo(instructions, task="custom"),
        threshold=threshold,
        stages=stages,
        context=context,
        sources=sources,
        whole_text=whole_text,
        backend=backend,
        version=version,
    )


def sentiment(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no: the text is hostile or strongly negative."""

    _version(version, "sentiment.v1")
    return _yesno(
        "sentiment",
        SENTIMENT_V1,
        threshold=threshold,
        version="sentiment.v1",
        backend=backend,
    )


def emotions(
    blocked: Iterable[str] | None = None,
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """One yes/no question per emotion. The default is the v1 negative set."""

    _version(version, "emotions.v1")
    selected = tuple(EMOTIONS_V1 if blocked is None else blocked)
    if not selected or any(not item.strip() for item in selected):
        raise PolicyError("emotions requires at least one emotion")
    questions = {
        emotion: YesNo(EMOTION_V1.format(emotion=emotion), task="custom") for emotion in selected
    }
    for emotion in questions:
        validate_identifier(emotion, field="emotion")
    return judge(
        "emotions",
        questions,
        threshold=threshold,
        backend=backend,
        version="emotions.v1",
    )


def gibberish(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no: the text is not meaningful language."""

    _version(version, "gibberish.v1")
    return _yesno(
        "gibberish",
        GIBBERISH_V1,
        threshold=threshold,
        version="gibberish.v1",
        backend=backend,
    )


def bias(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no, output only: the text demeans or stereotypes a group."""

    _version(version, "bias.v1")
    return _yesno(
        "bias",
        BIAS_V1,
        threshold=threshold,
        version="bias.v1",
        stages=("output",),
        backend=backend,
    )


def refusal(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no, output only: the text refuses the request."""

    _version(version, "refusal.v1")
    return _yesno(
        "refusal",
        REFUSAL_V1,
        threshold=threshold,
        version="refusal.v1",
        stages=("output",),
        backend=backend,
    )


def language(
    allowed: Iterable[str],
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Choice over allowed language codes plus ``other``. ``other`` violates."""

    _version(version, "language.v1")
    codes = tuple(allowed)
    if not codes:
        raise PolicyError("language requires at least one allowed code")
    if "other" in codes or len(set(codes)) != len(codes):
        raise PolicyError("language codes must be unique and must not include other")
    for code in codes:
        validate_identifier(code, field="language code")
    options = {code: code for code in codes}
    options["other"] = "another language"
    return judge(
        "language",
        Choice(LANGUAGE_V1, options, task="custom"),
        threshold=threshold,
        violating=("other",),
        backend=backend,
        version="language.v1",
    )


def language_same(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no, output, required prompt: the reply is in a different language."""

    _version(version, "language_same.v1")
    return _yesno(
        "language_same",
        LANGUAGE_SAME_V1,
        threshold=threshold,
        version="language_same.v1",
        stages=("output",),
        context="required",
        whole_text=True,
        backend=backend,
    )


def code(
    mode: str,
    languages: Iterable[str],
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Choice over a fixed language set plus ``not_code``.

    ``mode="ban"`` treats the listed languages as violations.
    ``mode="allow"`` treats every other programming language as a violation.
    """

    _version(version, "code.v1")
    if mode not in {"ban", "allow"}:
        raise PolicyError("code mode must be ban or allow")
    selected = tuple(languages)
    if not selected:
        raise PolicyError("code requires at least one language")
    unknown = [item for item in selected if item not in CODE_LANGUAGES_V1 or item == "not_code"]
    if unknown or len(set(selected)) != len(selected):
        raise PolicyError("code languages must be unique known languages")
    if mode == "ban":
        violating = selected
    else:
        violating = tuple(
            item for item in CODE_LANGUAGES_V1 if item not in selected and item != "not_code"
        )
    options = {item: item for item in CODE_LANGUAGES_V1}
    return judge(
        "code",
        Choice(CODE_V1, options, task="custom"),
        threshold=threshold,
        violating=violating,
        backend=backend,
        version="code.v1",
    )


def _url_items(text: str) -> Iterable[Item]:
    for match in _URL.finditer(text):
        yield Item(match.group(), Span(match.start(), match.end()))


def malicious_urls(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Judge each absolute http(s) URL string. More than 20 blocks before any call."""

    _version(version, "malicious_urls.v1")
    return judge(
        "malicious_urls",
        YesNo(MALICIOUS_URL_V1, task="custom"),
        threshold=threshold,
        items=_url_items,
        max_policy_items=_MAX_URLS,
        item_overflow_label="too_many_urls",
        backend=backend,
        version="malicious_urls.v1",
    )


def relevance(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no, whole output, required prompt: the reply does not address the prompt."""

    _version(version, "relevance.v1")
    return _yesno(
        "relevance",
        RELEVANCE_V1,
        threshold=threshold,
        version="relevance.v1",
        stages=("output",),
        context="required",
        whole_text=True,
        backend=backend,
    )


def factual_consistency(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    backend: Backend | None = None,
) -> _JudgePolicy:
    """Yes/no, whole output, required prompt and sources when they are passed."""

    _version(version, "factual_consistency.v1")
    return _yesno(
        "factual_consistency",
        FACTUAL_CONSISTENCY_V1,
        threshold=threshold,
        version="factual_consistency.v1",
        stages=("output",),
        context="required",
        sources=True,
        whole_text=True,
        backend=backend,
    )
