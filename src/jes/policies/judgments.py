"""The built-in judgments. Each is one ``judge`` call with frozen questions."""

from __future__ import annotations

import re
from collections.abc import Iterable

from jes.backend import ModelSpec
from jes.errors import PolicyError
from jes.policies.base import Judgment, judge
from jes.policies.prompts import (
    HAZARD_CODES,
    INDIRECT_INJECTION_V1,
    INJECTION_V1,
    TOOL_SAFETY_V1,
    TOXICITY_LABELS,
    hazard_instruction,
    topic_instruction,
    toxicity_instruction,
)
from jes.questions import Question, Threshold, YesNo

_SLUG = re.compile(r"[^A-Za-z0-9_-]+")


def injection(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """Text that tries to override or extract an assistant's instructions.

    Runs on user input, retrieved text, and tool call arguments.
    """

    return judge(
        "injection",
        YesNo(INJECTION_V1),
        threshold=threshold,
        stages=("input", "untrusted", "tool_call"),
        model=model,
    )


def indirect_injection(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """Instructions aimed at the assistant inside retrieved text or a tool result."""

    return judge(
        "indirect_injection",
        YesNo(INDIRECT_INJECTION_V1),
        threshold=threshold,
        stages=("untrusted", "tool_result"),
        context="optional",
        model=model,
    )


def tool_safety(*, threshold: Threshold | float, model: ModelSpec | None = None) -> Judgment:
    """A tool call that is destructive, exfiltrates data, or goes beyond the request."""

    return judge(
        "tool_safety",
        YesNo(TOOL_SAFETY_V1),
        threshold=threshold,
        stages=("tool_call",),
        context="optional",
        model=model,
    )


def hazards(
    categories: Iterable[str] | None = None,
    *,
    threshold: Threshold | float,
    model: ModelSpec | None = None,
) -> Judgment:
    """Requests for or help with the hazards S1 to S14, or only the ``categories`` given.

    One question per category. A finding names each category at or above the threshold.
    """

    codes = tuple(HAZARD_CODES if categories is None else categories)
    unknown = [code for code in codes if code not in HAZARD_CODES]
    if not codes or unknown:
        raise PolicyError(f"unknown hazard category: {unknown[0] if unknown else 'none given'}")
    questions: dict[str, Question] = {code: YesNo(hazard_instruction(code)) for code in codes}
    return judge(
        "hazards",
        questions,
        threshold=threshold,
        stages=("input", "output"),
        context="optional",
        model=model,
    )


def toxicity(
    labels: Iterable[str] | None = None,
    *,
    threshold: Threshold | float,
    model: ModelSpec | None = None,
) -> Judgment:
    """Toxic, obscene, threatening, insulting, or hateful text, or only the ``labels`` given."""

    selected = tuple(TOXICITY_LABELS if labels is None else labels)
    unknown = [label for label in selected if label not in TOXICITY_LABELS]
    if not selected or unknown:
        raise PolicyError(f"unknown toxicity label: {unknown[0] if unknown else 'none given'}")
    questions: dict[str, Question] = {
        label: YesNo(toxicity_instruction(label)) for label in selected
    }
    return judge(
        "toxicity", questions, threshold=threshold, stages=("input", "output"), model=model
    )


def topics(
    deny: Iterable[str],
    *,
    threshold: Threshold | float,
    model: ModelSpec | None = None,
) -> Judgment:
    """Text about any topic in ``deny``. A finding is named after the topic."""

    names = tuple(deny)
    if not names or not all(name.strip() for name in names):
        raise PolicyError("topics needs at least one non-empty topic")
    questions: dict[str, Question] = {}
    for topic in names:
        questions[_topic_id(topic, questions)] = YesNo(topic_instruction(topic))
    return judge("topics", questions, threshold=threshold, stages=("input", "output"), model=model)


def _topic_id(topic: str, taken: dict[str, Question]) -> str:
    """A readable question id for a topic, such as ``crypto_trading``."""

    slug = _SLUG.sub("_", topic.strip()).strip("_-")[:50]
    if not slug or not slug[0].isalpha():
        slug = f"topic_{slug}" if slug else "topic"
    candidate, index = slug, 2
    while candidate in taken:
        candidate, index = f"{slug}_{index}", index + 1
    return candidate


__all__ = ["hazards", "indirect_injection", "injection", "tool_safety", "topics", "toxicity"]
