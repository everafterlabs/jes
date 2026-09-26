"""PII span metrics. No judgment threshold is produced."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import cast

from jes import Guard
from jes.policies import pii
from jes.policies._protocols import Policy
from jes.types import ScanResult


@dataclass(frozen=True, slots=True)
class GoldSpan:
    text: str
    entity: str | None
    start: int | None
    end: int | None
    evasion: bool = False


@dataclass(frozen=True, slots=True)
class PiiReport:
    ner: str
    available: bool
    exact_recall: float | None
    overlap_recall: float | None
    evasion_recall: float | None
    support: int


def _spans(result: ScanResult) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for finding in result.findings:
        for location in finding.locations:
            found.append((location.span.start, location.span.end, finding.label))
    return found


def _score(cases: tuple[GoldSpan, ...], ner: str) -> PiiReport:
    try:
        policy = pii(entities=("EMAIL_ADDRESS",), ner=ner, restore=False)
    except Exception:
        return PiiReport(ner, False, None, None, None, len(cases))
    guard = Guard([cast(Policy, policy)])
    exact = 0
    overlap = 0
    positives = 0
    evasion_hits = 0
    evasions = 0
    for case in cases:
        result = guard.check_input(case.text)
        detected = _spans(result)
        if case.entity is None:
            continue
        positives += 1
        if case.evasion:
            evasions += 1
        exact_hit = any(
            label == case.entity and start == case.start and end == case.end
            for start, end, label in detected
        )
        overlap_hit = any(
            label == case.entity
            and case.start is not None
            and case.end is not None
            and start < case.end
            and end > case.start
            for start, end, label in detected
        )
        hidden = case.text.lower() not in result.sanitized.lower()
        if not detected and hidden:
            overlap_hit = True
        if exact_hit:
            exact += 1
        if overlap_hit:
            overlap += 1
            if case.evasion:
                evasion_hits += 1
    return PiiReport(
        ner=ner,
        available=True,
        exact_recall=None if positives == 0 else exact / positives,
        overlap_recall=None if positives == 0 else overlap / positives,
        evasion_recall=None if evasions == 0 else evasion_hits / evasions,
        support=positives,
    )


def smoke_cases() -> tuple[GoldSpan, ...]:
    email = "ada@example.com"
    prefix = "mail "
    evasive = "ada\u200b@example.com"
    return (
        GoldSpan(
            f"{prefix}{email}",
            "EMAIL_ADDRESS",
            len(prefix),
            len(prefix) + len(email),
            False,
        ),
        GoldSpan(
            f"{prefix}{evasive}",
            "EMAIL_ADDRESS",
            len(prefix),
            len(prefix) + len(evasive),
            True,
        ),
        GoldSpan("no address here", None, None, None, False),
    )


def evaluate_pii() -> tuple[PiiReport, PiiReport]:
    cases = smoke_cases()
    spacy = _score(cases, "spacy")
    if importlib.util.find_spec("transformers") is None:
        transformer = PiiReport("transformer", False, None, None, None, len(cases))
    else:
        transformer = _score(cases, "dslim/bert-base-NER")
    return spacy, transformer


__all__ = ["GoldSpan", "PiiReport", "evaluate_pii", "smoke_cases"]
