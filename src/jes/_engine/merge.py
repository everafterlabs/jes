"""Deterministic result merging."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from jes.errors import BackendError
from jes.types import Finding, FindingLocation, ScoreResult

_ACTION_RANK = {"flag": 0, "redact": 1, "block": 2}


def _location_key(location: FindingLocation) -> tuple[object, ...]:
    return (
        location.target,
        -1 if location.index is None else location.index,
        location.role or "",
        -1 if location.item_ordinal is None else location.item_ordinal,
        location.span.start,
        location.span.end,
    )


def merge_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    merged: dict[tuple[str, str | None, str], Finding] = {}
    order: list[tuple[str, str | None, str]] = []
    for finding in findings:
        key = (finding.policy, finding.question, finding.label)
        previous = merged.get(key)
        if previous is None:
            merged[key] = finding
            order.append(key)
            continue

        action = (
            finding.action
            if _ACTION_RANK[finding.action] > _ACTION_RANK[previous.action]
            else previous.action
        )
        score = previous.score
        if score is None or (finding.score is not None and finding.score.value > score.value):
            score = finding.score
        locations = tuple(
            sorted(
                set((*previous.locations, *finding.locations)),
                key=_location_key,
            )
        )
        chunks = tuple(sorted(set((*previous.chunks, *finding.chunks))))
        merged[key] = Finding(
            policy=finding.policy,
            label=finding.label,
            action=action,
            question=finding.question,
            score=score,
            locations=locations,
            chunks=chunks,
        )
    return tuple(merged[key] for key in order)


def merge_scores(
    score_maps: Iterable[Mapping[str, ScoreResult]],
) -> Mapping[str, ScoreResult]:
    merged: dict[str, ScoreResult] = {}
    for scores in score_maps:
        for question_id, score in scores.items():
            previous = merged.get(question_id)
            if previous is not None:
                if (
                    previous.kind != score.kind
                    or previous.provenance.decision_profile != score.provenance.decision_profile
                ):
                    raise BackendError(
                        score.provenance.backend,
                        "incompatible chunk score provenance",
                        question_ids=(question_id,),
                    )
                if previous.value >= score.value:
                    continue
            merged[question_id] = score
    return merged
