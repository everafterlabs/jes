"""Fake-backend smoke runner. It does not download datasets or write defaults."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from evals.cache import CachingBackend
from evals.datasets import assert_download_allowed
from evals.metrics import (
    BUNDLE_FPR,
    Row,
    average_precision,
    choose_threshold,
    rank_key,
    roc_auc,
)
from evals.pii_metrics import evaluate_pii
from evals.topics import topic_report
from jes import Guard
from jes.backends import Backend, BackendResult, RequestContext
from jes.errors import BackendError
from jes.policies._candidates import _Candidate, injection_candidate
from jes.policies._protocols import Policy
from jes.policies.prompts import verify_prompt_bytes
from jes.questions import Question, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import ScanResult, Stage, State

Phase = Literal["development", "selection", "calibration", "audit", "holdout"]
_ERROR_LABELS = frozenset(
    {
        "backend_error",
        "deadline_exceeded",
        "input_too_long",
        "response_too_large",
        "too_many_requests",
    }
)


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    group_id: str
    text: str
    label: bool
    stage: Stage = "input"


@dataclass(frozen=True, slots=True)
class Observation:
    example_id: str
    group_id: str
    label: bool
    score: float
    kind: str
    complete: bool
    decision: str
    error: str | None
    cached: bool
    latency_ms: float
    request_profile: str
    decision_profile: str


class SmokeBackend(FakeBackend):
    """Maps a smoke marker in the text to a fixed violation score."""

    def __init__(self) -> None:
        super().__init__(max_units=100_000)
        self._score_lock = threading.Lock()

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        if "boom" in state.text:
            raise BackendError(self.name, "connection_error", question_ids=questions)
        score = 0.95 if "positive" in state.text else 0.05
        with self._score_lock:
            self.register_answer("violation", YesNoAnswer(score, "probability"))
            return super().decide(state, questions, request)


def _summary(result: ScanResult, policy_name: str) -> tuple[float, str, str, str]:
    scores = {
        key.split(".", 1)[1]: score
        for key, score in result.scores.items()
        if key.startswith(f"{policy_name}.")
    }
    if not scores:
        return 1.0, "missing", "", ""
    chosen = scores.get("any") or scores.get("violation")
    if chosen is None:
        chosen = max(scores.values(), key=lambda item: item.value)
    profile = next(iter(scores.values()))
    return (
        chosen.value,
        chosen.kind,
        profile.provenance.request_profile,
        profile.provenance.decision_profile,
    )


def _error_label(result: ScanResult) -> str | None:
    for finding in result.findings:
        if finding.policy == "jes" and finding.label in _ERROR_LABELS:
            return finding.label
    if not result.complete:
        return "incomplete"
    return None


def _check(guard: Guard, example: Example) -> ScanResult:
    if example.stage == "untrusted":
        return guard.check_untrusted(example.text)
    if example.stage == "output":
        return guard.check_output(example.text, prompt="synthetic prompt")
    return guard.check_input(example.text)


def run_phase(
    examples: list[Example],
    policy: _Candidate,
    backend: Backend,
    phase: Phase,
    *,
    workers: int = 1,
    cache: CachingBackend | None = None,
    deadline_s: float | None = None,
) -> tuple[list[Observation], CachingBackend]:
    if phase != "development":
        verify_prompt_bytes()
    if cache is None:
        cache = CachingBackend(backend, enabled=phase == "development")
    guard = Guard(
        [cast(Policy, policy)],
        backend=cache,
        on_backend_error="block",
        deadline_s=30.0 if deadline_s is None else deadline_s,
    )

    def once(example: Example) -> Observation:
        hits = cache.hits
        started = time.perf_counter()
        result = _check(guard, example)
        elapsed = (time.perf_counter() - started) * 1_000
        score, kind, request_profile, decision_profile = _summary(result, policy.name)
        return Observation(
            example_id=example.example_id,
            group_id=example.group_id,
            label=example.label,
            score=score,
            kind=kind,
            complete=result.complete,
            decision=result.decision,
            error=_error_label(result),
            cached=cache.hits > hits,
            latency_ms=0.0 if cache.hits > hits else elapsed,
            request_profile=request_profile,
            decision_profile=decision_profile,
        )

    if workers <= 1:
        observations = [once(example) for example in examples]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            observations = list(pool.map(once, examples))
    return observations, cache


def _rows(observations: list[Observation]) -> list[Row]:
    return [
        Row(item.group_id, item.label, item.score, item.complete, item.kind)
        for item in observations
    ]


def phase_report(observations: list[Observation], phase: Phase) -> dict[str, object]:
    rows = _rows(observations)
    choice = choose_threshold(rows)
    completed = [item for item in observations if item.complete]
    uncached = [item for item in observations if not item.cached]
    errors: dict[str, int] = {}
    for item in observations:
        if item.error is None:
            continue
        errors[item.error] = errors.get(item.error, 0) + 1
    label_only = bool(completed) and all(item.kind == "label" for item in completed)
    return {
        "phase": phase,
        "examples": len(observations),
        "benign_groups": choice.benign_groups,
        "positive_groups": choice.positive_groups,
        "qualified": False,
        "reason": (
            choice.reason
            if not choice.qualified
            else "milestone_6_does_not_publish_defaults"
        ),
        "calibrated_threshold": choice.threshold,
        "default_threshold": None,
        "upper_fpr": choice.upper_fpr,
        "catch": choice.catch,
        "completion": (len(completed) / len(observations)) if observations else 0.0,
        "errors": errors,
        "roc_auc": None if label_only else roc_auc(rows),
        "average_precision": None if label_only else average_precision(rows),
        "label_only": label_only,
        "latency_samples": len(uncached),
        "request_profile": observations[0].request_profile if observations else "",
        "decision_profile": observations[0].decision_profile if observations else "",
        "bundle": {
            "qualified": False,
            "reason": "insufficient_support",
            "target_fpr": BUNDLE_FPR,
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Evaluation smoke",
        "",
        "No default threshold is published from this run.",
        "",
        f"Phase: {report.get('phase', '')}",
        f"Qualified: {report.get('qualified')}",
        f"Reason: {report.get('reason', '')}",
        f"Completion: {report.get('completion')}",
        f"ROC AUC: {report.get('roc_auc')}",
    ]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, report: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "report.md").write_text(render_markdown(report), encoding="utf-8")


def smoke_examples() -> list[Example]:
    rows: list[Example] = []
    for index in range(4):
        rows.append(Example(f"benign-{index}", f"benign-{index}", f"benign note {index}", False))
        rows.append(
            Example(f"positive-{index}", f"positive-{index}", f"positive note {index}", True)
        )
    rows.append(Example("boom", "boom", "boom connection", True))
    return rows


def smoke(directory: Path) -> dict[str, object]:
    """Run the plumbing path and withhold every default."""

    verify_prompt_bytes()
    try:
        assert_download_allowed(Path(__file__).with_name("datasets.toml"))
        downloadable = True
    except RuntimeError:
        downloadable = False
    policy = injection_candidate(threshold=0.5)
    backend = SmokeBackend()
    examples = smoke_examples()
    first, cache = run_phase(examples, policy, backend, "development")
    second, _replay = run_phase(examples, policy, backend, "development", cache=cache)
    audit, audit_cache = run_phase(examples, policy, backend, "audit")
    body = phase_report(first, "development")
    spacy, transformer = evaluate_pii()
    body["cache_hits_on_replay"] = cache.hits
    body["audit_cache_hits"] = audit_cache.hits
    body["audit_examples"] = len(audit)
    body["datasets_downloadable"] = downloadable
    body["finalists"] = []
    completion_value = body["completion"]
    completion = completion_value if isinstance(completion_value, float) else 0.0
    body["ranking"] = [
        rank_key(
            body["catch"] if isinstance(body["catch"], float) else None,
            completion,
            "smoke",
        )
    ]
    body["topics"] = topic_report()
    body["pii"] = {
        "spacy_available": spacy.available,
        "spacy_overlap_recall": spacy.overlap_recall,
        "spacy_evasion_recall": spacy.evasion_recall,
        "transformer_available": transformer.available,
        "default_threshold": None,
    }
    body["replay_cached"] = sum(item.cached for item in second)
    write_report(directory, body)
    return body


__all__ = [
    "Example",
    "Observation",
    "SmokeBackend",
    "phase_report",
    "render_markdown",
    "run_phase",
    "smoke",
    "smoke_examples",
    "write_report",
]
