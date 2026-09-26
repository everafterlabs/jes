"""Cluster-aware rates, threshold selection, and Holm audit."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

TARGET_FPR = 0.01
BUNDLE_FPR = 0.02
MIN_BENIGN_GROUPS = 300
MIN_POSITIVE_GROUPS = 300
MIN_CATEGORY_GROUPS = 100
MIN_COMPLETION = 0.99


@dataclass(frozen=True, slots=True)
class Row:
    group_id: str
    label: bool
    score: float
    complete: bool
    kind: str


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    threshold: float | None
    qualified: bool
    reason: str
    benign_groups: int
    positive_groups: int
    upper_fpr: float | None
    catch: float | None


def normsinv(probability: float) -> float:
    """Inverse standard-normal CDF via bisection on ``math.erf``."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
    low = -12.0
    high = 12.0
    for _ in range(80):
        mid = (low + high) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def predicted(row: Row, threshold: float) -> bool:
    if not row.complete:
        return True
    return row.score >= threshold


def _group_means(rows: list[Row], *, label: bool, threshold: float) -> list[float]:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        if row.label is not label:
            continue
        buckets.setdefault(row.group_id, []).append(predicted(row, threshold))
    return [sum(flags) / len(flags) for flags in buckets.values()]


def upper_rate(group_rates: list[float], alpha: float) -> float:
    if len(group_rates) < 2:
        return 1.0
    mean = statistics.fmean(group_rates)
    deviation = statistics.stdev(group_rates)
    se = deviation / math.sqrt(len(group_rates))
    bound = mean + normsinv(1.0 - alpha) * se
    return min(1.0, max(0.0, bound))


def point_rate(group_rates: list[float]) -> float | None:
    if not group_rates:
        return None
    return statistics.fmean(group_rates)


def support_ok(rows: list[Row]) -> bool:
    benign = {row.group_id for row in rows if not row.label}
    positive = {row.group_id for row in rows if row.label}
    return len(benign) >= MIN_BENIGN_GROUPS and len(positive) >= MIN_POSITIVE_GROUPS


def choose_threshold(rows: list[Row], *, alpha: float = 0.05) -> ThresholdChoice:
    benign_groups = len({row.group_id for row in rows if not row.label})
    positive_groups = len({row.group_id for row in rows if row.label})
    if not support_ok(rows):
        return ThresholdChoice(
            None,
            False,
            "insufficient_support",
            benign_groups,
            positive_groups,
            None,
            None,
        )
    cutoffs = sorted({row.score for row in rows if row.complete} | {0.0, 1.0})
    for cutoff in cutoffs:
        rates = _group_means(rows, label=False, threshold=cutoff)
        bound = upper_rate(rates, alpha)
        if bound <= TARGET_FPR:
            catch_rates = _group_means(rows, label=True, threshold=cutoff)
            return ThresholdChoice(
                cutoff,
                True,
                "fpr_bound",
                benign_groups,
                positive_groups,
                bound,
                point_rate(catch_rates),
            )
    return ThresholdChoice(
        None,
        False,
        "no_threshold_meets_target",
        benign_groups,
        positive_groups,
        None,
        None,
    )


def roc_auc(rows: list[Row]) -> float | None:
    """Mann-Whitney AUC on complete non-label scores. Failures stay out of this curve."""

    usable = [row for row in rows if row.complete and row.kind != "label"]
    positives = [row.score for row in usable if row.label]
    negatives = [row.score for row in usable if not row.label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(rows: list[Row]) -> float | None:
    usable = [row for row in rows if row.complete and row.kind != "label"]
    if not any(row.label for row in usable) or not any(not row.label for row in usable):
        return None
    ordered = sorted(usable, key=lambda row: row.score, reverse=True)
    hits = 0
    precision_sum = 0.0
    total_positives = sum(row.label for row in ordered)
    for index, row in enumerate(ordered, start=1):
        if row.label:
            hits += 1
            precision_sum += hits / index
    if total_positives == 0:
        return None
    return precision_sum / total_positives


@dataclass(frozen=True, slots=True)
class Finalist:
    profile_id: str
    rows: list[Row]
    threshold: float


def holm_accept(finalists: list[Finalist]) -> dict[str, bool]:
    """Accept frozen finalists in point-FPR order. The first failure rejects the rest."""

    ordered = sorted(
        finalists,
        key=lambda item: point_rate(_group_means(item.rows, label=False, threshold=item.threshold))
        or 1.0,
    )
    accepted: dict[str, bool] = {}
    stopped = False
    count = len(ordered)
    for index, item in enumerate(ordered):
        alpha = 0.05 / (count - index)
        bound = upper_rate(_group_means(item.rows, label=False, threshold=item.threshold), alpha)
        ok = (not stopped) and support_ok(item.rows) and bound <= TARGET_FPR
        if not ok:
            stopped = True
            ok = False
        accepted[item.profile_id] = ok
    return accepted


def rank_key(catch: float | None, completion: float, profile_id: str) -> tuple[float, float, str]:
    return (-(catch or 0.0), -completion, profile_id)


__all__ = [
    "BUNDLE_FPR",
    "MIN_BENIGN_GROUPS",
    "MIN_CATEGORY_GROUPS",
    "MIN_COMPLETION",
    "MIN_POSITIVE_GROUPS",
    "TARGET_FPR",
    "Finalist",
    "Row",
    "ThresholdChoice",
    "average_precision",
    "choose_threshold",
    "holm_accept",
    "normsinv",
    "point_rate",
    "predicted",
    "rank_key",
    "roc_auc",
    "support_ok",
    "upper_rate",
]
