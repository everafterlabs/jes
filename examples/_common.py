"""Printing and .env loading shared by the lessons. No jes logic lives here."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

from jes._env import load_project_env
from jes.types import ScanResult


def require_env(*names: str) -> None:
    """Load ``.env`` and exit with a clear message if a key is missing."""

    load_project_env()
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not set. Add it to the environment or .env.")


class Check(NamedTuple):
    """One jes check: where it ran and what it returned."""

    stage: str
    result: ScanResult


@dataclass
class Run:
    """One agent scenario: every check, the tools that really ran, the reply."""

    checks: list[Check] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    reply: str = ""


def show(name: str, checks: Iterable[Check], reply: str | None = None) -> None:
    """Print one scenario: each check's decision, top score, and onward text."""

    print(f"== {name}")
    for stage, result in checks:
        top = max(result.scores.items(), key=lambda item: item[1].value, default=None)
        score = f" {top[0]}={top[1].value:.2f}" if top else ""
        print(f"  {stage}: {result.decision}{score}")
        if not result.ok:
            print(f"    onward: {result.onward}")
    if reply is not None:
        print(f"  reply: {reply}")
