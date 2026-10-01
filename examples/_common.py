"""Printing and .env loading shared by the lessons. No jes logic lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from jes import Result
from jes.agents.env import apply_env_file

# The lessons read keys from the .env at the root of this checkout.
_DOTENV = Path(__file__).resolve().parents[1] / ".env"


def require_env(*names: str) -> None:
    """Load ``.env`` and exit with a clear message if a key is missing."""

    if _DOTENV.is_file():
        apply_env_file(_DOTENV, os.environ)
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not set. Add it to the environment or .env.")


class Check(NamedTuple):
    label: str
    result: Result


@dataclass
class Run:
    """What happened in one agent scenario."""

    checks: list[Check] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    reply: str = ""


def print_check(label: str, result: Result) -> None:
    """The decision and top score, plus the onward text if jes blocked or flagged it."""

    top = max(result.scores.items(), key=lambda item: item[1].value, default=None)
    score = f" {top[0]}={top[1].value:.2f}" if top else ""
    print(f"  {label}: {result.decision}{score}")
    if not result.ok or result.findings:
        print(f"    onward: {result.onward}")


def print_run(name: str, run: Run) -> None:
    print(f"== {name}")
    for label, result in run.checks:
        print_check(label, result)
    print(f"  reply: {run.reply}")
