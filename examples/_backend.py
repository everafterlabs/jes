"""Live wiring shared by the lessons: keys, models, and one print format.

Lessons call real services by default. ``--mock`` swaps in ``examples/_mocks.py``.
Keys come from the environment or the project ``.env``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple

from jes._env import load_project_env
from jes.types import ScanResult

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_typesafe import TypeSafeClassifier

    from jes.questions import Answer

# Pin a release such as "jev-1.13.0" in production once thresholds are tuned.
JEV_MODEL = "jev-latest"
# Any init_chat_model id works, e.g. EXAMPLE_CHAT_MODEL=ollama:qwen3:1.7b.
DEFAULT_CHAT_MODEL = "openai:gpt-5.4-mini"
OLLAMA_URL = "http://localhost:11434"
# A local Jev-style decision model served by Ollama. Also: tev1:0.8b, nimble.
TEV_MODEL = "tev1"


def parse_mock(argv: list[str] | None = None) -> bool:
    """True when ``--mock`` is on the command line."""

    return "--mock" in (sys.argv[1:] if argv is None else argv)


def require_env(*names: str) -> None:
    """Load ``.env`` and exit with a clear message if a key is missing."""

    load_project_env()
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not set. Add it to the environment or .env, or run with --mock.",
        )


def decision_model(
    mock: bool,
    scores: Mapping[str, float | Answer] | None = None,
    *,
    when: str | None = None,
    local: bool = False,
) -> Any:
    """The decision model a Guard asks.

    Live: hosted Jev, or ``tev1`` on Ollama with ``local=True``. With ``--mock``:
    fixed ``scores`` such as ``{"injection.violation": 0.95}`` (others score 0),
    applied only to text containing ``when`` if given.
    """

    if mock:
        from examples._mocks import mock_jev

        return mock_jev(scores or {}, when)
    if local:
        return tev()
    require_env("TYPESAFE_API_KEY")
    return JEV_MODEL


@cache
def tev() -> TypeSafeClassifier:
    """The local tev1 decision model on Ollama. No key leaves the machine."""

    from langchain_typesafe import TypeSafeClassifier

    # Ollama ignores the key. Passing one stops the classifier from sending
    # your real TYPESAFE_API_KEY to localhost.
    return TypeSafeClassifier(model=TEV_MODEL, base_url=OLLAMA_URL, api_key="ollama", timeout=120)


@cache
def chat_model() -> BaseChatModel:
    """The LangChain chat model for live runs, created once."""

    from langchain.chat_models import init_chat_model

    load_project_env()
    name = os.environ.get("EXAMPLE_CHAT_MODEL", "").strip() or DEFAULT_CHAT_MODEL
    if name.startswith("openai:"):
        require_env("OPENAI_API_KEY")
    if name.startswith("ollama:"):
        # ChatOllama ignores timeout=; its HTTP client takes it instead.
        return init_chat_model(name, client_kwargs={"timeout": 60})
    return init_chat_model(name, timeout=60, max_retries=2)


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


# Every agent lesson runs these three, with the same text live and mocked.
SCENARIOS = ("clean", "attack", "poisoned_tool")


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
