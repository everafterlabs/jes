"""Run every lesson in examples/ against real services. Not part of CI.

    uv run --frozen --group examples pytest -m live tests/test_examples_live.py --no-cov

A lesson is skipped when a key it needs is missing or Ollama is not running.
EXAMPLE_CHAT_MODEL=ollama:qwen3:1.7b swaps the LangChain chat model.
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from pathlib import Path

import pytest

from jes._env import load_project_env

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[1]
_OLLAMA = "ollama"
_JEV = ("TYPESAFE_API_KEY",)
_CHAT = "OPENAI_API_KEY"  # only when EXAMPLE_CHAT_MODEL is an openai: model
_WEB = ("TAVILY_API_KEY",)
_SECRETS = ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY")
_KEY = re.compile(r"(sk|tvly|ts)[-_][A-Za-z0-9*_-]{8,}|Bearer [A-Za-z0-9._-]{8,}")

# lesson, what it needs, text every live run must print
_CASES: list[tuple[str, tuple[str, ...], str]] = [
    ("01_first_check", _JEV, "Blocked: injection."),
    ("02_model_call", (*_JEV, _CHAT), "Blocked"),
    ("03_tool_calls", (*_JEV, *_WEB), "Tool result blocked."),
    ("04_pii", (_CHAT,), "ada@example.com"),
    ("05_secrets_canary", (_CHAT,), "Blocked: canary."),
    ("06_topics_toxicity", _JEV, "Blocked"),
    ("07_custom_questions", _JEV, "Blocked"),
    ("08_recipes", _JEV, "Blocked: sentiment."),
    ("09_async", _JEV, "allow"),
    ("10_failures", (), "Blocked: input_too_long."),
    ("11_openai_sdk", (*_JEV, "OPENAI_API_KEY", *_WEB), "Blocked: injection."),
    ("12_openai_agents_sdk", (*_JEV, "OPENAI_API_KEY", *_WEB), "Blocked: injection."),
    ("13_langchain_agent", (*_JEV, _CHAT, *_WEB), "Blocked: injection."),
    ("14_langgraph", (*_JEV, _CHAT, *_WEB), "Blocked: injection."),
    ("15_deep_agents", (*_JEV, _CHAT, *_WEB), "Blocked: injection."),
    ("16_ollama", (_OLLAMA,), "Blocked: injection."),
    ("17_langgraph_local", (_OLLAMA,), "Blocked: injection."),
]


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=2):
            return True
    except OSError:
        return False


def _skip_reason(needs: tuple[str, ...]) -> str | None:
    chat = os.environ.get("EXAMPLE_CHAT_MODEL", "")
    for need in needs:
        if need == _OLLAMA:
            if not _ollama_up():
                return "Ollama is not running"
        elif need == _CHAT and chat and not chat.startswith("openai:"):
            continue
        elif not os.environ.get(need, "").strip():
            return f"{need} not set"
    return None


@pytest.mark.parametrize(("name", "needs", "expected"), _CASES, ids=[c[0] for c in _CASES])
def test_lesson_runs_live(name: str, needs: tuple[str, ...], expected: str) -> None:
    load_project_env()
    reason = _skip_reason(needs)
    if reason:
        pytest.skip(reason)
    # --frozen: never rewrite uv.lock. The examples group pins every package.
    command = ["uv", "run", "--frozen", "--group", "examples", "python", "-m", f"examples.{name}"]
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=900, check=False
    )
    # Mask keys before anything can land in a CI log; never assert on raw output.
    stdout, output = _mask(done.stdout), _mask(done.stdout + done.stderr)
    code = done.returncode
    del done
    assert code == 0, output[-4000:]
    assert expected in stdout, output[-4000:]


def _mask(text: str) -> str:
    for name in _SECRETS:
        value = os.environ.get(name, "").strip()
        if len(value) >= 8:
            text = text.replace(value, "<key>")
    return _KEY.sub("<key>", text)
