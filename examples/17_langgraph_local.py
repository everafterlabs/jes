"""Lesson 17: The lesson 14 graph, fully local.

Same graph, questions and checks as lesson 14. Only the models and the search
change: qwen3:1.7b on Ollama replies and tev1 on Ollama scores every check, so
nothing leaves the machine. Prints every check for clean, attack, poisoned_tool.

Run:   uv run --group examples python -m examples.17_langgraph_local
Mock:  add --mock to run offline, no keys needed.
Needs: Ollama with qwen3:1.7b and tev1
"""

from __future__ import annotations

import importlib

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from examples._backend import OLLAMA_URL, SCENARIOS, Run, decision_model, parse_mock, show

# Module names that start with a digit need import_module instead of `import`.
lesson14 = importlib.import_module("examples.14_langgraph")

THRESHOLD = 0.5  # tev1 scores sit in a narrower band than Jev's; lesson 01 explains thresholds.
LOCAL_CHAT_MODEL = "ollama:qwen3:1.7b"
# QUESTIONS, SYSTEM, MOCK_SCORES and MOCK_REPLIES come from lesson 14.
MOCK_SCORES = lesson14.MOCK_SCORES


@tool("search")
def local_search(query: str) -> str:
    """Search a local copy of the web. No network."""

    return f"Results for {query!r}: Python 3.14 is the latest stable release (python.org)."


def run(name: str, *, mock: bool) -> Run:
    # ChatOllama ignores timeout=; its HTTP client takes it instead.
    # reasoning=False turns off qwen3 thinking, so replies are short and fast.
    chat = None
    if not mock:
        chat = init_chat_model(
            LOCAL_CHAT_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,
            reasoning=False,
            client_kwargs={"timeout": 120},
        )
    return lesson14.run(
        name,
        mock=mock,
        model=decision_model(mock, MOCK_SCORES.get(name, {}), local=True),
        chat=chat,  # None with --mock: lesson 14's scripted replies.
        search=local_search,
        threshold=THRESHOLD,
    )


def main(*, mock: bool = False) -> dict[str, Run]:
    return {name: run(name, mock=mock) for name in SCENARIOS}


if __name__ == "__main__":
    for name, run in main(mock=parse_mock()).items():
        show(name, run.checks, run.reply)
