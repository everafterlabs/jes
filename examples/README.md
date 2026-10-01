# Learn jes in 17 lessons

jes checks the text around an AI agent: what the user typed, what a tool
returned, what the model is about to say. Each check asks a small decision
model (Jev, or `tev1` running locally) a yes/no question and blocks above your
threshold. You get back `result.ok` (continue or not) and `result.onward` (the
text to pass on: the original, a redacted copy, or a refusal).

Read the lessons in order. Each one teaches one idea and runs on its own.

## Setup

```bash
uv sync --group examples
cp .env.example .env   # add TYPESAFE_API_KEY, and the others a lesson asks for
```

Every lesson calls real services. Add `--mock` to run it offline with fixed
scores and a scripted model, no keys needed:

```bash
uv run python -m examples.01_first_check          # live, on Jev
uv run python -m examples.01_first_check --mock   # offline
```

## Part 1: Guard basics

| # | Lesson | You learn |
| --- | --- | --- |
| 01 | [`01_first_check.py`](01_first_check.py) | A `Guard`, one policy, a threshold, `ok` and `onward` |
| 02 | [`02_model_call.py`](02_model_call.py) | Three places to check: the input, a retrieved page, the model's reply |
| 03 | [`03_tool_calls.py`](03_tool_calls.py) | Allow only some tools, then check what a tool returns |
| 04 | [`04_pii.py`](04_pii.py) | Hide PII from the model and restore it in the reply |
| 05 | [`05_secrets_canary.py`](05_secrets_canary.py) | Keep secrets from the model and catch a leaked canary |
| 06 | [`06_topics_toxicity.py`](06_topics_toxicity.py) | Built-in judgments: denied topics and toxicity |
| 07 | [`07_custom_questions.py`](07_custom_questions.py) | Ask your own yes/no, choice and score questions |
| 08 | [`08_recipes.py`](08_recipes.py) | Ready-made recipes from the catalog |
| 09 | [`09_async.py`](09_async.py) | The same checks with `AsyncGuard` |
| 10 | [`10_failures.py`](10_failures.py) | When the decision model fails: fail closed or open |

## Part 2: In your agent

Same three runs in every lesson: a clean request, a prompt injection, and a
tool that returns a poisoned page.

| # | Lesson | You learn |
| --- | --- | --- |
| 11 | [`11_openai_sdk.py`](11_openai_sdk.py) | A hand-written tool loop with every check in plain sight |
| 12 | [`12_openai_agents_sdk.py`](12_openai_agents_sdk.py) | jes as OpenAI Agents SDK guardrails |
| 13 | [`13_langchain_agent.py`](13_langchain_agent.py) | One middleware guards a LangChain agent |
| 14 | [`14_langgraph.py`](14_langgraph.py) | The same checks as LangGraph nodes |
| 15 | [`15_deep_agents.py`](15_deep_agents.py) | Guard a Deep Agent and the subagent it hands work to |

## Part 3: Fully local

Nothing leaves your machine. Ollama runs both the chat model and the `tev1`
decision model: `ollama pull qwen3:1.7b && ollama pull tev1`.

| # | Lesson | You learn |
| --- | --- | --- |
| 16 | [`16_ollama.py`](16_ollama.py) | The plain `ollama` client with a local decision model |
| 17 | [`17_langgraph_local.py`](17_langgraph_local.py) | Lesson 14's graph, fully local |

## Good to know

- Thresholds in the lessons are application choices, not library defaults.
  Tune them on your own traffic, then pin the model (`jev-1.13.0`).
- `EXAMPLE_CHAT_MODEL` picks the LangChain chat model, for example
  `EXAMPLE_CHAT_MODEL=ollama:qwen3:1.7b`. The default is `openai:gpt-5.4-mini`.
- The shared files `_backend.py`, `_mocks.py` and `_middleware.py` hold the
  wiring, so each lesson shows only the jes calls.
- Run every lesson live:
  `uv run --frozen --group examples pytest -m live tests/test_examples_live.py --no-cov`.
