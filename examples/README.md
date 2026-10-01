# Learn jes in 15 lessons

jes checks the text around an AI agent: what the user typed, what a tool
returned, what the model is about to say. Each check asks a small decision
model a yes/no question and blocks above your threshold. You get back
`result.ok` (continue or not) and `result.onward` (the text to pass on: the
original, a redacted copy, or a refusal).

Every lesson is a folder with two files that teach the same thing:

- **`jev.py`**: hosted Jev decides, OpenAI writes replies, Tavily searches.
- **`local.py`**: `tev1` decides and `qwen3:1.7b` writes replies, both on Ollama. Nothing leaves your machine.

Diff the two files and only the models change. Read the lessons in order.

## Setup

```bash
uv sync --group examples
cp .env.example .env              # for jev.py: TYPESAFE_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
ollama pull tev1 && ollama pull qwen3:1.7b   # for local.py
```

```bash
uv run python -m examples.01_first_check.jev
uv run python -m examples.01_first_check.local
```

## Part 1: Guard basics

| # | Lesson | You learn |
| --- | --- | --- |
| 01 | [jev](01_first_check/jev.py) · [local](01_first_check/local.py) | A `Guard`, one policy, a threshold, `ok` and `onward` |
| 02 | [jev](02_model_call/jev.py) · [local](02_model_call/local.py) | Three places to check: the input, a retrieved page, the model's reply |
| 03 | [jev](03_tool_calls/jev.py) · [local](03_tool_calls/local.py) | Allow only some tools, then check what a tool returns |
| 04 | [jev](04_pii/jev.py) · [local](04_pii/local.py) | Hide PII from the model and restore it in the reply |
| 05 | [jev](05_secrets_canary/jev.py) · [local](05_secrets_canary/local.py) | Keep secrets from the model and catch a leaked canary |
| 06 | [jev](06_topics_toxicity/jev.py) · [local](06_topics_toxicity/local.py) | Built-in judgments: denied topics and toxicity |
| 07 | [jev](07_custom_questions/jev.py) · [local](07_custom_questions/local.py) | Ask your own yes/no, choice and score questions |
| 08 | [jev](08_recipes/jev.py) · [local](08_recipes/local.py) | Ready-made recipes from the catalog |
| 09 | [jev](09_async/jev.py) · [local](09_async/local.py) | The same checks with `AsyncGuard` |
| 10 | [jev](10_failures/jev.py) · [local](10_failures/local.py) | When the decision model is down: fail closed or open |

## Part 2: In your agent

Same three runs in every lesson: a clean request, a prompt injection, and a
tool that returns a poisoned page.

| # | Lesson | You learn |
| --- | --- | --- |
| 11 | [jev](11_openai_sdk/jev.py) · [local](11_openai_sdk/local.py) | A hand-written tool loop with every check in plain sight |
| 12 | [jev](12_openai_agents_sdk/jev.py) · [local](12_openai_agents_sdk/local.py) | jes as OpenAI Agents SDK guardrails |
| 13 | [jev](13_langchain_agent/jev.py) · [local](13_langchain_agent/local.py) | One middleware guards a LangChain agent |
| 14 | [jev](14_langgraph/jev.py) · [local](14_langgraph/local.py) | The same checks as LangGraph nodes |
| 15 | [jev](15_deep_agents/jev.py) · [local](15_deep_agents/local.py) | Guard a Deep Agent and the subagent it hands work to |

## Good to know

- Thresholds are application choices, not library defaults: 0.72 for Jev,
  0.5 for `tev1`. Tune them on your own traffic, then pin the model (`jev-1.13.0`).
- `_common.py` only prints results and loads `.env`. `_middleware.py` is the
  LangChain middleware that lessons 13 and 15 share.
- Run every lesson live:
  `uv run --frozen --group examples pytest -m live tests/test_examples_live.py --no-cov`.
