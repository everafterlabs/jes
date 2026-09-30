<p align="center">
  <a href="https://getjes.dev">
    <img src="docs/assets/banner.png" alt="jes: open-source guardrails for AI agents. Prompt injection protection for every prompt, skill, subagent, tool call and reply" width="100%">
  </a>
</p>

<p align="center">
  <strong>Open-source guardrails for AI agents, powered by a decision model, not an LLM.</strong><br>
  Stop prompt injection, jailbreaks, data leaks and risky tool calls at every step of your agent.
</p>

<p align="center">
  <a href="https://getjes.dev">Website</a> ·
  <a href="https://docs.getjes.dev">Docs</a> ·
  <a href="https://docs.getjes.dev/getting-started/quickstart">Quickstart</a> ·
  <a href="https://docs.getjes.dev/cookbook">Cookbook</a> ·
  <a href="https://pypi.org/project/jes/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/jes/"><img src="https://img.shields.io/pypi/v/jes?color=136CE0" alt="PyPI"></a>
  <a href="https://pypi.org/project/jes/"><img src="https://img.shields.io/pypi/pyversions/jes?color=136CE0" alt="Python 3.11+"></a>
  <a href="https://github.com/everafterlabs/jes/actions/workflows/ci.yml"><img src="https://github.com/everafterlabs/jes/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-FCB02A" alt="Apache-2.0"></a>
</p>

## Why jes

- **A decision model, not an LLM.** Every judgment runs on [Jev](https://typesafe.ai/), TypeSafe's System One decision model. It classifies instead of generating, so injected text can't talk it out of its verdict.
- **Every step of the agent.** Prompt, retrieved page, skill, subagent, tool call, tool result and reply.
- **Secrets stay local.** Secrets and PII are redacted in your process before any model sees the text.
- **Your thresholds.** No magic defaults. Pin the model (`jev-1.13.0`) once you've tuned them.

## Works with

<p>
  <a href="examples/langchain_agent.py"><img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" height="28"></a>
  <a href="examples/langgraph_agent.py"><img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik01IDE5SDEwQTUgNSAwIDExNSAxNFpNMTkgMTRBNSA1IDAgMTExNCAxOUgxOVpNMTAgNUE1IDUgMCAxMDUgMTBWNVpNMTkgNVYxMEE1IDUgMCAxMDE0IDVaIi8%2BPC9zdmc%2B" alt="LangGraph" height="28"></a>
  <a href="https://docs.getjes.dev"><img src="https://img.shields.io/badge/OpenAI%20Agents%20SDK-412991?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0yMi4yODE5IDkuODIxMWE1Ljk4NDcgNS45ODQ3IDAgMCAwLS41MTU3LTQuOTEwOCA2LjA0NjIgNi4wNDYyIDAgMCAwLTYuNTA5OC0yLjlBNi4wNjUxIDYuMDY1MSAwIDAgMCA0Ljk4MDcgNC4xODE4YTUuOTg0NyA1Ljk4NDcgMCAwIDAtMy45OTc3IDIuOSA2LjA0NjIgNi4wNDYyIDAgMCAwIC43NDI3IDcuMDk2NiA1Ljk4IDUuOTggMCAwIDAgLjUxMSA0LjkxMDcgNi4wNTEgNi4wNTEgMCAwIDAgNi41MTQ2IDIuOTAwMUE1Ljk4NDcgNS45ODQ3IDAgMCAwIDEzLjI1OTkgMjRhNi4wNTU3IDYuMDU1NyAwIDAgMCA1Ljc3MTgtNC4yMDU4IDUuOTg5NCA1Ljk4OTQgMCAwIDAgMy45OTc3LTIuOTAwMSA2LjA1NTcgNi4wNTU3IDAgMCAwLS43NDc1LTcuMDcyOXptLTkuMDIyIDEyLjYwODFhNC40NzU1IDQuNDc1NSAwIDAgMS0yLjg3NjQtMS4wNDA4bC4xNDE5LS4wODA0IDQuNzc4My0yLjc1ODJhLjc5NDguNzk0OCAwIDAgMCAuMzkyNy0uNjgxM3YtNi43MzY5bDIuMDIgMS4xNjg2YS4wNzEuMDcxIDAgMCAxIC4wMzguMDUydjUuNTgyNmE0LjUwNCA0LjUwNCAwIDAgMS00LjQ5NDUgNC40OTQ0em0tOS42NjA3LTQuMTI1NGE0LjQ3MDggNC40NzA4IDAgMCAxLS41MzQ2LTMuMDEzN2wuMTQyLjA4NTIgNC43ODMgMi43NTgyYS43NzEyLjc3MTIgMCAwIDAgLjc4MDYgMGw1Ljg0MjgtMy4zNjg1djIuMzMyNGEuMDgwNC4wODA0IDAgMCAxLS4wMzMyLjA2MTVMOS43NCAxOS45NTAyYTQuNDk5MiA0LjQ5OTIgMCAwIDEtNi4xNDA4LTEuNjQ2NHpNMi4zNDA4IDcuODk1NmE0LjQ4NSA0LjQ4NSAwIDAgMSAyLjM2NTUtMS45NzI4VjExLjZhLjc2NjQuNzY2NCAwIDAgMCAuMzg3OS42NzY1bDUuODE0NCAzLjM1NDMtMi4wMjAxIDEuMTY4NWEuMDc1Ny4wNzU3IDAgMCAxLS4wNzEgMGwtNC44MzAzLTIuNzg2NUE0LjUwNCA0LjUwNCAwIDAgMSAyLjM0MDggNy44NzJ6bTE2LjU5NjMgMy44NTU4TDEzLjEwMzggOC4zNjQgMTUuMTE5MiA3LjJhLjA3NTcuMDc1NyAwIDAgMSAuMDcxIDBsNC44MzAzIDIuNzkxM2E0LjQ5NDQgNC40OTQ0IDAgMCAxLS42NzY1IDguMTA0MnYtNS42NzcyYS43OS43OSAwIDAgMC0uNDA3LS42Njd6bTIuMDEwNy0zLjAyMzFsLS4xNDItLjA4NTItNC43NzM1LTIuNzgxOGEuNzc1OS43NzU5IDAgMCAwLS43ODU0IDBMOS40MDkgOS4yMjk3VjYuODk3NGEuMDY2Mi4wNjYyIDAgMCAxIC4wMjg0LS4wNjE1bDQuODMwMy0yLjc4NjZhNC40OTkyIDQuNDk5MiAwIDAgMSA2LjY4MDIgNC42NnpNOC4zMDY1IDEyLjg2M2wtMi4wMi0xLjE2MzhhLjA4MDQuMDgwNCAwIDAgMS0uMDM4LS4wNTY3VjYuMDc0MmE0LjQ5OTIgNC40OTkyIDAgMCAxIDcuMzc1Ny0zLjQ1MzdsLS4xNDIuMDgwNUw4LjcwNCA1LjQ1OWEuNzk0OC43OTQ4IDAgMCAwLS4zOTI3LjY4MTN6bTEuMDk3Ni0yLjM2NTRsMi42MDItMS40OTk4IDIuNjA2OSAxLjQ5OTh2Mi45OTk0bC0yLjU5NzQgMS40OTk3LTIuNjA2Ny0xLjQ5OTdaIi8%2BPC9zdmc%2B" alt="OpenAI Agents SDK" height="28"></a>
  <a href="https://docs.getjes.dev"><img src="https://img.shields.io/badge/Claude%20Agent%20SDK-D97757?style=for-the-badge&logo=claude&logoColor=white" alt="Claude Agent SDK" height="28"></a>
  <a href="https://docs.getjes.dev"><img src="https://img.shields.io/badge/FastMCP-2D2D2D?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjE2IDE2IDE2NCAxNjQiIGZpbGw9IndoaXRlIj48cGF0aCBkPSJNMTQ1Ljc0NyA0NC42MTFMMTQ1LjM1NSA0NC4zODc3TDE0NC45NiA0NC42MTFMODYuMDI4MyA3OC41Mjc2VjE3MS4yNjdMODYuNDAxNCAxNzEuNDk5TDk5LjY2NzQgMTc5LjY2N1Y4Ni4zODU5TDE1OSA1Mi4yMzc5TDE0NS43NDcgNDQuNjExWiIgLz4KICAgICAgPHBhdGggZD0iTTEyMS42MTYgMzAuMjcxNEwxMjEuMjI0IDMwLjA0NTRMMTIwLjgzMiAzMC4yNzE0TDYxLjg5NzUgNjQuMTg4VjE1Ni45MjhMNjIuMjczMiAxNTcuMTU2TDc1LjUzOTMgMTY1LjMyNVY3Mi4wNDYzTDEzNC44NjkgMzcuODk4M0wxMjEuNjE2IDMwLjI3MTRaIiAvPgogICAgICA8cGF0aCBkPSJNOTcuNDg5NCAxNi4zODE4TDk3LjA5NzMgMTYuMTU1OEw5Ni43MDI1IDE2LjM4MThMMzcuNzcwNSA1MC4zMDM4VjE0Mi4wNjZMNTEuNDA5NiAxNTAuNDYzVjU4LjE1NjdMMTEwLjc0MiAyNC4wMDg2TDk3LjQ4OTQgMTYuMzgxOFoiIC8%2BCiAgICAgIDxwYXRoIGQ9Ik0xMzEuMjMgMTEzLjY3MUwxMjQuOTc5IDExNy4yNjZMMTI0LjU4NCAxMTcuNDk0VjExNy41TDExNi43OTYgMTIxLjk4N0wxMTAuNTQ3IDEyNS41ODFMMTEwLjE1MiAxMjUuODA3VjE0MS41MUwxNDQuNTY0IDEyMS43MDlWMTIxLjY5OEwxNTguOTk5IDExMy4zOTRWOTcuNjg1MUwxMzkuMjc3IDEwOS4wMzRMMTMxLjIzIDExMy42NzFaIiAvPjwvc3ZnPg%3D%3D" alt="FastMCP" height="28"></a>
  <a href="https://docs.getjes.dev"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" height="28"></a>
</p>
<p>
  <a href="docs/guide.md#claude-code"><img src="https://img.shields.io/badge/Claude%20Code-D97757?style=for-the-badge&logo=claude&logoColor=white" alt="Claude Code" height="28"></a>
  <a href="docs/guide.md#codex"><img src="https://img.shields.io/badge/Codex-412991?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0yMi4yODE5IDkuODIxMWE1Ljk4NDcgNS45ODQ3IDAgMCAwLS41MTU3LTQuOTEwOCA2LjA0NjIgNi4wNDYyIDAgMCAwLTYuNTA5OC0yLjlBNi4wNjUxIDYuMDY1MSAwIDAgMCA0Ljk4MDcgNC4xODE4YTUuOTg0NyA1Ljk4NDcgMCAwIDAtMy45OTc3IDIuOSA2LjA0NjIgNi4wNDYyIDAgMCAwIC43NDI3IDcuMDk2NiA1Ljk4IDUuOTggMCAwIDAgLjUxMSA0LjkxMDcgNi4wNTEgNi4wNTEgMCAwIDAgNi41MTQ2IDIuOTAwMUE1Ljk4NDcgNS45ODQ3IDAgMCAwIDEzLjI1OTkgMjRhNi4wNTU3IDYuMDU1NyAwIDAgMCA1Ljc3MTgtNC4yMDU4IDUuOTg5NCA1Ljk4OTQgMCAwIDAgMy45OTc3LTIuOTAwMSA2LjA1NTcgNi4wNTU3IDAgMCAwLS43NDc1LTcuMDcyOXptLTkuMDIyIDEyLjYwODFhNC40NzU1IDQuNDc1NSAwIDAgMS0yLjg3NjQtMS4wNDA4bC4xNDE5LS4wODA0IDQuNzc4My0yLjc1ODJhLjc5NDguNzk0OCAwIDAgMCAuMzkyNy0uNjgxM3YtNi43MzY5bDIuMDIgMS4xNjg2YS4wNzEuMDcxIDAgMCAxIC4wMzguMDUydjUuNTgyNmE0LjUwNCA0LjUwNCAwIDAgMS00LjQ5NDUgNC40OTQ0em0tOS42NjA3LTQuMTI1NGE0LjQ3MDggNC40NzA4IDAgMCAxLS41MzQ2LTMuMDEzN2wuMTQyLjA4NTIgNC43ODMgMi43NTgyYS43NzEyLjc3MTIgMCAwIDAgLjc4MDYgMGw1Ljg0MjgtMy4zNjg1djIuMzMyNGEuMDgwNC4wODA0IDAgMCAxLS4wMzMyLjA2MTVMOS43NCAxOS45NTAyYTQuNDk5MiA0LjQ5OTIgMCAwIDEtNi4xNDA4LTEuNjQ2NHpNMi4zNDA4IDcuODk1NmE0LjQ4NSA0LjQ4NSAwIDAgMSAyLjM2NTUtMS45NzI4VjExLjZhLjc2NjQuNzY2NCAwIDAgMCAuMzg3OS42NzY1bDUuODE0NCAzLjM1NDMtMi4wMjAxIDEuMTY4NWEuMDc1Ny4wNzU3IDAgMCAxLS4wNzEgMGwtNC44MzAzLTIuNzg2NUE0LjUwNCA0LjUwNCAwIDAgMSAyLjM0MDggNy44NzJ6bTE2LjU5NjMgMy44NTU4TDEzLjEwMzggOC4zNjQgMTUuMTE5MiA3LjJhLjA3NTcuMDc1NyAwIDAgMSAuMDcxIDBsNC44MzAzIDIuNzkxM2E0LjQ5NDQgNC40OTQ0IDAgMCAxLS42NzY1IDguMTA0MnYtNS42NzcyYS43OS43OSAwIDAgMC0uNDA3LS42Njd6bTIuMDEwNy0zLjAyMzFsLS4xNDItLjA4NTItNC43NzM1LTIuNzgxOGEuNzc1OS43NzU5IDAgMCAwLS43ODU0IDBMOS40MDkgOS4yMjk3VjYuODk3NGEuMDY2Mi4wNjYyIDAgMCAxIC4wMjg0LS4wNjE1bDQuODMwMy0yLjc4NjZhNC40OTkyIDQuNDk5MiAwIDAgMSA2LjY4MDIgNC42NnpNOC4zMDY1IDEyLjg2M2wtMi4wMi0xLjE2MzhhLjA4MDQuMDgwNCAwIDAgMS0uMDM4LS4wNTY3VjYuMDc0MmE0LjQ5OTIgNC40OTkyIDAgMCAxIDcuMzc1Ny0zLjQ1MzdsLS4xNDIuMDgwNUw4LjcwNCA1LjQ1OWEuNzk0OC43OTQ4IDAgMCAwLS4zOTI3LjY4MTN6bTEuMDk3Ni0yLjM2NTRsMi42MDItMS40OTk4IDIuNjA2OSAxLjQ5OTh2Mi45OTk0bC0yLjU5NzQgMS40OTk3LTIuNjA2Ny0xLjQ5OTdaIi8%2BPC9zdmc%2B" alt="Codex" height="28"></a>
  <a href="docs/guide.md#hermes"><img src="https://img.shields.io/badge/Hermes-16181D?style=for-the-badge" alt="Hermes" height="28"></a>
  <a href="docs/guide.md#opencode"><img src="https://img.shields.io/badge/OpenCode-16181D?style=for-the-badge" alt="OpenCode" height="28"></a>
  <a href="docs/guide.md#openclaw"><img src="https://img.shields.io/badge/OpenClaw-16181D?style=for-the-badge" alt="OpenClaw" height="28"></a>
  <a href="docs/guide.md#pi"><img src="https://img.shields.io/badge/Pi-16181D?style=for-the-badge" alt="Pi" height="28"></a>
</p>

## Quickstart

```bash
pip install jes
export TYPESAFE_API_KEY=...
```

```python
from jes import Guard
from jes.policies import injection, secrets

guard = Guard([secrets(), injection(threshold=0.5)], model="jev-latest")

incoming = guard.check_input(user_text)
if not incoming.ok:
    return incoming.onward            # "Blocked: injection."

reply = llm(incoming.onward)          # the secret is already redacted
return guard.check_output(reply, prompt=incoming).onward
```

Check each step with `check_input`, `check_untrusted` (retrieved pages), `check_tool_call`, `check_tool_result` and `check_output`. Every result has `ok`, `decision`, `scores` and `onward`, the text to pass on.

<details>
<summary><b>Guard an agent's tool calls</b></summary>

```python
from jes.policies import allowed_tools, indirect_injection, tool_safety

guard = Guard(
    [allowed_tools(["search"]), tool_safety(threshold=0.5), indirect_injection(threshold=0.5)],
    model="jev-latest",
)

call = guard.check_tool_call("search", {"q": "quarterly notes"}, prompt=incoming)
if call.ok:
    result = guard.check_tool_result(run_search("quarterly notes"), name="search", prompt=incoming)
    feed_to_model(result.onward)      # a poisoned page becomes a refusal
```

</details>

## Write a custom guard

Ask Jev your own question with `judge()`:

```python
from jes import Guard
from jes.policies import judge
from jes.questions import YesNo

refunds = judge("refund", YesNo("The text asks for money back."), threshold=0.8, stages=("input",))

guard = Guard([refunds], model="jev-latest")
guard.check_input("Please refund my order.").ok   # False above the threshold
```

`Choice` and `Score` questions work the same way. See [`examples/custom_questions.py`](examples/custom_questions.py).

## Guardrails for Claude Code, Codex and other coding agents

```bash
uvx jes login            # saves your TypeSafe key and a default guard config
uvx jes claude-settings  # prints the hooks for ~/.claude/settings.json
```

Swap `claude` for `codex`, `hermes`, `opencode`, `openclaw` or `pi`. In Claude Code, the hooks also check skill loads, subagent launches and every tool call inside a subagent.

<details>
<summary><b>What each agent checks</b></summary>

| Agent | Prompt | Tool call | Tool result | Reply |
| --- | :---: | :---: | :---: | :---: |
| Claude Code | ✅ | ✅ | ✅ | ✅ on screen |
| Codex | ✅ | ✅ | ✅ | ✅ |
| Hermes | ✅ | ✅ | — | — |
| OpenCode | ✅ | ✅ | ✅ | — |
| OpenClaw | ✅ | ✅ | ✅ | ✅ |
| Pi | ✅ | ✅ | ✅ | — |

OpenCode, OpenClaw and Pi also need the file that `uvx jes runner-settings` prints. Details are in the [guide](docs/guide.md).

</details>

## Guards: prompt injection, PII, secrets and tool calls

| Guard | Catches |
| --- | --- |
| `injection`, `indirect_injection` | Instructions that try to take over the model, typed in or hidden in a page or tool result |
| `tool_safety`, `allowed_tools` | Tool calls that don't fit the request, or aren't on your list |
| `hazards`, `toxicity`, `topics` | The S1–S14 hazards, toxic content, topics you deny |
| `secrets`, `pii` | API keys and personal data, redacted locally (`jes[secrets]`, `jes[pii]`) |
| `invisible_text`, `canary` | Hidden characters, and a marker that must never leak |
| `regex`, `substrings`, `token_limit` | Your own patterns, terms and size limits |
| `judge` | Any question you write |

More in the [recipes](https://docs.getjes.dev/recipes) and the [cookbook](https://docs.getjes.dev/cookbook).

<details>
<summary><b>Examples</b></summary>

All run offline on `FakeBackend`, except `live_typesafe.py`.

| Example | Shows |
| --- | --- |
| [`one_check.py`](examples/one_check.py) | A prompt injection blocked at input |
| [`model_call.py`](examples/model_call.py) | User text, a retrieved page and the reply |
| [`tool_calls.py`](examples/tool_calls.py) | A disallowed tool and a poisoned tool result |
| [`pii_conversation.py`](examples/pii_conversation.py) | PII hidden on the way in, restored in the reply |
| [`secrets_canary.py`](examples/secrets_canary.py) | A redacted secret and a leaked canary |
| [`topics_toxicity.py`](examples/topics_toxicity.py) | Denied topics and toxicity |
| [`custom_questions.py`](examples/custom_questions.py) | Custom yes/no, choice and score questions |
| [`recipes.py`](examples/recipes.py) | Recipes from the catalog |
| [`langchain_agent.py`](examples/langchain_agent.py) | A LangChain agent |
| [`langgraph_agent.py`](examples/langgraph_agent.py) | A LangGraph graph |
| [`async_check.py`](examples/async_check.py) | `AsyncGuard` |
| [`failures.py`](examples/failures.py) | Backend errors: an incomplete result is never ok |
| [`live_typesafe.py`](examples/live_typesafe.py) | One live check against Jev |

</details>

<details>
<summary><b>FAQ</b></summary>

**What is jes?** An open-source (Apache-2.0) Python library that adds guardrails to AI agents. It checks the prompt, retrieved text, tool calls, tool results and the reply.

**How do I protect an AI agent from prompt injection?** Check more than the user's message. Most attacks are *indirect*: hidden in a web page, a file or a tool result. Use `indirect_injection` on `check_untrusted` and `check_tool_result`, and gate tools with `allowed_tools` and `tool_safety`.

**Why a decision model instead of an LLM-as-judge?** An LLM judge reads the attack as part of its own prompt, and the attack can steer its answer. Jev only answers typed questions with probabilities, and the checked text is never part of an instruction.

**Does jes send my secrets or personal data to a model?** No. `secrets` and `pii` run locally first, and judgments only see the redacted text.

**Does it cover subagents and skills?** In Claude Code, yes. They are tool calls, so the hooks check them, and the hooks also run inside subagents.

**What thresholds should I use?** jes publishes none. Start around `0.5`, measure on your own traffic, then pin the model version.

</details>

<details>
<summary><b>Install extras</b></summary>

`jes[pii]` (Presidio, plus a spaCy English model) · `jes[secrets]` (detect-secrets) · `jes[crypto]` (encrypted `Redactions`) · `jes[tokens]` (tiktoken) · `jes[regex]` · `jes[json]`. Python 3.11+.

</details>

## Learn more

[Prompt injection in AI agents](https://www.getjes.dev/blog/prompt-injection-ai-agents) · [LangChain and LangGraph guardrails](https://www.getjes.dev/blog/langchain-langgraph-guardrails) · [AI agent guardrails compared](https://www.getjes.dev/blog/ai-agent-guardrails-compared) · [What is Jev?](https://www.getjes.dev/blog/what-is-jev) · [FAQ](https://www.getjes.dev/faq)

## Scope

jes publishes no measured default thresholds; tune them on your traffic. A hook can refuse a tool call before it runs, but can't undo one that already ran. Guardrails reduce risk; they don't replace least privilege or sandboxing. Report vulnerabilities through [SECURITY.md](SECURITY.md).

## Development

```bash
uv sync --dev && uv run ruff check . && uv run pyright && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0. This project is independent. It isn't affiliated with [TypeSafe.ai](https://typesafe.ai).
