# jes

Docs: [docs.getjes.dev](https://docs.getjes.dev)

jes is a security toolkit for LLM applications. Checks run on a decision model, [Jev](https://typesafe.ai/), which returns a probability. You set the threshold. If the probability is over it, `onward` is a refusal (`Blocked: injection.`) and that text should not go to the LLM or to the user.

## Install

Python 3.11 or newer.

```bash
pip install jes
export TYPESAFE_API_KEY=...
```

Extras, if you need them:

* `jes[pii]` needs Presidio and a spaCy English model, `en_core_web_sm` or `en_core_web_lg`
* `jes[secrets]` uses detect-secrets
* `jes[crypto]` encrypts a `Redactions` store (`dumps` / `loads`)
* `jes[tokens]` uses tiktoken for `token_limit`

## How to use

```python
from jes import Guard
from jes.policies import injection

guard = Guard([injection(threshold=0.5)], model="jev-latest")

incoming = guard.check_input(user_text)
if not incoming.ok:
    return incoming.onward

# call the LLM with incoming.onward
outgoing = guard.check_output(llm_reply, prompt=incoming)
return outgoing.onward
```

* `check_input(text)` — user text, before the LLM
* `check_untrusted(text, question=)` — a retrieved page or file, before it enters the prompt
* `check_tool_call(name, arguments, prompt=)` — a tool call, before it runs
* `check_tool_result(text, name=, prompt=)` — what the tool returned
* `check_output(text, prompt=)` — the LLM reply, before you show it

`prompt=` and `question=` take the `check_input` result. `model="jev-1.13.0"` pins the Jev release. The client is LangChain's [TypeSafe classifier](https://docs.langchain.com/oss/python/integrations/providers/typesafe).

## Personal agents

`jes login` writes `TYPESAFE_API_KEY` into `~/.config/jes/.env` and, the first time, `~/.config/jes/config.json`. That file chooses which guards run and their settings. The default enables `injection`, `indirect_injection`, and `hazards` at threshold `0.5` on `jev-latest`. A later login updates the key and leaves the config file alone.

uv ships `uvx`, so the hooks do not need a separate jes install:

```bash
uvx jes login
uvx jes claude-settings
uvx jes codex-settings
uvx jes hermes-settings
uvx jes opencode-settings
uvx jes openclaw-settings
uvx jes pi-settings
uvx jes runner-settings
```

Claude Code, Codex, and Hermes run `uvx jes claude-hook`, `uvx jes codex-hook`, and `uvx jes hermes-hook`. OpenCode, OpenClaw, and Pi use the TypeScript plugins those settings commands print. The plugins import `./jes-runner.ts`, which `jes runner-settings` prints; save that file beside the plugin. The runner calls `uvx jes hook`.

## Guards

Pass any of these to `Guard`. Judgment guards take `threshold=`.

* `injection` — someone trying to override the model's instructions
* `indirect_injection` — those instructions hidden in retrieved text or a tool result
* `hazards` — the S1–S14 hazard list
* `toxicity`
* `topics` — a list of topics you name
* `pii` — hides personal data on the way in, restores it in the reply (`jes[pii]`)
* `secrets` — API keys and similar tokens (`jes[secrets]`)
* `invisible_text` — hidden and lookalike characters
* `regex`, `substrings`, `token_limit`
* `allowed_tools` — tool names you permit
* `tool_safety` — whether a tool call and its arguments are safe for the user's request. The threshold is required; `0.5` in the agent config is a starting value, not a measured recommendation
* `canary` — a marker that must not appear in the reply
* `judge` — a question you write

More are in the [cookbook](https://docs.getjes.dev/cookbook) and [Recipes](https://docs.getjes.dev/recipes).

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pyright
uv run pytest
```

## License

Apache-2.0.

This project is independent. It isn't affiliated with TypeSafe, Meta, or Protect AI. The name is one letter off Jev, and the package doesn't include their weights.
