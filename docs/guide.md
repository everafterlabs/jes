# Guide

jes checks text before it reaches a model and before a tool result or reply is trusted. A check returns a probability. If that probability is over your threshold, `onward` is a refusal such as `Blocked: injection.` Send `onward` next, including when the check allows it.

There are two ways to run those checks.

- **Agents.** Claude Code, Codex, Hermes, OpenCode, OpenClaw, and Pi call `uvx jes` from a hook or plugin. You do not import the library.
- **Applications.** Your Python code constructs a `Guard` and calls it around the model and tools.

Patterns for the library API are in [cookbook.md](cookbook.md). Recipes are at [docs.getjes.dev/recipes](https://docs.getjes.dev/recipes).

## Install

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). The agent hooks run `uvx`, which ships with uv, so they do not need a separate global `jes` install. The first `uvx jes` call downloads the package. Later calls use the cache.

```bash
uvx jes login
```

`jes login` asks for a TypeSafe API key and writes it to `~/.config/jes/.env` (mode `0600`, directory mode `0700`). The first login also writes `~/.config/jes/config.json`. A later login updates the key and does not replace that file. `XDG_CONFIG_HOME` replaces `~/.config` when it is set. The command does not install hooks.

To embed jes in an application instead:

```bash
pip install jes
```

Optional extras:

- `jes[pii]` hides personal data. It needs Presidio and a spaCy English model, `en_core_web_sm` or `en_core_web_lg`.
- `jes[secrets]` detects API keys and similar tokens.
- `jes[crypto]` encrypts a `Redactions` store.
- `jes[tokens]` uses tiktoken for `token_limit`.
- `jes[regex]` and `jes[json]` are for those policies.

From a git checkout, use `uv run jes` after `uv sync --dev`. `uvx --from . jes` runs the checkout without installing it.

## Configure the agent checks

The API key is an environment variable. The first value wins: the process environment, then `.env.local`, then `.env` at the git repository root, then `~/.config/jes/.env`. A blank line in a file does not override a value that is already set.

| Variable | Default | Effect |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | unset | Required. The TypeSafe key for Jev. |

Which guards run is `~/.config/jes/config.json`, not the environment. A hook with no config file exits 2 and tells you to run `jes login`. The model is `jev-latest`. A library `Guard` does not read this file. You pass policies and the model to it yourself.

`config.json` is one object, `guards`. Each built-in policy is a key. `"enabled": false` skips that guard. An unknown guard, an unknown field, or a wrong type is an error. `judge` is not in the file. A custom question stays in Python.

The file `jes login` writes enables three guards and lists the others turned off, with their factory defaults filled in so you can see the fields:

- `injection` and `indirect_injection`: `threshold` (default `0.5`)
- `hazards`: `threshold`. Omit `categories` to check every S1–S14, or set a list such as `["S1", "S2"]`
- `toxicity`: `threshold`. Omit `labels` for every toxicity label, or set a list of labels
- `topics`: `threshold` and `deny`, the topics to block. `deny` is required when the guard is enabled
- `invisible_text`: `mode` (`targeted` or `all`) and `block`
- `allowed_tools`: `names`, the tool names that may run. Required and non-empty when enabled
- `tool_safety`: `threshold`. Judges the tool name and arguments against the user's request. `0.5` is the unset starting value, not a measured recommendation. Required when the guard is enabled
- `canary`: `token`, the marker that must not appear. Required when enabled
- `regex`: `patterns`, `action` (`block` or `redact`), `match` (`search` or `fullmatch`), `require`, `fold`, `timeout_ms`
- `substrings`: `terms`, `action` (`block` or `redact`), `whole_words`, `fold`
- `token_limit`: `limit`, `encoding` (default `cl100k_base`), `mode` (`block` or `truncate`)
- `pii`: `entities` (omit for the built-in set), `input_mode` (`redact`, `mask`, or `block`), `untrusted_mode` (`mask`, `redact`, or `block`), `output_mode` (`flag`, `redact`, or `block`), `restore`
- `secrets`: `redact` (`all`, `partial`, or `hmac`). `hmac` also requires `key`, base64 for at least 32 bytes

`pii` needs `jes[pii]`. `secrets` needs `jes[secrets]`. `regex` needs `jes[regex]`. `token_limit` needs `jes[tokens]`. A missing extra fails the hook with that policy's error. Per-policy `stages`, `name`, and `version` stay at the factory defaults.

The hook stores the last allowed user prompt for a session under `~/.config/jes/sessions`. That file is the prompt text and nothing else.

## Use jes with an agent

Print a snippet, then save it where that agent loads hooks or plugins. Each snippet calls `uvx jes ...`, so uv must be on `PATH` when the agent runs.

```bash
uvx jes claude-settings
uvx jes codex-settings
uvx jes hermes-settings
uvx jes opencode-settings
uvx jes openclaw-settings
uvx jes pi-settings
uvx jes runner-settings
```

The OpenCode, OpenClaw, and Pi plugins import `./jes-runner.ts`. Save the runner beside the plugin. `jes runner-settings` prints that file. The runner calls `uvx jes hook`.

A blocked tool call is refused before the tool runs. A blocked tool result is replaced only when that host applies the hook's output. A shell command or file write that already ran is not undone.

### Claude Code

Merge the `hooks` object from `jes claude-settings` into `~/.claude/settings.json`. The `limits` array in the printed JSON is documentation. Claude does not read it.

| Event | What jes does |
| --- | --- |
| `UserPromptSubmit` | Checks the user prompt. A block returns `decision: block`. |
| `PreToolUse` | Checks the tool call before it runs. A block denies the call. |
| `PostToolUse` | Checks the tool result after the tool has run. A block replaces the output the model reads. |
| `MessageDisplay` | Checks the reply once the message is final. A block changes the text on screen. The transcript, and what Claude sees on the next turn, keep the original reply. |

### Codex

Copy the `hooks` object from `jes codex-settings` into `~/.codex/hooks.json`. Codex ignores a hook until you trust its exact text with `/hooks`.

| Event | What jes does |
| --- | --- |
| `UserPromptSubmit` | Checks the user prompt. |
| `PreToolUse` | Checks the tool call before it runs. |
| `PostToolUse` | Replaces the model-facing result with the refusal. It does not undo the command. Codex does not apply `updatedToolOutput`. |
| `Stop` | Checks `last_assistant_message` and continues the turn with the refusal. It does not change text already on screen. A second Stop with `stop_hook_active` is skipped so a refusal does not loop. |

### Hermes

Copy the `hooks` map from `jes hermes-settings` into `~/.hermes/config.yaml`. Trust the hook when Hermes asks.

| Event | What jes does |
| --- | --- |
| `pre_tool_call` | Prints `action: block` and exits 2. `fail_closed: true` is set so a crashed hook does not allow the call. |
| `pre_llm_call` | A blocked user message is injected as `context`. Hermes does not reject the user message. |

Hermes shell hooks cannot replace a tool result or the assistant reply. The printed config does not register those events.

### OpenCode

Save both files in the plugin directory. Global plugins live in `~/.config/opencode/plugins/`. Project plugins live in `.opencode/plugins/`.

```bash
mkdir -p ~/.config/opencode/plugins
uvx jes opencode-settings > ~/.config/opencode/plugins/opencode-plugin.ts
uvx jes runner-settings > ~/.config/opencode/plugins/jes-runner.ts
```

OpenCode loads TypeScript files in that directory. The plugin exports `JesGuard`.

| Hook | What jes does |
| --- | --- |
| `chat.message` | Checks the user prompt. A block throws the refusal. |
| `tool.execute.before` | Checks the tool call. A block throws the refusal before the tool runs. |
| `tool.execute.after` | Checks the tool result. A block sets `output.output` to the refusal. |

OpenCode has no display-only reply hook, so the plugin does not rewrite the assistant reply.

### OpenClaw

Save `openclaw-plugin.ts` and `jes-runner.ts` in one directory. The plugin's default export is `register`. Link that directory the way OpenClaw loads a local plugin (`openclaw plugins install --link`), then enable it. Restart the gateway after you change plugin code.

`before_agent_run` is a conversation hook. On a plugin that is not bundled with OpenClaw, set `plugins.entries.<name>.hooks.allowConversationAccess` to `true` or that hook does not run. See the OpenClaw plugin hook docs for the current enablement rules.

| Hook | What jes does |
| --- | --- |
| `before_agent_run` | Checks the user prompt. A block returns `outcome: "block"` with the refusal as `reason` and `message`. |
| `before_tool_call` | Checks the tool call. A block returns `block: true` and `blockReason`. |
| `tool_result_persist` | Checks the tool result. A block returns `content` set to the refusal. |
| `message_sending` | Checks the reply that is about to be delivered. A block returns `content` set to the refusal. |

### Pi

Pi loads `~/.pi/agent/extensions/*.ts` and `~/.pi/agent/extensions/*/index.ts`. Put the extension in a subdirectory so Pi loads `index.ts` and does not treat the runner as a second extension. Project-local extensions use `.pi/extensions/` the same way.

```bash
mkdir -p ~/.pi/agent/extensions/jes
uvx jes pi-settings > ~/.pi/agent/extensions/jes/index.ts
uvx jes runner-settings > ~/.pi/agent/extensions/jes/jes-runner.ts
```

Reload with `/reload`.

| Hook | What jes does |
| --- | --- |
| `before_agent_start` | Checks the user prompt. A block injects the refusal as a message. Pi does not erase the original prompt. |
| `tool_call` | Checks the tool call. A block returns `block: true` and `reason`. |
| `tool_result` | Checks the tool result. A block replaces `content` with the refusal. Array content becomes a single text block. |

## Use jes from Python

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

`result.ok` is false when the check blocks. `result.onward` is the text to send next. `result.decision` is `"allow"` or `"block"`.

| Method | When | Required | Optional |
| --- | --- | --- | --- |
| `check_input(text)` | User text, before the LLM. | `text` | `redactions=`, `history=` |
| `check_untrusted(text, question=)` | A retrieved page or file, before it enters the prompt. | `text` | `question=`, `redactions=` |
| `check_tool_call(name, arguments, prompt=)` | A tool call, before it runs. | `name`, `arguments`, `prompt=` | `redactions=` |
| `check_tool_result(text, name=, prompt=)` | What the tool returned. | `text`, `name=` | `prompt=`, `redactions=` |
| `check_output(text, prompt=)` | The LLM reply, before you show it. | `text`, `prompt=` | `sources=`, `history=`, `redactions=` |

`prompt=` and `question=` take the `check_input` result. A plain string also works.

`prompt=` is required on `check_tool_call` and `check_output` because those checks judge the text against what the user asked. `tool_safety` asks whether the call "goes beyond what the user asked": `delete_file("/data/reports")` is fine after "clean up old reports" and a block after "summarize this PDF". Without the prompt, the judge can't tell those apart. Passing the `check_input` result also carries the conversation's `Redactions` store, so PII hidden on the way in lines up with later checks and you don't need `redactions=`. `injection`, `secrets`, `pii` and `allowed_tools` look only at the tool name and arguments. `model="jev-1.13.0"` pins a Jev release. The client is LangChain's [TypeSafe classifier](https://docs.langchain.com/oss/python/integrations/providers/typesafe). Set `TYPESAFE_API_KEY` in the environment. A library `Guard` does not call `jes login` for you. Examples run live by default; `--mock` runs them offline. The live test suite is `uv run pytest -m live tests/test_examples_live.py --no-cov`. It is not part of CI.

Pass any of these to `Guard`. Judgment policies take `threshold=`.

- `injection` — someone trying to override the model's instructions
- `indirect_injection` — those instructions hidden in retrieved text or a tool result
- `hazards` — the S1–S14 hazard list
- `toxicity`
- `topics` — a list of topics you name
- `pii` — hides personal data on the way in and restores it in the reply (`jes[pii]`)
- `secrets` — API keys and similar tokens (`jes[secrets]`)
- `invisible_text` — hidden and lookalike characters
- `regex`, `substrings`, `token_limit`
- `allowed_tools` — tool names you permit
- `canary` — a marker that must not appear in the reply
- `judge` — a question you write

`AsyncGuard` is the async twin of `Guard`. Tests should pass `jes.testing.FakeBackend` instead of calling Jev.

## Develop jes

```bash
uv sync --dev
uv run ruff check .
uv run pyright
uv run pytest
```

Plugin tests need Node 22:

```bash
node --experimental-strip-types --test tests/plugins/*.test.ts
```

CI runs ruff, pyright, and pytest on Python 3.11 through 3.14, and runs the plugin tests on Node 22. CI must not download model weights or call a live provider. Use `FakeBackend` and synthetic text. Fixtures must not contain hazardous content, secrets, or real personal data.

The library surface is `src/jes`. Policies live in `src/jes/policies`. The guard engine is `src/jes/_engine`. Agent stdin and stdout adapters are `src/jes/claude.py`, `src/jes/codex.py`, and `src/jes/hermes.py`. They share payload parsing in `src/jes/payload.py` and the check loop in `src/jes/hook.py`. The TypeScript plugins are in `src/jes/data/`.

Public APIs stay typed under strict pyright. A change to a name in a module's `__all__` also needs an update in the API reference, which lives in the separate `jes-docs` repository. The site build pulls `docs/cookbook.md`, `docs/design.md`, `CHANGELOG.md`, and `examples/*.py` from this repo.
