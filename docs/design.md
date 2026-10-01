# jes — design

This is how jes 2 works: what it protects, the guarantees it keeps, and the parts that keep them. The [guide](guide.md) covers setup and the agent hooks, and the [cookbook](cookbook.md) covers patterns.

## 1. Purpose

jes checks the text around an AI agent: the user's input, retrieved text, each tool call and tool result, and the reply. A check runs local **transforms** first (redaction, invisible characters, patterns, token limits), then asks a decision model **questions** and compares each answer with a **threshold** you choose. The result says whether to continue (`ok`) and what to pass on (`onward`).

The judge is a decision model, such as TypeSafe's Jev, reached through LangChain's `TypeSafeClassifier`. It classifies rather than generates, and the checked text is never part of an instruction it follows. Any object that implements the backend protocol (section 8) can take its place.

jes publishes no default thresholds. Every judgment takes `threshold=`.

## 2. Threat model

**Protected:** the application's instructions, personal data and secrets in prompts and tool traffic, the people who read model output, and the systems that act on tool calls.

**Attackers and failure sources:**

1. A user: direct injection, jailbreaks, requests for hazardous content.
2. A third party whose text reaches the model through retrieval, browsing, email, or tool results: indirect injection.
3. The model itself: harmful, leaking, or off-policy output and tool calls.
4. Anyone targeting jes: invisible and lookalike characters, padding that pushes an attack past a context window, text addressed to the judge, oversized inputs, forged placeholders, and links that would carry a restored value to another server.
5. For the agent hooks: the project the agent is working in. Its files are untrusted, including any `.env`.

**Out of scope:** detecting attacks spread across turns, decoding obfuscated payloads (base64, ROT13), images and audio, restoring values into tool arguments or rich renderers, and an attacker who already runs code in the application's process. Custom policy code runs in-process and is trusted with whatever it is given.

jes does not guarantee that a judgment is correct. It lowers risk and is one layer of defense.

## 3. Guarantees

The tests named after each one are in `tests/`.

- **Backends never receive a value that pii, secrets, or canary found.** That holds for the checked text and every context value, including tool call arguments in the judges' view. Every standalone occurrence of a found value is replaced, not just the one a detector reported (`test_pii_flow.py::test_every_standalone_occurrence_is_replaced`, `test_judges_never_see_secrets_in_tool_call_arguments`, `test_context_values_are_sanitized_before_judges_see_them`).
- **Every character of a judgment's text reaches a backend, or the check is incomplete.** Chunks overlap and cover the whole text. A check that cannot judge everything returns `complete=False`, and `ok` is false (`test_guard.py::test_chunks_always_cover_the_text`).
- **Values do not cross conversations.** A token is an HMAC of its value under one store's random secret. A reply restores only tokens that are in its store and appeared in that check's context (`test_pii_flow.py::test_tokens_the_prompt_did_not_carry_stay_unrestored`, `test_restore.py::test_only_authorized_tokens_in_the_store_are_restored`).
- **A restored value never goes to another origin.** A token inside a URL restores only when the URL is an absolute http or https URL whose origin is in `restore_origins`. Otherwise the reply blocks, or with `on_placeholder_in_url="allow"` passes with the token left in place (`test_restore.py::test_tokens_in_urls_restore_only_for_allowed_origins`).
- **Work is bounded.** Input, context, normalized and restored text, chunks, items, requests, redactions, and the wall-clock time of a check have limits (section 10). Text folding runs in linear time (`test_fold.py::test_adversarial_unicode_folds_in_linear_time`).
- **Errors and reprs never contain checked text.** They carry sizes, labels, and ids only (`test_types.py::test_message_and_state_reprs_hide_text`, `test_result.py`, `test_redactions.py`).
- **Failures fail closed unless you choose otherwise.** A backend failure raises by default. With `on_backend_error="block"` or `"allow"`, the result is incomplete, so `ok` is false either way. `"allow"` only keeps `decision` at allow (`test_guard.py::test_every_backend_failure_follows_on_backend_error`).

## 4. A check

```text
check_input / check_untrusted / check_tool_call / check_tool_result / check_output
        │
        ▼
prepare   resolve the conversation's Redactions store
          check sizes
          sanitize context: prompt, question, sources, history
          run transforms on the text, by phase: normalize, detect, limit
        │
        ▼
plan      pick the judgments for this stage
          batch their questions per backend and context mode
          fit context into the request, newest history first
          chunk the text, or extract items, so every request fits
        │
        ▼
execute   Guard: a thread pool, waiting at most deadline_s
          AsyncGuard: asyncio tasks, cancelled at the deadline
        │
        ▼
finish    read answers, compare scores with thresholds, merge findings
          for a reply: put back local markers, restore authorized tokens
          commit new tokens to the store only if the check allowed and completed
        │
        ▼
Result    ok, decision, complete, onward, original, sanitized, findings, scores, usage
```

`prepare`, `plan`, and `finish` live in `jes/engine/pipeline.py` and do no I/O. `Guard` and `AsyncGuard` (`jes/guard.py`) differ only in how they send requests.

| Component | Where the text goes |
| --- | --- |
| Transforms: invisible text, regex, substrings, token limit, pii, secrets, canary | Your process |
| Judgments | The backend, after transforms |

## 5. Stages and context

| Method | Stage | Context it can carry |
| --- | --- | --- |
| `check_input(text)` | `input` | `history` |
| `check_untrusted(text, question=)` | `untrusted` | the retrieval question |
| `check_tool_call(name, arguments, prompt=)` | `tool_call` | the user's request |
| `check_tool_result(text, name=, prompt=)` | `tool_result` | the user's request |
| `check_output(text, prompt=)` | `output` | the request, `sources`, `history` |

A judgment's `context` mode decides what it sees. `"none"` sends the text alone. `"optional"` adds the request, sources if asked, and as much recent history as fits. It flags `context_dropped` or `history_truncated` when something does not fit. `"required"` needs the request and sources to fit, and records `context_too_long` when they do not. It is allowed on output only.

Context may take at most half of a request's room. History is a contiguous run of the newest messages. A context value that is a `Result` from the same store is used as sanitized, and one that was not `ok` blocks with `context_not_ok`. Any other value is sanitized again, with the store, so its values get the same tokens.

## 6. Policies

**Transforms** implement `name`, `stages`, `phase`, and `apply(text, context) -> TransformOutcome(edits, findings)`. The engine applies the edits and maps each finding back to the original text, through every earlier edit, with `TextMap` (`jes/text/textmap.py`). A transform runs on text whose origin stage is in its stages, so a prompt passed as context is cleaned the way input is. Tool call arguments pass on unchanged: there, an edit changes only what judges see, and a redaction finding becomes a block.

**Sensitive policies** (`pii`, `secrets`, `canary`) report hits. The engine chooses the replacement:

| Mode | Replacement | Used for |
| --- | --- | --- |
| `token` | `[JES_PII_<id>]`, restorable | pii in input |
| `partial` | the first and last two characters, or all stars for 8 characters or fewer | pii in retrieved text, `secrets("partial")` |
| `mask` | `******` | `secrets("all")` |
| `hmac` | an HMAC-SHA256 under your key, in hex | `secrets("hmac")` |
| `local` | a random marker, for judges only | pii in a reply, `output_mode="flag"` |
| `remove` | `[REDACTED_<ENTITY>]` | canary, and everything else |

Replies and tool call arguments never get tokens, and only the reply being checked gets markers. Where a token or marker is not allowed, the value is removed. An occurrence inside a longer word or number is a different value and stays.

**Judgments** come from `judge()`: questions (`YesNo`, `Choice`, `Score`), a `Threshold`, stages, a context mode, and optionally an item extractor. A yes/no question's score is P(yes). A choice sums its violating options. A score sums the violation level and every level above. A score at or above `block_at` blocks, and one at or above `flag_at` flags. The built-in judgments are `judge()` calls with frozen question text (`jes/policies/prompts.py`). Changing that text needs a new id, and `tests/test_prompts.py` holds a hash of each.

Judgments that share a backend and context needs share one request. Their question ids are namespaced as `policy.question`.

## 7. Redaction

A `Redactions` store belongs to one conversation. Create one, such as `Redactions(scope=conversation_id.encode())`, and pass it to each check as `redactions=`. Or let `check_input` create one and pass its result on as `prompt=`. A store holds 1,000 values and 8 MiB by default.

- New tokens are staged during a check. They are committed only when the check allows and completes, so a blocked input leaves nothing behind.
- Token-shaped text in incoming text is escaped to `[JES_LITERAL_...]` and flagged, so a forged token never restores.
- `dumps` and `loads` save a store with AES-256-GCM (`jes[crypto]`). Loading needs the same key, scope, and associated data, and checks every entry.
- A reply's own personal data, under `output_mode="flag"`, is swapped for random markers while judges run, then swapped back. A later edit that cuts a marker drops that value.

## 8. Backends

A backend has `name`, `model`, and `headroom(state, questions) -> int | None`, plus `decide(Request) -> Reply`, `async adecide(Request) -> Reply`, or both. `Request` carries the state, the questions, and the seconds left. `Reply` carries answers and token counts. `TypeSafe` (`jes/backend.py`) is the built-in backend. A model id, a `TypeSafeClassifier`, or any backend works as `model=`. `jes.testing.FakeBackend` answers offline.

Every backend failure becomes a `BackendError` outcome: an exception, a `None` reply, missing answers, or answers of the wrong type. `on_backend_error` decides what follows. Requests still running at the deadline count as `deadline_exceeded`.

## 9. Results

`Result.ok` is the flag to act on: the check allowed the text and finished every judgment. `onward` is then the text to pass on:

- input, retrieved text, and tool results: the sanitized text
- tool calls: the arguments, unchanged
- replies: the text with markers put back and authorized tokens restored

When `ok` is false, `onward` is a refusal: `Tool call blocked.`, `Tool result blocked.`, or `Blocked:` with the blocking findings' names. Findings name the policy and label, the action, the score, and spans in the original text, or in the context value given by `target` and `index`.

## 10. Limits

`Limits` is passed to a guard as `limits=`.

| Limit | Default | Finding |
| --- | --- | --- |
| `max_input_bytes` | 1 MiB | `input_too_long` |
| `max_context_bytes` | 2 MiB | `context_too_long` |
| `max_chunks` | 32 per judgment | `too_many_chunks` |
| `max_items` | 100 per check | `too_many_items` |
| `max_requests` | 128 per check | `too_many_requests` |
| `max_concurrency` | 8 per guard | none: requests wait their turn |
| `max_redactions` | 1,000 per check | `too_many_redactions` |

A size limit stops the check before any judgment runs, with a blocking `jes` finding and an incomplete result. That covers the input, the context, more than 1,024 context values (`too_many_context_items`), text that cannot be encoded (`invalid_unicode`), redactions, and a full store (`redaction_store_full`). Normalized and restored text may grow to four times the input limit (`normalized_too_long`, `restored_output_too_large`).

A planning limit belongs to one judgment: too many chunks or items, context that `"required"` cannot fit, or a `whole_text` judgment that does not fit one request (`text_too_long`). The finding carries the judgment's name, it blocks or flags by the judgment's `on_overflow`, and the check is incomplete either way.

A check takes at most `deadline_s` seconds, 30 by default.

## 11. Agent hooks

`jes/agents` is the `jes` command: `login`, the settings and plugin printers, and one hook per agent. Each agent's protocol is an adapter (`jes/agents/adapters.py`) that turns an event into one check and the response back into that agent's JSON and exit code.

- The hooks read `~/.config/jes/config.json` for guards and the model. They read the API key from the environment or `~/.config/jes/.env` only, never from the project.
- An input check that fails exits 2, as does a tool call check. Result and reply checks answer with a refusal. A tool call with no stored prompt for its session is refused.
- The session store keeps each session's last allowed prompt and the parts of streamed replies, in 0600 files under `~/.config/jes/sessions`.
- Printed settings pin the jes release, so an upload to PyPI never runs unannounced.

## 12. Package layout

| Path | What it holds |
| --- | --- |
| `jes/guard.py` | `Guard`, `AsyncGuard` |
| `jes/engine/` | the pipeline, chunking, redaction, restoration |
| `jes/policies/` | transforms, sensitive policies, judgments, frozen prompts |
| `jes/recipes.py` | ready-made policies without evaluated thresholds |
| `jes/text/` | `TextMap`, Unicode folding, vendored Unicode 18 tables |
| `jes/backend.py` | the backend protocol and `TypeSafe` |
| `jes/redactions.py`, `jes/result.py`, `jes/types.py`, `jes/questions.py`, `jes/limits.py` | the values a check uses and returns |
| `jes/agents/` | the `jes` command, hooks, config, sessions, and the TypeScript plugins |

## 13. Testing and versioning

CI runs ruff, `ruff format --check`, strict pyright, and pytest with a 90% coverage floor on Python 3.11 to 3.14, plus the plugin tests on Node 22. Tests never call a live model or download weights: they use `FakeBackend` and synthetic text.

jes follows semantic versioning. 2.0 changed the Python API, and the hook config, commands, and JSON protocol stayed compatible.
