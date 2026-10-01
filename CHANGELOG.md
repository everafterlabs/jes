# Changelog

All notable changes to jes will be documented here.

The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 2.0.0

A rewrite of the core. The Python API changed. The agent hooks did not: `config.json`, the `jes` commands, and the `jes hook` JSON protocol work as before. Print your hook settings again after upgrading so they pin this release.

### Upgrading from 1.x

| 1.x | 2.0 |
| --- | --- |
| `InputResult`, `ScanResult` | `Result`, from every check |
| `result.text` | `result.original` |
| `result.timings` | `result.duration_ms` |
| `result.sanitization` | Removed |
| `Finding.locations`, `chunks`, `question` | `Finding.spans`, `target`, `index` |
| `Finding.score`, a `ScoreResult` | `Finding.score`, a float. Full scores are in `result.scores` |
| `ScoreResult.kind`, `provenance` | `ScoreResult.model` |
| `Usage.backend`, `request`, `attempt` | Removed |
| `FindingLocation`, `Provenance`, `ThresholdProvenance`, `Timings`, and `History` and `State` from `jes` | Removed |
| `Guard(max_input_bytes=..., ...)`, 21 limits | `Guard(limits=Limits(...))`, 7 limits |
| `Guard(trace=True)` | Removed |
| `version=` on judgments and recipes | Removed. Question text is frozen per id |
| `judge(on_context_overflow=, on_text_overflow=, on_items_overflow=)` | `judge(on_overflow=)` |
| `judge(max_policy_items=, item_overflow_label=)` | `judge(max_items=)`. The finding is `too_many_items` |
| `judge(version=, interpretation_version=)` | Removed |
| `toxicity()` without `threshold=` | `threshold=` is required |
| `pii(language=, ner=)`, `jes[pii-ner]` | Removed. `PERSON` uses Presidio with an installed spaCy model |
| `json_check(0, True)` | `json_check(0, repair=True)` |
| `jes.judge`: `Judge`, `resolve_judge`, profiles, capabilities | `jes.backend`: `TypeSafe`, `resolve_model`, `Request`, `Reply` |
| `FakeBackend(default_answer=, max_units=, tasks=, ...)` | `FakeBackend(answers, default=, rule=, max_request_bytes=, delay_s=)` |
| `check_backend_contract`, `assert_semantic_parity`, `fake_sensitive`, `FakeSensitiveTransform`, `FakeRequestBudget`, `register_default`, `clear_defaults` | Removed |
| `TransformPolicy`, `JudgmentPolicy`, `CallContext`, `TransformEdit` | `Transform`, `Judgment`, `TransformContext`, `Edit` |
| A transform returns `TransformOutcome(text=, edits=, findings=)` | It returns `TransformOutcome(edits, findings)`, and the engine applies the edits |
| `jes.cli`, `jes.claude`, `jes.codex`, `jes.hermes`, `jes.hook`, `jes.config` | `jes.agents.*`. `python -m jes` and the `jes` command are unchanged |
| `[JES_v1_PII_…]` tokens, store blobs version 1 | `[JES_PII_…]` tokens, version 2. A 1.x blob does not load |
| `Redactions(max_scope_bytes=, max_value_bytes=)`, `.id`, `.view()`, `copy()`, and `max_scope_bytes=` and `max_associated_data_bytes=` on `dumps` and `loads` | Removed. A store cannot be copied or pickled. Save it with `dumps` |

### Changed

- Every judgment requires `threshold=`.
- `hazards(["S1", "S2"], threshold=...)` asks about those categories only, one question each. There is no `hazard.any` question.
- `topics` names each question after its topic, such as `topics.medication_dosage`, instead of `topic_0`.
- `on_backend_error` covers every backend failure: an exception, a `None` reply, and missing or mistyped answers. With `"block"` or `"allow"` the result is incomplete, so `ok` is false either way.
- `Guard` sends a check's requests in parallel and stops at `deadline_s`. `AsyncGuard` cancels the requests still running at the deadline, or when the caller cancels the check.
- There is no cap on findings. 1.x blocked a check with `too_many_findings` at 1,000.
- `"optional"` context that does not fit is flagged `context_dropped`, and history that does not fit `history_truncated`.
- `canary` matches lookalike, cased, and invisible-character spellings, and checks tool calls as well as replies.
- `malicious_urls` takes `max_urls`, runs on input, retrieved text, tool results, and replies, and finds URLs with an uppercase scheme.
- `refusal_phrases` checks replies only.
- `json_check(repair=True)` repairs the JSON at the first bracket instead of skipping to a valid array inside it. Without repair, it tries at most 100 bracket positions.
- `token_limit` counts text that looks like a special token, such as `<|endoftext|>`, instead of failing.
- Text folding runs in linear time. 1.x took seconds on a megabyte of plain text, and longer on crafted Unicode.
- `ConfigError` moved to `jes.errors`, under `JesError`, and is exported from `jes`.

### PII and secrets

- `pii` checks email, phone, credit card, US SSN, IBAN, crypto address, and person by default. `UUID`, `IP_ADDRESS`, and `US_BANK_NUMBER` are opt-in.
- Every standalone occurrence of a found value is replaced, so a repeated value never reaches a judge. A value inside a longer number no longer fails the check.
- `PERSON` raises `PolicyError` without Presidio and a spaCy model. 1.x skipped it without a word.
- Phone numbers need separators, and IP addresses stop at 255 in each octet.
- `pii(tool_call_mode="flag")` lets a tool call that carries personal data through, with a finding. The default still blocks.
- Judges no longer see secrets or personal data inside tool call arguments.
- `secrets` finds OpenAI project keys, Anthropic keys, GitHub `gho_`, `ghu_`, `ghs_`, `ghr_`, and fine-grained tokens, Google API keys, Stripe keys, AWS `ASIA` keys, and private key blocks. Public IP addresses are no longer reported as secrets.
- A link with an invalid port, or a missing `idna` package, no longer makes `check_output` raise. A token right after a word and a colon, such as `id:[JES_PII_…]`, is no longer taken for a link.

### Agent hooks

- `config.json` takes an optional `model`, `jev-latest` by default.
- A `pii` guard without `entities` checks the default entities except `PERSON`, because uvx installs no spaCy model. `tool_call_mode` is a new field.
- Claude Code keeps the parts of each streamed reply per message, so two replies at once no longer clear each other's parts.
- The printed settings no longer carry a `limits` array. The guide has that text.
- Hermes transform events are allowed without a check. Hermes drops their replacement.
- The TypeScript runner blocks when jes takes over 60 seconds or prints no decision.

### Removed

- `evals/`, the evaluation harness and its protocol.
- The `jes[pii-ner]` extra and the `presidio-anonymizer` dependency.

## 1.0.5

Security fixes for the agent hooks. Print your hook settings again after upgrading so they pin this release.

- The hooks no longer read `.env` or `.env.local` from the project the agent is working in. The first value wins: the process environment, then `~/.config/jes/.env`. Before, a repository could set `TYPESAFE_BASE_URL` and receive your API key, prompts, and tool calls, and answer every check with "allow".
- Claude Code: a Bash tool result is checked on stdout and stderr. A blocked result also clears stderr.
- `jes claude-settings`, `codex-settings`, `hermes-settings`, and `runner-settings` pin the hooks to the installed release, such as `uvx jes@1.0.5 claude-hook`. An unpinned `uvx jes` ran whatever version was newest on PyPI.
- The API key, config, and session prompt files are created with mode 0600 instead of being tightened after they are written. jes no longer changes the permissions of a directory it did not create.

## 1.0.4

- Add the `jes` command. Agent hooks run with `uvx jes`.
- Add Claude Code, Codex, and Hermes hook commands, plus OpenCode, OpenClaw, and Pi plugins.
- `jes login` saves the TypeSafe API key and writes `~/.config/jes/config.json`. Hooks read that file for which guards run and their settings.
- Add `tool_safety`, a judgment of whether a tool call and its arguments are safe for the user's request. It is off in the default config, and the threshold is required.

## 1.0.3

- Point the project URLs at everafterlabs/jes.
- Raise the minimum versions of cryptography, json-repair, tiktoken, and hypothesis.

## 1.0.2

- Describe the package as "jes is a security toolkit for AI agents."

## 1.0.1

- Judge with TypeSafe through LangChain. The earlier model backends are removed.
- Add tool-call and tool-result checks.
- Add the cookbook and an offline README quickstart.
- Move the package to `src/jes` and ship only that tree in the source distribution.
- Point the docs at docs.getjes.dev.

## 1.0.0

- Freeze the public factory signatures and the v1 question ids.
- Add the README limitations. Every judgment still
  passes `threshold=`. No default or recommended backend is published.

## 0.3.0

- Release the unevaluated recipe catalog: sentiment, emotions, gibberish, bias,
  refusal, language, code, competitors, malicious URLs, relevance, factual
  consistency, reading time, and JSON. Judgment recipes require `threshold=`.
  URLReachability is not included.
- Allow transformers 5.x for the Prompt Guard 2 and Llama Guard 4 extras.

## 0.2.0

- Add Prompt Guard 2 and Llama Guard 4 adapters. Endpoints replay fixture-backed
  provider profiles; local mode accepts canned logits and completions so CI does
  not download weights.
- Add the evaluation harness: frozen candidate question bytes, private
  injection/hazard/topic/toxicity candidates, cluster-aware threshold rules, and
  a fake-backend smoke run. No measured default is published.
- Export `injection`, `indirect_injection`, `hazards`, `topics`, and `toxicity`
  with the frozen v1 bytes. Every call passes `threshold=`.
- Add unevaluated recipes for the remaining LLM Guard scanners. They are not
  part of this release's contract. Judgment recipes require `threshold=`.
  URLReachability is omitted.

## 0.1.0

- Establish the repository, packaging, test, typing, and CI baseline.
- Implement the first policy-engine contracts.
- Add folding maps and the core text transforms: invisible_text, regex, substrings, and token_limit.
- Add secrets, PII, canary, encrypted store dumps, and complete-reply restoration.
- Add System One and LiteLLM backends with fixture-replayed HTTP, capped
  responses, and `judge()` on an application-chosen threshold.
