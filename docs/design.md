# jes — design

Package name: **jes** (import `jes`). This document specifies a new library. It is not a change to Protect AI’s LLM Guard, and it moves into the new repository in Milestone 0.

Status: 0.2.0 and 0.3.0 are tagged. Published thresholds stay deferred. In this document, “v1” means the 1.x release line.

## 1. Purpose

jes is a Python library that checks text on the way into a model, text retrieved from untrusted sources, and text on the way out. It covers the practical abilities of three existing tools behind one API:

| Source | Ability jes must provide |
| --- | --- |
| LLM Guard | Reversible PII placeholders, secrets, prompt injection, toxicity, banned topics and substrings, invisible text, regex, token limits, and the long tail: language, code, competitors, gibberish, sentiment, emotion, bias, refusals, relevance, factual consistency, JSON, reading time, malicious URLs |
| Llama Prompt Guard 2 | Low-latency detection of explicit attempts to override an AI system’s instructions (injection and jailbreaks), in user text and in untrusted content |
| Llama Guard 4 | Safe/unsafe classification against the hazard list S1–S14, for prompts and model replies |

The abilities are **policies**. The models are **backends**. General backends (Laya, Jev, and capability-qualified models through LiteLLM) can answer arbitrary policy questions; Prompt Guard 2 and Llama Guard 4 run only the supported tasks they declare. The evaluation recommends backends and supplies thresholds for exact, measured configurations (section 11); the caller still chooses a backend.

## 2. Why jes

LLM Guard was archived in July 2026. The maintained alternatives solve adjacent problems:

| Project | What it is | What it leaves open |
| --- | --- | --- |
| Guardrails AI | Validators from a hub, structured output, re-asking | Each validator brings its own model or service; checks do not share a judge |
| NeMo Guardrails | Dialog and tool-flow control in Colang, with pluggable rails | An orchestration layer, not a scanner library; even a single check needs a rails configuration. jes can run inside it as a custom action. |
| LlamaFirewall | Meta’s scanners for agents: Prompt Guard 2, AlignmentCheck, CodeShield | Built around Meta’s models; no reversible PII placeholders; no general policy catalog |
| OpenAI Guardrails | A drop-in wrapper for the OpenAI client with preflight, input, and output checks: Presidio PII, moderation, and LLM-based jailbreak and custom-prompt checks | Masks PII with generic tokens such as `<EMAIL_ADDRESS>`, so values cannot be restored in the reply; LLM checks use a confidence number the chat model writes |

jes is one policy API over interchangeable judges, including cheap typed-decision models. Prompt Guard–style injection, Llama Guard’s hazard list, and reversible PII that works across turns and complete plain-text replies are first-class policies, and every decision profile with a default threshold has published evaluation numbers.

## 3. Decisions

These hold for v1 unless this document is revised.

1. **Python 3.11+.** The core depends only on `httpx`. No torch in the core.
2. **Apache-2.0.** Meta weights are never shipped. Local Meta adapters download them under the user’s Hugging Face account.
3. **One pipeline, three entry points:** `check_input`, `check_untrusted`, `check_output`. `Guard` and `AsyncGuard` share all planning and interpretation code; only request execution differs.
4. **Two policy kinds.** Transforms (exact rules and span edits) and judgments (questions for a backend). All transforms run before any judgment, in fixed phases (normalize, detect, limit). Checked text and every context value get a backend-safe projection; a context block that leaves sensitive text in place prevents backend I/O.
5. **Backends are peers** behind one protocol: System One (Laya, Jev), LiteLLM, Prompt Guard 2, Llama Guard 4. Each backend declares which tasks it can answer.
6. **Recommendations and defaults are measured.** No backend is recommended and no threshold is a default until evaluation measures the exact request and per-policy decision profiles. A decision configuration that differs from an evaluated profile needs an explicit threshold.
7. **No silent client-side truncation in strict mode.** Every character of each judgment’s logical subject—the full transformed text or each extracted `Item.text`—is included in at least one payload jes submits, or the check raises or returns a blocking, incomplete result. Actual model consumption is guaranteed only for verified non-truncating local/attested profiles; remote providers may reject or violate their contracts.
8. **Redaction stores belong to one conversation.** The caller creates a `Redactions` store per conversation and passes it to each check; a guard never holds one. Stores and results carry immutable random identities. Placeholders are unguessable, authenticated, exact-case tokens, and only tokens authorized by the context of one generation are restored.
9. **Every built-in yes/no question is phrased so true means violation.** Question text is versioned, selectable, and recorded in findings.
10. **Completion is explicit.** Results carry both an allow/block decision and whether every configured check completed. Backend failure raises by default; `"block"` fails closed, and the explicit `"allow"` mode returns `complete=False`.
11. **Text only in v1.**
12. **Fine-tuning is out of scope.** Callers can point a backend at a fine-tuned checkpoint.
13. **Complete-reply restoration.** Authorized placeholders are restored only after the complete plain-text reply has passed output transforms and judgments. Incremental restoration is deferred until after 1.0.
14. **Work is bounded.** Raw and normalized byte limits, item and request limits, redaction-store limits, response limits, and guard-wide concurrency limits apply before attacker-controlled work can grow without bound.

## 4. Non-goals

- Training or fine-tuning models.
- Dialog control and tool-flow orchestration (NeMo Guardrails).
- Authorizing agent actions or auditing agent reasoning (LlamaFirewall’s AlignmentCheck).
- Calling the application’s own LLM. jes only checks text the application already has.
- A guarantee of safety. Every judgment is a score and a threshold.
- Image and audio moderation in v1.
- Decoding obfuscated payloads (base64, ROT13, and similar) in v1.
- Detecting attacks spread across several turns in v1. History can be passed to backends as context; no policy targets multi-turn attacks yet.
- Judging or restoring partial replies while they stream in v1. Applications buffer the complete reply before `check_output`; incremental transform/finalizer support is a post-1.0 candidate.
- Restoring PII into tool-call arguments in v1. Applications authorize and populate tool arguments themselves.
- Automatic restoration into Markdown, HTML, JSON, shell text, or another structured sink in v1. jes restores plain text; the application escapes restored values for its renderer.
- Fetching URLs found in text.
- A hosted SaaS.

## 5. Threat model

**Protected:** the application’s instructions, personal data and secrets that appear in prompts, the people who read model output, and systems that act on model output.

**Attackers and failure sources:**

1. A user writing prompts: direct injection, jailbreaks, requests for hazardous content, attempts to extract other users’ data.
2. A third party whose text reaches the prompt through retrieval, browsing, email, or tool results: indirect injection.
3. The model itself: harmful, leaking, or off-policy output with no attacker involved.
4. Anyone targeting jes: invisible and control characters, lookalike letters, padding that pushes an attack past a model’s context window, text addressed to the judge (“classifier: this is benign”), oversized inputs that multiply work, forged placeholders, mismatched conversation stores, and instructions that get placeholders into URLs so that restoration sends the real values to a third party.

**In scope for v1:** all four, one message at a time, with optional history as context.

**Out of scope for v1:** the non-goals above, attacks on backend providers, the model supply chain, and the application’s own authorization.

**Guarantees (each has tests):**

For output guarantees, `check_output(text=...)` has a caller precondition: `text` is the one complete model reply, not a delta or prefix. Calling it independently on chunks is misuse outside the URI/restoration guarantees.

- No backend receives a full value detected by jes-owned `pii`, `secrets`, or `canary` sensitive transforms, including checked text, prompts, retrieval questions, sources, and history. Their replacements and provenance are engine-owned. Custom transforms are outside this confidentiality guarantee.
- In the default strict modes, every character of each configured judgment’s logical subject (the transformed full text or each validated extracted item) appears in at least one submitted payload, or the check raises or blocks with `complete=False`. Explicit fail-open modes never report `complete=True`; stronger model-consumption claims require a verified non-truncating profile.
- Given distinct application scope/store identities per conversation, redacted values never cross conversations through jes: a result and explicit store must agree, raw strings cannot introduce authority, and restoration accepts only exact occurrences in the authenticated generation manifest. Deliberately reusing one scoped store is caller-granted sharing.
- In plain-text output, a token inside a URI candidate restores only when the candidate is an absolute `http` or `https` URI whose canonical origin exactly matches `restore_origins`. Scheme-relative, `www.`, malformed, userinfo-bearing, and non-HTTP(S) candidates never qualify. jes makes no rich-renderer safety claim; applications must escape restored values before rendering.
- Raw input, normalized text, provider responses, serialized stores, planned requests, extracted items, findings, redaction stores, restored output, and guard-wide concurrency are bounded by configured limits.
- jes’s own logs and exception messages never contain checked text, context, history, or redacted values.

These guarantees cover the engine and built-ins. Custom Python can perform its own I/O, retain or log arguments, ignore deadlines, or return unsafe rewrites; jes bounds and wraps its interface where possible but cannot sandbox it and grants it no confidentiality guarantee.

The acceptance-test mapping is explicit:

- confidentiality projection → engine-owned sensitive-edit properties for `pii`, `secrets`, and `canary` across every subject/context target;
- submitted-payload coverage → chunk/item coverage properties, exact renderer tests, and strict/fail-open completion tests;
- conversation isolation → store-id mismatch, forged stamp/token, authorization-snapshot, and concurrent-store tests;
- URI restoration → complete-reply canonical-origin and parser-differential matrix tests;
- resource bounds → exact-boundary and atomic-failure tests for every cap;
- log/exception secrecy → DEBUG-log, safe-`repr`, built-in failure, and custom failure canary tests.

jes does not guarantee that any judgment is correct. Detection quality is measured and published per backend (section 11).

## 6. Architecture

```text
check_input(text, redactions=None, history=())                        → InputResult
check_untrusted(text, question=None, redactions=None)                 → ScanResult
check_output(text, prompt, sources=(), redactions=None, history=())   → ScanResult
              │
              ▼
1. transforms                    normalize → detect → limit; list order within a phase
              │                  rewrite text and context, emit findings, record redactions
              │                  fail_fast: stop here on a block
              ▼
2. judgments                     plan requests per (backend, context mode, chunk)
              │                  send only requests whose rendered form fits
              │                  Guard: sequential · AsyncGuard: concurrent
              │                  interpret each chunk, merge findings
              ▼
3. finalizers (output only)      restore exact authorized tokens under the plain-text URI policy
              │
              ▼
           result
```

Planning and interpretation do no I/O. `Guard` and `AsyncGuard` differ only in how they execute the planned backend requests.

In LLM Guard every check owns a model — more than a dozen Hugging Face models across the catalog — and 14 of its 22 output scanners are wrapper classes around input scanners. In jes a judgment is a set of questions plus an interpretation; compatible policies may share a backend request, while policy-atomic partitions preserve interpretation and threshold identity.

| Backend | Answers | Runs |
| --- | --- | --- |
| `SystemOne` | Any typed question | Laya through `laya-serve` or in-process, hosted Jev, any compatible endpoint |
| `LiteLLMJudge` | Any typed question | Any model LiteLLM can call, hosted or local (Ollama, vLLM) |
| `PromptGuard2` | `injection` only | In-process, or a provider-specific classifier endpoint with a fixture-backed profile |
| `LlamaGuard4` | `hazard.any` and `hazard.S1`–`hazard.S14` only | A fixture-backed chat endpoint (for example Groq or vLLM) or in-process |

Where checked text goes:

| Component | Destination |
| --- | --- |
| Transforms (regex, substrings, invisible text, Presidio, detect-secrets) | Local process |
| `SystemOne.in_process`, local `PromptGuard2` and `LlamaGuard4` | Local process |
| `SystemOne.local` pointed at localhost | Local machine |
| `SystemOne.hosted`, endpoint adapters, `LiteLLMJudge` | The configured host, after transforms |

The README shows this table in the quickstart.

## 7. Public API

### 7.1 Guard

```python
from jes import Guard
from jes.backends import SystemOne
from jes.policies import (
    hazards,
    indirect_injection,
    injection,
    invisible_text,
    pii,
    secrets,
    token_limit,
    topics,
)

guard = Guard(
    [
        invisible_text(),
        secrets(),
        pii(),
        token_limit(4096),
        injection(threshold=0.8),
        indirect_injection(threshold=0.8),
        hazards(threshold=0.8),
        topics(["medical advice"], threshold=0.7),
    ],
    backend=SystemOne.local(
        "http://127.0.0.1:8000",
        model="english",
        revision="pinned-checkpoint-revision",
        artifact_digest="sha256:…",
    ),
)

incoming = guard.check_input(user_text)
if not incoming.ok:
    return refuse(incoming.findings)

documents = [guard.check_untrusted(doc, question=incoming) for doc in retrieved]
safe_documents = [d for d in documents if d.ok]
context = [d.sanitized for d in safe_documents]

reply = call_model(incoming.sanitized, context)
outgoing = guard.check_output(reply, prompt=incoming, sources=safe_documents)
```

The backend and thresholds in this example are illustrative application choices, not recommendations. 0.2.0 publishes no default thresholds, so every judgment passes `threshold=`. Omitting it succeeds only for a later audited decision-profile fingerprint. The checkpoint revision identifies weights, independently of the installed Laya package version. Otherwise construction fails (section 7.4).

A multi-turn chat keeps one `Redactions` store per conversation, passes earlier results as history, and restores placeholders after checking the complete reply:

```python
from jes import Redactions

redactions = Redactions(scope=conversation_id.encode())  # one per conversation, held by caller
history = []                       # earlier InputResults and output ScanResults, oldest first

incoming = guard.check_input(user_text, redactions=redactions, history=history)

reply = call_model(
    incoming.sanitized,
    [turn.sanitized for turn in history],
)
outgoing = guard.check_output(
    reply,
    prompt=incoming,
    redactions=redactions,
    history=history,
)
if not outgoing.ok:
    return refuse(outgoing.findings)
show(outgoing.text)
history += [incoming, outgoing]
```

Applications buffer model output until `check_output` returns. Restoration produces plain text only. jes does not restore tool-call arguments; the application parses, authorizes, and populates tool inputs itself.

```python
Guard(
    policies: Sequence[Policy],
    *,
    backend: SyncBackend | None = None,
    fail_fast: bool = False,
    on_backend_error: Literal["raise", "block", "allow"] = "raise",
    max_input_bytes: int = 1_048_576,
    max_context_bytes: int = 2_097_152,
    max_context_items: int = 256,
    max_normalized_bytes: int = 2_097_152,
    max_normalized_context_bytes: int = 2_097_152,
    max_chunks: int = 32,
    max_items: int = 100,
    max_requests: int = 128,
    max_findings: int = 1_000,
    max_locations: int = 4_096,
    max_authorities: int = 2_048,
    max_metadata_bytes: int = 65_536,
    max_call_redactions: int = 1_000,
    max_call_redaction_bytes: int = 8_388_608,
    max_restorations: int = 1_000,
    max_restored_output_bytes: int = 2_097_152,
    max_response_bytes: int = 1_048_576,
    max_active_checks: int = 32,
    max_concurrency: int = 8,          # guard-wide physical backend permits
    max_worker_threads: int = 8,       # guard-wide bounded sync executor
    max_queued_work: int = 32,
    deadline_s: float | None = 30.0,       # duration; converted per check to monotonic deadline
    trace: bool = False,
)
```

`AsyncGuard` takes the same arguments, accepts sync or async backends, prefers `adecide` when a backend implements both, and runs built-in transforms and sync-only backends in worker threads so they do not block the event loop. `Guard` rejects an async-only backend at construction.

```python
class AsyncGuard:
    async def check_input(
        self, text: str, *, redactions: Redactions | None = None, history: Sequence[History] = ()
    ) -> InputResult: ...

    async def check_untrusted(
        self,
        text: str,
        *,
        question: str | InputResult | None = None,
        redactions: Redactions | None = None,
    ) -> ScanResult: ...

    async def check_output(
        self,
        text: str,
        *,
        prompt: str | InputResult,
        sources: Sequence[str | ScanResult] = (),
        redactions: Redactions | None = None,
        history: Sequence[History] = (),
    ) -> ScanResult: ...
```

`History = InputResult | ScanResult | Message`. `Guard` exposes the same signatures without `async`.

| Method | Stage | Context available to policies that use it |
| --- | --- | --- |
| `check_input(text, *, redactions=None, history=())` | `input` | Earlier turns |
| `check_untrusted(text, *, question=None, redactions=None)` | `untrusted` | The user question the text was retrieved for |
| `check_output(text, *, prompt, sources=(), redactions=None, history=())` | `output` | The prompt, retrieved sources, and earlier turns |

- `redactions` is the conversation’s store (section 7.6). Without one, `check_input` creates a fresh store and keeps it on the result, which is enough for a single turn. `check_untrusted` and `check_output` derive the store from an `InputResult` question or prompt when possible. An explicit store whose identity differs from that result raises `RedactionError`; jes never silently picks one.
- Context is sanitized before any backend sees it. `prompt`, `question`, `sources`, and `history` accept jes results or raw strings (`Message(role, text)` in history). A result fast path requires `ok=True` plus a verified stamp, exact text/findings/authority digests, stage, configuration, and store/scope identity. A blocked or incomplete result produces blocking `context_not_ok`. A failed stamp is treated as raw untrusted text: reserved token syntax is neutralized, authority is discarded, and current transforms run again.
- Raw context strings and input/untrusted subjects are untrusted: reserved placeholder syntax is neutralized before transforms. A raw output subject is the one exception because an application model must be able to return an authorized context token; its token-shaped substrings are preserved for exact validation and unauthorized ones never restore. Context transforms run for their origin stage: `untrusted` for sources, `input` for user questions and user history, and `output` for assistant history. A context block propagates to the check; it is never discarded while unrevised text continues to a backend.
- When a raw input-origin context needs reversible PII redaction, it may stage tokens for the matching conversation store. Without a store, jes uses irreversible masks. Output-origin context (for example raw assistant history) is always irreversibly masked for backend projection; it neither creates output-local edits nor adds caller-visible restoration authority.
- `history` lists earlier turns, oldest first: `InputResult`s, output `ScanResult`s, or `Message`s with role `"user"` or `"assistant"`.
- `check_output.text` must be the complete model reply in one call. jes cannot infer whether an arbitrary `str` is a prefix; chunk-by-chunk calls are outside output restoration guarantees.
- `check_output` is the only restoration entry point in v1. It restores complete plain-text model replies after output transforms and judgments. There is no generic or incremental restoration API.
- `deadline_s` is a per-check duration. At method entry jes computes `time.monotonic() + deadline_s` and passes that absolute deadline to backends and transforms. Transport, retry-wait, permit, cooperative CPU, and `AsyncGuard` wait expiry all become `DeadlineExceeded`, a `BackendError` that follows `on_backend_error`. Hard resource caps are not backend errors and always block. Python cannot forcibly stop a synchronous custom callback or third-party CPU call such as an in-flight Presidio analysis; it may finish in a bounded worker after `AsyncGuard` has discarded it. `Guard` can overrun wall time and applies the same deadline outcome when the call returns.
- The byte limits are checked before expensive transforms and after every transform. Call-local redaction count/bytes cover output-local edits and irreversible masks independently of a conversation store. `max_requests` counts physical provider calls, including retries and planned question partitions. Exceeding any resource limit blocks with `complete=False` before more work is scheduled.
- A guard holds no per-call or per-conversation state. Built-in backends and transforms are safe to call from several threads; custom backends document whether they are.

Construction raises `PolicyError` when:

- two policies share a name;
- a policy/question/option/label identifier is invalid, duplicated, or exceeds its bound;
- a judgment has no backend (neither its own nor the guard’s);
- a backend does not support the task of a question routed to it;
- a choice or score question exceeds the resolved backend capability profile;
- a judgment has no threshold (neither explicit nor an evaluated default for its exact decision profile);
- a judgment with `context="required"` is registered for a stage other than `output`;
- item mode lacks an extractor, text mode supplies item-limit settings, an item cap is invalid, its overflow label is undeclared, or item mode is combined with `whole_text`;
- a backend partition splits one policy’s questions, or one policy’s complete question set cannot fit;
- the combined questions of a planned batch leave fewer than 64 units for the text (the guard probes `headroom` on each complete group with empty text and no context);
- two `pii` policies disagree on restore settings.

### 7.2 Results

```python
Stage = Literal["input", "untrusted", "output"]
Action = Literal["flag", "redact", "block"]
ScoreKind = Literal["probability", "label", "verbalized"]
Decision = Literal["allow", "block"]

@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int

@dataclass(frozen=True, slots=True)
class ThresholdProvenance:
    source: Literal["explicit", "default"]
    block_at: float
    flag_at: float | None
    fingerprint: str
    vector_fingerprint: str
    evaluation_run: str | None

@dataclass(frozen=True, slots=True)
class Provenance:
    backend: str            # adapter, e.g. "system_one"
    model: str              # immutable model revision or artifact digest
    request_profile: str    # renderer/backend/planner fingerprint
    decision_profile: str   # policy interpretation/merge fingerprint
    prompt_version: str     # e.g. "injection.v1"
    threshold: ThresholdProvenance

@dataclass(frozen=True, slots=True)
class ScoreResult:
    value: float
    kind: ScoreKind
    confidence: float | None
    provenance: Provenance

@dataclass(frozen=True, slots=True)
class SanitizationStamp:
    result_id: str
    stage: Stage
    config_digest: str       # ordered normalize/detect configuration
    text_digest: str         # SHA-256 of exact sanitized UTF-8 bytes
    store_id: str | None
    scope_id: str | None
    decision: Decision
    complete: bool
    findings_digest: str
    authority_digest: str
    tag: str                 # full HMAC-SHA256 under this Guard instance's private stamp key

@dataclass(frozen=True, slots=True, repr=False)
class _TokenAuthority:
    token: str
    span: Span               # exact occurrence in sanitized
    origin_result: str
    reusable: bool

@dataclass(frozen=True, slots=True, repr=False)
class _TokenAuthorityManifest:
    entries: tuple[_TokenAuthority, ...]
    digest: str

LocationTarget = Literal["subject", "prompt", "question", "source", "history"]

@dataclass(frozen=True, slots=True)
class FindingLocation:
    target: LocationTarget
    span: Span
    index: int | None = None
    role: Literal["user", "assistant"] | None = None
    item_ordinal: int | None = None

@dataclass(frozen=True, slots=True)
class Finding:
    policy: str
    label: str              # e.g. "S9", "EMAIL_ADDRESS", "input_too_long"
    action: Action
    question: str | None = None
    score: ScoreResult | None = None
    locations: tuple[FindingLocation, ...] = ()
    chunks: tuple[int, ...] = ()

@dataclass(frozen=True, slots=True)
class Usage:
    backend: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    request: int             # canonical public index assigned after execution
    attempt: int             # 0 for first attempt, then retries

@dataclass(frozen=True, slots=True)
class Timings:
    transforms_ms: float
    judgments_ms: float
    finalizers_ms: float

@dataclass(frozen=True, slots=True, repr=False)
class ScanResult:
    stage: Stage
    text: str
    sanitized: str
    decision: Decision
    complete: bool
    findings: tuple[Finding, ...]
    scores: Mapping[str, ScoreResult]
    sanitization: SanitizationStamp
    _authority: _TokenAuthorityManifest = field(
        default=_EMPTY_AUTHORITY, init=False, repr=False, compare=False
    )
    usage: tuple[Usage, ...] = ()
    timings: Timings | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def ok(self) -> bool:
        return self.allowed and self.complete

@dataclass(frozen=True, slots=True, repr=False)
class InputResult(ScanResult):
    redactions: Redactions = field(kw_only=True)
```

- `decision` is `"block"` when any finding has action `block`. An explicit fail-open mode merely declines to add a block for its own failure; another blocking finding still wins. Such a path returns `complete=False`, and `ok` is intentionally false.
- `text` has policy rewrites applied. On output it contains restorations only for a complete allowed pre-finalizer result; otherwise it equals the backend-safe `sanitized` projection with tokens/markers unrestored.
- `sanitized` is the backend-safe projection, with placeholders unrestored. It is what jes uses for matching stamped context and what a caller sends to its model.
- `findings` lists only flags, redactions, and blocks. A check that passes adds no finding.
- `scores` holds the violation score and its kind and provenance for every judgment question asked, keyed `"<policy>.<question>"`, including questions that passed. The mapping is copied into an immutable view.
- Score provenance records request and decision fingerprints plus the exact applied threshold, whether it was explicit/default, evaluation run, and complete bundle-threshold-vector fingerprint. A threshold change therefore changes provenance even when policy/backend profiles do not.
- A probability is a calibrated or normalized probability, a label is 0 or 1, and a verbalized score is a number written by the model. All use the range [0, 1], but jes does not call labels or verbalized numbers probabilities.
- `FindingLocation` carries offsets and no text. Its span is in the original string identified by its target and index: the check subject, prompt, retrieval question, a source, or a history message. An item judgment maps back to `target="subject"` and carries its engine ordinal separately. The engine maps spans through edits; a merged judgment finding unions locations and chunk ids in deterministic order.
- A guard owns a random private stamp key. Its full-length HMAC authenticates the result id, stage, exact sanitized bytes, store/scope ids, ordered context-transform configuration, decision, completion, complete findings digest, and private token-authority digest. Authority is derived from the manifest, never a caller-set boolean. The fast path accepts only a complete allowed result whose HMAC and every digest verify. A blocked/incomplete result supplied as context produces `context_not_ok`; a copied or altered stamp, another guard instance, and a custom transform without a stable `fingerprint` force re-sanitization with no inherited authority.
- Stamp verification is ordered to bound hostile public result objects: validate fixed field types/lengths and constant-time HMAC first without traversing findings or authority entries; reject oversized tuple counts/metadata/locations/authorities next; only then recompute text, findings, and authority digests. A failed early check falls back to untrusted text without walking attacker-sized structures.
- The public result constructor always installs `_EMPTY_AUTHORITY`; only an engine-private factory can attach a non-empty manifest and valid stamp. User-constructed results remain usable as untrusted text containers but can never take the fast path or grant restoration authority.
- `InputResult.redactions` is the store the check used (section 7.6).

### 7.3 Policies

```python
ContextTarget = Literal["subject", "context"]

class RedactionView(Protocol):
    id: str
    scope_id: str
    size: int

@dataclass(frozen=True, slots=True)
class CallContext:
    call_stage: Stage
    origin_stage: Stage
    target: ContextTarget
    redactions: RedactionView | None
    deadline: float | None             # absolute time.monotonic value

@dataclass(frozen=True, slots=True, repr=False)
class TransformEdit:
    start: int
    end: int
    replacement: str

@dataclass(frozen=True, slots=True)
class TransformFinding:
    label: str
    action: Action
    spans: tuple[Span, ...] = ()       # indices in this transform's input

@dataclass(frozen=True, slots=True, repr=False)
class TransformOutcome:
    text: str
    findings: tuple[TransformFinding, ...] = ()
    edits: tuple[TransformEdit, ...] = ()

_SensitiveMode = Literal[
    "conversation_token",
    "irreversible",
    "mask_all",
    "mask_partial",
    "hmac",
    "output_local",
]

@dataclass(frozen=True, slots=True, repr=False)
class _SensitiveEdit:
    span: Span
    entity: str
    mode: _SensitiveMode
    action: Action

@dataclass(frozen=True, slots=True, repr=False)
class _SensitiveOutcome:
    edits: tuple[_SensitiveEdit, ...]
    findings: tuple[TransformFinding, ...] = ()

class _SensitiveTransform(Protocol):
    def _apply_sensitive(
        self, text: str, call: CallContext, transaction: _RedactionTransaction
    ) -> _SensitiveOutcome: ...

@dataclass(frozen=True, slots=True, repr=False)
class Item:
    text: str
    span: Span                              # indices in the current transformed subject

@dataclass(frozen=True, slots=True)
class InterpretationContext:
    provenance: Provenance
    chunk: int
    item_ordinal: int | None
    location: FindingLocation

@dataclass(frozen=True, slots=True)
class JudgmentOutcome:
    findings: tuple[Finding, ...]
    scores: Mapping[str, ScoreResult]

class TransformPolicy(Protocol):
    name: str
    labels: frozenset[str]                  # every label the policy may emit
    stages: frozenset[Stage]
    phase: Literal["normalize", "detect", "limit"]
    fingerprint: str | None                # stable config id; None disables stamped fast path

    def apply(self, text: str, call: CallContext) -> TransformOutcome: ...

class JudgmentPolicy(Protocol):
    kind: str                                   # e.g. "injection"; key for default thresholds
    name: str                                   # defaults to kind; unique within a guard
    labels: frozenset[str]                      # every non-engine label it may emit
    stages: frozenset[Stage]
    subject_mode: Literal["text", "items"]
    max_policy_items: int | None
    item_overflow_label: str
    on_items_overflow: Literal["block", "allow"]
    context: Literal["none", "optional", "required"]
    on_context_overflow: Literal["block", "allow"]  # allow is explicit incomplete fail-open
    sources: bool                               # required context also includes check_output's sources
    whole_text: bool                            # never chunked
    on_text_overflow: Literal["block", "allow"] # used only when whole_text is true
    backend: Backend | None                     # overrides the guard's backend
    threshold: Threshold | None                 # None: use the evaluated default
    version: str                                # e.g. "injection.v1"
    interpretation_version: str                 # versioned score/finding aggregation logic

    def questions(self, tasks: frozenset[str] | None) -> Mapping[str, Question]: ...
    def items(self, text: str) -> Iterable[Item]: ...
    def interpret(
        self,
        answers: Mapping[str, Answer],
        threshold: Threshold,
        context: InterpretationContext,
    ) -> JudgmentOutcome: ...

Policy = TransformPolicy | JudgmentPolicy
```

- Transforms run by phase: `normalize` (`invisible_text`), then `detect` (the default: `secrets`, `pii`, `regex`, `substrings`, and custom transforms), then `limit` (`token_limit`, `reading_time`). List order applies within a phase, so a detector never sees text a normalizer has not cleaned, and truncation never cuts text before redaction.
- `CallContext` gives a custom transform `call_stage`, `origin_stage`, `target` (`"subject"` or `"context"`), an absolute monotonic deadline, and a metadata-only `RedactionView`. It exposes no raw value lookup, token issuance, or mutation. jes-owned sensitive transforms receive a separate private transaction. Deterministic staged tokens may appear in backend-safe payloads but gain restoration authority only when the transaction commits. The engine commits once, immediately before returning a complete allowed result; every block, incomplete path, exception, or limit failure discards it and leaves the store unchanged.
- `TransformEdit` and `TransformFinding.spans` use Python string indices into that transform’s input. The policy does not construct final locations: the engine knows the subject/context identity and accumulated offset map, and decorates each local finding with policy name and original `FindingLocation`. Edits are sorted, non-overlapping, and may insert with `start == end`; applying them must reproduce `TransformOutcome.text` exactly. Malformed custom outcomes raise `PolicyExecutionError` before backend I/O.
- Custom transforms may return ordinary edits, flags, redactions, and blocks, but jes makes no confidentiality claim about them.
- `pii`, `secrets`, and `canary` are registered jes-owned `_SensitiveTransform`s in the `detect` phase. The engine invokes their private `_apply_sensitive` channel instead of public `apply`; it returns `_SensitiveOutcome` directives, never replacement strings. Registration is a closed internal table, not structural runtime matching, so a third party cannot opt in by defining a similarly named method. The engine reads any HMAC key from trusted built-in configuration, applies a replacement from the closed mode set, records provenance, and verifies that the complete detected value is absent. A blocking sensitive edit is still replaced before the block result. Private types are not importable or implementable by third-party policies in v1.
- Sensitive edits in one outcome must be non-overlapping. A built-in detector resolves recognizer overlaps deterministically by recognizer score, then longer span, then configured entity order; the engine rejects any unresolved overlap. That rule and recognizer ordering are part of the request profile.
- An `output_local` edit stores its original value only in a non-copyable private finalizer map keyed by the exact marker occurrence. Every later transform edit is composed against that map. Any overlap, deletion, split, movement, or truncation permanently tombstones the entry; only an unchanged marker at its mapped position can restore after judgments.
- `InterpretationContext` supplies immutable provenance and location, so policy code can build complete `ScoreResult` and `Finding` values without mutable backend state. `JudgmentOutcome` contains one score for every question.
- The guard calls `questions` once, at construction, with the resolved backend’s declared tasks (`None` for general backends). It rejects an empty mapping. Questions never depend on checked text, so attacker text never enters instructions.
- Policy names, question ids, option labels, and declared finding labels must match `[A-Za-z][A-Za-z0-9_-]{0,63}` and are copied at construction. A transform or interpretation that returns an undeclared/dynamic label fails with `PolicyExecutionError`; checked text can never become metadata. Engine-owned limit/error labels are a separate closed enum. `max_metadata_bytes` and `max_locations` cap aggregate result metadata independently of text limits.
- `subject_mode` is immutable. In `"text"` mode the engine never calls `items`; in `"items"` mode `items` returns an iterable (empty means no subjects, never “fall back to text”). `max_policy_items=None` means the guard-wide cap. A finite policy cap must be no greater than `guard.max_items`; when it is lower, policy overflow emits its declared label/action, while reaching the guard cap always emits strict blocking `too_many_items`. The engine consumes only the applicable cap plus one sentinel item. `Item(text, span)` must match the transformed subject. The engine assigns a unique ordinal, maps to original-subject locations, and rejects item mode with `whole_text=True`.
- The guard resolves each judgment’s request and decision profile separately for every registered stage. An explicit `threshold` applies to all stages; without one, every stage must have its own default entry. At call time the engine passes the active stage’s resolved threshold and provenance to `interpret`. Policy objects and copied question mappings stay immutable.

Custom judgments use the same machinery as built-ins:

```python
from jes.policies import judge
from jes.questions import YesNo

refund = judge(
    "refund_request",
    YesNo("The text asks for money back."),
    threshold=0.8,
    stages=("input",),
)
```

`judge` has the following complete factory surface:

```python
judge(
    name: str,
    questions: Question | Mapping[str, Question],
    *,
    threshold: Threshold | float,
    violating: Collection[str] | Mapping[str, Collection[str]] | None = None,
    violation_level: int | Mapping[str, int] | None = None,
    stages: Iterable[Stage] = ("input", "output"),
    context: Literal["none", "optional", "required"] = "none",
    on_context_overflow: Literal["block", "allow"] = "block",
    sources: bool = False,
    whole_text: bool = False,
    on_text_overflow: Literal["block", "allow"] = "block",
    items: Callable[[str], Iterable[Item]] | None = None,
    max_policy_items: int | None = None,
    item_overflow_label: str = "too_many_items",
    on_items_overflow: Literal["block", "allow"] = "block",
    backend: Backend | None = None,
    version: str = "v1",
    interpretation_version: str = "v1",
) -> JudgmentPolicy
```

`threshold` is always required. `violating` applies to `Choice`; `violation_level` applies to `Score`, and each must cover exactly the applicable question ids. `items=None` creates text mode; supplying an extractor creates item mode and is incompatible with `whole_text=True`. Custom questions use task `"custom"`, which only general backends answer.


### 7.4 Questions, scores, and thresholds

```python
@dataclass(frozen=True, slots=True)
class YesNo:
    instructions: str
    true: str | None = None
    false: str | None = None
    task: str = "custom"

@dataclass(frozen=True, slots=True)
class Choice:
    instructions: str
    options: Mapping[str, str | None]    # label → description; at least 2
    task: str = "custom"

@dataclass(frozen=True, slots=True)
class Score:
    instructions: str
    levels: tuple[str, ...]              # 2 to 10, ordered low to high
    task: str = "custom"

@dataclass(frozen=True, slots=True)
class YesNoAnswer:
    score: float
    kind: ScoreKind
    confidence: float | None = None

@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    scores: Mapping[str, float]
    kind: ScoreKind
    confidence: float | None = None

@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    scores: tuple[float, ...]
    kind: ScoreKind
    confidence: float | None = None

Question = YesNo | Choice | Score
Answer = YesNoAnswer | ChoiceAnswer | ScoreAnswer
```

- The top choice and the expected score level are computed properties, not stored fields.
- Scores are finite numbers in [0, 1]. Choice keys must equal the options, and choice and ordered-level scores must sum to 1 ± 1e-3. A `label` answer is one-hot. A `probability` answer is a backend probability or a documented normalization over candidates. A `verbalized` answer is a distribution written by the model. Violations raise `BackendError`.
- Task ids: `custom`, `injection`, `indirect_injection`, `hazard.any`, `hazard.S1` through `hazard.S14`, `topic`, `toxicity.<label>`. General backends (`tasks=None`) answer any task from the instruction text. Fixed-task backends answer only the tasks they list and ignore instruction text.
- `YesNo` is the name System One calls `noul`; the System One adapter maps it on the wire.

**Violation score** is the number compared to thresholds and reported in `scores`:

| Question | Violation score |
| --- | --- |
| YesNo | Score of true |
| Choice | Sum of the scores of the options the policy marks as violating |
| Score | Sum of the scores of levels at or above the policy’s violation level |

Using the total matters: text split across two violating options (0.45 and 0.45) has a violation score of 0.9, even though neither option wins alone.

```python
@dataclass(frozen=True, slots=True)
class Threshold:
    block_at: float
    flag_at: float | None = None
```

- A violation score at or above `block_at` blocks.
- If `flag_at` is set, a violation score from `flag_at` up to `block_at` flags.
- `threshold=0.8` means `Threshold(block_at=0.8)`.
- Construction rejects booleans, NaN, infinities, and values outside `0 <= flag_at <= block_at <= 1`.
- jes never softens a decision because a backend is unsure: an attacker can make a judge unsure. Escalating uncertain answers to a second backend is a candidate feature after 1.0.

**Request and decision profiles.** Construction first computes a `RequestProfile` for every state-independent question partition. It fingerprints ordered subject/context transforms; engine and HTTP-serialization versions; exact behavior-affecting dependency/model/data assets; renderer/planner versions; backend capabilities/profile/attestation; stage/context/source/subject modes; exact partition; chunk rules; total-window budget; template/tokenizer; and generation settings. It describes what is sent, not a policy threshold.

For each policy in that request, construction computes a `DecisionProfile`. It adds policy kind/version/subset, stage, question and answer-kind schema, interpretation version, and merge version to the request-profile hash. `jes/policies/defaults.py`, generated from evaluation calibration data, maps each decision-profile fingerprint—not a shared request fingerprint—to one threshold and evaluation run id. A bundled request therefore resolves a threshold vector keyed by policy name.

Any decision-profile mismatch requires an explicit threshold. That includes a floating model, different policy subset, changed transforms/question partition/chunk rules, provider/template/scorer, or interpretation/merge version. Recipes and `judge()` always require explicit thresholds. Examples pin immutable weight revisions or digests (`laya:english@hf:<commit>#sha256:<digest>`, `jev-1.13.0`, or an Ollama digest); aliases such as `jev-latest` never match.

### 7.5 Errors

| Exception | Raised | Examples |
| --- | --- | --- |
| `PolicyError` | At construction | Duplicate names, missing threshold, unsupported task, too many options, questions that leave too little room for the text |
| `PolicyExecutionError` | During a check, before or after backend I/O | Invalid transform edits/items, undeclared labels, or malformed `interpret()` output |
| `RedactionError` | Before backend I/O or restoration | Result/store scope mismatch; invalid token authority object; serialized blob scope/authentication failure |
| `BackendError` | During a check | Transport failure, non-2xx response, missing or mistyped answer, invalid scores, missing required candidate log-probabilities, cooperative deadline exceeded |
| `DeadlineExceeded` | During a check; subclass of `BackendError` | Transport, retry wait, permit wait, cooperative CPU, or async wait reaches the check deadline |

`BackendError` messages carry the backend name, status code, and question ids, never request or response bodies (section 14).

`PolicyExecutionError` always raises because it indicates a policy implementation/contract bug; `on_backend_error` does not apply. The engine still closes transactions and releases admission in `finally`.

Conditions that depend on input size, content, context state, or admission are findings, not exceptions: `invalid_unicode`, `input_too_long`, `context_too_long`, `context_not_ok`, `text_too_long`, `response_too_large`, `guard_busy`, `too_many_context_items`, `too_many_items`, `too_many_requests`, `too_many_findings`, `too_many_locations`, `too_many_authorities`, `metadata_too_large`, `redaction_limit_exceeded`, `restored_output_too_large`, `placeholder_in_input`, `invalid_placeholder`, and `placeholder_in_url`. They set `complete=False` only when a configured transform, judgment, or finalizer did not finish; successfully neutralized placeholder findings are complete as specified below.

`placeholder_in_input` and `invalid_placeholder` have `action="flag"` and leave `complete=True` after successful neutralization. `placeholder_in_url` blocks by default or flags under its explicit allow mode; either way the finalizer completed, the token remains unrestored, and completeness stays true.

`on_backend_error`:

- `"raise"` (default): the check raises `BackendError`.
- `"block"`: fail closed. The result gets a finding `label="backend_error"`, `action="block"`, and `complete=False`.
- `"allow"`: explicit fail open. Judgments from the failed backend are missing from `scores`, a warning is logged, no block is added for that failure, `complete=False`, and a `backend_error` flag is present. `decision` is `"allow"` only if no unrelated finding blocks; `ok` is false either way.

For `DeadlineExceeded`, block/allow modes use `label="deadline_exceeded"` instead of `backend_error`. The action and completion semantics are otherwise identical.

### 7.6 Redaction stores

```python
class Redactions:
    """Authenticated placeholders and values for one conversation."""

    def __init__(
        self,
        *,
        scope: bytes | None = None,                 # non-empty application scope; None → random in-memory scope
        max_scope_bytes: int = 1_024,
        max_entries: int = 1_000,
        max_value_bytes: int = 65_536,
        max_bytes: int = 8_388_608,
    ) -> None: ...

    @property
    def id(self) -> str: ...                         # random public store identity
    @property
    def scope_id(self) -> str: ...                   # digest, never the scope bytes
    def __len__(self) -> int: ...
    def dumps(
        self, key: bytes, *, associated_data: bytes, max_associated_data_bytes: int = 4_096
    ) -> bytes: ...
    @classmethod
    def loads(
        cls,
        blob: bytes,
        key: bytes,
        *,
        scope: bytes,
        associated_data: bytes,
        max_scope_bytes: int = 1_024,
        max_associated_data_bytes: int = 4_096,
        max_entries: int = 1_000,
        max_value_bytes: int = 65_536,
        max_plaintext_bytes: int = 8_388_608,
        max_blob_bytes: int = 8_392_704,             # plaintext cap + fixed 4 KiB envelope allowance
    ) -> Redactions: ...
```

- A conversation uses one store. The caller creates it with a non-empty tenant/application/conversation scope and passes it to every check, or, for an in-memory single turn, lets `check_input` create a random scope. A guard never holds one. Every store has a random 128-bit public id, a scope digest, and a separate 256-bit secret. Isolation guarantees assume distinct scope/store identities; deliberately sharing one store is explicit caller authority.
- A token has the fixed-shape ASCII form `[JES_v1_PII_{ID}_{TAG}]`. `ID` is the first 128 bits of HMAC-SHA256 under the store secret over `"id" || exact value`; `TAG` is the first 128 bits over `"tag" || version || store id || ID`. Both are base64url without padding. The store collision-checks and maps the complete exact token to the value. The keyed deterministic id is unguessable without the store secret, gives one token per exact value regardless of recognizer label or concurrency, and avoids provisional-token remapping.
- There are two disjoint restoration authorities. A **context token** restores only when its HMAC validates, it exists in the selected store, its exact case-sensitive spelling occurs in the raw model reply, and an authenticated authority manifest authorizes that occurrence for this generation. An **output-local edit** replaces newly detected output PII with an irreversible `[REDACTED_{ENTITY}]` marker in the backend/result `sanitized` text and keeps the original value only in a private call-local span map. The complete-output finalizer may put that value into `text` only at the creating span in that same check. It is never inserted into `Redactions`, is non-reusable, and need not occur as a token in the raw reply.
- Raw user, source, and `Message` strings never carry restoration authority. Any substring matching reserved `\[JES_v[0-9]+_...\]` syntax is rewritten as `⟦JES_LITERAL:{base64url(UTF-8 original)}⟧` before transforms and gets `placeholder_in_input`; this form contains no ASCII `[JES_` prefix and is never decoded automatically. The raw model-reply subject is exempt from rewriting so its exact authorized tokens can be validated. Copying a valid token through any other raw channel does not bypass the rule.
- Token authority is occurrence-based, not inferred by scanning a stamped string. Engine-issued conversation-token edits create stamped context occurrences. From the exact context results actually supplied to the application model, jes derives a private generation token set. An exact token spelling found in the raw output receives authority at that raw-output span only when its token identity is in that set. After every transform, overlapping authority is tombstoned and every untracked token-shaped occurrence—including one synthesized by Unicode normalization—is neutralized before backend I/O. Its literal spelling may remain in final `text`, but `sanitized` contains the neutralized form, the result manifest grants no authority, and `invalid_placeholder` is recorded.
- Values leave a store only through the complete-output finalizer and encrypted serialization. There is no API that restores arbitrary application, tool, or partial-stream text. `repr` shows only the store id and counts. `pickle`, JSON serialization, `copy.copy`, and `copy.deepcopy` raise; consequently `dataclasses.asdict` of a result containing a store also fails instead of copying secrets.
- Store byte accounting is the exact length of its canonical plaintext serialization: fixed header, length prefixes, UTF-8 tokens and values, and metadata. Under the store lock, a prospective insertion validates `max_entries`, one value against `max_value_bytes`, and resulting total bytes before mutating either direction of the map. An overflow is atomic: jes emits `redaction_limit_exceeded`, blocks with `complete=False`, exposes no unmatched value to a backend, and leaves the store unchanged.
- Scope and associated data are non-empty and bounded before hashing or allocation. `dumps` requires a 32-byte key from `[crypto]`; its canonical format includes a fixed-size envelope/header plus bounded plaintext (version, scope digest, store id, limits, entries, secret). AES-256-GCM uses a fresh 96-bit nonce and authenticates every header plus associated data. `loads` rejects oversized scope/AAD/blob before decryption, then refuses embedded entry/value/plaintext limits above caller ceilings and rejects unknown versions, scope mismatch, or authentication failure. The default blob cap is the plaintext cap plus a fixed 4 KiB envelope allowance. Key rotation and replay counters are caller responsibilities.
- Store mutations take a short internal lock; jes never holds it across a transform or backend I/O. One call-scoped transaction stages every allocation, performs a read-only prospective cap/collision check before backend scheduling, then commits only for a complete allowed result. Final commit rechecks for concurrent changes. Concurrent insertion of the same value computes the same token; an impossible HMAC-id collision with another value blocks atomically. A closed or timed-out transaction rejects late operations. Entries are immutable after insertion, so a check snapshots its authorized map before releasing the lock. Stores, transactions, authority manifests/snapshots, and output-local maps have redacted `repr` and reject pickle, JSON, copy, and deepcopy.

## 8. Backends

### 8.1 Protocol

```python
@dataclass(frozen=True, slots=True, repr=False)
class Message:
    role: Literal["user", "assistant"]
    text: str

@dataclass(frozen=True, slots=True, repr=False)
class State:
    stage: Stage
    text: str
    prompt: str | None = None       # output stage
    question: str | None = None     # untrusted stage
    sources: tuple[str, ...] = ()   # output stage, policies with sources=True
    history: tuple[Message, ...] = ()

@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    tasks: frozenset[str] | None
    max_options: int | None
    max_attempts: int                       # total physical attempts, initial call included
    score_kinds: Mapping[str, frozenset[ScoreKind]]  # task id or "*" → supported kinds
    budget_fidelity: Literal["exact", "conservative"]

@dataclass(frozen=True, slots=True)
class BackendProfile:
    adapter: str
    adapter_version: str
    scorer_version: str
    parser_version: str
    provider: str
    provider_profile: str
    model: str
    revision: str | None
    artifact_digest: str | None
    tokenizer_revision: str | None
    template_revision: str | None
    mode: str
    generation_settings: Mapping[str, str | int | float | bool | None]
    context_window_tokens: int | None
    max_request_bytes: int | None
    output_reserve: int
    budget_attestation: str | None
    dependency_versions: Mapping[str, str]

@dataclass(frozen=True, slots=True, repr=False)
class _ProfileAttestation:
    profile_fingerprint: str
    source: Literal["installed_artifact", "provider_registry", "signed_deployment"]
    evidence_digest: str
    signature: str

@dataclass(frozen=True, slots=True)
class RequestProfile:
    backend: BackendProfile
    attestation_digest: str | None
    transform_digest: str
    engine_dependency_digest: str
    transform_asset_digest: str
    renderer_version: str
    planner_version: str
    stage: Stage
    context_mode: Literal["none", "optional", "required"]
    sources: bool
    subject_mode: Literal["text", "items"]
    question_partition_digest: str
    chunk_config_digest: str
    budget_digest: str
    fingerprint: str

@dataclass(frozen=True, slots=True)
class DecisionProfile:
    request_profile: str
    stage: Stage
    policy_kind: str
    policy_version: str
    subset_digest: str
    interpretation_version: str
    merge_version: str
    score_kinds_digest: str
    fingerprint: str

class RequestPermit(Protocol):
    permit_id: str                    # opaque admission id visible only to backend/engine
    deadline: float | None

class RequestBudget(Protocol):
    def acquire(
        self, deadline: float | None, *, logical_index: int, attempt: int
    ) -> ContextManager[RequestPermit]: ...
    def aacquire(
        self, deadline: float | None, *, logical_index: int, attempt: int
    ) -> AsyncContextManager[RequestPermit]: ...

@dataclass(frozen=True, slots=True)
class RequestContext:
    logical_index: int
    deadline: float | None             # absolute time.monotonic value
    budget: RequestBudget              # shared by every retry and split in this check
    max_response_bytes: int

@dataclass(frozen=True, slots=True)
class BackendUsage:
    permit_id: str
    attempt: int
    input_tokens: int | None
    output_tokens: int | None

@dataclass(frozen=True, slots=True)
class BackendResult:
    answers: Mapping[str, Answer]
    usage: tuple[BackendUsage, ...] = ()

class Backend(Protocol):
    name: str
    capabilities: BackendCapabilities
    profile: BackendProfile

    def count_units(self, text: str) -> int:
        """Tokenizer units for exact profiles, UTF-8 bytes for conservative profiles."""

    def partition_questions(
        self, questions: Mapping[str, Question]
    ) -> tuple[Mapping[str, Question], ...]:
        """Pure deterministic, state-independent split performed at construction."""

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        """Units left after rendering the submitted payload, escaping and output reserve
        included. Negative: it does not fit. None: no declared model limit."""

class SyncBackend(Backend, Protocol):
    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult: ...

class AsyncBackend(Backend, Protocol):
    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult: ...
```

- Each backend renders `State` for its model. The engine never builds prompt strings.
- `max_attempts >= 1` is immutable, fingerprinted in canonical capabilities, and includes the initial call. `partition_questions` is pure and depends only on the immutable backend profile/static schema. A partition may split only between policy namespaces; every question for one policy stays together. If one policy’s set cannot fit, construction raises. Backends expose every deterministic split at construction and never exceed `max_attempts` inside `decide`/`adecide`.
- `decide` returns exactly one answer per question id plus immutable private usage for every physical provider request. Usage never lives in mutable backend “last response” state.
- Built-in HTTP backends implement both `decide` and `adecide` on top of shared, pure request and response code. In-process backends implement `decide`, and `AsyncGuard` runs it in a worker thread.
- The engine sends a request only when its `headroom` is 0 or more. Headroom is recomputed on the complete combined question batch after rendering, so text that grows when escaped (JSON turns one control character into six) cannot push the submitted payload past the adapter’s declared limit.
- Before every provider call—including a retry—the backend enters `request.budget.acquire(..., logical_index=..., attempt=...)` or `aacquire`. `RequestPermit` and `BackendUsage` are public backend-extension types, but their opaque permit id is never copied into `ScanResult`. The engine verifies it, then assigns canonical public `Usage.request` indices by logical/attempt slot.
- `RequestContext.deadline` is cooperative for custom sync code. Built-in transports derive every connect/read/write timeout and retry sleep from the remaining time. `AsyncGuard` may stop awaiting a custom sync callback at the deadline, but the callback can continue in its worker; the docs make that limitation explicit and never claim thread cancellation.
- Built-in HTTP adapters stream and count success and error bodies and stop at `max_response_bytes` before buffering or parsing; overflow emits `response_too_large` and is a strict resource block. Custom backends receive the cap in `RequestContext` and must pass its contract test to claim bounded behavior.
- The mappings in capabilities and profiles are defensively copied into immutable views. `BackendProfile` has no caller-settable verification flag. Closed built-in adapter factories register `_ProfileAttestation` in an engine-owned identity registry that public/custom objects cannot populate: verified installed bytes, a shipped immutable provider registry entry, or a deployment manifest signed by a configured trusted key and matched by handshake. The registry supplies `RequestProfile.attestation_digest`. Custom backends are explicit-threshold-only in v1; caller labels or private-looking attributes never qualify.
- `jes.testing.check_backend_contract(backend)` checks deterministic partitioning, answer coverage, permit use for retries/splits, immutable per-call usage, task-specific score kinds, capability limits, deterministic rendering, exact or conservative headroom on control characters/emoji/quotes, and deadline cooperation. Every built-in backend passes it.

### 8.2 Token budgets

- A constructor accepts a total `context_window_tokens` or an attested provider total, never a “text budget.” `headroom` derives usable subject space by rendering state, questions, answer format, escaping, templates and special tokens, then reserving completion tokens.
- An `exact` profile uses a pinned tokenizer and pinned chat template: the `[tokenizers]` extra for Laya checkpoints, or the model’s own pinned tokenizer for in-process adapters.
- A `conservative` profile counts UTF-8 bytes after rendering and compares them with an explicit conservative payload budget. It guarantees only what jes submits; a remote provider may add hidden template tokens and reject the request. Such a profile cannot receive a default threshold unless that exact provider profile was live-tested and evaluated.
- A verified remote profile also attests that server-side truncation is disabled and over-limit requests fail rather than silently truncate. Without that evidence, the profile can guarantee submitted-payload coverage only and is ineligible for the stronger model-consumption claim.
- jes always sends an explicit Laya checkpoint name, so `laya-serve`’s router never picks a checkpoint with a smaller window.
- Any model not in the table below requires either a pinned tokenizer/template plus total context profile or a conservative `max_request_bytes=` contract. Output reserve is mandatory.

| Backend model | Total context / nominal state share | Basis |
| --- | --- | --- |
| Laya `english` | 512 total / about 320 state | 192-token question head; exact adapter headroom also counts model special tokens |
| Laya `multilingual` | 1,024 total / about 768 state | 256-token question head; exact adapter headroom also counts model special tokens |
| Laya `typed-decisions` | 1,024 total / about 768 state | 256-token question head; exact adapter headroom also counts model special tokens |
| Jev `jev-1.13.0` | 32,000 tokens minus the longest question; state plus all questions at most 64,000 | TypeSafe’s documented limits |
| Prompt Guard 2 (22M, 86M) | 512 total | Model card; classifier special tokens reduce subject headroom |
| Llama Guard 4 | Provider/model total (131,072 on Groq) | Conversation template, category text, special tokens, and completion reserve reduce subject headroom |

The manual live-test job checks each entry against a real server.

### 8.3 System One

```python
SystemOne.local(
    base_url="http://127.0.0.1:8000",
    model="english",
    provider_profile="laya-serve.v1",
    revision=None,
    artifact_digest=None,
    deployment_manifest=None,
    trusted_manifest_keys=(),
    api_key=None,
    context_window_tokens=None,
    max_request_bytes=None,
    timeout_s=5.0,
)
SystemOne.hosted(
    model="jev-1.13.0",
    base_url="https://api.typesafe.ai",
    api_key=None,
    provider_profile="typesafe.jev.v1",
    timeout_s=10.0,
)
SystemOne.in_process(
    checkpoint="english",
    provider_profile="laya.in_process.v1",
    revision=None,
    artifact_digest=None,
)   # extra [laya]
```

- `local` and `hosted` share one HTTP client. `hosted` reads `TYPESAFE_API_KEY` when `api_key` is `None` and sends `Authorization: Bearer`. `local` sends a key only when given one (matching `LAYA_API_KEY` on the server).
- `in_process` calls the `laya` package directly, using the same request and response mapping.
- The backend profile records every field in section 8.1. `in_process` hashes and verifies installed checkpoint metadata. Stock `laya-serve` reports a serving alias/model but not an artifact digest, so `SystemOne.local` has no attestation unless a controlled deployment manifest covers laya/laya-serve variants/versions, checkpoint commit/digest, tokenizer/config, total/head budgets, aliases, truncation behavior, and server configuration; verifies against `trusted_manifest_keys`; and matches a live handshake. Caller labels or self-signed manifests without a configured trust key never qualify.
- Known verified Laya profiles fill `context_window_tokens`; unknown models require a pinned tokenizer/template with a total context or `max_request_bytes`. Supplying incompatible budget modes is an error.
- The adapter renders the state to a string itself and sends that string: the text alone when only `text` is present, otherwise compact JSON (`ensure_ascii=False`) with the keys `text`, `prompt`, `question`, `sources`, and `history`. Headroom is counted on that string, so the model sees exactly what was measured.
- Built-in questions refer to those keys by name, so key names are part of each prompt version.
- Request: `{"model", "state", "questions": {id: {"type", "instructions", "criteria"}}}`. A `YesNo` becomes type `noul`, with criteria describing `true` and `false`; choice criteria map option labels to descriptions; score criteria list the levels. Hosted Jev posts that body to `/v1/systemone`. Local System One posts it to `/v1/decide`.
- Response: `answers[id]` carries the yes/no probability, choice probabilities, or ordered-level probabilities plus optional confidence. Hosted Jev names those fields `noul` and `probabilities`. Local fixtures name them `score` and `scores`. The parser accepts both. The verified provider profile—not an untrusted response field—assigns `kind="probability"`. The parser returns `BackendResult`; each permitted provider call contributes private usage that the engine validates and converts to canonical public `Usage`.
- `max_attempts=3`: initial call plus up to 2 retries on 429/5xx, with exponential backoff and jitter within the deadline. No retry on other 4xx. Deadline-derived timeouts raise `DeadlineExceeded`.
- `partition_questions` splits a Jev logical batch before chunking when its questions would exceed the 64,000-token request limit. Each planned partition is headroom-checked and each attempt acquires its own request permit.
- System One answers are `kind="probability"` when the verified wire profile supplies calibrated distributions. Laya’s multilingual checkpoint ships without fitted calibration; evaluation measures it separately.

### 8.4 LiteLLM judge

```python
LiteLLMJudge(
    "ollama/llama3.1:8b",
    mode="logprobs",
    provider="ollama",
    provider_profile="ollama.chat.v1",
    revision="sha256:…",
    tokenizer_revision="llama3.1@…",
    template_revision="ollama-chat@…",
    context_window_tokens=8192,                   # optional application cap on the total window
    max_request_bytes=None,
    top_logprobs=5,
    timeout_s=10.0,
    num_retries=0,
    fallbacks=None,
    cache=False,
    http_client=None,                             # jes injects a bounded client by default
    completion_fn=None,
)
```

- The system message holds the questions and answer format. The user message holds only rendered state fields. A delimiter is derived deterministically from the canonical request hash and a counter; the adapter increments until the delimiter is absent from every field. This is reproducible for caching and still prevents text from closing its own block.
- `mode` is required and fixed at construction. It is part of the backend profile; jes never switches modes per response.
- **`mode="logprobs"`**: the provider capability profile must state its supported `top_logprobs` range and supply a pinned tokenizer. Construction verifies that every answer label is one token in every accepted leading-space form and that the complete candidate set fits the provider limit. LiteLLM’s portable interface documents at most five; a provider-specific profile may document another verified limit. There is no universal `max_options=20`.
- At runtime every candidate must be present at its answer position. The adapter sums documented token variants and renormalizes over the complete candidate set; the pre-normalization candidate mass is `confidence`. Missing candidates, a response without log-probabilities, or mass below 0.5 raises `BackendError`. This score is a candidate-normalized probability, not an unconditional model probability, and is calibrated only under its exact profile.
- **`mode="verbalized"`**: the model writes a score distribution for each question. Answers have `kind="verbalized"`, and evaluation reports them separately. Use it for providers that return no log-probabilities (Groq, Anthropic) and for models that reason before answering, such as gpt-oss-safeguard.
- `max_attempts=2`: temperature 0 where accepted; strict parsing; one jes-scheduled retry for unknown/missing/malformed answers, then `BackendError`.
- LiteLLM-internal retries, fallbacks, hedging, routing failover, and caching are disabled. The one documented parse retry is scheduled by jes and acquires another permit. Supported LiteLLM versions must accept jes’s decoded-body-capped httpx transport; otherwise construction fails. `completion_fn` replaces model completion in parser tests but does not substitute for separate transport-cap tests.
- `revision=` must name an immutable provider version or digest. It is sent when the provider supports pinning and verified from response metadata when available; caller-only labels that cannot be verified make the profile ineligible for defaults.
- `completion_fn` replaces the LiteLLM call in tests.

### 8.5 Prompt Guard 2

```python
PromptGuard2.endpoint(
    base_url="https://provider.example/v1",
    model="meta-llama/llama-prompt-guard-2-86m",
    provider_profile="provider.prompt_guard_2.v1",
    revision="immutable-provider-revision",
    tokenizer_revision="prompt-guard-2@…",
    api_key=None,
    context_window_tokens=512,
)
PromptGuard2.local(
    "meta-llama/Llama-Prompt-Guard-2-86M",
    provider_profile="transformers.sequence_classification.v1",
    revision="pinned-hugging-face-revision",
    artifact_digest="sha256:…",
)   # extra [prompt-guard]
```

- `tasks={"injection"}`. Renders only `state.text`.
- Its model card includes explicit attempts in user text and in untrusted third-party data that try to supersede existing instructions, plus jailbreak techniques. jes therefore routes `injection` at both the input and untrusted stages. It does not route the broader `indirect_injection` policy, which also asks whether benign-worded text is addressed to an assistant; that narrower routing is a measured jes decision, not a claim that the model card excludes third-party injection.
- Local mode returns `kind="probability"` from softmax over the malicious class, with the label id read from pinned model config. There is no standard OpenAI-compatible classifier response. Endpoint support exists only for a named provider profile whose request, response, score kind, limits, and immutable revision were captured in fixtures; otherwise construction fails.
- Local mode has `max_attempts=1`; each endpoint profile declares and tests a fixed total attempt bound.
- The 22M checkpoint is the English option; 86M is multilingual.

### 8.6 Llama Guard 4

```python
LlamaGuard4.endpoint(
    base_url="https://api.groq.com/openai/v1",
    model="meta-llama/llama-guard-4-12b",
    provider_profile="groq.llama_guard_4.v1",
    revision="immutable-provider-revision",
    tokenizer_revision="llama4@…",
    template_revision="groq-llama-guard-4@…",
    api_key=None,
    logprobs=False,                                  # Groq returns no log-probabilities
    context_window_tokens=131_072,
)
LlamaGuard4.endpoint(
    base_url="http://127.0.0.1:8001/v1",             # vLLM
    model="meta-llama/Llama-Guard-4-12B",
    provider_profile="vllm.llama_guard_4.v1",
    revision="pinned-hugging-face-revision",
    artifact_digest="sha256:…",
    template_revision="llama-guard-4@…",
    logprobs=True,
    context_window_tokens=None,                      # read and verified from model config
)
LlamaGuard4.local(
    "meta-llama/Llama-Guard-4-12B",
    provider_profile="transformers.causal_lm.v1",
    revision="pinned-hugging-face-revision",
    artifact_digest="sha256:…",
)   # extra [llama-guard]
```

- `tasks` is `hazard.any` and `hazard.S1` through `hazard.S14`. The adapter renders history, `prompt` as the user turn, and `text` as the turn being classified (the assistant turn on output, the user turn on input).
- When the provider accepts a custom category list, the adapter sends only the categories the policy asked for. Otherwise it classifies all categories and drops the rest.
- With `logprobs=True`, construction verifies the pinned tokenizer’s `safe` and `unsafe` candidate forms. `hazard.any` is their normalized first-token score on every request, including a generated `safe`, and has `kind="probability"`. A missing candidate raises. Category answers remain `kind="label"`: 1.0 for listed codes and 0.0 for the rest. Score kind is answer-specific, never backend-wide.
- With `logprobs=False`, `hazard.any` and category answers are labels. Groq currently returns no log-probabilities for this model; use a verified vLLM or local profile for probability scores. Probability and label profiles are evaluated and thresholded separately.
- Local mode has `max_attempts=1`; each endpoint profile declares and tests a fixed total attempt bound.
- Output parsing: `safe`, or `unsafe` followed by comma-separated codes. Unknown codes raise `BackendError`.
- Llama Guard can itself be jailbroken; also run `injection` on the same text.
- “About 24 GB” is only the BF16 weight size, not a runtime memory promise; KV cache, framework overhead, and long context require more. The docs report measured hardware per evaluated local profile.
- The adapter docs point to the Llama 4 Community License and its attribution requirements. jes itself distributes no Llama materials.

## 9. Engine rules

### 9.1 Phases

1. Before normalization or another expensive operation, jes checks the subject’s UTF-8 size and counts every prompt, question, source, and history entry—including stamped results and empty strings—against `max_context_items`. It checks each effective context value and their aggregate against `max_context_bytes`. Reserved token syntax is neutralized in raw contexts and input/untrusted subjects, but preserved in an output subject only for comparison with the prior generation manifest. The expanded neutralized form is size-checked immediately before the first transform.
2. Transforms for the subject run by phase (`normalize`, `detect`, `limit`), in list order within a phase; each receives the previous output. After every edit, the engine remaps existing authority spans, tombstones any overlap, scans all reserved token syntax, and preserves only engine-issued conversation tokens or exact untouched raw-output occurrences authorized by the prior manifest. Every other occurrence is neutralized before the next transform. The result is checked against `max_normalized_bytes` after neutralization and after every transform.
3. Context is projected as described in sections 7.1 and 9.2. A stamped fast path first verifies its full HMAC, result state, findings/authority/configuration digests, store/scope ids, and exact sanitized bytes. Every effective projection still counts toward context caps. Other values run applicable `normalize` and `detect` transforms with the same post-edit authority scan, and aggregate size is checked after every neutralization/transform against `max_normalized_context_bytes`. Context findings identify their origin. Flags and redactions remain findings, and any context block blocks the call before backend I/O.
4. A jes-owned sensitive transform whose engine-applied edit cannot remove the complete detected value prevents backend I/O regardless of `fail_fast`. With `fail_fast=True`, any other subject block ends the check before planning judgments and sets `complete=False` because applicable judgments were skipped. With it false, judgments may run only on a backend-safe subject; the final decision remains block.
5. Judgments are grouped by their complete logical request key, chunked, sent, interpreted per chunk, and merged. Async execution may complete out of order; results are sorted back into logical plan order.
6. Output restoration runs only when all pre-finalizer work produced `decision="allow", complete=True`. The engine computes the entire prospective plain-text result atomically from manifest-authorized context tokens and surviving output-local markers under section 10.2. Any preexisting block/incomplete path returns zero restorations; an internal finalizer/cap failure applies zero restorations and blocks incomplete. URI policy findings are successful finalizer outcomes governed separately by `on_placeholder_in_url`.

Judgments always see fully transformed backend-safe text and context, and finalizers run after judgments, so no backend sees a restored value. `complete=True` means every applicable transform, judgment, and required finalizer completed. A context block, fail-fast skip, resource stop, or other early exit that omits one sets it false.

Before scheduling judgments, a read-only transaction preflight checks prospective store limits/collisions; a known failure blocks without I/O. The transaction remains staged through judgments/finalization. The engine first computes final text/findings, builds the complete result and authority manifest, and signs all authenticated fields in memory. A short synchronous commit is the final linearization point before returning that complete allowed result. It rechecks concurrent state and cannot await or be cancelled midway. Any block, incomplete path, exception, finalizer failure, cancellation, or commit failure closes/discards the transaction in `finally`, releases admission/permits, returns zero restorations, and leaves the store unchanged. A commit failure emits `redaction_limit_exceeded` in a separately signed block result.

### 9.2 Context and history

Each judgment declares `context`:

- `"none"`: the state holds only the text.
- `"optional"`: add the prompt or retrieval question first, then the newest whole history messages that fit. Selected history is rendered in chronological order. With a finite backend budget, context may use at most half of empty-text headroom and must leave at least 64 units for text. When `headroom` is `None`, every optional value fitting the guard’s item and byte caps is included. Omitted optional context does not make the result incomplete.
- `"required"`: the prompt, plus all sources for policies with `sources=True`, must fit whole and leave at least 64 units for text. If they do not, the policy is not asked and gets `context_too_long`. The default adds a block and sets `complete=False`; `on_context_overflow="allow"` adds no block for this overflow but still sets `complete=False`.

Required-context policies run only on output, where a prompt is always present. When no sources are passed, a `sources=True` policy uses the prompt alone. Context is never chunked; only the subject is. Fitting calls `headroom` repeatedly on the complete rendered state and question batch; it never estimates fit by adding separately counted fields.

Only reusable input-origin tokens from authenticated context results enter a generation authority manifest. Output-local edits leave irreversible markers in `sanitized`; an assistant-history result therefore carries no output-local value or authority into a later generation.

### 9.3 Chunking

- If `headroom` returns `None`, the text is sent whole.
- Otherwise the planner proposes an end offset from `count_units`, verifies the complete request, and shrinks geometrically until it fits. It may make at most eight bounded forward probes to enlarge that candidate; it does not promise the globally longest prefix or assume token counts are monotonic. If one code point cannot fit, it blocks. Cuts prefer whitespace and fall back to a code-point boundary. Planning is linear in source length plus a bounded number of renders per chunk.
- Overlap is at most the smaller of 32 backend units and one quarter of the actual fitted chunk’s units. The next chunk must extend past the previous chunk’s ending source offset. If escaping or tokenization prevents positive source progress, planning blocks with `input_too_long` instead of looping.
- Every character of the text appears in at least one chunk.
- If any logical group needs more than `max_chunks`, the check blocks with `input_too_long`, sets `complete=False`, and sends no judgment requests.
- A `whole_text` policy is never chunked. When its text does not fit one request, `text_too_long` blocks by default; `on_text_overflow="allow"` is the explicit incomplete alternative.
- `Item`s are validated against transformed-subject spans, assigned ordinals, mapped to original-subject locations, and chunked separately. The engine consumes each extractor only through its effective policy/global cap. Policy overflow uses its declared label/action; aggregate call overflow uses `too_many_items`. Both stop before judgment I/O.

### 9.4 Merging chunks

Each chunk is interpreted on its own. Findings merge per `(policy, question, label)`: the most severe action wins (`block` over `flag`), the highest score value wins, and equal winners union sorted locations and chunk ids. `scores` take the maximum across chunks. Values merged under one question must have the same score kind and decision-profile fingerprint; otherwise the backend contract is broken. Adding a chunk can never lower a score, erase provenance, or soften an action.

### 9.5 Batching

Batching has two keys. The **compatibility key**, computed before questions are combined, contains:

1. backend instance and immutable backend profile;
2. stage and complete sanitized context projection;
3. context mode and whether sources are included;
4. subject kind (`text`, `whole_text`, or `item`) and engine-owned item ordinal; and
5. whole-text and overflow behavior.

Only policies with equal static compatibility fields are candidates to batch. At construction, their namespaced questions are combined and `backend.partition_questions` produces state-independent, policy-atomic partitions. At runtime the sanitized context/subject identity completes the compatibility key and each partition gets its own chunk plan; because a policy appears in exactly one partition, `interpret` sees all its answers for one chunk range. The **finalized request key** adds the partition’s complete question hash and chunk source range.

Direct-text policies may batch; whole-text and item policies are separate, and each item is separate even when two item strings match. The guard validates every finalized partition’s capability limits and empty-text headroom. A combined partition may use defaults only when every policy has an evaluated decision-profile entry for that request profile, yielding a complete threshold vector; otherwise the planner partitions to evaluated per-policy requests or requires explicit thresholds. Recommended bundles publish all vector entries.

Question ids are `"<policy>.<question>"`; each segment obeys the bounded identifier rule in section 7.3, so namespacing cannot collide. Each finalized partition has one `RequestContext`; every retry acquires another internal permit, carries the same deadline, reports private backend usage, and is merged before policy interpretation. Public usage indices are assigned canonically afterward.

Before I/O, the planner reserves deterministic attempt slots for each finalized request using the backend profile’s fixed retry limit. `max_requests` must cover the worst-case planned slots or the check blocks before any call. Unused retry slots cost no provider request. A physical call is identified semantically by `(logical request, attempt)`; scheduling/acquisition order never decides which request is allowed to retry.

### 9.6 Limits, concurrency, and deadlines

- Every Python string is first checked for Unicode scalar values. An unpaired surrogate emits blocking `invalid_unicode`; jes never calls an encoder that can retain the source string in `UnicodeEncodeError.object`. UTF-8 size uses an allocation-free code-point counter that stops as soon as the configured cap is crossed.
- `max_requests` covers the deterministic worst-case attempt slots for planned question partitions. Insufficient slots emit `too_many_requests` before I/O. Deadline expiry while waiting for a guard-wide permit raises `DeadlineExceeded` and follows `on_backend_error`.
- `max_findings` must be at least 1. jes reserves its final slot for `too_many_findings`: when ordinary findings would consume that slot, processing stops and the terminal blocking finding occupies it. No returned tuple exceeds the cap and no already returned finding is replaced.
- Policy/question/label/profile metadata is counted as UTF-8 against `max_metadata_bytes`; locations and token-authority entries are capped by `max_locations` and `max_authorities`. The terminal `metadata_too_large`, `too_many_locations`, or `too_many_authorities` block occupies the same reserved finding slot.
- Call-local redaction count and canonical UTF-8 bytes are checked prospectively before an edit is accepted; overflow is atomic and follows the same strict block behavior as a conversation-store overflow.
- Before finalization, jes computes the complete number of context-token plus output-local restorations and the prospective UTF-8 result length. Crossing `max_restorations` or `max_restored_output_bytes` emits `restored_output_too_large`, applies no restoration, blocks incomplete, and never returns a partly restored string.
- One guard-shared semaphore limits physical backend calls across all concurrent checks to `max_concurrency`; `max_active_checks` limits admitted checks. `AsyncGuard` also owns a bounded `max_worker_threads` executor and `max_queued_work` queue for sync callbacks, so timed-out work cannot create unbounded threads. Admission saturation emits `guard_busy`, blocks incomplete, and starts no work. `Guard` executes one check’s requests in deterministic plan order. Both APIs canonicalize findings, scores, and public usage identically.
- jes-owned transform loops poll the absolute deadline, and timeout-capable regex/HTTP operations receive the remaining duration. Synchronous third-party CPU calls and custom callbacks are cooperative: after an async timeout their bounded worker may continue, but its result is discarded and its private redaction transaction is closed. `Guard` cannot return until a non-cooperative call returns; v1 makes no stronger wall-clock guarantee.
- No store lock is held during a backend call. A timed-out or discarded request cannot add restoration authority because authorization is an immutable snapshot created before I/O.

## 10. Policy catalog

### 10.1 Question rules

- Every built-in yes/no question is phrased so true means violation. `relevance`, for example, asks whether the output fails to address the prompt.
- Instructions refer to state fields by name: “the text”, “the prompt”, “the question”, “the sources”.
- Instruction text lives in `jes/policies/prompts.py` under ids such as `injection.v1`. A snapshot test fails on any edit; a wording change is a new version.
- Callers pick a version with `version=`. Findings record it.

### 10.2 Core transforms

| Policy | Default stages | Behavior |
| --- | --- | --- |
| `invisible_text(mode="targeted", block=False)` | All | See below. Finding `redact` when anything was removed; `block=True` blocks instead. |
| `regex(patterns, action="block", match="search", require=False, fold=False, timeout_ms=50)` | All | Caller patterns through the timeout-capable `regex` package (extra `[regex]`). `match="search"` looks anywhere; `"fullmatch"` needs the whole text. `action` is `block` or `redact`; with `require=True`, text matching none of the patterns is blocked. |
| `substrings(terms, action="block", whole_words=False, fold=True)` | All | Matches on a folded copy, so lookalike letters and hidden characters cannot split a banned term. |
| `token_limit(limit, encoding="cl100k_base", mode="block")` | Input | tiktoken count. `mode="truncate"` cuts to the limit and flags. Extra `[tokens]`. |
| `secrets(redact="all", key=None)` | All | detect-secrets plus a documented set of high-value token patterns; see below. `all` → `******`; `partial` → first two and last two characters; `hmac` → HMAC-SHA256 hex under `key`, which that mode requires. A plain hash would let anyone confirm a guessed secret. Extra `[secrets]`. |
| `pii(...)` | All | Presidio; see below. Extra `[pii]`. |
| `canary(token)` | Output | Jes-owned sensitive transform: irreversibly removes and blocks a canary token leaked from an application system prompt. |

**`invisible_text`.** `"targeted"` mode removes the characters attacks use, taken from Unicode properties rather than a hand-written list:

- every `Default_Ignorable_Code_Point` character (tag characters, zero-width spaces, word joiners, invisible operators, soft hyphens, byte-order marks, Hangul fillers, and more), except those kept below;
- every `Bidi_Control` character, including the marks U+200E, U+200F, and U+061C;
- C0 and C1 control characters except tab, line feed, and carriage return. In model output they carry ANSI escape sequences into terminals, and JSON escapes each one to six characters.

It keeps the zero-width joiner and non-joiner in display text, so emoji and Persian or Indic text survive. A variation selector is kept only when its complete sequence occurs in the vendored Unicode standardized-variation or emoji-variation data; arbitrary selectors and selector runs are removed. Subdivision flag emoji lose their tag characters, which is accepted.

Property data is vendored from the Unicode Character Database with its license notice. `"all"` mode also removes every remaining character in Unicode categories Cf, Co, and Cn, joiners included, as LLM Guard did.

**Folding** (`fold=True`) applies NFKC, case folding, removal of default-ignorable characters (soft hyphens, zero-width characters), and the Unicode TR39 confusables skeleton. The folded copy keeps a map back to original offsets, so redaction edits the original text. Confusables data is vendored with the Unicode license notice.

Natural-language PII recognition uses an NFKC copy because named-entity recognition depends on letter case. Machine-identifier recognizers for email addresses, account numbers, URLs, and secrets use a second detector-only copy that also strips joiners, variation selectors, and other default-ignorables inside candidates. Both copies retain maps to original offsets. Thus `a\u200Dlice@example.com` cannot evade an email recognizer even though a legitimate ZWJ remains in display text.

**`secrets`.** Text is scanned in memory and nothing is written to disk. Every occurrence of a detected secret is redacted, not only the first. detect-secrets keeps its settings in a process-wide singleton, so jes configures it once per process and scans under a lock, and a second guard with different detect-secrets settings raises `PolicyError`.

**`pii(entities=None, language="en", ner="spacy", input_mode="redact", untrusted_mode="mask", output_mode="flag", restore=True, restore_origins=(), on_placeholder_in_url="block")`:**

- Default entities follow LLM Guard: credit card, crypto wallet, email, IBAN, IP address, person, phone, US SSN, US bank number, UUID. v1 supports `language="en"` only and raises `PolicyError` for another value. Adding a language requires a later design revision with explicit NLP models, recognizers, installation, and evaluation; no Chinese support is implied.
- `ner` picks the named-entity model: `"spacy"` (Presidio’s default) or a Hugging Face model id, which Presidio runs through its transformers engine (extra `[pii-ner]`). LLM Guard’s default was a DeBERTa model fine-tuned on ai4privacy data, so recall differs by choice; the evaluation measures both (section 11).
- Detection uses the NFKC and machine-identifier views described above, with offsets mapped back to the original text.
- Input: with `restore=True`, `redact` asks the engine to replace each value with its authenticated store token; a value already in the store keeps its exact token. With `restore=False`, `redact` uses an irreversible `[REDACTED_{ENTITY}]` marker, issues no conversation token, and disables context-token restoration. `mask` asks the engine to keep the first and last characters only when that removes the complete value and otherwise uses `*`; `block` still replaces the sensitive backend projection before returning a block.
- Token syntax in raw input, untrusted text, or context is never trusted, even when it is a valid live token. jes rewrites it visibly as literal text and emits `placeholder_in_input`. A model-reply output subject preserves syntax only so the finalizer can validate exact occurrences against the generation authority manifest; well-formed text alone grants no authority.
- Untrusted: `mask` by default. Untrusted text gets no placeholders, so it cannot collide with the store.
- Output: every newly detected value is replaced by an irreversible marker in `sanitized` before a judgment backend sees it, regardless of `output_mode`. `flag` keeps the original only in a private call-local span map and restores it into final `text` after judgments when plain-text rules permit; `redact` leaves the marker in `text`; `block` blocks. Exact reusable context tokens restore only when authorized. Changed-case, partial, fuzzy, unknown, and merely well-formed context tokens remain literal and emit `invalid_placeholder`.
- The complete-reply plain-text finalizer uses one private strict URI parser. Candidates begin with an RFC 3986 scheme, `//`, or `www.` and continue until Unicode whitespace, a C0/C1 control, or one of `<`, `>`, `"`, `'`, or backtick; other punctuation is conservatively part of the candidate. Scheme-relative, `www.`, malformed, userinfo-bearing, backslash-containing, percent-encoded-host, Unicode-dot, IPv6-zone-id, and non-HTTP(S) candidates never qualify for restoration.
- `restore_origins` entries are parsed at construction and must be absolute `http`/`https` origins with no userinfo, path other than `/`, query, or fragment. Scheme and DNS host are lowercased; DNS uses a pinned `idna` package in strict IDNA2008/non-transitional STD3 mode; one ASCII trailing dot is removed; IPv4 is canonical dotted decimal; IPv6 is bracketed and canonicalized with `ipaddress`; default ports are normalized. A token inside a URI restores only on exact `(scheme, canonical host, effective port)` equality. Subdomains, alternate numeric forms, and redirects are not implied. Nonmatching tokens remain and get `placeholder_in_url` (`block` by default; `"allow"` adds a flag); the finalizer completes in either mode.
- Candidate detection records every syntactically containing URI range, including nested/overlapping starts. A token restores only if every containing candidate is a qualifying absolute origin; ambiguity is deny-by-default. Tests include strings such as `https://allowed.example/https://evil.example/[TOKEN]`.
- Restoration produces complete plain text only. An application that renders Markdown, HTML, JSON, or another grammar must escape values or insert them as text nodes after parsing. Tool arguments, arbitrary application strings, and partial streams are never restored by jes.
- Presidio needs a spaCy model download; the README documents it.

### 10.3 Core judgments

| Policy | Default stages | Context | Questions | Task |
| --- | --- | --- | --- | --- |
| `injection()` | Input, untrusted | None | One yes/no question: the text tries to override, ignore, or reveal an AI assistant’s instructions, or to make it abandon its rules | `injection` |
| `indirect_injection()` | Untrusted | Optional (the question) | One yes/no question: the text contains instructions addressed to an AI assistant rather than information for a reader | `indirect_injection` |
| `hazards(categories=None)` | Input, output | Optional (prompt, history) | One yes/no question per category, S1–S14, plus `hazard.any` on backends that list it in `tasks` | `hazard.S<n>`, `hazard.any` |
| `topics(deny, threshold=...)` | Input, output | None | One yes/no question per denied topic | `topic` |
| `toxicity(labels=None)` | Input, output | None | One yes/no question per label: toxicity, severe toxicity, obscene, threat, insult, identity attack, sexual explicit | `toxicity.<label>` |

- `topics` asks one yes/no question per topic, not one choice, because text can touch several topics. It always requires a threshold, since caller-defined topics cannot have a measured default.
- `hazards` scores the text by its “any hazard” score: the backend’s `hazard.any` answer when its `tasks` list that task (Llama Guard 4), otherwise the highest category score. The threshold applies to that score. Findings name the categories whose own scores cross the threshold, or the codes Llama Guard lists (`unattributed` when it lists none).
- Hazard instructions restate the public category definitions in jes’s own words.
- Final wording is tuned on development data, assigned exact `v1` ids before calibration in Milestone 6, then verified and published unchanged in Milestone 7.

| Code | Hazard |
| --- | --- |
| S1 | Violent crimes |
| S2 | Non-violent crimes |
| S3 | Sex-related crimes |
| S4 | Child sexual exploitation |
| S5 | Defamation |
| S6 | Specialized advice |
| S7 | Privacy |
| S8 | Intellectual property |
| S9 | Indiscriminate weapons |
| S10 | Hate |
| S11 | Suicide and self-harm |
| S12 | Sexual content |
| S13 | Elections |
| S14 | Code interpreter abuse |

### 10.4 Recipes (`jes.recipes`)

Recipes cover the rest of LLM Guard’s catalog. They are built only from public APIs (`judge`, `substrings`, custom transforms), tested with the fake backend, and documented. They are not evaluated, carry no default thresholds, and every judgment-based recipe requires `threshold=`. Their public factory signatures and question ids join the 1.0 stability promise; “recipe” means unevaluated, not unstable. Promotion to core adds evaluation/defaults without silently changing the existing recipe contract.

| Recipe | Built from |
| --- | --- |
| `sentiment(threshold=...)` | Yes/no: the text is hostile or strongly negative |
| `emotions(blocked=..., threshold=...)` | One yes/no question per emotion; defaults to LLM Guard’s negative subset of GoEmotions |
| `gibberish(threshold=...)` | Yes/no: the text is not meaningful language |
| `bias(threshold=...)` | Yes/no (output): the text demeans or stereotypes a group |
| `refusal(threshold=...)` | Yes/no (output): the text refuses the request. `refusal_phrases` provides a threshold-free substring variant. |
| `language(allowed, threshold=...)` | Choice over allowed language codes plus `other`; violating option: `other` |
| `language_same(threshold=...)` | Yes/no (output, context required): the text is in a different language than the prompt |
| `code(mode, languages, threshold=...)` | Choice over programming languages plus `not_code`; violating options: the banned languages |
| `competitors(names)` | `substrings` with redaction |
| `malicious_urls(threshold=...)` | URLs extracted as `Item`s with transformed-subject spans and engine ordinals, then mapped to original locations; duplicate URLs remain distinct states (at most 20; more blocks with `too_many_urls`). Judges the URL string only; it cannot know what a host serves. |
| `relevance(threshold=...)` | Yes/no (output, context required, whole text): the text does not address the prompt |
| `factual_consistency(threshold=...)` | Yes/no (output, context required, `sources=True`): the text makes claims that the sources, or the prompt when no sources are passed, contradict or do not support |
| `reading_time(max_minutes)` | Transform: 200 words per minute; block or truncate |
| `json_check(required_elements=0, repair=False)` | Transform: find and validate JSON; repair with extra `[json]` |

### 10.5 Mapping from LLM Guard and Meta

| Source | jes | Tier |
| --- | --- | --- |
| InvisibleText | `invisible_text` | Core transform |
| Regex | `regex` | Core transform |
| BanSubstrings | `substrings` | Core transform |
| TokenLimit | `token_limit` | Core transform |
| Secrets | `secrets` | Core transform |
| Anonymize / Deanonymize | `pii` (redact on input, restore on output) | Core transform |
| Sensitive | `pii(output_mode=...)` | Core transform |
| PromptInjection | `injection`, `indirect_injection` | Core judgment |
| BanTopics | `topics` | Core judgment |
| Toxicity | `toxicity` | Core judgment |
| Llama Prompt Guard 2 | `injection` with the `PromptGuard2` backend | Core judgment |
| Llama Guard 4 | `hazards` with any backend, including `LlamaGuard4` | Core judgment |
| BanCompetitors | `recipes.competitors` | Recipe |
| ReadingTime | `recipes.reading_time` | Recipe |
| JSON | `recipes.json_check` | Recipe |
| Sentiment | `recipes.sentiment` | Recipe |
| EmotionDetection | `recipes.emotions` | Recipe |
| Gibberish | `recipes.gibberish` | Recipe |
| Language | `recipes.language` | Recipe |
| LanguageSame | `recipes.language_same` | Recipe |
| Bias | `recipes.bias` | Recipe |
| NoRefusal / NoRefusalLight | `recipes.refusal` | Recipe |
| BanCode / Code | `recipes.code` | Recipe |
| MaliciousURLs | `recipes.malicious_urls` | Recipe |
| Relevance | `recipes.relevance` | Recipe |
| FactualConsistency | `recipes.factual_consistency` | Recipe |
| URLReachability | Dropped: outbound requests to model-chosen URLs are an SSRF risk, and the check adds little | — |

## 11. Evaluation

The evaluation recommends request profiles and supplies default thresholds for exact per-policy decision profiles. It never silently selects a backend at runtime, and its numbers are published.

- **Location:** `evals/` in the repository, not in the package.
- **Data:** datasets are downloaded at run time from pinned revisions. Licenses, provider terms, allowed hosted processing, and retention constraints are recorded in `evals/datasets.toml` and reviewed before use, since several popular safety sets are non-commercial. Raw texts are never committed; published results are aggregate numbers only.
- **Categories:**
    - direct injection and jailbreaks;
    - indirect injection in documents and tool output (candidate: BIPIA);
    - hazards labeled with S1–S14;
    - toxicity;
    - benign text, including long documents and prompts that look unsafe but are not (candidate: XSTest), to measure false positives;
    - judge-directed attacks, where the text addresses the classifier, generated from templates;
    - padding attacks, where an attack follows benign padding at several offsets, generated from templates, with prose padding and with control-character padding;
    - PII detection: recall and precision per entity type for `pii`, with `ner="spacy"` and with a transformer model, including joiner and variation-selector evasions.
- **Discipline:** development tunes wording and explores models; a selection split ranks request/decision-profile candidates; calibration fits thresholds for the selected candidates; and a sealed audit split is opened once for final acceptance/reporting. If a candidate fails audit, it gets no default/recommendation; trying another requires fresh holdout data. Every split is grouped by source document or dataset-defined group, never row. Near-duplicates, padded variants, and one template family stay together. Known training contamination is reported separately.
- **Repeated releases:** sealed audit data is never used to pick wording, models, ranking, or thresholds and is not repeatedly reused for changed profiles. Any changed prompt, transform, planner/interpreter/merge version, adapter/provider/model/dependency, tokenizer/template, generation setting, question partition, budget, or threshold uses a fresh rolling holdout for acceptance; historical data is comparability-only.
- **Backends:**
    - immutable Laya `english` and `multilingual` artifacts in process or through a controlled manifest-attested `laya-serve` deployment, and a pinned Jev provider profile;
    - gpt-oss-safeguard through LiteLLM verbalized mode, because it classifies against a policy written at inference time;
    - at least one hosted and one local LiteLLM profile, with logprob and verbalized modes treated as different profiles;
    - pinned Prompt Guard 2 22M and 86M local artifacts, plus endpoint profiles only after provider-specific fixtures exist;
    - pinned Llama Guard 4 profiles, with answer-specific probability and label scores reported separately.
- **Operational completeness:** every attempted check contributes to denominators. Reports include completion rate and counts/rates for transport errors, timeouts, malformed answers, missing logprob candidates, retries, and resource blocks. The evaluation runner uses `on_backend_error="block"` for operational safety metrics, so an incomplete check counts as a block; a parallel raise-mode diagnostic classifies the underlying error. Model-only conditional metrics may be secondary but never replace operational metrics. A logprob profile whose valid response omits a required candidate is unsupported rather than filtered.
- **Metrics per (policy, decision profile):** independent-group and row counts; ROC AUC and PR AUC for non-label scores; catch rate at 1% and 5% false-positive rate; false-positive rate at 95% catch rate; false-positive rate by text length; expected calibration error by score kind; uncached latency p50/p95; physical requests/retries per check; completion/error rates; and public cost. Label-only profiles report their operating point rather than meaningless curves.
- **Uncertainty:** rates and AUCs use cluster-aware intervals over the independent split group, never row bootstrap. `evals/protocol.md` pre-registers power-derived minimum row and independent-cluster counts for benign, positive, and category subgroups; meeting 10,000 rows alone is never sufficient. Recommendation comparisons use a pre-registered simultaneous-confidence procedure such as Holm correction, with every support count and interval published.
- **Pre-registration:** before Milestone 6, maintainers accept the target false-positive rate, operational failure treatment, profile ranking, minimum independent-group support, simultaneous audit procedure, and tie-breakers. The proposed target is 1%. Selection data ranks candidates and freezes a finalist set before audit. The sealed audit accepts/rejects that set with simultaneous one-sided bounds and never reranks or substitutes a failed finalist; a new candidate needs fresh holdout data. If none qualifies, recommend none.
- **Default thresholds:** on calibration, enumerate observed score cutoffs and choose the lowest threshold whose cluster-aware one-sided 95% upper false-positive bound meets the target. The exact decision profile spans its evaluated policy, stage, question set, interpretation, and merge behavior. Sealed audit/fresh rolling holdout must independently pass. Defaults are block-only (`flag_at=None`); a flag band needs its own protocol. Any decision-profile mismatch requires another evaluation or explicit threshold.
- **Bundles:** a bundle has one request profile and a vector of decision profiles/thresholds. `evals/protocol.md` pre-registers its guard-level false-positive/catch targets and completion requirement; individually calibrated policies cannot substitute for that gate. A bundle is recommended only when its exact threshold-vector fingerprint passes sealed audit/fresh holdout. Bundle results and every component profile are published.
- **Topics:** caller-defined topics never receive a default. A fixed representative topic suite evaluates prompt wording and robustness before `topics.v1` is frozen, without claiming its threshold transfers to arbitrary topics.
- **PII metrics:** PII is not forced into backend/profile ROC metrics. Each recognizer configuration records exact model/config revisions, supported entities and language, exact-span and overlap-span micro/macro precision/recall/F1, per-entity support, evasion recall, and latency p50/p95. It produces no judgment threshold entry.
- **Report:** JSON under `evals/results/<run-id>/` and a generated markdown table copied into the README.
- **Cache:** development caching keys the canonical rendered state, full question schema, request/decision profiles, and generation settings. It stores validated answers only—not inputs, raw provider bodies, latency, retries, failures, or `Usage.request`. Replays are marked cached, receive run-local canonical usage metadata, and are excluded from latency/completion/cost acceptance metrics. Release acceptance metrics run uncached.
- **CI:** a smoke run against the fake backend tests the plumbing. Real runs are manual.
- **When to rerun:** before a minor release that publishes or changes a measured default, and whenever that request profile, decision profile, threshold vector, or recommended bundle changes. 0.1.0 and 0.2.0 publish none. Their quickstart uses an explicit threshold.

## 12. Package layout and dependencies

```text
jes/
  __init__.py            # Guard, AsyncGuard, Redactions, results, Threshold, errors
  errors.py
  types.py               # results, scores, provenance, items, messages, state
  questions.py           # YesNo, Choice, Score, answers, validation
  redactions.py          # Redactions, token authority, complete-reply URI policy
  testing.py             # FakeBackend, check_backend_contract
  py.typed
  backends/
    __init__.py          # Protocols, profiles/results/usage, SystemOne, LiteLLMJudge, PromptGuard2, LlamaGuard4
    _profiles.py         # BackendProfile and verified deployment manifests
    _tokens.py           # token counters
    system_one.py
    litellm_judge.py
    prompt_guard.py
    llama_guard.py
  policies/
    __init__.py          # core factories and judge()
    _protocols.py        # TransformPolicy, JudgmentPolicy
    _fold.py             # folding and NFKC copies with offset maps; vendored confusables and Unicode property data
    prompts.py           # versioned instruction text
    defaults.py          # evaluated decision-profile thresholds (generated)
    transforms.py        # invisible_text, regex, substrings, token_limit, canary
    secrets.py
    pii.py
    judgments.py         # injection, indirect_injection, hazards, topics, toxicity
  recipes/
  _engine/
    authority.py         # stamps, token occurrence lineage, redaction transactions
    sensitive.py         # private jes-owned SensitiveEdit modes and finalizer maps
    limits.py            # guard admission, per-check budgets, metadata/response caps
    plan.py              # phases, profiles, context fitting, grouping, chunking, span mapping (pure)
    merge.py             # per-chunk interpretation and merging (pure)
    run_sync.py
    run_async.py
evals/
tests/
```

The core depends only on `httpx`. Everything else is an extra, imported inside the module that needs it, so `import jes` works with the core alone. Endpoint adapters can operate without model weights in conservative byte-budget mode; an exact verified endpoint profile needs `[tokenizers]` unless the attested provider contract supplies equivalent token-count metadata.

| Extra | Brings | Used by |
| --- | --- | --- |
| `regex` | regex | Timeout-bounded `regex` |
| `secrets` | detect-secrets | `secrets` |
| `pii` | presidio-analyzer, presidio-anonymizer, idna | English `pii`, strict origin canonicalization; README installs the tested spaCy English model separately |
| `pii-ner` | presidio-analyzer[transformers], presidio-anonymizer, idna, transformers, torch, paired spaCy model | `pii(ner=<model id>)` |
| `crypto` | cryptography | `Redactions.dumps` and `loads` |
| `tokens` | tiktoken | `token_limit` |
| `tokenizers` | tokenizers | Exact token counts for Laya checkpoints |
| `laya` | laya | `SystemOne.in_process` |
| `litellm` | litellm | `LiteLLMJudge` |
| `prompt-guard` | transformers, torch | `PromptGuard2.local` |
| `llama-guard` | transformers, torch | `LlamaGuard4.local` |
| `json` | json-repair | `recipes.json_check(repair=True)` |

Tooling: uv for environments and the lockfile, hatchling, ruff, pyright in strict mode on `jes/`, pytest with hypothesis.

API-sensitive extras have tested compatibility bounds in `pyproject.toml`, established from fixtures before their milestone. The uv lock records exact CI versions, but defaults do not rely on the lock alone: each request profile records exact installed versions and asset hashes for jes planner/renderer, httpx serialization, regex, detect-secrets/plugins, tiktoken, Presidio, spaCy model, idna data, LiteLLM, Laya, transformers, torch, tokenizers, and model packages as applicable. A different behavior-affecting version cannot reuse a default until that exact profile is evaluated.

## 13. Testing

- The engine and policies are tested with `jes.testing.FakeBackend`, which returns registered answers and records every `State` and question it receives. No network and no GPU.
- Property tests (hypothesis):
    - chunks cover every character of each full-text or item logical subject, every chunk advances, and every finalized request fits its complete rendered payload, including control characters, emoji, and quotes;
    - merging never lowers a score or softens an action;
    - folding offsets map back into the original text, and redacting through them removes exactly the match;
    - locations from any transform order map to the correct original subject or context value and cover the characters they describe;
    - context-token and output-local PII round trips obey their distinct authority rules;
    - later edits either preserve an output-local marker exactly or permanently tombstone its restoration authority.
- `check_backend_contract` runs against every built-in backend.
- HTTP backends replay fixtures captured once from real servers, through `httpx.MockTransport`. LiteLLM uses `completion_fn`. Local Meta adapters use canned logits and completions. CI never downloads model weights.
- Sync/async parity uses `jes.testing.assert_semantic_parity` with deterministic fakes and no wall-clock/transport race. It compares decision, completion, text projections, findings, scores, canonical usage slots, and value-equivalent redactions while excluding random result/store ids, HMAC tags, private manifests, timings, and permit ids. Under real deadlines or provider races, both APIs follow the same rules but are not promised identical external outcomes.
- Isolation: concurrent calls on one guard with different stores, or with none, never see each other’s values; a prompt result paired with a different store fails before backend I/O; forged or altered sanitization stamps never take the fast path.
- Authority lineage: invisible-character normalization cannot synthesize authority; every untracked token is neutralized after every edit; unauthorized output cannot gain authority through stamped history; stamp HMACs cover decision, completeness, findings, scope, and the occurrence manifest.
- Confidentiality: each jes-owned sensitive edit removes the complete detected value from the backend-safe projection for every action; an impossible edit prevents all backend I/O.
- Exfiltration: raw strings containing copied or forged live-token syntax are neutralized; a valid token not authorized for the current generation is not restored; and a token inside a nonallowlisted plain-text URI candidate is never restored.
- Resource bounds: raw/context-item and post-every-transform byte limits fire before more expensive work; response, blob, item, request, finding, location, metadata, store, restored-output, and guard-wide concurrency caps cannot be exceeded or partially mutate protected state.
- Planning: subject mode never changes at runtime, one policy is never split across question partitions, and public usage ordering is invariant under concurrent completion/retry order.
- Deadlines: transport, retry, permit, cooperative CPU, and async wait expiry all exercise identical `DeadlineExceeded` behavior in raise/block/allow modes; resource-cap failures remain strict blocks.
- No fixture contains hazardous content. Hazard and toxicity findings are exercised by returning high violation scores from the fake backend.
- Coverage gate: 90% line and branch coverage on `jes/`.
- CI runs lint, type checks, and tests on Python 3.11–3.14. Extras run on the Python versions their dependencies support. A manual job runs live backend tests when secrets are configured.

## 14. Observability

- Logging goes through the stdlib logger `jes`. Records include bounded policy id, action, score value/kind, request/decision profile ids, and latency. They never include checked text, context, history, redacted values, rendered requests, or provider bodies.
- jes exception type/args, cause, context, notes, and library-owned metadata carry only allowlisted backend/status/policy/question identifiers, never text or provider bodies. A handler drops the original reference, exits `except`, then raises a fresh wrapper with no cause/context/notes or inherited internal traceback. Python caller traceback-frame locals are outside this guarantee; applications must not serialize arbitrary frame locals.
- Every text-bearing type (`State`, `Message`, `Item`, transform outcomes/edits, results, and redaction stores) implements redacted `repr`; only findings, spans, counts, bounded static ids, and profile metadata are shown. Secret-bearing stores, sessions, authority manifests/snapshots, and output-local maps additionally reject pickle, JSON, copy, and deepcopy.
- `Guard`, built-in backends, and policies with API/HMAC/encryption credentials use redacted representations and reject pickle/deepcopy; copying is either rejected or explicitly returns the same thread-safe credential owner without exposing material.
- Supported LiteLLM versions must expose enforceable payload-log suppression; jes configures it per call and installs narrowly scoped redaction filters on adapter-owned logger handlers. A version that cannot suppress payloads is rejected by the `[litellm]` compatibility check. Application-added handlers and arbitrary custom-component logging remain outside jes’s guarantee and are called out in the README.
- Tests capture records emitted by `jes` and its supported httpx/LiteLLM adapter configuration at DEBUG during success, retries, malformed bodies, built-in failures, and custom exception wrapping from every subject/context channel. Unique canary text and values must be absent from reprs, records, exception args/cause/context/notes, and library-owned metadata; tests explicitly exclude arbitrary caller traceback locals.
- `trace=True` fills `ScanResult.timings`. `ScanResult.usage` reports each physical request’s token usage in canonical logical/attempt order whenever the backend returns it.
- No OpenTelemetry dependency in v1.

## 15. Versioning and releases

- Semantic versioning. Public modules: `jes`, `jes.types`, `jes.questions`, `jes.backends`, `jes.policies`, `jes.recipes`, and `jes.testing`. `jes` re-exports the common result, message, threshold, redaction, and error types; `jes.questions` owns `Question`/`Answer` and their variants; `jes.policies` exports the public policy protocols, outcomes, `Item`, and factories. Everything else is private.
- **Prompt versions.** A wording change adds a new version id. Before 1.0, a new default version can land in a minor release, with its evaluation numbers in the changelog. From 1.0, the old version stays selectable until the next major release; removing an id is a major change.
- **Default thresholds.** A newly audit-qualified decision profile adds entries in a minor release. Changing its request profile, decision semantics, or threshold requires fresh acceptance data and a changelog note.
- **Pinned profiles.** Examples pin provider, model revision or digest, adapter/scorer version, tokenizer, template, and mode where applicable. Findings record both request and decision fingerprints. Floating or caller-labeled-but-unverified profiles never receive defaults (section 7.4).
- Release candidates and pipeline tests publish to TestPyPI. Functional releases publish to PyPI through trusted publishing. jes does not publish an empty name-holder: PEP 541 treats a project with no functionality as name squatting.

| Release | After | Contents |
| --- | --- | --- |
| TestPyPI 0.0.1 | Milestone 0 | Disposable pipeline proof; never uploaded to public PyPI |
| 0.1.0 | Milestones 0–4 | First public release: engine, `Guard` and `AsyncGuard`, core transforms, secrets and PII with authenticated complete-reply restoration, System One and LiteLLM backends, `judge()` |
| 0.2.0 | Milestones 5–7 | Meta adapters, evaluation harness, and core judgment policies. Every call passes an explicit threshold. No published defaults and no recommended backend. |
| 0.3.0 | Milestone 8 | Recipes for the rest of LLM Guard’s catalog |
| 1.0.0 | Milestone 9 | Frozen API and question ids, migration guide |

## 16. Milestones

Each milestone ends with its tests passing offline, `ruff` clean, and `pyright` clean on the code it added.

### Milestone 0 — Repository baseline

Depends on: nothing.

Steps:

1. Establish and commit the existing repository baseline with this file at `docs/design.md`.
2. Add `LICENSE` (Apache-2.0), `CHANGELOG.md`, and a `README.md` with the purpose, a “design in progress, no release” line, and a plain statement of jes’s relationship to TypeSafe and Meta: independent if it is, the relationship if not. The evaluation publishes numbers comparing Jev with its competitors, so this line must be literally true.
3. Add `SECURITY.md`: private reporting through GitHub security advisories, supported versions, and what counts as a vulnerability (for example, a bypass of a guarantee in section 5).
4. Add `CONTRIBUTING.md` (how to run tests, no model downloads in CI, no hazardous fixtures, and the prompt-version rule) and `CODE_OF_CONDUCT.md`.
5. Add `pyproject.toml` (hatchling; name `jes`; `requires-python = ">=3.11"`; dependency `httpx`), a uv lockfile, and configuration for ruff, pyright (strict on `jes/`), pytest, and coverage.
6. Add CI for lint, type checks, and tests on Python 3.11–3.14, with every GitHub Action pinned by commit SHA and Dependabot keeping the pins current. Add a TestPyPI workflow for Milestone 0 and a separate manual public release workflow using PyPI trusted publishing, disabled until a functional release.
7. Add `jes/__init__.py` exposing `__version__`.
8. Build, upload, install, and import version 0.0.1 on TestPyPI to prove packaging and trusted publishing. Do not upload it to public PyPI.

Done when: a clean clone passes CI, `import jes` works with only `httpx` installed, and the TestPyPI artifact can be installed in a clean environment. The first public upload waits for functional 0.1.0.

Out of scope: engine, policy, and backend code.

### Milestone 1 — Engine

Depends on: Milestone 0.

Steps:

1. Types from sections 7.2, 7.3, 7.6, and 8.1; question and answer types with score-kind and finite-number validation (section 7.4).
2. `PolicyError`, `PolicyExecutionError`, `RedactionError`, `BackendError`, and `DeadlineExceeded`.
3. Backend capabilities/profiles, `RequestProfile`, `DecisionProfile`, policy-atomic question partitioning, shared admission plus per-check request budgets, private backend usage, `BackendResult`, and `RequestContext`.
4. Policy protocols with static subject mode/labels/questions, complete transform-edit invariants, typed engine-ordinal `Item`s, `whole_text`, and `sources`; `judge()`; and violation-score thresholds.
5. `Redactions` without concrete PII token issuance or encryption: random scoped identity/secret, authority manifests/stamps, whole-call transactions, limits, mismatch checks, safe representations, and copy/serialization refusal. Concrete token format and crypto arrive with Milestone 3.
6. Private fake `_SensitiveEdit` modes and engine-owned replacements/finalizer maps for testing; only Milestone 3 supplies real PII/secrets implementations.
7. Pure planner: pre/post-neutralization limits, stage/origin filtering, authority-aware context sanitization, request/decision profiles, static compatibility and policy-atomic partitions, identifier/metadata/location validation, exact rendered context fitting, source-progress chunking, and item/request caps.
8. Pure interpreter: deterministic per-chunk interpretation, decision-profile provenance, merging, completion tracking, and canonical public usage assignment.
9. `Guard` and `AsyncGuard` with all admission/resource caps, bounded shared executor/semaphores, `fail_fast`, unified deadline behavior, `on_backend_error`, canonical sync/async ordering, and `trace`. Concrete store/restoration caps become active with Milestone 3.
10. `check_backend_contract` in `jes.testing`, run against `FakeBackend`.

Done when tests cover:

- A transform followed by a judgment: the backend sees the transformed text.
- A `normalize` transform listed after a `detect` transform still runs first.
- Raw prompt, question, source, and history strings reach a backend only after transforms for their origin stage; a block propagates. Fast paths require `ok` plus HMAC-authenticated state/findings/authority/config/text/store/scope fields; blocked, incomplete, forged, altered, and another-guard results cannot carry authority.
- Post-edit scanning neutralizes synthesized/untracked token syntax and preserves only engine-issued or prior-manifest occurrences in the exact mapped span.
- Private fake sensitive edits model jes-owned PII/secrets/canary modes; the engine removes the complete value, owns replacement strings, and tombstones overlapped output-local edits.
- Subject mode is static; text/item fallback and `whole_text + items` fail construction. One policy’s questions never split across partitions.
- A policy registered for `input` does not run in `check_output`.
- Duplicate or invalid names, an empty question mapping, a missing threshold, an unsupported task/score kind, invalid `max_attempts`, too many options, and a finalized partition leaving under 64 units raise `PolicyError` at construction.
- A multi-stage policy resolves one decision profile/default per stage; an explicit threshold applies to all, while one missing stage default fails construction.
- `questions` is called once, at construction, with the backend’s tasks.
- A two-chunk text whose second chunk violates blocks, whatever the first chunk says, for yes/no, choice, and score policies.
- A choice split across two violating options (0.45 and 0.45) blocks at `block_at=0.8`.
- Every submitted payload has headroom of 0 or more on the fake backend’s complete rendering, including questions, output reserve, control-character escaping, emoji, and quotes.
- Every full-text and item chunk advances its ending offset; every character of each logical subject is covered; overlap obeys the actual fitted-chunk bound; a non-advancing renderer blocks without looping.
- More than `max_chunks`, `max_items`, or `max_requests` blocks with `complete=False` and makes no judgment calls when the excess is knowable during planning.
- Required context and whole-text overflow block by default. Their explicit allow modes and backend fail-open add no block for that failure and set `complete=False`; any unrelated block still determines `decision`.
- Optional context drops oldest history while preserving chronological rendering and never takes more than half the empty-text headroom.
- `Item` transformed-subject spans are validated and map to original `FindingLocation`s; malformed custom edits or items fail before backend I/O.
- A finding from a transform that runs after another transform removed characters has a location in the correct original subject or context value.
- Private backend usage remains attached to its permit under concurrency; the engine assigns equal canonical public indices in sync and async despite different acquisition/completion order.
- The planner reserves every partition’s worst-case `max_attempts` slots before I/O; retries acquire only their assigned slots, cannot exceed per-check/guard-wide caps, and bounded timed-out workers cannot accumulate.
- `on_backend_error` in all three modes, including completeness and decision fields.
- Transport, retry, permit, cooperative CPU, and async-wait expiry all become `DeadlineExceeded` and follow raise/block/allow semantics. A late sync result is discarded and cannot commit a closed transaction.
- `fail_fast` skips judgments after a transform block, returns `complete=False`, and preserves `decision="block"`.
- `scores` includes passing questions with score kind, confidence, request/decision ids, exact threshold source/value/run, and bundle-vector fingerprint; changing only a threshold changes provenance.
- Invalid surrogates fail without `UnicodeEncodeError`; allocation-free raw/context/post-neutralization/transform counters and response, active-check, queue/worker, item, request, location, authority, metadata, concurrency, and finding caps fail at exact boundaries. `max_findings` reserves one terminal slot.
- `repr` is safe, while `pickle`, copy, deepcopy, `dataclasses.asdict`, and JSON serialization fail without exposing a store’s values.
- Custom transform/backend exceptions containing canary text are wrapped outside the handler; text is absent from wrapper args/cause/context/notes, library-owned metadata/frames, and adapter-owned logs (caller frame locals excluded).
- The chunk-coverage, merge-monotonicity, transform-edit mapping, sync/async parity, request-budget, and guard-isolation properties from section 13. Folding, token issuance, PII round trips, and URI handling remain owned by Milestones 2 and 3.

Out of scope: real backends and real policies.

### Spike — zero-shot check

Depends on: nothing in the package. Runs in parallel with Milestone 1.

A throwaway script, not committed: score a few hundred examples from one public injection set and one benign set with Laya `english` through `laya-serve`, Jev, and Prompt Guard 2 86M, and report ROC AUC and catch rate at 5% false-positive rate. `docs/spike.md` records the datasets, immutable revisions, and numbers.

Done when `docs/spike.md` exists. Among backends implemented by Milestone 4, its numbers choose the illustrative 0.1.0 quickstart and whether the README leads with cheap decision models; Prompt Guard results inform Milestones 5–7 only. The quickstart uses an explicitly application-chosen threshold and labels it non-default. Spike numbers are not published as evaluation results; Milestone 6 produces those.

### Milestone 2 — Text transforms

Depends on: Milestone 1.

Steps:

1. Folding and NFKC copies with offset maps; the vendored TR39 confusables data and Unicode property data (`Default_Ignorable_Code_Point`, `Bidi_Control`).
2. `invisible_text` (`targeted` and `all` modes), timeout-bounded `regex` (extra `[regex]`, including `require=True`), `substrings`, and `token_limit` (extra `[tokens]`).
3. `EXPLOIT_TERMS`, LLM Guard’s exploit-phrase list, as opt-in data for `substrings`.

Done when table-driven tests cover:

- Evasion: fullwidth letters, Cyrillic lookalikes, soft-hyphen splits, tag-character smuggling, bidi overrides, bidi marks (U+200E, U+200F), Hangul fillers, C0 and C1 control characters, ANSI escape sequences, arbitrary variation selectors, and per-character selectors.
- Legitimate text left intact in `targeted` mode: emoji ZWJ sequences, registered standardized and emoji variation sequences, Persian with ZWNJ, Hindi, and tab, newline, and carriage return.
- Empty input, no match, redact versus block, whole-word versus substring, `regex(require=True)` on text that matches no pattern, and catastrophic patterns ending at `timeout_ms` without blocking the process.
- Folding property tests.

Out of scope: secrets and PII.

### Milestone 3 — Secrets and PII

Depends on: Milestone 2.

Steps:

1. `secrets` (extra `[secrets]`) with engine-owned `all`, `partial`, and `hmac` sensitive-edit modes, plus core `canary(token)`: in-memory scanning, every occurrence handled, and detect-secrets configured once per process and scanned under a lock.
2. `pii` (extra `[pii]`): NFKC natural-language detection, stricter machine-identifier view, private sensitive edits for conversation tokens/irreversible markers/output-local values, exact authorized complete-reply restoration, and `ner=` with a transformer model (extra `[pii-ner]`).
3. Scoped store identity/secret; deterministic keyed-id-plus-MAC tokens; exact occurrence authority manifests; post-transform neutralization of untracked syntax; whole-call transactional commit; mismatch rejection.
4. The complete-reply strict URI parser, pinned IDNA behavior, canonical `restore_origins`, and `placeholder_in_url`.
5. Output-local edit tombstoning through every later transform and limit edit. Generic, streamed, and tool-argument restoration remain out of scope.
6. `Redactions.dumps` and `loads` with AES-256-GCM from `[crypto]`, authenticated headers, required scope binding, pre-decryption size limits, and copy/serialization refusal.
7. Establish tested Presidio/spaCy/transformers/torch bounds; verify `[pii]`, `[pii-ner]`, and `[crypto]` independently in clean environments.

Done when tests cover:

- Two emails get distinct authenticated tokens and both restore; the same exact value gets one deterministic token across threads/turns and recognizer labels.
- Across two turns sharing one scope/store, values retain tokens and a reply can restore both. A different scope/store or non-ok context result blocks before backend I/O.
- Changed-case, partial, well-formed-but-forged, cross-store, and valid-but-not-authorized tokens never restore and produce the specified finding.
- A user or retrieved document containing an exact live token has it visibly neutralized before backend I/O.
- A zero-width or confusable token spelling that normalization turns into a valid live token remains unauthorized and is neutralized; an unauthorized output token cannot gain authority through assistant history.
- A context token restores only when it occurs exactly in the complete raw reply and authenticated authority manifest. Newly detected output PII round-trips only through its private output-local edit span in `check_output`; assistant history cannot reuse it.
- A token in scheme-relative, `www.`, malformed, userinfo, and non-HTTP(S) candidates stays unrestored. Absolute HTTP(S) restores only for an exact canonical origin, with tests for pinned IDNA behavior, percent-encoded hosts, IPv4/IPv6, zone ids, backslashes, Unicode dots, trailing dots, default/nondefault ports, subdomains, delimiters, and redirects.
- Store insertions at, below, and above `max_entries`, `max_value_bytes`, and canonical `max_bytes` boundaries are atomic. Overflow returns `redaction_limit_exceeded`, no backend call, no leaked value, and no partial map mutation.
- A later context-size, metadata, location, item, or planning failure discards every staged allocation from that call; no earlier transform partially mutates the store.
- Output-local redactions obey `max_call_redactions` and `max_call_redaction_bytes` at exact boundaries and fail atomically with the same finding.
- Finalization over `max_restorations` or `max_restored_output_bytes` applies no restoration and returns a blocking incomplete result.
- `restore=False` issues no conversation token, uses irreversible input markers, disables context restoration, and still protects output PII from judgment backends.
- `alice\u200b@example.com`, `a\u200Dlice@example.com`, per-character variation selectors, and `alice＠example.com` are redacted, whatever order the transforms are listed in.
- The English configuration works in a clean install; another language fails construction in v1.
- A secret that appears twice is redacted twice, and nothing is written to disk.
- `canary(token)` removes every occurrence from the backend projection and blocks even with `fail_fast=False`.
- Concurrent calls on one guard with distinct stores never see each other’s values; shared-store mutation never holds a lock during backend I/O.
- The fake backend’s recorded state contains placeholders, not the original email or a fake API token, including when the email arrived in a string history message.
- `dumps` and `loads` round-trip only with the same key, non-empty scope, and associated data. Oversized blobs fail before decryption; changed scope/header/ciphertext/data/version fails without exposing values. Stores, transactions, manifests/snapshots, and output-local maps reject copy and serialization.

Out of scope: Faker substitution, additional languages, and recognizers beyond the configured Presidio English set.

### Milestone 4 — System One and LiteLLM backends (0.1.0)

Depends on: Milestone 1. It can run in parallel with Milestones 2 and 3; the tag waits for all three.

Steps:

1. Pure request builders and response parsers, with httpx sync and async transports.
2. `SystemOne.local`, `.hosted`, and `.in_process` (extra `[laya]`): immutable backend/deployment profiles; total-window headroom with pinned tokenizer/template counting (extra `[tokenizers]`) or conservative bytes; optional controlled `laya-serve` manifest attestation; per-call `BackendResult`; deadline retries; permit use; and deterministic Jev policy partitions.
3. `LiteLLMJudge` (extra `[litellm]`): required `logprobs` or `verbalized` mode; provider capability profiles; pinned candidate tokenization; deterministic collision-free boundaries; complete-candidate normalization and confidence; strict parsing; one retry; and `completion_fn`.
4. Capture fixtures once from `laya-serve` and Jev, and replay them with `httpx.MockTransport`.
5. README quickstart: transforms, PII, and one `judge()` policy on the Milestone 4 backend the zero-shot spike favored, with an explicitly application-chosen non-default threshold, the data-flow table from section 6, a multi-turn `Redactions` example, complete-reply restoration, sink-escaping guidance, and the absence of streamed/tool-argument restoration.
6. Establish tested Laya, LiteLLM, tokenizer, and httpx compatibility bounds from those fixtures and test each extra in a clean environment.
7. Tag 0.1.0.

Done when:

- Fixture tests cover each question type and score kind, a batch, private-permit usage/canonical public ordering, 429 then success, 500 then failure, 422 without retry, oversized success/error bodies, a malformed/missing answer, and deadline expiry between retries.
- `check_backend_contract` passes for both adapters, sync and async.
- A long text padded with control characters, emoji, and quotes, checked with prompt context, reaches the mock transport as payloads covering every character and fitting the total Laya window after question head, special tokens, and output reserve.
- A stock `laya-serve` HTTP profile without attestation remains default-ineligible; only a trusted-key-signed manifest plus matching live handshake produces a private attestation, and any server/model/config mismatch invalidates it.
- LiteLLM logprob construction rejects candidates that are not one token or exceed the provider’s declared top-logprob limit. Runtime raises when any candidate is absent. Verbalized answers report `kind="verbalized"`. Identical canonical requests render identically, while text containing an attempted boundary cannot forge a field.
- LiteLLM performs no hidden retry/fallback/cache/hedge; every physical call acquires a jes permit, and its injected transport enforces success/error response caps.
- A `BackendError` raised for a 422 whose body echoes the input contains none of that input.

Out of scope: starting servers in CI, and live calls.

### Milestone 5 — Prompt Guard 2 and Llama Guard 4 adapters

Depends on: Milestone 4.

Steps:

1. `PromptGuard2.local` (extra `[prompt-guard]`) with `tasks={"injection"}`; add `.endpoint` only for each named provider profile after its classifier contract has fixtures.
2. `LlamaGuard4.endpoint` and `.local` (extra `[llama-guard]`): immutable provider/model/tokenizer/template profiles; total context windows with derived headroom; `hazard.any` and category tasks; conversation rendering; category subsets where supported; answer-specific score kinds; and fixed logprob mode.
3. Provider-specific endpoint fixtures (Groq label profile and vLLM probability-plus-label profile where supported); canned logits and completions for local mode.
4. License documentation for both adapters.
5. Establish tested transformers, torch, tokenizer, provider-client, and serving-profile bounds; record them in fixtures and clean-install tests.

Done when:

- A completion `unsafe\nS1,S9` gives S1 and S9 label scores of 1.0 and other categories 0.0. A verified logprob completion gives `hazard.any` a probability score while categories remain labels; a label-only provider gives every answer `kind="label"`. Missing `safe`/`unsafe` candidates or logprobs raise `BackendError`.
- Prompt Guard runs on explicit override attempts in input and untrusted stages; the broader `indirect_injection` task remains unsupported. An endpoint without a named fixture-backed profile fails construction.
- Routing a `topics` question to either adapter raises `PolicyError` at construction.
- Prompt Guard’s 512-token and Groq’s 131,072-token totals yield smaller verified subject headroom after classifier/conversation special tokens, categories, templates, and output reserve; no request relies on those totals as text budgets.
- Both adapters pass `check_backend_contract`.
- `import jes` still works without torch.

Out of scope: images, quantization, ONNX.

### Milestone 6 — Evaluation harness

Depends on: Milestones 3, 4, and 5.

Entry criterion: `evals/protocol.md` accepts the target false-positive rate, operational failure treatment, grouped development/selection/calibration/sealed-audit rules, independent-cluster support, simultaneous-confidence method, recommendation metric, and tie-breakers from section 11. These are no longer open once data inspection begins.

Steps:

1. `evals/datasets.toml` with pinned sources, licenses, hosted-processing terms, and retention review; grouped development, selection, calibration, sealed audit, and future rolling-holdout splits with duplicate/template/padding-family isolation.
2. Template generators for judge-directed and padding attacks, including control-character padding.
3. Tune draft question text on development splits only. Before calibration, copy the selected bytes unchanged to exact candidate ids such as `injection.v1` and freeze their hashes; calibration, audit, and cache keys use those exact ids and bytes.
4. Implement private, non-exported candidate policies for `injection`, `indirect_injection`, `hazards`, `topics`, and `toxicity` with their real task ids, stages, subsets, interpretation/merge versions, and frozen question bytes. Fixed-task adapters evaluate these candidates; `judge()` remains task `"custom"`.
5. A runner with bounded concurrency, development-only answer caching, uncached acceptance measurement, every completion/error metric, confidence/support rule, request/decision profile, threshold vector, bundle result, and withholding rule from section 11; JSON results and a generated report.
6. PII metrics for both `ner` choices, including machine-identifier Unicode evasions.
7. A representative fixed suite for `topics.v1` wording and robustness, without generating a default threshold for caller-defined topics.
8. A smoke run against the fake backend in CI.

Done when a manual run ranks on selection data, freezes a pre-registered finalist set, calibrates it, then reports every applicable operational/model metric on the once-opened sealed audit, including all failed attempts, padding, and judge-directed attacks, for finalists drawn from the following backend families (“LiteLLM” includes gpt-oss-safeguard):

- injection: Laya, Jev, LiteLLM, Prompt Guard 2;
- indirect injection: Laya, Jev, LiteLLM, and Prompt Guard 2 (through `injection`) as a baseline;
- hazards: Laya, Jev, LiteLLM, and each Llama Guard 4 request/decision profile, with answer-specific score kinds;
- toxicity: Laya, Jev, LiteLLM;
- topics: prompt-robustness results on the fixed representative suite.

The same run reports the separate PII metric contract from section 11 for `ner="spacy"` and one transformer configuration.

The run must contain enough grouped support for its confidence-bound rules. Profiles below support remain exploratory and cannot generate defaults.

Out of scope: fine-tuning, and committing raw data.

### Milestone 7 — Core judgment policies (0.2.0)

Depends on: Milestones 3 and 6.

Steps:

1. Promote the exact private candidate implementations, question ids/bytes/hashes, interpretation versions, and task routing evaluated in Milestone 6 into public factories without behavioral changes. Any change returns to Milestone 6 with fresh acceptance data.
2. Leave `jes/policies/defaults.py` empty for this tag. A later audit may add block-only (`flag_at=None`) entries for qualified decision-profile fingerprints, with request-profile reference, threshold, run id, independent support, completion/error rates, and confidence metadata.
3. Export factories with category and label subsets, `version=`, and `backend=` overrides. A subset uses a default only when that exact decision profile was evaluated.
4. README: no results table and no recommended backend. Every shown threshold is an application choice.
5. Tag 0.2.0.

Done when:

- Snapshot tests pin every `v1` instruction.
- Threshold boundary tests pass for each policy with the fake backend.
- A hazard response with two violating categories yields two findings.
- `hazards` on a backend that answers `hazard.any` blocks on that score and names the listed codes; with no codes listed, it reports `unattributed`.
- Construction without a threshold fails for a floating, unverifiable, changed, or missing decision profile and succeeds only for an exact fingerprint in `defaults.py`; bundles require a complete threshold vector.
- 0.2.0 ships with `THRESHOLDS` empty. A README-recommended bundle’s exact request profile and threshold-vector fingerprint pass the pre-registered guard-level sealed-audit gate; otherwise no bundle recommendation is published.

Out of scope: recipes.

### Milestone 8 — Recipes (0.3.0)

Depends on: Milestone 7.

Steps:

1. Implement each recipe in section 10.4 from public APIs only, one pull request per recipe, each with its versioned question text, fake-backend tests, and a docs page.
2. Tag 0.3.0. The recipe modules are the release. They stay unevaluated.

Done when: every LLM Guard scanner maps to a core policy or a recipe (section 10.5), and URLReachability is documented as dropped. Every judgment recipe rejects a missing threshold; `malicious_urls` judges each URL as its own item, `relevance` runs on the whole text, and `factual_consistency` uses sources when passed.

Out of scope: evaluating recipes. Promotion to core requires evaluation data.

### Milestone 9 — 1.0.0

Depends on: Milestone 8.

Steps:

1. `docs/migration.md`: each LLM Guard class and its jes equivalent, with notes that thresholds do not carry over and that PII recall differs unless `ner=` names a transformer model like LLM Guard’s default.
2. `__all__` on public modules; pyright strict with no ignores outside the optional-import shims.
3. README limitations: short context on local Laya, label-only scores without log-probabilities, explicit incomplete fail-open results, output judgments/restoration only on complete replies, plain-text-only restoration, no streamed/tool-argument restoration, no images, no multi-turn detection, no decoding of obfuscated payloads, and no safety guarantee.
4. Freeze core and recipe question ids and every public factory signature. Rerun the evaluation on the release candidate and publish it.
5. Tag 1.0.0.

Done when: a new user can follow the README and see a blocked injection plus an email redacted/restored using documented imports. The example uses a recommended default when one qualified; if evaluation recommends none, it uses an audit-evaluated profile with a clearly labeled application-chosen explicit threshold and makes no recommendation/default claim.

### Milestone 10 — HTTP service (after 1.0.0)

Build this only if a deployment needs a sidecar like `llm_guard_api`.

Depends on: Milestone 9.

Steps:

1. Extra `[server]`: FastAPI, with one guard per process built from a YAML config.
2. `POST /check/input`, `/check/untrusted`, and `/check/output`; `GET /healthz`, which loads no models; an optional bearer token. Middleware enforces content-length and streaming body caps before JSON/Base64 parsing; validation errors and access logs never echo bodies.
3. Successful context-bearing check responses carry the conversation’s `Redactions` store as an opaque blob. Health, authentication failures, and pre-conversation errors do not. The service constructs stores with tenant/application/conversation scope; `dumps` uses a rotated key and key-id/purpose AAD, and `loads` applies pre-parse/decryption caps, expiry, and scope binding.
4. Every reusable jes result carries a bounded opaque authority envelope produced by a private canonical codec. It authenticates every field required by the core fast path: codec/engine/config versions, tenant/conversation scope, store/result ids, stage, sanitized digest, decision, completion, findings digest, full bounded authority entries/digest, and expiry. Another worker validates fixed fields/HMAC/caps before decoding entries, then issues a local stamp; client-written fields never become authority.
5. Context envelopes are intentionally reusable within their scope until expiry. A client-carried store blob is a replayable snapshot; the stateless service does not claim fork/replay prevention and requires serialized requests per conversation. Deployments needing concurrent same-conversation updates add an external atomic version/CAS store and bind its version into blob/envelope AAD.
6. No policy logic in the server; it calls the library.

Done when multi-worker tests create input on worker A, validate prompt/history envelopes and restore on worker B, reject cross-scope/tampered/expired/oversized data before deep parsing, permit legitimate in-scope context reuse, document snapshot replay semantics, and keep workers free of plaintext per-conversation state between requests.

### Milestone 11 — Incremental output (2.0 candidate)

Depends on: Milestone 9.

Build this only in a 2.0 design; v1 means the complete 1.x line and excludes streamed restoration.

1. Define an incremental transform/finalizer protocol for every output transform that can alter token or URI boundaries.
2. Permit streamed restoration only when every configured output transform implements that protocol; otherwise construction rejects it.
3. Property-test every stream split against the complete-reply projection and finalizer, including Unicode/control-character and URI parser differentials.
4. Bound total stream input/output, restoration count/bytes, findings/locations, pending state, and abandoned worker work. Output judgments still run on the complete reply unless another evaluated design changes that rule.

Done when every configured stream-compatible transform/finalizer passes split-equivalence and prefix-safety properties, incompatible guards fail construction, and no 1.x compatibility or guarantee is retroactively weakened.

## 17. First implementation session

Do Milestones 0 and 1 only, then stop. Milestone 1’s tests are the contract every backend and policy must satisfy. The zero-shot spike can run alongside; it needs a `laya-serve` instance, a Jev key, and Prompt Guard 2 access, not package code.

## 18. Acceptance

Accepted: package name `jes`, Apache-2.0, evaluated decision-profile thresholds over pinned request profiles, caller-selected recommended backends, scoped authenticated redaction stores, strict completion semantics, complete plain-text restoration in v1, no streamed or generic tool restoration, and the milestone order above. The name appeared available on PyPI when checked on 2026-09-25, but Milestone 0 uses TestPyPI only and the first public release is functional 0.1.0. jes is one letter from Jev, so the README states its relationship to TypeSafe plainly.

Open points that do not block Milestones 0–5:

- The target false-positive rate and backend-recommendation protocol (proposed: 1% on the benign set with the section 11 rule). They must be accepted in `evals/protocol.md` before Milestone 6, but do not block Milestones 0–5.
- The benign and hazard datasets, chosen after license review in Milestone 6.
- The two LiteLLM models besides gpt-oss-safeguard (proposed: one cheap hosted non-reasoning model that returns log-probabilities, and one local model through Ollama).
- The 0.1.0 quickstart backend, decided by the zero-shot spike before Milestone 4’s README step.
