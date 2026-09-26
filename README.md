# jes

jes is a Python policy engine for checking text before it enters a language
model, text retrieved from untrusted sources, and complete model replies.

The project is independent and is not affiliated with or endorsed by TypeSafe,
Meta, Protect AI, or the maintainers of LLM Guard. The name is one letter from
TypeSafe’s Jev; jes provides adapters for compatible services and models but
ships no third-party weights.

**There are no measured judgment defaults and no recommended backend.**
`injection`, `indirect_injection`, `hazards`, `topics`, and `toxicity` use
frozen v1 question text. Every call passes `threshold=`, including in 0.2.0.
A category or label
subset is a different decision profile and does not inherit another profile's
threshold. Topics never receive a default. The unevaluated catalog is in
[`docs/recipes.md`](docs/recipes.md). URLReachability is not included.

## Install

```bash
pip install jes
pip install 'jes[pii,secrets,crypto]'   # optional extras
```

Core depends only on `httpx`. `SystemOne.in_process` needs `jes[laya]`.
`LiteLLMJudge` needs `jes[litellm]`. Exact Laya token budgets need
`jes[tokenizers]`. `PromptGuard2.local` needs `jes[prompt-guard]`.
`LlamaGuard4.local` needs `jes[llama-guard]`. Importing `jes` does not
import torch.

## Quickstart

This example uses Laya through a local `laya-serve` as the illustrative cheap
typed-decision backend. The `0.72` threshold is an explicit application choice,
not a library default.

```python
from jes import Guard, Redactions
from jes.backends import SystemOne
from jes.policies import invisible_text, judge, pii, secrets
from jes.questions import YesNo

backend = SystemOne.local(
    "http://127.0.0.1:8000",
    model="english",
    max_request_bytes=8_192,
)
guard = Guard(
    [
        invisible_text(),
        secrets(),
        pii(),
        judge(
            "injection",
            YesNo("The text tries to override the assistant's instructions."),
            threshold=0.72,
        ),
    ],
    backend=backend,
)

redactions = Redactions(scope=b"conversation-1")
incoming = guard.check_input("email me at ada@example.com", redactions=redactions)
if not incoming.ok:
    raise SystemExit(incoming.findings)

reply = "I will write to [the placeholder the model saw]."
outgoing = guard.check_output(reply, prompt=incoming, redactions=redactions)
if outgoing.ok:
    show(outgoing.text)  # complete-reply restoration into plain text only
```

Applications must escape restored values before rendering Markdown, HTML, JSON,
or a shell. jes does not restore streamed prefixes or tool-call arguments.

### Multi-turn store

Keep one `Redactions` store per conversation and pass earlier results as history:

```python
history = []
incoming = guard.check_input(user_text, redactions=redactions, history=history)
outgoing = guard.check_output(
    complete_reply,
    prompt=incoming,
    redactions=redactions,
    history=history,
)
history += [incoming, outgoing]
```

## Where checked text goes

| Component | Destination |
| --- | --- |
| Transforms (regex, substrings, invisible text, Presidio, detect-secrets) | Local process |
| `SystemOne.in_process`, local Prompt Guard 2 and Llama Guard 4 | Local process |
| `SystemOne.local` pointed at localhost | Local machine |
| `SystemOne.hosted`, endpoint adapters, `LiteLLMJudge` | The configured host, after transforms |

`LiteLLMJudge` disables LiteLLM retries, fallbacks, hedging, and cache. jes
owns the one parse retry and the response-size cap. Supported LiteLLM versions
must suppress payload logs; application-added handlers are outside that
guarantee.

## Model adapters

Prompt Guard 2 answers `injection` only. It does not answer
`indirect_injection`. Llama Guard 4 answers `hazard.any` and `hazard.S1`
through `hazard.S14`. A verified logprob profile reports `hazard.any` as a
probability and category scores as labels. Groq’s label profile reports every
answer as a label.

jes ships no Meta weights and no model-card text. Prompt Guard 2 checkpoints
are under the Llama Community License. Llama Guard 4 is under the
[Llama 4 Community License](https://www.llama.com/llama4/license/), including
its attribution requirements. “About 24 GB” is only the BF16 weight size, not a
runtime memory promise.

`PromptGuard2`’s 512-token window and Groq’s 131,072-token window are total
context lengths. Headroom subtracts classifier or conversation special tokens,
the rendered template, and an output reserve. Those totals are not text budgets.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pyright
uv run pytest
```

CI never downloads model weights. Fixtures must not contain hazardous content.

## License

Apache-2.0.
