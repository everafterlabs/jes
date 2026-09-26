# Contributing

## Setup

```bash
uv sync --dev
uv run ruff check .
uv run pyright
uv run pytest
```

## Rules

- CI must not download model weights or call live model providers.
- Fixtures must not contain hazardous content, secrets, or real personal data.
  Use fake backend scores and synthetic non-sensitive markers.
- Built-in question text is versioned. Changing wording requires a new prompt
  version; do not mutate an existing version in place.
- Public APIs must remain typed under strict pyright.
- New built-in backends must pass `jes.testing.check_backend_contract`.
- Security-sensitive edits need tests for all applicable guarantees in section 5
  of `docs/design.md`.

## Pull requests

Keep changes small, explain observable behavior, and include tests. Generated
evaluation results must identify their immutable request and decision profiles.
