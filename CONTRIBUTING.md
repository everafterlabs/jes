# Contributing

## Setup

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
node --experimental-strip-types --test tests/plugins/*.test.ts   # Node 22
```

## Rules

- CI must not download model weights or call live model providers. Use
  `jes.testing.FakeBackend`, or a fake classifier as in `tests/test_backend.py`.
- Fixtures must not contain hazardous content, secrets, or real personal data.
  Use fake backend scores and synthetic non-sensitive markers.
- Built-in question text is frozen in `src/jes/policies/prompts.py`, and
  `tests/test_prompts.py` records a hash of each question. A wording change
  needs a new constant and id, such as `INJECTION_V2`. Do not edit an existing
  one in place.
- Public APIs must remain typed under strict pyright.
- A new built-in backend implements the protocol in `src/jes/backend.py` and
  turns every failure into `BackendError`.
- Security-sensitive edits need tests for all applicable guarantees in section 3
  of `docs/design.md`.

## Documentation

The documentation site, [docs.getjes.dev](https://docs.getjes.dev), is built
from a separate repository, `jes-docs`. Guides, the API reference, and the recipe catalog live there. It pulls these files from
this repo at build time, so edit them here:

- `examples/*.py`, which the cookbook embeds, and `docs/cookbook.md`
- `CHANGELOG.md`
- `docs/design.md`

A change to a name in any module's `__all__` also needs an update to the API
reference in `jes-docs`, whose CI checks for missing names.

## Pull requests

Keep changes small, explain observable behavior, and include tests.
