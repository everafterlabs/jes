# Security policy

## Reporting

Please report suspected vulnerabilities privately through GitHub Security
Advisories. Do not open a public issue for a vulnerability or include checked
text, credentials, PII, model prompts, or provider response bodies in a report.

## Supported versions

| Version | Security fixes |
| --- | --- |
| 2.x | Yes |
| 1.x | No. Upgrade to 2.x. |

## What counts as a vulnerability

Security issues include, but are not limited to:

- a bypass of a guarantee in section 3 of `docs/design.md`;
- checked text or redacted values appearing in jes-owned logs, exceptions, or reprs;
- a backend receiving a value that `pii`, `secrets`, or `canary` found;
- text that no judgment saw, in a result that is still `complete`;
- restoring a value from another conversation, a forged placeholder, or a token
  inside a URL whose origin is not in `restore_origins`;
- a coding-agent hook that reads settings or credentials from the project it
  runs in, or writes its state where other users can read it;
- a hook that allows an event after its check failed, where that agent lets the
  hook block.

Incorrect model judgments without a contract bypass are quality bugs rather than
security vulnerabilities.
