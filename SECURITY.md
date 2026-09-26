# Security policy

## Reporting

Please report suspected vulnerabilities privately through GitHub Security
Advisories. Do not open a public issue for a vulnerability or include checked
text, credentials, PII, model prompts, or provider response bodies in a report.

## Supported versions

No public version is currently supported. This section will list supported
release lines before the first public release.

## What counts as a vulnerability

Security issues include, but are not limited to:

- a bypass of a guarantee in section 5 of `docs/design.md`;
- checked text or redacted values appearing in jes-owned logs or exceptions;
- cross-conversation redaction restoration;
- silent client-side truncation in strict mode;
- a backend receiving a value removed by a jes-owned sensitive transform;
- accepting a forged sanitization stamp, authority manifest, or backend profile
  attestation.

Incorrect model judgments without a contract bypass are quality bugs rather than
security vulnerabilities.
