# Changelog

All notable changes to jes will be documented here.

The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 1.0.0

- Freeze the public factory signatures and the v1 question ids.
- Add `docs/migration.md` and the README limitations. Every judgment still
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
