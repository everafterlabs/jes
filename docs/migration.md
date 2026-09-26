# Migrating from LLM Guard

jes is not a drop-in replacement. A scanner becomes a policy. The model is a
backend you pass in. An LLM Guard threshold does not carry over: pass a new
`threshold=` for every judgment. There is no published jes default and no
recommended backend.

PII recall will differ from LLM Guard unless `ner=` names a transformer model.
The default `ner="spacy"` is Presidio’s English spaCy model. LLM Guard’s
default anonymizer was a DeBERTa model fine-tuned on ai4privacy data.

Output scanners that only wrapped an input scanner are the same jes policy,
registered for the `output` stage. There is no second class to construct.

`URLReachability` is not implemented. jes does not fetch a URL chosen by model
output.

| LLM Guard | jes |
| --- | --- |
| InvisibleText | `invisible_text` |
| Regex | `regex` |
| BanSubstrings | `substrings` |
| TokenLimit | `token_limit` |
| Secrets | `secrets` |
| Anonymize / Deanonymize | `pii` with `restore=True` |
| Sensitive | `pii(output_mode=...)` |
| PromptInjection | `injection` and `indirect_injection` |
| BanTopics | `topics` |
| Toxicity | `toxicity` |
| Llama Prompt Guard 2 | `injection` with a `PromptGuard2` backend |
| Llama Guard 4 | `hazards` with a `LlamaGuard4` backend, or any general backend |
| BanCompetitors | `jes.recipes.competitors` |
| ReadingTime | `jes.recipes.reading_time` |
| JSON | `jes.recipes.json_check` |
| Sentiment | `jes.recipes.sentiment` |
| EmotionDetection | `jes.recipes.emotions` |
| Gibberish | `jes.recipes.gibberish` |
| Language | `jes.recipes.language` |
| LanguageSame | `jes.recipes.language_same` |
| Bias | `jes.recipes.bias` |
| NoRefusal / NoRefusalLight | `jes.recipes.refusal` and `jes.recipes.refusal_phrases` |
| BanCode / Code | `jes.recipes.code` |
| MaliciousURLs | `jes.recipes.malicious_urls` |
| Relevance | `jes.recipes.relevance` |
| FactualConsistency | `jes.recipes.factual_consistency` |
| URLReachability | Dropped |

Core and recipe question ids are frozen at `v1` in 1.0. A wording change needs
a new id. Factory signatures in `jes.policies` and `jes.recipes` are frozen
the same way.
