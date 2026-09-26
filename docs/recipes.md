# Recipes

Recipes are unevaluated policies built from `judge`, `substrings`, and custom
transforms. They have no default thresholds and no recommended backend.
Promotion into core requires a fresh evaluation. Question text is version `v1`.

URLReachability is not implemented. Fetching a URL chosen by model output is an
SSRF risk, and the check adds little once `malicious_urls` has judged the URL
string itself. jes does not request those hosts.

| Recipe | Threshold | What it checks |
| --- | --- | --- |
| `sentiment` | required | The text is hostile or strongly negative. |
| `emotions` | required | One yes/no question per emotion. The default set is the v1 negative GoEmotions labels. |
| `gibberish` | required | The text is not meaningful language. |
| `bias` | required | Output only. The text demeans or stereotypes a group. |
| `refusal` | required | Output only. The text refuses the request. |
| `refusal_phrases` | none | Blocks on the v1 refusal-phrase list. |
| `language` | required | Choice of allowed language codes plus `other`. `other` is the violation. |
| `language_same` | required | Whole output, required prompt. The reply is in a different language. |
| `code` | required | `mode="ban"` or `"allow"` over a fixed language set plus `not_code`. |
| `competitors` | none | Redacts the given names. |
| `malicious_urls` | required | Each `http`/`https` URL is its own item, at most 20. The question is about the URL string only. |
| `relevance` | required | Whole output, required prompt. The reply does not address the prompt. |
| `factual_consistency` | required | Whole output, required prompt, and sources when they are passed. |
| `reading_time` | none | 200 words per minute. `mode="block"` or `"truncate"`. |
| `json_check` | none | One JSON array or object. `repair=True` needs `jes[json]`. |
