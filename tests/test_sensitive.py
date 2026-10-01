"""pii, secrets, and canary detection, and how each origin chooses its replacement."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from jes.errors import PolicyError
from jes.policies import TransformContext, canary, pii, secrets, sensitive as sensitive_module
from jes.policies.base import SensitiveHit
from jes.types import Stage

NO_PERSON = ("EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE", "CRYPTO")


def _context(origin: Stage, stage: Stage | None = None) -> TransformContext:
    return TransformContext(stage=stage or origin, origin=origin, target="text")


def _found(hits: list[SensitiveHit], text: str) -> list[tuple[str, str]]:
    return [(hit.entity, text[hit.span.start : hit.span.end]) for hit in hits]


def test_default_pii_needs_a_spacy_model_for_person() -> None:
    # B9: PERSON was silently skipped without a model. Now pii() says so.
    with pytest.raises(PolicyError, match="spaCy English model"):
        pii()


def test_pii_finds_each_default_entity() -> None:
    text = (
        "mail ada@example.com, call (555) 123-4567 or 555.123.4567, card 4111 1111 1111 1111, "
        "ssn 078-05-1120, iban DE89370400440532013000, btc 1BoatSLRHtKNngkdXEeobR76b53LETtpyT"
    )
    hits = pii(NO_PERSON).detect(text, _context("input"))
    assert _found(hits, text) == [
        ("EMAIL_ADDRESS", "ada@example.com"),
        ("PHONE_NUMBER", "(555) 123-4567"),
        ("PHONE_NUMBER", "555.123.4567"),
        ("CREDIT_CARD", "4111 1111 1111 1111"),
        ("US_SSN", "078-05-1120"),
        ("IBAN_CODE", "DE89370400440532013000"),
        ("CRYPTO", "1BoatSLRHtKNngkdXEeobR76b53LETtpyT"),
    ]


def test_pii_skips_lookalikes_of_numbers() -> None:
    # A bare timestamp is not a phone number, and a card number must pass the Luhn check.
    text = "at 1727800000 the order 4111 1111 1111 1112 shipped"
    assert pii(NO_PERSON).detect(text, _context("input")) == []


def test_uuid_ip_and_bank_numbers_are_opt_in() -> None:
    text = (
        "container 3f2b1c4e-1111-2222-3333-444455556666 on 10.0.0.255 acct 123456789, not 999.1.1.1"
    )
    assert pii(NO_PERSON).detect(text, _context("input")) == []
    opted = pii(("UUID", "IP_ADDRESS", "US_BANK_NUMBER")).detect(text, _context("input"))
    assert _found(opted, text) == [
        ("UUID", "3f2b1c4e-1111-2222-3333-444455556666"),
        ("IP_ADDRESS", "10.0.0.255"),
        ("US_BANK_NUMBER", "123456789"),
    ]


def test_hidden_characters_do_not_hide_an_email() -> None:
    text = "write to ada\u200b@exa\u00admple.com today"
    (hit,) = pii(["EMAIL_ADDRESS"]).detect(text, _context("input"))
    assert text[hit.span.start : hit.span.end] == "ada\u200b@exa\u00admple.com"


def test_overlapping_matches_keep_the_earlier_longer_one() -> None:
    text = "mail 555-123-4567@example.com"
    hits = pii(["EMAIL_ADDRESS", "PHONE_NUMBER"]).detect(text, _context("input"))
    assert _found(hits, text) == [("EMAIL_ADDRESS", "555-123-4567@example.com")]


@pytest.mark.parametrize(
    ("settings", "origin", "mode", "action"),
    [
        ({}, "input", "token", "redact"),
        ({"restore": False}, "input", "remove", "redact"),
        ({"input_mode": "mask"}, "input", "partial", "redact"),
        ({"input_mode": "block"}, "input", "remove", "block"),
        ({}, "untrusted", "partial", "redact"),
        ({"untrusted_mode": "redact"}, "tool_result", "remove", "redact"),
        ({"untrusted_mode": "block"}, "untrusted", "partial", "block"),
        ({}, "output", "local", "flag"),
        ({"output_mode": "redact"}, "output", "remove", "redact"),
        ({"output_mode": "block"}, "output", "remove", "block"),
        ({}, "tool_call", "remove", "block"),
        ({"tool_call_mode": "flag"}, "tool_call", "remove", "flag"),
    ],
)
def test_pii_replacement_by_origin(
    settings: dict[str, object], origin: Stage, mode: str, action: str
) -> None:
    policy = pii(["EMAIL_ADDRESS"], **settings)  # type: ignore[arg-type]
    (hit,) = policy.detect("ada@example.com", _context(origin))
    assert (hit.mode, hit.action) == (mode, action)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"entities": []}, "at least one entity"),
        ({"entities": ["PASSPORT"]}, "unknown pii entity: PASSPORT"),
        ({"entities": ["EMAIL_ADDRESS"], "input_mode": "hide"}, "input_mode"),
        ({"entities": ["EMAIL_ADDRESS"], "untrusted_mode": "hide"}, "untrusted_mode"),
        ({"entities": ["EMAIL_ADDRESS"], "output_mode": "hide"}, "output_mode"),
        ({"entities": ["EMAIL_ADDRESS"], "tool_call_mode": "redact"}, "tool_call_mode"),
        ({"entities": ["EMAIL_ADDRESS"], "on_placeholder_in_url": "flag"}, "on_placeholder_in_url"),
        (
            {"entities": ["EMAIL_ADDRESS"], "restore_origins": ["app.example.com"]},
            "restore_origins",
        ),
        ({"entities": ["EMAIL_ADDRESS"], "stages": ()}, "at least one stage"),
    ],
)
def test_pii_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        pii(**kwargs)  # type: ignore[arg-type]


def test_restore_origins_are_parsed() -> None:
    policy = pii(["EMAIL_ADDRESS"], restore_origins=["https://app.example.com"])
    assert policy.restore_origins == {("https", "app.example.com", 443)}
    assert policy.hmac_key is None


@dataclass
class _Entity:
    start: int
    end: int
    entity_type: str


class _Analyzer:
    """Stands in for Presidio: finds the capitalized names it is given."""

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def analyze(self, *, text: str, language: str, entities: list[str]) -> list[_Entity]:
        assert language == "en" and entities == ["PERSON"]
        found = [
            _Entity(text.find(name), text.find(name) + len(name), "PERSON") for name in self.names
        ]
        return [*found, _Entity(0, 0, "PERSON"), _Entity(0, 2, "LOCATION")]


@pytest.fixture
def people(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sensitive_module, "_installed_spacy_model", lambda: "en_core_web_sm")
    monkeypatch.setattr(sensitive_module._People, "_analyzer", _Analyzer(["Ada Lovelace"]))
    yield


def test_person_detection_maps_back_through_nfkc(people: None) -> None:
    text = "From \uff21da Lovelace, ada@example.com"
    hits = pii().detect(text, _context("input"))
    assert _found(hits, text) == [
        ("PERSON", "\uff21da Lovelace"),
        ("EMAIL_ADDRESS", "ada@example.com"),
    ]


def test_person_needs_presidio(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "presidio_analyzer" else real(name, *args),
    )
    with pytest.raises(PolicyError, match=r"jes\[pii\]"):
        pii(["PERSON"])


def test_spacy_model_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sensitive_module._installed_spacy_model() is None
    spacy_util = sys.modules.get("spacy.util")
    assert spacy_util is not None
    monkeypatch.setattr(spacy_util, "is_package", lambda name: name == "en_core_web_md")
    assert sensitive_module._installed_spacy_model() == "en_core_web_md"
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "spacy" else real(name, *args),
    )
    assert sensitive_module._installed_spacy_model() is None


def test_presidio_is_built_once_with_the_installed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []

    class Provider:
        def __init__(self, nlp_configuration: dict[str, object]) -> None:
            built.append(nlp_configuration)

        def create_engine(self) -> str:
            return "engine"

    class Engine:
        def __init__(self, nlp_engine: str, supported_languages: list[str]) -> None:
            built.append((nlp_engine, supported_languages))

    nlp = ModuleType("presidio_analyzer.nlp_engine")
    nlp.NlpEngineProvider = Provider  # type: ignore[attr-defined]
    analyzer = ModuleType("presidio_analyzer")
    analyzer.AnalyzerEngine = Engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "presidio_analyzer", analyzer)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp)
    monkeypatch.setattr(sensitive_module._People, "_analyzer", None)
    monkeypatch.setattr(sensitive_module, "_installed_spacy_model", lambda: None)
    with pytest.raises(PolicyError, match="spaCy English model"):
        sensitive_module._People.analyzer()

    monkeypatch.setattr(sensitive_module, "_installed_spacy_model", lambda: "en_core_web_lg")
    first = sensitive_module._People.analyzer()
    assert sensitive_module._People.analyzer() is first
    assert built == [
        {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        },
        ("engine", ["en"]),
    ]


SECRETS = {
    "openai": "sk-proj-" + "A1b2C3d4" * 6,
    "anthropic": "sk-ant-api03-" + "x" * 40,
    "github": "ghp_" + "a" * 36,
    "github_pat": "github_pat_" + "B" * 30,
    "slack": "xoxb-1234567890-abcdefghij",
    "aws": "AKIA" + "ABCDEFGHIJKLMNOP",
    "google": "AIza" + "Z" * 35,
    "stripe": "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
}


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_secrets_find_current_key_formats(name: str) -> None:
    value = SECRETS[name]
    text = f"key: {value} end"
    hits = secrets().detect(text, _context("input"))
    assert [(hit.entity, text[hit.span.start : hit.span.end]) for hit in hits] == [
        ("secret", value)
    ]
    assert (hits[0].mode, hits[0].action) == ("mask", "redact")


def test_secrets_find_private_keys_and_passwords_but_not_public_ips() -> None:
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
    text = f'config:\n{key}\npassword = "hunter2-correct"\nserver 8.8.8.8\n'
    found = [
        text[hit.span.start : hit.span.end] for hit in secrets().detect(text, _context("input"))
    ]
    assert key in found
    assert "hunter2-correct" in found
    assert not any("8.8.8.8" in item for item in found)


def test_secret_modes_and_tool_calls() -> None:
    value = SECRETS["github"]
    assert secrets("partial").detect(value, _context("input"))[0].mode == "partial"
    keyed = secrets("hmac", key=b"k" * 32)
    assert keyed.hmac_key == b"k" * 32
    assert "k" * 32 not in repr(keyed)
    assert keyed.detect(value, _context("input"))[0].mode == "hmac"
    assert secrets().detect(value, _context("tool_call"))[0].action == "block"
    with pytest.raises(PolicyError, match="32 bytes"):
        secrets("hmac", key=b"short")
    with pytest.raises(PolicyError, match="only for hmac"):
        secrets("all", key=b"k" * 32)
    with pytest.raises(PolicyError, match="redact must be"):
        secrets("some")  # type: ignore[arg-type]


def test_secrets_need_their_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "detect_secrets" else real(name, *args),
    )
    with pytest.raises(PolicyError, match=r"jes\[secrets\]"):
        secrets()


def test_secrets_skip_plugins_without_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    class Empty:
        def analyze_line(self, **kwargs: object) -> list[object]:
            return [SimpleNamespace(secret_value=None)]

    monkeypatch.setattr(sensitive_module, "_secret_plugins", lambda: (Empty(),))
    assert secrets().detect("nothing here", _context("input")) == []


def test_canary_catches_disguised_spellings() -> None:
    policy = canary("CANARY-7f3a")
    assert policy.stages == {"output", "tool_call"}
    for text in (
        "leak: CANARY-7f3a",
        "leak: canary-7f3a",
        "leak: CA\u200bNARY-7f3a",
        "leak: \uff23\uff21\uff2e\uff21\uff32\uff39-7f3a",
    ):
        (hit,) = policy.detect(text, _context("output"))
        assert (hit.entity, hit.mode, hit.action) == ("canary", "remove", "block")
        assert text[hit.span.start : hit.span.end] == text[len("leak: ") :]
    assert policy.detect("nothing to see", _context("output")) == []
    assert "7f3a" not in repr(policy)
    with pytest.raises(PolicyError, match="must not be empty"):
        canary("  ")
    with pytest.raises(PolicyError, match="folds to nothing"):
        canary("\u200b")
