from __future__ import annotations

import copy
import pickle
import threading

import pytest

from jes import Guard, Redactions
from jes.errors import PolicyError, RedactionError
from jes.policies import canary, invisible_text, pii, secrets
from jes.policies.secrets import _reset_secrets_config
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend
from tests.helpers import yesno_policy

_FAKE_KEY = "sk-" + ("a" * 20)
_EMAIL_A = "alice@example.com"
_EMAIL_B = "bob@example.com"


def _backend() -> FakeBackend:
    return FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})


def test_two_emails_get_distinct_tokens_and_restore() -> None:
    backend = _backend()
    store = Redactions(scope=b"conversation-1")
    guard = Guard(
        [invisible_text(), pii(restore_origins=["https://allowed.example"]), yesno_policy()],
        backend=backend,
    )
    incoming = guard.check_input(f"mail {_EMAIL_A} and {_EMAIL_B}", redactions=store)
    assert incoming.ok
    assert _EMAIL_A not in incoming.sanitized
    assert _EMAIL_B not in incoming.sanitized
    values = incoming.redactions.snapshot_values()
    assert set(values.values()) == {_EMAIL_A, _EMAIL_B}
    tokens = list(values)
    assert tokens[0] != tokens[1]
    reply = " ".join(tokens)
    outgoing = guard.check_output(reply, prompt=incoming, redactions=store)
    assert outgoing.ok
    assert _EMAIL_A in outgoing.text
    assert _EMAIL_B in outgoing.text
    assert _EMAIL_A not in outgoing.sanitized
    assert _EMAIL_A not in backend.calls[0][0].text


def test_same_value_is_one_token_across_threads_and_turns() -> None:
    backend = _backend()
    store = Redactions(scope=b"shared")
    guard = Guard([pii()], backend=backend)
    incoming = guard.check_input(f"first {_EMAIL_A}", redactions=store)
    token = next(iter(incoming.redactions.snapshot_values()))

    def later() -> str:
        again = Guard([pii()], backend=backend).check_input(
            f"again {_EMAIL_A}",
            redactions=store,
        )
        return next(iter(again.redactions.snapshot_values()))

    threads = [threading.Thread(target=later) for _ in range(4)]
    for thread in threads:
        thread.start()
    tokens = [later() for _ in range(2)]
    for thread in threads:
        thread.join()
    assert set(tokens) == {token}
    assert len(store) == 1


def test_forged_and_unauthorized_tokens_do_not_restore() -> None:
    backend = _backend()
    store = Redactions(scope=b"scope-a")
    guard = Guard([pii()], backend=backend)
    incoming = guard.check_input(f"mail {_EMAIL_A}", redactions=store)
    token = next(iter(incoming.redactions.snapshot_values()))
    other = Guard([pii()], backend=backend).check_output(
        token,
        prompt="raw prompt",
        redactions=Redactions(scope=b"scope-b"),
    )
    assert _EMAIL_A not in other.text
    changed = token[:-2] + "x]"
    blocked = guard.check_output(changed, prompt=incoming, redactions=store)
    assert _EMAIL_A not in blocked.text


def test_live_token_in_user_text_is_neutralized() -> None:
    backend = _backend()
    store = Redactions(scope=b"scope")
    guard = Guard([pii(), yesno_policy()], backend=backend)
    incoming = guard.check_input(f"mail {_EMAIL_A}", redactions=store)
    token = next(iter(incoming.redactions.snapshot_values()))
    leaked = guard.check_input(f"copied {token}", redactions=store)
    assert "JES_LITERAL" in leaked.sanitized
    assert token not in leaked.sanitized
    assert any(finding.label == "placeholder_in_input" for finding in leaked.findings)
    assert token not in backend.calls[-1][0].text


def test_evasion_emails_are_redacted() -> None:
    backend = _backend()
    guard = Guard([invisible_text(), pii()], backend=backend)
    cases = [
        "alice\u200b@example.com",
        "a\u200dlice@example.com",
        "".join(char + "\ufe00" for char in _EMAIL_A),
        "alice\uff20example.com",
    ]
    for text in cases:
        result = guard.check_input(text)
        assert _EMAIL_A not in result.sanitized or "alice" not in result.sanitized.lower()
        assert "example.com" not in result.sanitized or "[JES_v1_PII_" in result.sanitized


def test_restore_false_uses_irreversible_markers() -> None:
    backend = _backend()
    guard = Guard([pii(restore=False)], backend=backend)
    incoming = guard.check_input(f"mail {_EMAIL_A}")
    assert incoming.ok
    assert len(incoming.redactions) == 0
    assert "[REDACTED_EMAIL_ADDRESS]" in incoming.sanitized
    outgoing = guard.check_output(incoming.sanitized, prompt=incoming)
    assert _EMAIL_A not in outgoing.text


def test_canary_blocks_and_removes() -> None:
    backend = _backend()
    guard = Guard(
        [canary("CANARY-TOKEN"), yesno_policy()],
        backend=backend,
        fail_fast=False,
    )
    incoming = Guard([yesno_policy()], backend=backend).check_input("prompt")
    outgoing = guard.check_output("leaked CANARY-TOKEN here", prompt=incoming)
    assert outgoing.decision == "block"
    assert "CANARY-TOKEN" not in outgoing.sanitized
    assert "CANARY-TOKEN" not in backend.calls[-1][0].text


def test_secret_is_redacted_twice_in_memory() -> None:
    _reset_secrets_config()
    backend = _backend()
    guard = Guard([secrets()], backend=backend)
    result = guard.check_input(f"one {_FAKE_KEY} two {_FAKE_KEY}")
    assert _FAKE_KEY not in result.sanitized
    assert result.sanitized.count("******") == 2


def test_uri_policy_and_restore_origins() -> None:
    backend = _backend()
    store = Redactions(scope=b"uri")
    guard = Guard(
        [pii(restore_origins=["https://allowed.example"]), yesno_policy()],
        backend=backend,
    )
    incoming = guard.check_input(f"mail {_EMAIL_A}", redactions=store)
    token = next(iter(incoming.redactions.snapshot_values()))
    allowed = guard.check_output(
        f"https://allowed.example/path/{token}",
        prompt=incoming,
        redactions=store,
    )
    assert _EMAIL_A in allowed.text
    blocked = guard.check_output(
        f"https://evil.example/{token}",
        prompt=incoming,
        redactions=store,
    )
    assert _EMAIL_A not in blocked.text
    assert any(finding.label == "placeholder_in_url" for finding in blocked.findings)
    nested = guard.check_output(
        f"https://allowed.example/https://evil.example/{token}",
        prompt=incoming,
        redactions=store,
    )
    assert _EMAIL_A not in nested.text


def test_dumps_loads_and_copy_refusal() -> None:
    store = Redactions(scope=b"dump-scope")
    backend = _backend()
    Guard([pii()], backend=backend).check_input(f"mail {_EMAIL_A}", redactions=store)
    key = b"k" * 32
    blob = store.dumps(key, associated_data=b"aad")
    loaded = Redactions.loads(blob, key, scope=b"dump-scope", associated_data=b"aad")
    assert loaded.snapshot_values() == store.snapshot_values()
    with pytest.raises(RedactionError):
        Redactions.loads(blob, key, scope=b"other-scope", associated_data=b"aad")
    with pytest.raises(RedactionError):
        Redactions.loads(blob, key, scope=b"dump-scope", associated_data=b"nope")
    with pytest.raises(RedactionError):
        Redactions.loads(b"\x00" * 8_400_000, key, scope=b"dump-scope", associated_data=b"aad")
    with pytest.raises(TypeError):
        copy.copy(store)
    with pytest.raises(TypeError):
        pickle.dumps(store)


def test_redaction_overflow_is_atomic() -> None:
    backend = _backend()
    store = Redactions(scope=b"tiny", max_entries=1)
    guard = Guard([pii()], backend=backend)
    first = guard.check_input(f"mail {_EMAIL_A}", redactions=store)
    assert first.ok
    second = guard.check_input(f"mail {_EMAIL_B}", redactions=store)
    assert second.decision == "block"
    assert any(finding.label == "redaction_limit_exceeded" for finding in second.findings)
    assert len(store) == 1
    assert backend.calls[-1][0].text.find(_EMAIL_B) < 0 if backend.calls else True


def test_language_and_factory_validation() -> None:
    with pytest.raises(PolicyError):
        pii(language="zh")
    with pytest.raises(PolicyError):
        canary("")
    with pytest.raises(PolicyError):
        secrets(redact="hmac")
    with pytest.raises(PolicyError):
        pii(restore_origins=["//evil.example"])
    with pytest.raises(PolicyError):
        pii(entities=())
    with pytest.raises(PolicyError):
        pii(ner="not-spacy")
    with pytest.raises(PolicyError):
        secrets(redact="other")  # type: ignore[arg-type]


def test_pii_mask_block_and_output_modes() -> None:
    backend = _backend()
    masked = Guard([pii(input_mode="mask")], backend=backend).check_input(f"mail {_EMAIL_A}")
    assert _EMAIL_A not in masked.sanitized
    blocked = Guard([pii(input_mode="block")], backend=backend).check_input(f"mail {_EMAIL_A}")
    assert blocked.decision == "block"
    incoming = Guard([yesno_policy()], backend=backend).check_input("p")
    flagged = Guard([pii(output_mode="flag")], backend=backend).check_output(
        f"see {_EMAIL_A}",
        prompt=incoming,
    )
    assert _EMAIL_A not in flagged.sanitized
    assert _EMAIL_A in flagged.text
    redacted = Guard([pii(output_mode="redact")], backend=backend).check_output(
        f"see {_EMAIL_A}",
        prompt=incoming,
    )
    assert "[REDACTED_EMAIL_ADDRESS]" in redacted.text
    denied = Guard([pii(output_mode="block")], backend=backend).check_output(
        f"see {_EMAIL_A}",
        prompt=incoming,
    )
    assert denied.decision == "block"
    untrusted = Guard([pii()], backend=backend).check_untrusted(f"doc {_EMAIL_A}")
    assert _EMAIL_A not in untrusted.sanitized


def test_secrets_partial_hmac_and_uri_rejects() -> None:
    _reset_secrets_config()
    backend = _backend()
    key = b"k" * 32
    partial = Guard([secrets(redact="partial")], backend=backend).check_input(f"tok {_FAKE_KEY}")
    assert _FAKE_KEY not in partial.sanitized
    assert partial.sanitized.startswith("tok ")
    _reset_secrets_config()
    hashed = Guard([secrets(redact="hmac", key=key)], backend=backend).check_input(
        f"tok {_FAKE_KEY}"
    )
    assert _FAKE_KEY not in hashed.sanitized
    assert len(hashed.sanitized.split()[-1]) == 64

    from jes._engine.uri import canonical_origin, parse_restore_origin, uri_candidates

    assert canonical_origin("//evil.example/x") is None
    assert canonical_origin("www.evil.example/x") is None
    assert canonical_origin("https://user:pass@evil.example/") is None
    assert canonical_origin("https://evil.example/a\\b") is None
    assert canonical_origin("ftp://evil.example/") is None
    with pytest.raises(PolicyError):
        parse_restore_origin("https://evil.example/path")
    assert uri_candidates("see //evil.example/x")


def test_dumps_rejects_bad_inputs() -> None:
    store = Redactions(scope=b"dump-scope")
    key = b"k" * 32
    with pytest.raises(RedactionError):
        store.dumps(b"short", associated_data=b"aad")
    with pytest.raises(RedactionError):
        store.dumps(key, associated_data=b"")
    blob = store.dumps(key, associated_data=b"aad")
    tampered = blob[:-1] + bytes([(blob[-1] + 1) % 256])
    with pytest.raises(RedactionError):
        Redactions.loads(tampered, key, scope=b"dump-scope", associated_data=b"aad")
    with pytest.raises(RedactionError):
        Redactions.loads(
            b"JESR" + b"\x02" + blob[5:],
            key,
            scope=b"dump-scope",
            associated_data=b"aad",
        )
    with pytest.raises(RedactionError):
        Redactions.loads(blob, b"x" * 32, scope=b"dump-scope", associated_data=b"aad")


def test_uri_ipv_restore_caps_and_secret_lock() -> None:
    import importlib

    from jes._engine.uri import canonical_origin

    secrets_mod = importlib.import_module("jes.policies.secrets")

    assert canonical_origin("https://allowed.example.") == ("https", "allowed.example", 443)
    assert canonical_origin("http://127.0.0.1/") == ("http", "127.0.0.1", 80)
    assert canonical_origin("https://[::1]/") == ("https", "[::1]", 443)
    assert canonical_origin("https://[fe80::1%eth0]/") is None
    assert canonical_origin("https://example.com\u3002evil/") is None
    _reset_secrets_config()
    secrets()
    with pytest.raises(PolicyError):
        secrets_mod._configure("other-settings")
    _reset_secrets_config()
    backend = _backend()
    store = Redactions(scope=b"cap")
    incoming = Guard([pii(), yesno_policy()], backend=backend).check_input(
        f"mail {_EMAIL_A}",
        redactions=store,
    )
    token = next(iter(incoming.redactions.snapshot_values()))
    capped = Guard(
        [pii(), yesno_policy()],
        backend=backend,
        max_restored_output_bytes=1,
    ).check_output(token, prompt=incoming, redactions=store)
    assert capped.decision == "block"
    assert any(finding.label == "restored_output_too_large" for finding in capped.findings)
    assert _EMAIL_A not in capped.text
    card = Guard([pii()], backend=backend).check_input("card 4111111111111112")
    assert "4111111111111112" in card.sanitized
