"""The conversation redaction store: tokens, limits, and encrypted saves."""

from __future__ import annotations

import copy
import json
import pickle
import secrets
import sys

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from jes.errors import PolicyError, RedactionError
from jes.redactions import TOKEN_RE, Redactions

KEY = b"k" * 32
AAD = b"conversation-1"


def test_tokens_are_stable_per_store_and_differ_across_stores() -> None:
    store = Redactions(scope=b"one")
    token = store._token("ada@example.com")
    assert TOKEN_RE.fullmatch(token)
    assert store._token("ada@example.com") == token
    assert store._token("bob@example.com") != token
    assert Redactions(scope=b"one")._token("ada@example.com") != token


def test_staged_tokens_count_only_after_commit() -> None:
    store = Redactions()
    token = store._token("ada@example.com")
    staged = {token: "ada@example.com"}
    store._check_room(staged)
    assert store._value(token) is None
    assert len(store) == 0
    store._commit(staged)
    store._commit(staged)
    assert store._value(token) == "ada@example.com"
    assert len(store) == 1


def test_limits_are_enforced_before_and_at_commit() -> None:
    store = Redactions(max_entries=1)
    first = {store._token("a"): "a"}
    store._commit(first)
    store._check_room(first)
    with pytest.raises(RedactionError, match="entry limit"):
        store._check_room({store._token("b"): "b"})
    tight = Redactions(max_bytes=40)
    with pytest.raises(RedactionError, match="byte limit"):
        tight._commit({tight._token("x" * 20): "x" * 20})
    assert len(tight) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_entries": 0}, "max_entries"),
        ({"max_bytes": True}, "max_bytes"),
        ({"scope": b""}, "scope"),
        ({"scope": b"x" * 1_025}, "scope"),
    ],
)
def test_bad_store_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(RedactionError, match=message):
        Redactions(**kwargs)  # type: ignore[arg-type]


def test_the_store_never_shows_values_and_cannot_be_copied() -> None:
    store = Redactions(scope=b"one")
    store._commit({store._token("secret-value"): "secret-value"})
    assert "secret-value" not in repr(store)
    assert "entries=1" in repr(store)
    assert store.scope_id == Redactions(scope=b"one").scope_id
    for attempt in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="Redactions"):
            attempt(store)


def test_dumps_and_loads_round_trip() -> None:
    store = Redactions(scope=b"one")
    values = ["ada@example.com", "Ada Lovelace", "line\nbreak \u0001"]
    store._commit({store._token(value): value for value in values})
    blob = store.dumps(KEY, associated_data=AAD)
    assert b"ada@example.com" not in blob

    loaded = Redactions.loads(blob, KEY, scope=b"one", associated_data=AAD)
    assert len(loaded) == 3
    for value in values:
        assert loaded._token(value) == store._token(value)
        assert loaded._value(store._token(value)) == value


def test_loads_rejects_the_wrong_key_scope_or_associated_data() -> None:
    store = Redactions(scope=b"one")
    store._commit({store._token("v"): "v"})
    blob = store.dumps(KEY, associated_data=AAD)
    with pytest.raises(RedactionError, match="does not match"):
        Redactions.loads(blob, b"x" * 32, scope=b"one", associated_data=AAD)
    with pytest.raises(RedactionError, match="does not match"):
        Redactions.loads(blob, KEY, scope=b"one", associated_data=b"other")
    with pytest.raises(RedactionError, match="another scope"):
        Redactions.loads(blob, KEY, scope=b"two", associated_data=AAD)
    tampered = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(RedactionError, match="does not match"):
        Redactions.loads(tampered, KEY, scope=b"one", associated_data=AAD)


def test_loads_rejects_malformed_blobs() -> None:
    blob = Redactions(scope=b"one").dumps(KEY, associated_data=AAD)
    with pytest.raises(RedactionError, match="32 bytes"):
        Redactions.loads(blob, b"short", scope=b"one", associated_data=AAD)
    with pytest.raises(RedactionError, match="not a Redactions blob"):
        Redactions.loads(b"JESR", KEY, scope=b"one", associated_data=AAD)
    with pytest.raises(RedactionError, match="not a Redactions blob"):
        Redactions.loads(b"XXXX" + blob[4:], KEY, scope=b"one", associated_data=AAD)
    with pytest.raises(RedactionError, match="version"):
        Redactions.loads(blob[:4] + b"\x01" + blob[5:], KEY, scope=b"one", associated_data=AAD)
    with pytest.raises(RedactionError, match="larger than max_bytes"):
        Redactions.loads(blob + b"\0" * 2_000, KEY, scope=b"one", associated_data=AAD, max_bytes=1)


def _seal(payload: object) -> bytes:
    """Encrypt a hand-made payload the way dumps does."""

    nonce = secrets.token_bytes(12)
    header = b"JESR\x02" + nonce
    plaintext = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return header + AESGCM(KEY).encrypt(nonce, plaintext, header + AAD)


def test_loads_rejects_payloads_that_are_not_stores() -> None:
    store = Redactions(scope=b"one")
    good = {
        "version": 2,
        "scope_id": store.scope_id,
        "secret": "A" * 43,
        "entries": {},
    }
    bad_payloads: list[object] = [
        b"\xff not json",
        ["a", "list"],
        {**good, "version": 1},
        {**good, "secret": 7},
        {**good, "secret": "A" * 10},
        {**good, "secret": "!!!"},
        {**good, "secret": "A"},
        {**good, "entries": {"not-a-token": "v"}},
        {**good, "entries": {"[JES_PII_" + "A" * 22 + "]": 5}},
    ]
    for payload in bad_payloads:
        with pytest.raises(RedactionError, match="does not hold a store"):
            Redactions.loads(_seal(payload), KEY, scope=b"one", associated_data=AAD)
    forged = {**good, "entries": {"[JES_PII_" + "A" * 22 + "]": "value"}}
    with pytest.raises(RedactionError, match="does not match its value"):
        Redactions.loads(_seal(forged), KEY, scope=b"one", associated_data=AAD)


def test_loads_applies_the_callers_limits() -> None:
    store = Redactions(scope=b"one")
    store._commit({store._token(str(index)): str(index) for index in range(3)})
    blob = store.dumps(KEY, associated_data=AAD)
    with pytest.raises(RedactionError, match="entry limit"):
        Redactions.loads(blob, KEY, scope=b"one", associated_data=AAD, max_entries=2)


def test_dumps_checks_its_arguments() -> None:
    store = Redactions()
    with pytest.raises(RedactionError, match="32 bytes"):
        store.dumps(b"short", associated_data=AAD)
    with pytest.raises(RedactionError, match="associated_data"):
        store.dumps(KEY, associated_data=b"")


def test_saving_without_the_crypto_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives.ciphers.aead", None)
    with pytest.raises(PolicyError, match=r"jes\[crypto\]"):
        Redactions().dumps(KEY, associated_data=AAD)
