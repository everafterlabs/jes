"""Placeholder tokens and the values they stand for, one store per conversation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
from collections.abc import Mapping
from typing import Never, Protocol, SupportsIndex, cast

from jes.errors import PolicyError, RedactionError

TOKEN_RE = re.compile(r"\[JES_PII_[A-Za-z0-9_-]{22}\]")

_BLOB_MAGIC = b"JESR"
_BLOB_VERSION = 2
_NONCE_BYTES = 12
_HEADER_BYTES = len(_BLOB_MAGIC) + 1 + _NONCE_BYTES
_TAG_BYTES = 16
# Each entry costs its token and its value in UTF-8, plus this allowance for JSON around them.
_ENTRY_OVERHEAD = 8
# JSON escapes a control character in six bytes, so a full store can serialize this much larger.
_JSON_GROWTH = 6


class _Cipher(Protocol):
    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes: ...

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes: ...


def _cipher(key: bytes) -> _Cipher:
    if len(key) != 32:
        raise RedactionError("the key must be 32 bytes")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise PolicyError("saving a Redactions store requires the jes[crypto] extra") from None
    return AESGCM(key)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _cost(token: str, value: str) -> int:
    return len(token) + len(value.encode("utf-8")) + _ENTRY_OVERHEAD


class Redactions:
    """The placeholder tokens of one conversation and the values they replace.

    Create one store per conversation and pass it to each check. A token is an HMAC of
    its value under this store's random secret: the same value always gets the same
    token here, and no other store can produce or restore it.

    ``scope`` names the conversation. A saved store loads only under the same scope.
    """

    __slots__ = (
        "_bytes",
        "_lock",
        "_max_bytes",
        "_max_entries",
        "_scope_id",
        "_secret",
        "_values",
    )

    def __init__(
        self,
        *,
        scope: bytes | None = None,
        max_entries: int = 1_000,
        max_bytes: int = 8_388_608,
    ) -> None:
        for name, limit in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if type(limit) is not int or limit < 1:
                raise RedactionError(f"{name} must be a positive integer")
        scope = secrets.token_bytes(32) if scope is None else scope
        if not scope or len(scope) > 1_024:
            raise RedactionError("scope must be 1 to 1024 bytes")
        self._scope_id = hashlib.sha256(scope).hexdigest()
        self._secret = secrets.token_bytes(32)
        self._values: dict[str, str] = {}
        self._bytes = 0
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def scope_id(self) -> str:
        """A hash of the scope: safe to log, and equal for stores of the same conversation."""

        return self._scope_id

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __repr__(self) -> str:
        return f"Redactions(scope_id={self._scope_id[:12]!r}, entries={len(self)})"

    def __copy__(self) -> Never:
        raise TypeError("Redactions cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        raise TypeError("Redactions cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        raise TypeError("Redactions cannot be pickled; use dumps()")

    # Engine-facing. A check stages its new tokens and commits them only when it allows.

    def _token(self, value: str) -> str:
        digest = hmac.digest(self._secret, b"jes-token\0" + value.encode("utf-8"), "sha256")
        return f"[JES_PII_{_b64(digest[:16])}]"

    def _value(self, token: str) -> str | None:
        with self._lock:
            return self._values.get(token)

    def _check_room(self, staged: Mapping[str, str]) -> None:
        """Raise RedactionError if committing ``staged`` would cross a limit."""

        with self._lock:
            self._room(staged)

    def _commit(self, staged: Mapping[str, str]) -> None:
        with self._lock:
            added = self._room(staged)
            for token, value in staged.items():
                self._values.setdefault(token, value)
            self._bytes += added

    def _room(self, staged: Mapping[str, str]) -> int:
        new = {token: value for token, value in staged.items() if token not in self._values}
        if len(self._values) + len(new) > self._max_entries:
            raise RedactionError("redaction store entry limit reached")
        added = sum(_cost(token, value) for token, value in new.items())
        if self._bytes + added > self._max_bytes:
            raise RedactionError("redaction store byte limit reached")
        return added

    # Saving and loading.

    def dumps(self, key: bytes, *, associated_data: bytes) -> bytes:
        """Encrypt the store with AES-256-GCM. ``associated_data`` must match on load."""

        cipher = _cipher(key)
        if not associated_data or len(associated_data) > 4_096:
            raise RedactionError("associated_data must be 1 to 4096 bytes")
        with self._lock:
            payload = {
                "version": _BLOB_VERSION,
                "scope_id": self._scope_id,
                "secret": _b64(self._secret),
                "entries": dict(self._values),
            }
        plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        nonce = secrets.token_bytes(_NONCE_BYTES)
        header = _BLOB_MAGIC + bytes([_BLOB_VERSION]) + nonce
        return header + cipher.encrypt(nonce, plaintext.encode("utf-8"), header + associated_data)

    @classmethod
    def loads(
        cls,
        blob: bytes,
        key: bytes,
        *,
        scope: bytes,
        associated_data: bytes,
        max_entries: int = 1_000,
        max_bytes: int = 8_388_608,
    ) -> Redactions:
        """Decrypt a store saved by ``dumps``. ``scope`` and ``associated_data`` must match.

        The limits bound the loaded store. A blob that holds more fails to load.
        """

        store = cls(scope=scope, max_entries=max_entries, max_bytes=max_bytes)
        cipher = _cipher(key)
        if len(blob) > _HEADER_BYTES + _TAG_BYTES + 1_024 + _JSON_GROWTH * max_bytes:
            raise RedactionError("the blob is larger than max_bytes allows")
        if len(blob) < _HEADER_BYTES + _TAG_BYTES or not blob.startswith(_BLOB_MAGIC):
            raise RedactionError("not a Redactions blob")
        if blob[len(_BLOB_MAGIC)] != _BLOB_VERSION:
            raise RedactionError("unsupported Redactions blob version")
        from cryptography.exceptions import InvalidTag

        header = blob[:_HEADER_BYTES]
        try:
            plaintext = cipher.decrypt(
                header[-_NONCE_BYTES:], blob[_HEADER_BYTES:], header + associated_data
            )
        except InvalidTag:
            raise RedactionError("the blob, key, or associated_data does not match") from None
        scope_id, secret, entries = _parse_payload(plaintext)
        if scope_id != store._scope_id:
            raise RedactionError("the blob belongs to another scope")
        store._secret = secret
        for token, value in entries.items():
            if store._token(value) != token:
                raise RedactionError("the blob holds a token that does not match its value")
        store._commit(entries)
        return store


def _parse_payload(plaintext: bytes) -> tuple[str, bytes, dict[str, str]]:
    malformed = RedactionError("the blob does not hold a store")
    try:
        payload: object = json.loads(plaintext.decode("utf-8"))
    except ValueError:
        raise malformed from None
    if not isinstance(payload, dict):
        raise malformed
    fields = cast(dict[str, object], payload)
    scope_id, secret, entries = fields.get("scope_id"), fields.get("secret"), fields.get("entries")
    if (
        fields.get("version") != _BLOB_VERSION
        or not isinstance(scope_id, str)
        or not isinstance(secret, str)
        or not isinstance(entries, dict)
    ):
        raise malformed
    try:
        secret_bytes = _unb64(secret)
    except ValueError:
        raise malformed from None
    if len(secret_bytes) != 32:
        raise malformed
    checked: dict[str, str] = {}
    for token, value in cast(dict[object, object], entries).items():
        if (
            not isinstance(token, str)
            or not isinstance(value, str)
            or not TOKEN_RE.fullmatch(token)
        ):
            raise malformed
        checked[token] = value
    return scope_id, secret_bytes, checked


__all__ = ["TOKEN_RE", "Redactions"]
