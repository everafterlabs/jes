"""Conversation-scoped redaction storage.

Milestone 1 provides the bounded, transactional container. Encryption and the
production PII policy arrive in later milestones.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import dataclass
from typing import Never, SupportsIndex

from jes.errors import PolicyError, RedactionError

_TOKEN_VERSION = b"v1"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


@dataclass(frozen=True, slots=True)
class RedactionView:
    id: str
    scope_id: str
    size: int


class Redactions:
    """Authenticated token/value mappings for one conversation scope."""

    __slots__ = (
        "_id",
        "_lock",
        "_max_bytes",
        "_max_entries",
        "_max_value_bytes",
        "_scope_id",
        "_secret",
        "_token_by_value",
        "_value_by_token",
    )

    def __init__(
        self,
        *,
        scope: bytes | None = None,
        max_scope_bytes: int = 1_024,
        max_entries: int = 1_000,
        max_value_bytes: int = 65_536,
        max_bytes: int = 8_388_608,
    ) -> None:
        if max_scope_bytes < 1 or max_entries < 1 or max_value_bytes < 1 or max_bytes < 1:
            raise RedactionError("redaction limits must be positive")
        scope_value = secrets.token_bytes(32) if scope is None else scope
        if not scope_value or len(scope_value) > max_scope_bytes:
            raise RedactionError("invalid redaction scope")

        self._id = _b64(secrets.token_bytes(16))
        self._scope_id = hashlib.sha256(scope_value).hexdigest()
        self._secret = secrets.token_bytes(32)
        self._max_entries = max_entries
        self._max_value_bytes = max_value_bytes
        self._max_bytes = max_bytes
        self._value_by_token: dict[str, str] = {}
        self._token_by_value: dict[str, str] = {}
        self._lock = threading.RLock()

    @property
    def id(self) -> str:
        return self._id

    @property
    def scope_id(self) -> str:
        return self._scope_id

    def view(self) -> RedactionView:
        return RedactionView(id=self.id, scope_id=self.scope_id, size=len(self))

    def __len__(self) -> int:
        with self._lock:
            return len(self._value_by_token)

    def __repr__(self) -> str:
        return f"Redactions(id={self.id!r}, scope_id={self.scope_id!r}, entries={len(self)})"

    def __copy__(self) -> Redactions:
        raise TypeError("Redactions cannot be copied")

    def __deepcopy__(self, memo: object) -> Redactions:
        del memo
        raise TypeError("Redactions cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("Redactions cannot be pickled")

    def _token_for_value(self, value: str) -> str:
        value_bytes = value.encode("utf-8")
        token_id = hmac.digest(self._secret, b"id\0" + value_bytes, "sha256")[:16]
        tag = hmac.digest(
            self._secret,
            b"tag\0" + _TOKEN_VERSION + b"\0" + self._id.encode("ascii") + b"\0" + token_id,
            "sha256",
        )[:16]
        return f"[JES_v1_PII_{_b64(token_id)}_{_b64(tag)}]"

    def _canonical_size(self, additional: dict[str, str] | None = None) -> int:
        entries = dict(self._value_by_token)
        if additional is not None:
            entries.update(additional)
        # Fixed header allowance plus length prefixes for each UTF-8 field.
        return 128 + sum(
            8 + len(token.encode("ascii")) + len(value.encode("utf-8"))
            for token, value in entries.items()
        )

    def _validate_additions(self, additions: dict[str, str]) -> None:
        combined_count = len(set(self._value_by_token) | set(additions))
        if combined_count > self._max_entries:
            raise RedactionError("redaction entry limit exceeded")
        for value in additions.values():
            if len(value.encode("utf-8")) > self._max_value_bytes:
                raise RedactionError("redaction value limit exceeded")
        if self._canonical_size(additions) > self._max_bytes:
            raise RedactionError("redaction byte limit exceeded")
        for token, value in additions.items():
            existing = self._value_by_token.get(token)
            if existing is not None and existing != value:
                raise RedactionError("redaction token collision")

    def begin_transaction(self) -> RedactionTransaction:
        return RedactionTransaction(self)

    def snapshot_values(self) -> dict[str, str]:
        with self._lock:
            return dict(self._value_by_token)

    def transaction_existing_token(self, value: str) -> str | None:
        with self._lock:
            return self._token_by_value.get(value)

    def transaction_token_for_value(self, value: str) -> str:
        return self._token_for_value(value)

    def transaction_preflight(self, additions: dict[str, str]) -> None:
        with self._lock:
            self._validate_additions(additions)

    def dumps(
        self,
        key: bytes,
        *,
        associated_data: bytes,
        max_associated_data_bytes: int = 4_096,
    ) -> bytes:
        """Serialize the store with AES-256-GCM from the crypto extra."""

        if len(key) != 32:
            raise RedactionError("dumps requires a 32-byte key")
        if not associated_data or len(associated_data) > max_associated_data_bytes:
            raise RedactionError("invalid associated data")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise PolicyError("Redactions.dumps requires the jes[crypto] extra") from None
        with self._lock:
            secret = self._secret
            store_id = self._id
            scope_id = self._scope_id
            entries = dict(self._value_by_token)
            max_entries = self._max_entries
            max_value_bytes = self._max_value_bytes
            max_bytes = self._max_bytes
        payload = {
            "version": 1,
            "scope_id": scope_id,
            "store_id": store_id,
            "secret": _b64(secret),
            "max_entries": max_entries,
            "max_value_bytes": max_value_bytes,
            "max_bytes": max_bytes,
            "entries": entries,
        }
        plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        nonce = secrets.token_bytes(12)
        header = b"JESR" + bytes([1]) + nonce
        blob = header + AESGCM(key).encrypt(nonce, plaintext, header + associated_data)
        return blob

    @classmethod
    def loads(
        cls,
        blob: bytes,
        key: bytes,
        *,
        scope: bytes,
        associated_data: bytes,
        max_scope_bytes: int = 1_024,
        max_associated_data_bytes: int = 4_096,
        max_entries: int = 1_000,
        max_value_bytes: int = 65_536,
        max_plaintext_bytes: int = 8_388_608,
        max_blob_bytes: int = 8_392_704,
    ) -> Redactions:
        """Decrypt a store blob and bind it to the supplied scope."""

        if len(blob) > max_blob_bytes:
            raise RedactionError("redaction blob exceeds max_blob_bytes")
        if not scope or len(scope) > max_scope_bytes:
            raise RedactionError("invalid redaction scope")
        if not associated_data or len(associated_data) > max_associated_data_bytes:
            raise RedactionError("invalid associated data")
        if len(key) != 32:
            raise RedactionError("loads requires a 32-byte key")
        if len(blob) < 17 or blob[:4] != b"JESR":
            raise RedactionError("invalid redaction blob")
        version = blob[4]
        nonce = blob[5:17]
        ciphertext = blob[17:]
        if version != 1:
            raise RedactionError("unsupported redaction blob version")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise PolicyError("Redactions.loads requires the jes[crypto] extra") from None
        header = blob[:17]
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, header + associated_data)
        except Exception:
            raise RedactionError("redaction blob authentication failed") from None
        if len(plaintext) > max_plaintext_bytes:
            raise RedactionError("redaction plaintext exceeds max_plaintext_bytes")
        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception:
            raise RedactionError("invalid redaction plaintext") from None
        if payload.get("version") != 1:
            raise RedactionError("unsupported redaction blob version")
        scope_id = hashlib.sha256(scope).hexdigest()
        if payload.get("scope_id") != scope_id:
            raise RedactionError("redaction scope mismatch")
        embedded_entries = int(payload.get("max_entries", 0))
        embedded_value = int(payload.get("max_value_bytes", 0))
        embedded_bytes = int(payload.get("max_bytes", 0))
        if (
            embedded_entries > max_entries
            or embedded_value > max_value_bytes
            or embedded_bytes > max_plaintext_bytes
        ):
            raise RedactionError("embedded redaction limits exceed caller ceilings")
        store = cls(
            scope=scope,
            max_scope_bytes=max_scope_bytes,
            max_entries=max_entries,
            max_value_bytes=max_value_bytes,
            max_bytes=max_plaintext_bytes,
        )
        store._id = str(payload["store_id"])
        store._secret = base64.urlsafe_b64decode(str(payload["secret"]) + "==")
        entries = dict(payload.get("entries") or {})
        store._value_by_token = {str(token): str(value) for token, value in entries.items()}
        store._token_by_value = {value: token for token, value in store._value_by_token.items()}
        return store

    def transaction_commit(self, additions: dict[str, str]) -> None:
        with self._lock:
            self._validate_additions(additions)
            for token, value in additions.items():
                existing_token = self._token_by_value.get(value)
                if existing_token is not None and existing_token != token:
                    raise RedactionError("redaction value maps to a different token")
                self._value_by_token[token] = value
                self._token_by_value[value] = token


class RedactionTransaction:
    __slots__ = ("_closed", "_staged", "_store")

    def __init__(self, store: Redactions) -> None:
        self._store = store
        self._staged: dict[str, str] = {}
        self._closed = False

    @property
    def view(self) -> RedactionView:
        return self._store.view()

    def token_for(self, entity: str, value: str) -> str:
        del entity  # Entity is finding metadata; token identity is value-scoped.
        self._ensure_open()
        existing = self._store.transaction_existing_token(value)
        if existing is not None:
            return existing
        token = self._store.transaction_token_for_value(value)
        previous = self._staged.get(token)
        if previous is not None and previous != value:
            raise RedactionError("redaction token collision")
        self._staged[token] = value
        return token

    def preflight(self) -> None:
        self._ensure_open()
        self._store.transaction_preflight(self._staged)

    def commit(self) -> None:
        self._ensure_open()
        self._store.transaction_commit(self._staged)
        self.close()

    def close(self) -> None:
        self._staged.clear()
        self._closed = True

    def __repr__(self) -> str:
        return f"RedactionTransaction(store_id={self._store.id!r}, staged={len(self._staged)})"

    def __copy__(self) -> RedactionTransaction:
        raise TypeError("redaction transactions cannot be copied")

    def __deepcopy__(self, memo: object) -> RedactionTransaction:
        del memo
        raise TypeError("redaction transactions cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("redaction transactions cannot be pickled")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RedactionError("redaction transaction is closed")
