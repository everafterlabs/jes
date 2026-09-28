"""Sanitization stamps and occurrence-based token authority."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace

from jes.redactions import Redactions
from jes.types import (
    _EMPTY_AUTHORITY,
    Finding,
    SanitizationStamp,
    ScanResult,
    _TokenAuthority,
    _TokenAuthorityManifest,
)

_TOKEN_RE = re.compile(r"\[JES_v[0-9]+_[^\]\r\n]{1,256}\]")


def findings_digest(findings: Sequence[Finding]) -> str:
    payload = [
        {
            "policy": finding.policy,
            "label": finding.label,
            "action": finding.action,
            "question": finding.question,
            "locations": [
                {
                    "target": location.target,
                    "start": location.span.start,
                    "end": location.span.end,
                    "index": location.index,
                    "role": location.role,
                    "item": location.item_ordinal,
                }
                for location in finding.locations
            ],
        }
        for finding in findings
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def authority_digest(entries: Sequence[_TokenAuthority]) -> str:
    payload = [
        {
            "token": entry.token,
            "start": entry.span.start,
            "end": entry.span.end,
            "origin": entry.origin_result,
            "reusable": entry.reusable,
        }
        for entry in entries
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_manifest(entries: Iterable[_TokenAuthority]) -> _TokenAuthorityManifest:
    frozen = tuple(entries)
    return _TokenAuthorityManifest(entries=frozen, digest=authority_digest(frozen))


def empty_manifest() -> _TokenAuthorityManifest:
    return _EMPTY_AUTHORITY


def stamp_payload(stamp: SanitizationStamp) -> bytes:
    values = (
        stamp.result_id,
        stamp.stage,
        stamp.config_digest,
        stamp.text_digest,
        stamp.store_id or "",
        stamp.scope_id or "",
        stamp.decision,
        "1" if stamp.complete else "0",
        stamp.findings_digest,
        stamp.authority_digest,
    )
    return "\0".join(values).encode()


def sign_stamp(key: bytes, stamp: SanitizationStamp) -> str:
    return hmac.new(key, stamp_payload(stamp), "sha256").hexdigest()


def attach_authority(result: ScanResult, manifest: _TokenAuthorityManifest) -> None:
    object.__setattr__(result, "_authority", manifest)


def tokens_in(text: str) -> tuple[str, ...]:
    return tuple(match.group() for match in _TOKEN_RE.finditer(text))


def extract_generation_tokens(result: ScanResult) -> frozenset[str]:
    manifest = result._authority
    if not manifest.entries:
        return frozenset()
    present = set(tokens_in(result.sanitized))
    return frozenset(entry.token for entry in manifest.entries if entry.token in present)


def verify_stamp(
    key: bytes,
    result: ScanResult,
    *,
    config_digest: str,
    redactions: Redactions,
) -> bool:
    stamp = result.sanitization
    if len(stamp.tag) != 64:
        return False
    unsigned = replace(stamp, tag="")
    if not hmac.compare_digest(stamp.tag, sign_stamp(key, unsigned)):
        return False
    if stamp.config_digest != config_digest:
        return False
    if stamp.store_id != redactions.id or stamp.scope_id != redactions.scope_id:
        return False
    if stamp.decision != result.decision or stamp.complete != result.complete:
        return False
    if stamp.text_digest != hashlib.sha256(result.sanitized.encode()).hexdigest():
        return False
    if stamp.findings_digest != findings_digest(result.findings):
        return False
    return stamp.authority_digest == result._authority.digest
