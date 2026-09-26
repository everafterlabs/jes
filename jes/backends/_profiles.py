"""Known System One / LiteLLM profiles and the closed attestation registry."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jes.backends import Backend, BackendProfile, _ProfileAttestation
from jes.errors import PolicyError

ADAPTER_VERSION = "1"
SCORER_VERSION = "1"
PARSER_VERSION = "1"

_ATTESTATIONS: dict[str, _ProfileAttestation] = {}


@dataclass(frozen=True, slots=True)
class LayaWindow:
    context_window_tokens: int
    question_head: int
    output_reserve: int
    special_tokens: int


LAYA_WINDOWS: Mapping[str, LayaWindow] = {
    "english": LayaWindow(512, 192, 16, 8),
    "multilingual": LayaWindow(1024, 256, 16, 8),
    "typed-decisions": LayaWindow(1024, 256, 16, 8),
}

JEV_STATE_TOKENS = 32_000
JEV_REQUEST_TOKENS = 64_000
JEV_OUTPUT_RESERVE = 32
JEV_MODELS = frozenset({"jev-1.13.0"})

LITELLM_PORTABLE_TOP_LOGPROBS = 5

PROVIDER_PROFILES: Mapping[str, Mapping[str, object]] = {
    "laya-serve.v1": {"family": "laya", "handshake": True},
    "laya.in_process.v1": {"family": "laya", "handshake": False},
    "typesafe.jev.v1": {"family": "jev", "handshake": True},
    "ollama.chat.v1": {"family": "litellm", "top_logprobs_max": 5, "temperature": 0},
    "openai.chat.v1": {"family": "litellm", "top_logprobs_max": 5, "temperature": 0},
    "groq.chat.v1": {"family": "litellm", "top_logprobs_max": 0, "verbalized_only": True},
    "provider.prompt_guard_2.v1": {
        "family": "prompt_guard",
        "local": False,
        "max_attempts": 1,
        "context_window_tokens": 512,
    },
    "transformers.sequence_classification.v1": {
        "family": "prompt_guard",
        "local": True,
        "max_attempts": 1,
        "context_window_tokens": 512,
    },
    "groq.llama_guard_4.v1": {
        "family": "llama_guard",
        "local": False,
        "supports_logprobs": False,
        "custom_categories": False,
        "max_attempts": 1,
        "context_window_tokens": 131_072,
    },
    "vllm.llama_guard_4.v1": {
        "family": "llama_guard",
        "local": False,
        "supports_logprobs": True,
        "custom_categories": True,
        "max_attempts": 1,
    },
    "transformers.causal_lm.v1": {
        "family": "llama_guard",
        "local": True,
        "supports_logprobs": True,
        "custom_categories": True,
        "max_attempts": 1,
    },
}


def profile_fingerprint(profile: BackendProfile) -> str:
    return hashlib.sha256(repr(profile).encode()).hexdigest()


def register_attestation(profile: BackendProfile, attestation: _ProfileAttestation) -> None:
    _ATTESTATIONS[profile_fingerprint(profile)] = attestation


def attestation_digest_for(backend: Backend) -> str | None:
    item = _ATTESTATIONS.get(profile_fingerprint(backend.profile))
    return None if item is None else item.evidence_digest


def canonical_manifest_bytes(body: Mapping[str, object]) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def sign_manifest(body: Mapping[str, object], key: bytes) -> str:
    if not key:
        raise PolicyError("trusted manifest key must not be empty")
    return hmac.new(key, canonical_manifest_bytes(body), hashlib.sha256).hexdigest()


def verify_manifest_signature(
    body: Mapping[str, object],
    signature: str,
    trusted_keys: Mapping[str, bytes],
    key_id: str,
) -> None:
    if key_id not in trusted_keys:
        raise PolicyError("deployment manifest key is not trusted")
    expected = sign_manifest(body, trusted_keys[key_id])
    if not hmac.compare_digest(expected, signature):
        raise PolicyError("deployment manifest signature is invalid")


def require_known_provider_profile(name: str, family: str) -> Mapping[str, object]:
    spec = PROVIDER_PROFILES.get(name)
    if spec is None or spec.get("family") != family:
        raise PolicyError(f"unknown provider profile: {name}")
    return spec


def laya_window(model: str) -> LayaWindow | None:
    return LAYA_WINDOWS.get(model)


def conservative_byte_budget(model: str, *, max_request_bytes: int | None) -> int:
    if max_request_bytes is not None:
        if max_request_bytes < 1:
            raise PolicyError("request byte budget must be positive")
        return max_request_bytes
    window = laya_window(model)
    if window is not None:
        # Conservative payload cap for HTTP without a pinned tokenizer.
        # Not a token-window claim and remains default-ineligible.
        return 16_384
    if model in JEV_MODELS:
        return JEV_REQUEST_TOKENS
    raise PolicyError("unknown model requires context_window_tokens or max_request_bytes")


def output_reserve_for(model: str) -> int:
    window = laya_window(model)
    if window is not None:
        return window.output_reserve
    if model in JEV_MODELS:
        return JEV_OUTPUT_RESERVE
    return 16


AttestationSource = Literal["installed_artifact", "provider_registry", "signed_deployment"]


def make_attestation(
    profile: BackendProfile,
    *,
    source: AttestationSource,
    evidence: bytes,
    signature: str,
) -> _ProfileAttestation:
    digest = hashlib.sha256(evidence).hexdigest()
    item = _ProfileAttestation(
        profile_fingerprint=profile_fingerprint(profile),
        source=source,
        evidence_digest=digest,
        signature=signature,
    )
    register_attestation(profile, item)
    return item


__all__ = [
    "ADAPTER_VERSION",
    "JEV_MODELS",
    "JEV_REQUEST_TOKENS",
    "JEV_STATE_TOKENS",
    "LAYA_WINDOWS",
    "LITELLM_PORTABLE_TOP_LOGPROBS",
    "PARSER_VERSION",
    "PROVIDER_PROFILES",
    "SCORER_VERSION",
    "attestation_digest_for",
    "canonical_manifest_bytes",
    "conservative_byte_budget",
    "laya_window",
    "make_attestation",
    "output_reserve_for",
    "profile_fingerprint",
    "register_attestation",
    "require_known_provider_profile",
    "sign_manifest",
    "verify_manifest_signature",
]
