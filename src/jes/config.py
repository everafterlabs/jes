"""Agent guard config at ``~/.config/jes/config.json``."""

from __future__ import annotations

import base64
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from jes.errors import PolicyError
from jes.payload import ConfigError
from jes.policies import (
    Policy,
    allowed_tools,
    canary,
    hazards,
    indirect_injection,
    injection,
    invisible_text,
    pii,
    regex,
    secrets,
    substrings,
    token_limit,
    topics,
    toxicity,
)

_JUDGMENTS = frozenset({"injection", "indirect_injection", "hazards", "toxicity", "topics"})
_ORDER = (
    "injection",
    "indirect_injection",
    "hazards",
    "toxicity",
    "topics",
    "invisible_text",
    "allowed_tools",
    "canary",
    "regex",
    "substrings",
    "token_limit",
    "pii",
    "secrets",
)
_FIELDS: dict[str, frozenset[str]] = {
    "injection": frozenset({"enabled", "threshold"}),
    "indirect_injection": frozenset({"enabled", "threshold"}),
    "hazards": frozenset({"enabled", "threshold", "categories"}),
    "toxicity": frozenset({"enabled", "threshold", "labels"}),
    "topics": frozenset({"enabled", "threshold", "deny"}),
    "invisible_text": frozenset({"enabled", "mode", "block"}),
    "allowed_tools": frozenset({"enabled", "names"}),
    "canary": frozenset({"enabled", "token"}),
    "regex": frozenset({"enabled", "patterns", "action", "match", "require", "fold", "timeout_ms"}),
    "substrings": frozenset({"enabled", "terms", "action", "whole_words", "fold"}),
    "token_limit": frozenset({"enabled", "limit", "encoding", "mode"}),
    "pii": frozenset(
        {"enabled", "entities", "input_mode", "untrusted_mode", "output_mode", "restore"}
    ),
    "secrets": frozenset({"enabled", "redact", "key"}),
}

_Action = Literal["block", "redact"]
_Match = Literal["search", "fullmatch"]
_Invisible = Literal["targeted", "all"]
_Token = Literal["block", "truncate"]
_Redact = Literal["all", "partial", "hmac"]
_PiiInput = Literal["redact", "mask", "block"]
_PiiUntrusted = Literal["mask", "redact", "block"]
_PiiOutput = Literal["flag", "redact", "block"]


def config_path() -> Path:
    """``~/.config/jes/config.json``, honoring ``XDG_CONFIG_HOME``."""

    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    home = Path(raw) if raw else Path.home() / ".config"
    return home / "jes" / "config.json"


def default_document() -> dict[str, object]:
    """The config ``jes login`` writes. Three guards on, the rest shown and off."""

    guards: dict[str, object] = {
        "injection": {"enabled": True, "threshold": 0.5},
        "indirect_injection": {"enabled": True, "threshold": 0.5},
        "hazards": {"enabled": True, "threshold": 0.5},
        "toxicity": {"enabled": False, "threshold": 0.5},
        "topics": {"enabled": False, "threshold": 0.5, "deny": []},
        "invisible_text": {"enabled": False, "mode": "targeted", "block": False},
        "allowed_tools": {"enabled": False, "names": []},
        "canary": {"enabled": False, "token": ""},
        "regex": {
            "enabled": False,
            "patterns": [],
            "action": "block",
            "match": "search",
            "require": False,
            "fold": False,
            "timeout_ms": 50,
        },
        "substrings": {
            "enabled": False,
            "terms": [],
            "action": "block",
            "whole_words": False,
            "fold": True,
        },
        "token_limit": {
            "enabled": False,
            "limit": 8000,
            "encoding": "cl100k_base",
            "mode": "block",
        },
        "pii": {
            "enabled": False,
            "input_mode": "redact",
            "untrusted_mode": "mask",
            "output_mode": "flag",
            "restore": True,
        },
        "secrets": {"enabled": False, "redact": "all"},
    }
    return {"guards": guards}


def write_default_config(path: Path | None = None) -> bool:
    """Write the default config when it is absent. Return whether it was written."""

    target = config_path() if path is None else path
    if target.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    target.write_text(json.dumps(default_document(), indent=2) + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return True


def load_policies(path: Path | None = None) -> Sequence[Policy]:
    """Read the config file and build the enabled policies."""

    target = config_path() if path is None else path
    if not target.is_file():
        raise ConfigError("missing config.json; run jes login")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError("invalid config.json") from error
    if not isinstance(value, dict):
        raise ConfigError("config.json must be an object")
    return policies_from_config(cast(dict[str, object], value))


def policies_from_config(document: Mapping[str, object]) -> Sequence[Policy]:
    """Build policies from a config object."""

    unknown = set(document) - {"guards"}
    if unknown:
        raise ConfigError(f"unknown config field {sorted(unknown)[0]}")
    raw = document.get("guards")
    if not isinstance(raw, dict):
        raise ConfigError("config.json must contain guards")
    guards = cast(dict[str, object], raw)
    extra = set(guards) - set(_ORDER)
    if extra:
        raise ConfigError(f"unknown guard {sorted(extra)[0]}")
    chosen: list[object] = []
    for name in _ORDER:
        if name not in guards:
            continue
        body = _body(name, guards[name])
        if body is None:
            continue
        try:
            chosen.append(_build(name, body))
        except PolicyError as error:
            raise ConfigError(str(error)) from error
    return cast(Sequence[Policy], chosen)


def _body(name: str, value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    body = cast(dict[str, object], value)
    extra = set(body) - _FIELDS[name]
    if extra:
        raise ConfigError(f"unknown {name} field {sorted(extra)[0]}")
    enabled = body.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"{name} enabled must be a boolean")
    _check_present(name, body)
    if not enabled:
        return None
    return body


def _check_present(name: str, body: Mapping[str, object]) -> None:
    if name in _JUDGMENTS and "threshold" in body:
        _threshold(name, body["threshold"])
    for field in ("categories", "labels", "deny", "names", "patterns", "terms", "entities"):
        if field in body:
            _strings(name, field, body[field])
    if "mode" in body and name == "invisible_text":
        _choice(name, "mode", body["mode"], {"targeted", "all"})
    if "block" in body and not isinstance(body["block"], bool):
        raise ConfigError(f"invalid {name} block")
    if "action" in body:
        _choice(name, "action", body["action"], {"block", "redact"})
    if "match" in body:
        _choice(name, "match", body["match"], {"search", "fullmatch"})
    for field in ("require", "fold", "whole_words", "restore"):
        if field in body and not isinstance(body[field], bool):
            raise ConfigError(f"invalid {name} {field}")
    if "timeout_ms" in body:
        _positive_int(name, "timeout_ms", body["timeout_ms"])
    if "limit" in body:
        _positive_int(name, "limit", body["limit"])
    if "encoding" in body and (not isinstance(body["encoding"], str) or not body["encoding"]):
        raise ConfigError("invalid token_limit encoding")
    if "token" in body and not isinstance(body["token"], str):
        raise ConfigError("invalid canary token")
    if name == "token_limit" and "mode" in body:
        _choice(name, "mode", body["mode"], {"block", "truncate"})
    if "input_mode" in body:
        _choice(name, "input_mode", body["input_mode"], {"redact", "mask", "block"})
    if "untrusted_mode" in body:
        _choice(name, "untrusted_mode", body["untrusted_mode"], {"mask", "redact", "block"})
    if "output_mode" in body:
        _choice(name, "output_mode", body["output_mode"], {"flag", "redact", "block"})
    if "redact" in body:
        _choice(name, "redact", body["redact"], {"all", "partial", "hmac"})
    if "key" in body:
        _hmac_key(body["key"])


def _build(name: str, body: Mapping[str, object]) -> object:
    if name == "injection":
        return injection(threshold=_threshold(name, _required(body, name, "threshold")))
    if name == "indirect_injection":
        return indirect_injection(threshold=_threshold(name, _required(body, name, "threshold")))
    if name == "hazards":
        categories = _optional_strings(name, body, "categories")
        return hazards(
            categories,
            threshold=_threshold(name, _required(body, name, "threshold")),
        )
    if name == "toxicity":
        return toxicity(
            _optional_strings(name, body, "labels"),
            threshold=_threshold(name, _required(body, name, "threshold")),
        )
    if name == "topics":
        deny = _optional_strings(name, body, "deny")
        if not deny:
            raise ConfigError("topics requires deny")
        return topics(deny, threshold=_threshold(name, _required(body, name, "threshold")))
    if name == "invisible_text":
        mode = cast(
            _Invisible,
            _choice(name, "mode", body.get("mode", "targeted"), {"targeted", "all"}),
        )
        block = body.get("block", False)
        if not isinstance(block, bool):
            raise ConfigError("invalid invisible_text block")
        return invisible_text(mode, block=block)
    if name == "allowed_tools":
        names = _optional_strings(name, body, "names")
        if not names:
            raise ConfigError("allowed_tools requires names")
        return allowed_tools(names)
    if name == "canary":
        token = body.get("token", "")
        if not isinstance(token, str) or not token:
            raise ConfigError("canary requires token")
        return canary(token)
    if name == "regex":
        patterns = _optional_strings(name, body, "patterns")
        if not patterns:
            raise ConfigError("regex requires patterns")
        require = body.get("require", False)
        fold = body.get("fold", False)
        if not isinstance(require, bool) or not isinstance(fold, bool):
            raise ConfigError("invalid regex field")
        action = cast(
            _Action,
            _choice(name, "action", body.get("action", "block"), {"block", "redact"}),
        )
        match = cast(
            _Match,
            _choice(name, "match", body.get("match", "search"), {"search", "fullmatch"}),
        )
        return regex(
            patterns,
            action=action,
            match=match,
            require=require,
            fold=fold,
            timeout_ms=_positive_int(name, "timeout_ms", body.get("timeout_ms", 50)),
        )
    if name == "substrings":
        terms = _optional_strings(name, body, "terms")
        if not terms:
            raise ConfigError("substrings requires terms")
        whole_words = body.get("whole_words", False)
        fold = body.get("fold", True)
        if not isinstance(whole_words, bool) or not isinstance(fold, bool):
            raise ConfigError("invalid substrings field")
        action = cast(
            _Action,
            _choice(name, "action", body.get("action", "block"), {"block", "redact"}),
        )
        return substrings(terms, action=action, whole_words=whole_words, fold=fold)
    if name == "token_limit":
        encoding = body.get("encoding", "cl100k_base")
        if not isinstance(encoding, str) or not encoding:
            raise ConfigError("invalid token_limit encoding")
        mode = cast(
            _Token,
            _choice(name, "mode", body.get("mode", "block"), {"block", "truncate"}),
        )
        limit = _positive_int(name, "limit", _required(body, name, "limit"))
        return token_limit(limit, encoding=encoding, mode=mode)
    if name == "pii":
        return _pii(body)
    return _secrets(body)


def _pii(body: Mapping[str, object]) -> object:
    restore = body.get("restore", True)
    if not isinstance(restore, bool):
        raise ConfigError("invalid pii restore")
    entities = _optional_strings("pii", body, "entities")
    input_mode = cast(
        _PiiInput,
        _choice("pii", "input_mode", body.get("input_mode", "redact"), {"redact", "mask", "block"}),
    )
    untrusted_mode = cast(
        _PiiUntrusted,
        _choice(
            "pii",
            "untrusted_mode",
            body.get("untrusted_mode", "mask"),
            {"mask", "redact", "block"},
        ),
    )
    output_mode = cast(
        _PiiOutput,
        _choice("pii", "output_mode", body.get("output_mode", "flag"), {"flag", "redact", "block"}),
    )
    if entities is None:
        return pii(
            input_mode=input_mode,
            untrusted_mode=untrusted_mode,
            output_mode=output_mode,
            restore=restore,
        )
    return pii(
        entities,
        input_mode=input_mode,
        untrusted_mode=untrusted_mode,
        output_mode=output_mode,
        restore=restore,
    )


def _secrets(body: Mapping[str, object]) -> object:
    redact = cast(
        _Redact,
        _choice("secrets", "redact", body.get("redact", "all"), {"all", "partial", "hmac"}),
    )
    if redact != "hmac":
        if "key" in body:
            raise ConfigError("secrets key is only used when redact is hmac")
        return secrets(redact)
    if "key" not in body:
        raise ConfigError("secrets hmac requires key")
    return secrets("hmac", key=_hmac_key(body["key"]))


def _required(body: Mapping[str, object], name: str, field: str) -> object:
    if field not in body:
        raise ConfigError(f"{name} requires {field}")
    return body[field]


def _threshold(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"invalid {name} threshold")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError(f"invalid {name} threshold")
    return number


def _choice(name: str, field: str, value: object, options: set[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ConfigError(f"invalid {name} {field}")
    return value


def _strings(name: str, field: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"invalid {name} {field}")
    items: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ConfigError(f"invalid {name} {field}")
        items.append(item)
    return tuple(items)


def _optional_strings(
    name: str,
    body: Mapping[str, object],
    field: str,
) -> tuple[str, ...] | None:
    if field not in body:
        return None
    return _strings(name, field, body[field])


def _positive_int(name: str, field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"invalid {name} {field}")
    return value


def _hmac_key(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ConfigError("invalid secrets key")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ConfigError("invalid secrets key") from error
    if len(raw) < 32:
        raise ConfigError("secrets key must be at least 32 bytes")
    return raw
