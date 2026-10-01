"""The agent guard config, ``~/.config/jes/config.json``.

It holds ``guards``, one object per built-in policy, and an optional ``model``. Each
guard has ``enabled`` plus the fields in its spec below. Unknown guards, unknown fields,
and wrong types are errors, so a typo never silently turns a guard off.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

from jes.agents.files import jes_home, write_private
from jes.errors import ConfigError, PolicyError
from jes.policies import (
    DEFAULT_PII_ENTITIES,
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
    tool_safety,
    topics,
    toxicity,
)

DEFAULT_MODEL = "jev-latest"
# Hooks run under uvx, which installs no spaCy model, so their pii default leaves out
# PERSON. Listing PERSON in entities asks for it, and fails when it cannot run.
HOOK_PII_ENTITIES = tuple(entity for entity in DEFAULT_PII_ENTITIES if entity != "PERSON")

_Kind = Literal["bool", "unit", "text", "texts", "choice", "count", "key"]


@dataclass(frozen=True, slots=True)
class _Field:
    kind: _Kind
    choices: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True, slots=True)
class _Guard:
    fields: Mapping[str, _Field]
    build: Callable[[dict[str, object]], Policy]
    defaults: Mapping[str, object] = field(default_factory=dict[str, object])


def _threshold() -> _Field:
    return _Field("unit", required=True)


def _actions() -> _Field:
    return _Field("choice", ("block", "redact"))


def _secrets(values: dict[str, object]) -> Policy:
    redact = cast(Literal["all", "partial", "hmac"], values.get("redact", "all"))
    key = values.get("key")
    if redact != "hmac":
        if key is not None:
            raise ConfigError("secrets key is only used when redact is hmac")
        return secrets(redact)
    if key is None:
        raise ConfigError("secrets hmac requires key")
    return secrets("hmac", key=cast(bytes, key))


_GUARDS: dict[str, _Guard] = {
    "injection": _Guard({"threshold": _threshold()}, lambda v: injection(threshold=_f(v))),
    "indirect_injection": _Guard(
        {"threshold": _threshold()}, lambda v: indirect_injection(threshold=_f(v))
    ),
    "hazards": _Guard(
        {"threshold": _threshold(), "categories": _Field("texts")},
        lambda v: hazards(_texts(v, "categories"), threshold=_f(v)),
    ),
    "toxicity": _Guard(
        {"threshold": _threshold(), "labels": _Field("texts")},
        lambda v: toxicity(_texts(v, "labels"), threshold=_f(v)),
    ),
    "topics": _Guard(
        {"threshold": _threshold(), "deny": _Field("texts", required=True)},
        lambda v: topics(cast(list[str], v["deny"]), threshold=_f(v)),
    ),
    "invisible_text": _Guard(
        {"mode": _Field("choice", ("targeted", "all")), "block": _Field("bool")},
        lambda v: invisible_text(
            cast(Literal["targeted", "all"], v["mode"]), block=cast(bool, v["block"])
        ),
        {"mode": "targeted", "block": False},
    ),
    "allowed_tools": _Guard(
        {"names": _Field("texts", required=True)},
        lambda v: allowed_tools(cast(list[str], v["names"])),
    ),
    "tool_safety": _Guard({"threshold": _threshold()}, lambda v: tool_safety(threshold=_f(v))),
    "canary": _Guard(
        {"token": _Field("text", required=True)}, lambda v: canary(cast(str, v["token"]))
    ),
    "regex": _Guard(
        {
            "patterns": _Field("texts", required=True),
            "action": _actions(),
            "match": _Field("choice", ("search", "fullmatch")),
            "require": _Field("bool"),
            "fold": _Field("bool"),
            "timeout_ms": _Field("count"),
        },
        lambda v: regex(
            cast(list[str], v["patterns"]),
            action=cast(Literal["block", "redact"], v["action"]),
            match=cast(Literal["search", "fullmatch"], v["match"]),
            require=cast(bool, v["require"]),
            fold=cast(bool, v["fold"]),
            timeout_ms=cast(int, v["timeout_ms"]),
        ),
        {"action": "block", "match": "search", "require": False, "fold": False, "timeout_ms": 50},
    ),
    "substrings": _Guard(
        {
            "terms": _Field("texts", required=True),
            "action": _actions(),
            "whole_words": _Field("bool"),
            "fold": _Field("bool"),
        },
        lambda v: substrings(
            cast(list[str], v["terms"]),
            action=cast(Literal["block", "redact"], v["action"]),
            whole_words=cast(bool, v["whole_words"]),
            fold=cast(bool, v["fold"]),
        ),
        {"action": "block", "whole_words": False, "fold": True},
    ),
    "token_limit": _Guard(
        {
            "limit": _Field("count", required=True),
            "encoding": _Field("text"),
            "mode": _Field("choice", ("block", "truncate")),
        },
        lambda v: token_limit(
            cast(int, v["limit"]),
            encoding=cast(str, v["encoding"]),
            mode=cast(Literal["block", "truncate"], v["mode"]),
        ),
        {"encoding": "cl100k_base", "mode": "block"},
    ),
    "pii": _Guard(
        {
            "entities": _Field("texts"),
            "input_mode": _Field("choice", ("redact", "mask", "block")),
            "untrusted_mode": _Field("choice", ("mask", "redact", "block")),
            "output_mode": _Field("choice", ("flag", "redact", "block")),
            "tool_call_mode": _Field("choice", ("block", "flag")),
            "restore": _Field("bool"),
        },
        lambda v: pii(
            _texts(v, "entities"),
            input_mode=cast(Literal["redact", "mask", "block"], v["input_mode"]),
            untrusted_mode=cast(Literal["mask", "redact", "block"], v["untrusted_mode"]),
            output_mode=cast(Literal["flag", "redact", "block"], v["output_mode"]),
            tool_call_mode=cast(Literal["block", "flag"], v["tool_call_mode"]),
            restore=cast(bool, v["restore"]),
        ),
        {
            "entities": list(HOOK_PII_ENTITIES),
            "input_mode": "redact",
            "untrusted_mode": "mask",
            "output_mode": "flag",
            "tool_call_mode": "block",
            "restore": True,
        },
    ),
    "secrets": _Guard(
        {"redact": _Field("choice", ("all", "partial", "hmac")), "key": _Field("key")},
        _secrets,
    ),
}


def _f(values: dict[str, object]) -> float:
    return cast(float, values["threshold"])


def _texts(values: dict[str, object], name: str) -> list[str] | None:
    return cast(list[str] | None, values.get(name))


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """A parsed config: the model id, and the validated fields of each guard."""

    model: str
    guards: Mapping[str, Mapping[str, object]]

    def policies(self) -> list[Policy]:
        """The enabled guards, built in file order."""

        built: list[Policy] = []
        for name, values in self.guards.items():
            if not values["enabled"]:
                continue
            spec = _GUARDS[name]
            for field_name, field_spec in spec.fields.items():
                if field_spec.required and field_name not in values:
                    raise ConfigError(f"{name} requires {field_name}")
            arguments = {**spec.defaults, **values}
            try:
                built.append(spec.build(arguments))
            except PolicyError as error:
                raise ConfigError(f"{name}: {error}") from None
        return built


def config_path() -> Path:
    return jes_home() / "config.json"


def default_text() -> str:
    return files("jes.agents").joinpath("data", "config.json").read_text(encoding="utf-8")


def default_config() -> AgentConfig:
    return parse_config(json.loads(default_text()))


def write_default_config(path: Path | None = None) -> bool:
    """Write the default config if there is none. Return whether it was written."""

    target = config_path() if path is None else path
    if target.is_file():
        return False
    write_private(target, default_text())
    return True


def load_config(path: Path | None = None) -> AgentConfig:
    target = config_path() if path is None else path
    if not target.is_file():
        raise ConfigError("missing config.json; run jes login")
    try:
        document: object = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ConfigError("invalid config.json: not JSON") from None
    return parse_config(document)


def parse_config(document: object) -> AgentConfig:
    """Check a decoded config.json."""

    if not isinstance(document, dict):
        raise ConfigError("config.json must be an object")
    top = cast(dict[str, object], document)
    unknown = sorted(set(top) - {"guards", "model"})
    if unknown:
        raise ConfigError(f"unknown config field {unknown[0]}")
    model = top.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("model must be a model id")
    raw_guards = top.get("guards")
    if not isinstance(raw_guards, dict):
        raise ConfigError("config.json must contain guards")
    guards: dict[str, Mapping[str, object]] = {}
    for name, body in cast(dict[str, object], raw_guards).items():
        spec = _GUARDS.get(name)
        if spec is None:
            raise ConfigError(f"unknown guard {name}")
        guards[name] = _parse_guard(name, body, spec)
    return AgentConfig(model=model.strip(), guards=guards)


def _parse_guard(name: str, body: object, spec: _Guard) -> dict[str, object]:
    if not isinstance(body, dict):
        raise ConfigError(f"{name} must be an object")
    values = cast(dict[str, object], body)
    unknown = sorted(set(values) - {"enabled", *spec.fields})
    if unknown:
        raise ConfigError(f"unknown {name} field {unknown[0]}")
    if not isinstance(values.get("enabled"), bool):
        raise ConfigError(f"{name} requires enabled, true or false")
    parsed: dict[str, object] = {"enabled": values["enabled"]}
    for field_name, field_spec in spec.fields.items():
        if field_name in values:
            parsed[field_name] = _value(name, field_name, values[field_name], field_spec)
    return parsed


def _value(guard: str, name: str, value: object, spec: _Field) -> object:
    invalid = ConfigError(f"invalid {guard} {name}")
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise invalid
        return value
    if spec.kind == "unit":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise invalid
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise invalid
        return float(value)
    if spec.kind == "count":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise invalid
        return value
    if spec.kind in ("text", "choice"):
        if not isinstance(value, str) or (spec.kind == "choice" and value not in spec.choices):
            raise invalid
        return value
    if spec.kind == "texts":
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in cast(list[object], value)
        ):
            raise invalid
        return list(cast(list[str], value))
    return _key(value, invalid)


def _key(value: object, invalid: ConfigError) -> bytes:
    """An HMAC key: base64 for at least 32 bytes."""

    if not isinstance(value, str) or not value:
        raise invalid
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise invalid from None
    if len(raw) < 32:
        raise ConfigError("secrets key must be at least 32 bytes")
    return raw


__all__ = [
    "DEFAULT_MODEL",
    "HOOK_PII_ENTITIES",
    "AgentConfig",
    "config_path",
    "default_config",
    "default_text",
    "load_config",
    "parse_config",
    "write_default_config",
]
