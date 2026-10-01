"""Tool names, frozen arguments, and the payload fields the hook adapters share."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import cast

from jes.errors import PolicyError


class ConfigError(Exception):
    """A hook payload or environment setting is invalid. The message is safe to print."""


def require_tool_name(name: str) -> str:
    if not name or len(name) > 256 or name.strip() != name or any(ord(char) < 32 for char in name):
        raise PolicyError("tool name must be a non-empty single-line string")
    return name


def freeze_arguments(arguments: str | Mapping[str, object]) -> str:
    """Keep a string subject, or serialize a mapping one canonical way."""

    if isinstance(arguments, str):
        return arguments
    _require_json(arguments)
    return json.dumps(
        _plain_json(arguments),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def optional_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str):
        raise ConfigError(f"invalid {key}")
    return value


def required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"invalid {key}")
    return value


def tool_name(payload: Mapping[str, object]) -> str:
    name = payload.get("tool_name")
    if not isinstance(name, str):
        raise ConfigError("invalid tool")
    try:
        return require_tool_name(name)
    except PolicyError as error:
        raise ConfigError("invalid tool") from error


def tool_input(value: object) -> str:
    if value is None:
        value = {}
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ConfigError("invalid arguments")
    try:
        return freeze_arguments(cast(dict[str, object], value))
    except PolicyError as error:
        raise ConfigError("invalid arguments") from error


_TEXT_FIELDS = ("text", "content", "output")


def tool_text(response: object) -> str:
    """The text a tool result shows the model. A shell result is its stdout plus its stderr."""

    if isinstance(response, str):
        return response
    parsed = _as_dict(response)
    if parsed is not None:
        stdout = parsed.get("stdout")
        if isinstance(stdout, str):
            stderr = parsed.get("stderr")
            streams = (stdout, stderr if isinstance(stderr, str) else "")
            return "\n".join(stream for stream in streams if stream)
        for key in _TEXT_FIELDS:
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    try:
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except TypeError as error:
        raise ConfigError("invalid tool result") from error


def _require_json(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PolicyError("tool arguments must be JSON values")
        return
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        for key, item in items.items():
            if not isinstance(key, str):
                raise PolicyError("tool arguments must be JSON values")
            _require_json(item)
        return
    if isinstance(value, list):
        elements = cast(list[object], value)
        for item in elements:
            _require_json(item)
        return
    raise PolicyError("tool arguments must be JSON values")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        items = cast(Mapping[str, object], value)
        return {key: _plain_json(item) for key, item in items.items()}
    if isinstance(value, list):
        elements = cast(list[object], value)
        return [_plain_json(item) for item in elements]
    return value


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None
