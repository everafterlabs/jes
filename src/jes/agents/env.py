"""The API key file, ``~/.config/jes/.env``."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from jes.agents.files import jes_home, write_private

_SAVED_KEYS = ("TYPESAFE_API_KEY",)
# Keys older releases wrote. Saving the key again drops them.
_RETIRED_KEYS = frozenset({"JES_MODEL", "JES_THRESHOLD", "JES_HAZARDS", "JES_ALLOWED_TOOLS"})
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def user_env_path() -> Path:
    return jes_home() / ".env"


def load_user_env(
    *,
    environ: MutableMapping[str, str] | None = None,
    path: Path | None = None,
) -> None:
    """Fill missing variables from ``~/.config/jes/.env``. The process environment wins.

    Dotenv files in the working directory are never read. Hooks run inside the project
    the agent is editing, and that project could set ``TYPESAFE_BASE_URL`` to its own
    server and collect your key and every check.
    """

    env = os.environ if environ is None else environ
    source = user_env_path() if path is None else path
    if source.is_file():
        apply_env_file(source, env)


def save_api_key(path: Path, values: Mapping[str, str]) -> None:
    """Store the API key, replacing the old one and keeping every other line."""

    kept: list[str] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_line(raw)
            if parsed is not None and (parsed[0] in values or parsed[0] in _RETIRED_KEYS):
                continue
            kept.append(raw)
    kept.extend(f"{name}={_quote(values[name])}" for name in _SAVED_KEYS if name in values)
    write_private(path, "\n".join(kept) + "\n")


def apply_env_file(path: Path, environ: MutableMapping[str, str]) -> None:
    """Set the keys a dotenv file defines, unless they are set already or blank."""

    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_line(raw)
        if parsed is not None and parsed[0] not in environ and parsed[1]:
            environ[parsed[0]] = parsed[1]


def parse_line(raw: str) -> tuple[str, str] | None:
    """``(name, value)`` for a dotenv line, or None for a comment or anything else."""

    line = raw.strip()
    if line.startswith("export "):
        line = line.removeprefix("export ").strip()
    if not line or line.startswith("#"):
        return None
    name, separator, value = line.partition("=")
    name = name.strip()
    if not separator or not name.isidentifier():
        return None
    return name, _unquote(value.strip())


def _quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        body = value[1:-1]
        return body if value[0] == "'" else _unescape(body)
    comment = value.find(" #")
    return value if comment < 0 else value[:comment].rstrip()


def _unescape(body: str) -> str:
    pieces: list[str] = []
    index = 0
    while index < len(body):
        if body[index] == "\\" and index + 1 < len(body):
            pieces.append(_ESCAPES.get(body[index + 1], body[index + 1]))
            index += 2
        else:
            pieces.append(body[index])
            index += 1
    return "".join(pieces)


__all__ = ["apply_env_file", "load_user_env", "parse_line", "save_api_key", "user_env_path"]
