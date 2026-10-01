"""Load API keys from the environment and dotenv files."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from jes._files import write_private

_NAME_LINE = 'name = "jes"'
_CONFIG_KEYS = ("TYPESAFE_API_KEY",)
_RETIRED_KEYS = frozenset({"JES_MODEL", "JES_THRESHOLD", "JES_HAZARDS", "JES_ALLOWED_TOOLS"})


def load_project_env(start: Path | None = None) -> None:
    """Set missing variables from the ``.env`` next to this project's ``pyproject.toml``."""

    root = _project_root(Path.cwd() if start is None else start)
    if root is None:
        return
    path = root / ".env"
    if path.is_file():
        apply_env_file(path, os.environ)


def user_env_path() -> Path:
    """``~/.config/jes/.env``, honoring ``XDG_CONFIG_HOME``."""

    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    home = Path(raw) if raw else Path.home() / ".config"
    return home / "jes" / ".env"


def load_config(
    *,
    environ: MutableMapping[str, str] | None = None,
    user_env: Path | None = None,
) -> None:
    """Fill missing variables from ``~/.config/jes/.env``. The process environment wins.

    Dotenv files in the working directory are never read. Hooks run inside the project
    the agent is editing, and that project may set ``TYPESAFE_BASE_URL`` to its own server.
    """

    env = os.environ if environ is None else environ
    config = user_env if user_env is not None else user_env_path()
    if config.is_file():
        apply_env_file(config, env)


def write_config(path: Path, values: Mapping[str, str]) -> None:
    """Store dotenv keys, replacing previous values and keeping every other line."""

    kept: list[str] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(raw)
            if parsed is not None and (parsed[0] in values or parsed[0] in _RETIRED_KEYS):
                continue
            kept.append(raw)
    for name in _CONFIG_KEYS:
        if name in values:
            kept.append(f"{name}={_quote(values[name])}")
    write_private(path, "\n".join(kept) + "\n")


def apply_env_file(path: Path, environ: MutableMapping[str, str]) -> None:
    """Set missing keys from a dotenv file. Existing values and blank lines are kept."""

    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(raw)
        if parsed is None:
            continue
        key, value = parsed
        if key in environ or value == "":
            continue
        environ[key] = value


def _quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _project_root(start: Path) -> Path | None:
    current = start.resolve()
    for directory in (current, *current.parents):
        marker = directory / "pyproject.toml"
        if marker.is_file() and _is_jes_project(marker):
            return directory
    return None


def _is_jes_project(marker: Path) -> bool:
    text = marker.read_text(encoding="utf-8")
    return f"\n{_NAME_LINE}\n" in f"\n{text}\n"


def _parse_line(raw: str) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line.removeprefix("export ").strip()
    key, separator, value = line.partition("=")
    if not separator:
        return None
    name = key.strip()
    if not name.isidentifier():
        return None
    return name, _unquote(value.strip())


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        body = value[1:-1]
        if value[0] == "'":
            return body
        return _unescape(body)
    comment = value.find(" #")
    if comment != -1:
        value = value[:comment].rstrip()
    return value


def _unescape(body: str) -> str:
    pieces: list[str] = []
    index = 0
    while index < len(body):
        if body[index] == "\\" and index + 1 < len(body):
            escaped = body[index + 1]
            pieces.append(
                {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(escaped, escaped)
            )
            index += 2
            continue
        pieces.append(body[index])
        index += 1
    return "".join(pieces)
