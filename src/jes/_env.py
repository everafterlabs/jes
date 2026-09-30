"""Load API keys from the environment and dotenv files."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

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
    start: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    user_env: Path | None = None,
) -> None:
    """Fill missing variables from dotenv files.

    The first value wins: the process environment, ``.env.local`` then ``.env``
    at the git repo root, then ``~/.config/jes/.env``.
    """

    env = os.environ if environ is None else environ
    root = _repo_root(Path.cwd() if start is None else start)
    config = user_env if user_env is not None else user_env_path()
    for path in (root / ".env.local", root / ".env", config):
        if path.is_file():
            apply_env_file(path, env)


def write_config(path: Path, values: Mapping[str, str]) -> None:
    """Store dotenv keys, replacing previous values and keeping every other line."""

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
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
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


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


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / ".git").exists():
            return directory
    return current


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
