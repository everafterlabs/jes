"""Load the nearest project ``.env`` into the process environment."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

_NAME_LINE = 'name = "jes"'


def load_project_env(start: Path | None = None) -> None:
    """Set missing variables from the ``.env`` next to this project's ``pyproject.toml``."""

    root = _project_root(Path.cwd() if start is None else start)
    if root is None:
        return
    path = root / ".env"
    if path.is_file():
        apply_env_file(path, os.environ)


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
