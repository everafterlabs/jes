"""Private files under the jes config directory."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def make_private_dirs(directory: Path) -> None:
    """Create each missing directory down to ``directory`` with mode 0700.

    Directories that already exist keep their permissions. They may belong to the user.
    """

    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    for path in reversed(missing):
        path.mkdir(mode=0o700, exist_ok=True)


def write_private(path: Path, text: str) -> None:
    """Replace ``path`` with ``text``. The file is mode 0600 from the moment it exists."""

    make_private_dirs(path.parent)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
