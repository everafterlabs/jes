"""Per-session state the hooks keep between calls.

Each session's last allowed prompt, so later tool calls are judged against what the user
asked, and the parts of a streamed reply until it is complete. Everything lives under
``~/.config/jes/sessions`` in private files.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from pathlib import Path
from typing import Any

from jes.agents.files import jes_home, make_private_dirs, write_private

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock

    def _lock(handle: Any) -> None:
        del handle

else:

    def _lock(handle: Any) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")
# Parts of a reply that never finished are removed after this many seconds.
_STALE_SECONDS = 24 * 60 * 60


def default_session_dir() -> Path:
    return jes_home() / "sessions"


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def prompt(self, session_id: str) -> str | None:
        """The last prompt this session allowed, if any."""

        path = self._prompt_path(session_id)
        if path is None or not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def save_prompt(self, session_id: str, prompt: str) -> None:
        path = self._prompt_path(session_id)
        if path is not None:
            write_private(path, prompt)

    def append_display(self, session_id: str, message_id: str, delta: str) -> None:
        """Keep a part of a streamed reply until the reply is complete."""

        path = self._display_path(session_id, message_id)
        if path is not None:
            self._prune(path.parent)
            self._write(path, delta, finish=False)

    def finish_display(self, session_id: str, message_id: str, delta: str) -> str:
        """The whole reply: every kept part plus this last one. The parts are then removed."""

        path = self._display_path(session_id, message_id)
        if path is None:
            return delta
        return self._write(path, delta, finish=True)

    def _prompt_path(self, session_id: str) -> Path | None:
        if _ID.fullmatch(session_id) is None:
            return None
        return self.root / "prompts" / session_id

    def _display_path(self, session_id: str, message_id: str) -> Path | None:
        if _ID.fullmatch(session_id) is None or _ID.fullmatch(message_id) is None:
            return None
        return self.root / "display" / session_id / message_id

    def _write(self, path: Path, delta: str, *, finish: bool) -> str:
        make_private_dirs(path.parent)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            _lock(handle)
            handle.write(delta)
            handle.flush()
            handle.seek(0)
            text = handle.read()
            if finish:
                # Removed while still locked, so no part can slip in between.
                path.unlink(missing_ok=True)
        return text

    def _prune(self, directory: Path) -> None:
        """Remove the parts of replies that never finished. Other messages are left alone."""

        if not directory.is_dir():
            return
        cutoff = time.time() - _STALE_SECONDS
        for child in directory.iterdir():
            with contextlib.suppress(OSError):
                if child.is_file() and child.stat().st_mtime < cutoff:
                    child.unlink()


__all__ = ["SessionStore", "default_session_dir"]
