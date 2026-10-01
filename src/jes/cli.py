"""jes command line."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from jes import Guard, __version__, claude, codex, hermes
from jes._env import load_config, user_env_path, write_config
from jes.config import config_path, write_default_config
from jes.errors import PolicyError
from jes.hook import (
    ConfigError,
    SessionStore,
    data_text,
    default_session_dir,
    event_from_payload,
    load_object,
    open_guard,
    run_hook,
)
from jes.judge import ModelSpec

_SETTINGS = {
    "claude-settings": "claude-settings.json",
    "codex-settings": "codex-settings.json",
    "hermes-settings": "hermes-settings.yaml",
    "opencode-settings": "opencode-plugin.ts",
    "openclaw-settings": "openclaw-plugin.ts",
    "pi-settings": "pi-extension.ts",
    "runner-settings": "jes-runner.ts",
}
_VERSION_PLACEHOLDER = "__JES_VERSION__"


class _Adapter(Protocol):
    def prepare(
        self,
        payload: Mapping[str, object],
        sessions: SessionStore,
    ) -> tuple[dict[str, object], int] | None: ...

    def handle(
        self,
        payload: Mapping[str, object],
        *,
        guard: Guard,
        sessions: SessionStore,
    ) -> tuple[dict[str, object], int]: ...

    def closed_failure(self, payload: Mapping[str, object]) -> tuple[dict[str, object], int]: ...


_ADAPTERS: dict[str, _Adapter] = {
    "claude-hook": claude,
    "codex-hook": codex,
    "hermes-hook": hermes,
}


def main(argv: Sequence[str] | None = None, *, model: ModelSpec | None = None) -> int:
    """Run a jes hook command or print an agent settings snippet."""

    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    command = str(args.command)
    if command == "login":
        return _login()
    settings = _SETTINGS.get(command)
    if settings is not None:
        # Pin the hooks to this release, so a new upload to PyPI never runs unannounced.
        text = data_text(settings).replace(_VERSION_PLACEHOLDER, __version__)
        sys.stdout.write(text.rstrip("\n") + "\n")
        return 0
    session_dir = Path(str(args.session_dir)) if args.session_dir else default_session_dir()
    sessions = SessionStore(session_dir)
    raw = sys.stdin.read()
    adapter = _ADAPTERS.get(command)
    if adapter is None:
        return _hook(raw, sessions, model)
    return _agent(raw, sessions, model, adapter)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jes")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login", help="save the API key")
    for name in _SETTINGS:
        commands.add_parser(name, help="print the settings snippet or plugin")
    for name in ("hook", *_ADAPTERS):
        sub = commands.add_parser(name)
        sub.add_argument("--session-dir", default=None)
    return parser


def _hook(raw: str, sessions: SessionStore, model: ModelSpec | None) -> int:
    try:
        event = event_from_payload(load_object(raw))
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    load_config()
    try:
        with open_guard(model) as guard:
            response = run_hook(event, guard=guard, sessions=sessions)
    except (ConfigError, PolicyError) as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    _write_json(response.payload())
    return response.exit_code


def _agent(
    raw: str,
    sessions: SessionStore,
    model: ModelSpec | None,
    adapter: _Adapter,
) -> int:
    try:
        payload = load_object(raw)
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    try:
        early = adapter.prepare(payload, sessions)
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    if early is not None:
        body, code = early
        _write_json(body)
        return code
    try:
        load_config()
        with open_guard(model) as guard:
            body, code = adapter.handle(payload, guard=guard, sessions=sessions)
    except (ConfigError, PolicyError) as error:
        print(f"jes: {error}", file=sys.stderr)
        body, code = adapter.closed_failure(payload)
        if code == 2 and not body:
            return 2
        _write_json(body)
        return code
    _write_json(body)
    return code


def _login() -> int:
    key = getpass.getpass("TypeSafe API key: ").strip()
    if not key:
        print("jes: empty key", file=sys.stderr)
        return 2
    path = user_env_path()
    write_config(path, {"TYPESAFE_API_KEY": key})
    created = write_default_config()
    print(f"Saved API key to {path}")
    if created:
        print(f"Wrote {config_path()}")
    return 0


def _write_json(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    sys.stdout.write("\n")
