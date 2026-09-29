"""jes command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from jes.claude import closed_failure, handle, is_partial_display, remember_partial
from jes.errors import PolicyError
from jes.hook import (
    ConfigError,
    SessionStore,
    data_text,
    default_session_dir,
    event_from_payload,
    load_object,
    load_profile,
    open_guard,
    resolve_profile,
    run_hook,
)
from jes.judge import ModelSpec


def main(argv: Sequence[str] | None = None, *, model: ModelSpec | None = None) -> int:
    """Run ``jes hook``, ``jes claude-hook``, or ``jes claude-settings``."""

    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    command = str(args.command)
    if command == "claude-settings":
        sys.stdout.write(_settings_text())
        return 0
    session_dir = Path(str(args.session_dir)) if args.session_dir else default_session_dir()
    sessions = SessionStore(session_dir)
    raw = sys.stdin.read()
    if command == "claude-hook":
        return _claude(raw, sessions, _optional(args.profile), model)
    return _hook(raw, sessions, _optional(args.profile), model)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jes")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("claude-settings", help="print the Claude Code settings snippet")
    for name in ("hook", "claude-hook"):
        sub = commands.add_parser(name)
        sub.add_argument("--profile", default=None)
        sub.add_argument("--session-dir", default=None)
    return parser


def _hook(raw: str, sessions: SessionStore, profile: str | None, model: ModelSpec | None) -> int:
    try:
        event = event_from_payload(load_object(raw))
        guard_profile = load_profile(resolve_profile(profile))
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    try:
        with open_guard(guard_profile, model) as guard:
            response = run_hook(event, guard=guard, sessions=sessions)
    except PolicyError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    _write_json(response.payload())
    return response.exit_code


def _claude(raw: str, sessions: SessionStore, profile: str | None, model: ModelSpec | None) -> int:
    try:
        payload = load_object(raw)
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    if is_partial_display(payload):
        try:
            remember_partial(payload, sessions)
        except ConfigError as error:
            print(f"jes: {error}", file=sys.stderr)
            return 2
        _write_json({})
        return 0
    try:
        guard_profile = load_profile(resolve_profile(profile))
        with open_guard(guard_profile, model) as guard:
            body, code = handle(payload, guard=guard, sessions=sessions)
    except (ConfigError, PolicyError) as error:
        print(f"jes: {error}", file=sys.stderr)
        body, code = closed_failure(payload)
        if code == 2 and not body:
            return 2
        _write_json(body)
        return code
    _write_json(body)
    return code


def _settings_text() -> str:
    return data_text("claude-settings.json").rstrip("\n") + "\n"


def _write_json(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    sys.stdout.write("\n")


def _optional(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
