"""The jes command: hooks for coding agents, their settings, and login."""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import sys
import warnings
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path

from jes import __version__
from jes.agents.adapters import ADAPTERS, handle
from jes.agents.config import config_path, load_config, write_default_config
from jes.agents.env import load_user_env, save_api_key, user_env_path
from jes.agents.hook import event_from_payload, load_object, run_hook
from jes.agents.sessions import SessionStore, default_session_dir
from jes.backend import ModelSpec
from jes.errors import ConfigError, PolicyError
from jes.guard import Guard

_PRINTED = {
    "claude-settings": "claude-settings.json",
    "codex-settings": "codex-settings.json",
    "hermes-settings": "hermes-settings.yaml",
    "opencode-settings": "opencode-plugin.ts",
    "openclaw-settings": "openclaw-plugin.ts",
    "pi-settings": "pi-extension.ts",
    "runner-settings": "jes-runner.ts",
}
_VERSION_PLACEHOLDER = "__JES_VERSION__"


def main(argv: Sequence[str] | None = None, *, model: ModelSpec | None = None) -> int:
    """Run one jes command. ``model`` replaces the configured model, for tests."""

    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    command = str(args.command)
    if command == "login":
        return _login()
    if command in _PRINTED:
        # Pin hooks to this release, so a new upload to PyPI never runs unannounced.
        text = files("jes.agents").joinpath("data", _PRINTED[command]).read_text(encoding="utf-8")
        sys.stdout.write(text.replace(_VERSION_PLACEHOLDER, __version__).rstrip("\n") + "\n")
        return 0
    # Hooks print to the agent's transcript, and LangChain's beta notice is noise there.
    # Importing LangChain resets its warning filters, so the filter goes in after it.
    importlib.import_module("langchain_typesafe")
    warnings.filterwarnings("ignore", message=r"The class `TypeSafeClassifier` is in beta")
    sessions = SessionStore(Path(args.session_dir) if args.session_dir else default_session_dir())
    try:
        payload = load_object(sys.stdin.read())
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2

    def open_guard() -> Guard:
        load_user_env()
        config = load_config()
        return Guard(config.policies(), model=config.model if model is None else model)

    adapter = ADAPTERS.get(command)
    if adapter is not None:
        body, code = handle(adapter, payload, open_guard=open_guard, sessions=sessions)
        _write_json(body)
        return code
    try:
        event = event_from_payload(payload)
        with open_guard() as guard:
            response = run_hook(event, guard=guard, sessions=sessions)
    except (ConfigError, PolicyError) as error:
        print(f"jes: {error}", file=sys.stderr)
        return 2
    _write_json(response.payload())
    return response.exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jes", description="Guardrails for coding agents.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login", help="save your TypeSafe API key and a default config")
    for name in _PRINTED:
        commands.add_parser(name, help="print hook settings or a plugin for that agent")
    for name in ("hook", *ADAPTERS):
        hook = commands.add_parser(name, help="check one hook event read from stdin")
        hook.add_argument("--session-dir", default=None)
    return parser


def _login() -> int:
    key = getpass.getpass("TypeSafe API key: ").strip()
    if not key:
        print("jes: empty key", file=sys.stderr)
        return 2
    path = user_env_path()
    save_api_key(path, {"TYPESAFE_API_KEY": key})
    print(f"Saved API key to {path}")
    if write_default_config():
        print(f"Wrote {config_path()}")
    return 0


def _write_json(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["main"]
