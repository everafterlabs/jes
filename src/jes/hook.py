"""Normalized hook checks for personal-agent guards."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

from jes import Guard
from jes._engine.core import freeze_arguments, require_tool_name
from jes.errors import PolicyError
from jes.judge import ModelSpec
from jes.policies import Policy, allowed_tools, hazards, indirect_injection, injection
from jes.types import ScanResult, Stage

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_STAGES = frozenset({"input", "tool_call", "tool_result", "output"})
_FAIL_CLOSED = frozenset({"input", "tool_call"})
_FAILURE_TEXT: dict[Stage, str] = {
    "input": "Blocked.",
    "output": "Blocked.",
    "tool_call": "Tool call blocked.",
    "tool_result": "Tool result blocked.",
}


class ConfigError(Exception):
    """A profile or hook payload is invalid. The message is safe to print."""


@dataclass(frozen=True, slots=True)
class Profile:
    """A user-owned coding profile. Threshold and model are explicit."""

    model: str
    threshold: float
    hazards: tuple[str, ...] | None
    allowed_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HookEvent:
    """One normalized check. Arguments are already a frozen string."""

    stage: Stage
    text: str
    tool: str | None = None
    arguments: str | None = None
    prompt: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class HookResponse:
    """The JSON object the hook command writes, plus the process exit code."""

    ok: bool
    decision: Literal["allow", "block"]
    onward: str
    findings: tuple[str, ...]
    exit_code: int

    def payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "decision": self.decision,
            "onward": self.onward,
            "findings": list(self.findings),
        }


def data_text(name: str) -> str:
    """Read a file shipped in ``jes/data``."""

    return files("jes").joinpath("data", name).read_text(encoding="utf-8")


def config_home() -> Path:
    """The user config directory, honoring ``XDG_CONFIG_HOME``."""

    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".config"


def default_profile_path() -> Path:
    return config_home() / "jes" / "profile.toml"


def default_session_dir() -> Path:
    return config_home() / "jes" / "sessions"


def resolve_profile(explicit: str | None) -> Path:
    """Choose the profile file. The packaged example is never loaded by itself."""

    if explicit:
        return Path(explicit)
    env = os.environ.get("JES_PROFILE", "").strip()
    if env:
        return Path(env)
    default = default_profile_path()
    if default.is_file():
        return default
    raise ConfigError(
        "set --profile or JES_PROFILE, or copy jes/data/coding.toml to " + str(default)
    )


def load_profile(path: Path) -> Profile:
    """Load a profile. ``threshold`` and ``model`` must be set in the file."""

    if not path.is_file():
        raise ConfigError("profile not found")
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError("invalid profile") from error
    values = cast(dict[str, object], parsed)
    if "threshold" not in values:
        raise ConfigError("profile requires threshold")
    if "model" not in values:
        raise ConfigError("profile requires model")
    hazards_value = _hazard_categories(values)
    return Profile(
        model=_model_name(values["model"]),
        threshold=_threshold(values["threshold"]),
        hazards=hazards_value,
        allowed_tools=_name_list(values.get("allowed_tools"), field="allowed_tools"),
    )


def policies_from_profile(profile: Profile) -> Sequence[Policy]:
    """Build the coding-agent policies. An empty tool list omits the name gate."""

    threshold = profile.threshold
    chosen: list[object] = [
        injection(threshold=threshold),
        indirect_injection(threshold=threshold),
        hazards(profile.hazards, threshold=threshold)
        if profile.hazards is not None
        else hazards(threshold=threshold),
    ]
    if profile.allowed_tools:
        chosen.append(allowed_tools(profile.allowed_tools))
    return cast(Sequence[Policy], chosen)


def open_guard(profile: Profile, model: ModelSpec | None) -> Guard:
    chosen = profile.model if model is None else model
    return Guard(policies_from_profile(profile), model=chosen)


class SessionStore:
    """Last allowed user prompt for a session, plus in-flight display scratch.

    The prompt file contains that prompt and nothing else. Display scratch is
    deleted once the assistant message is checked.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, session_id: str) -> str | None:
        path = self._prompt_path(session_id)
        if path is None or not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def put(self, session_id: str, prompt: str) -> None:
        path = self._prompt_path(session_id)
        if path is None:
            return
        self._mkdir(path.parent)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(prompt, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)

    def append_display(self, session_id: str, message_id: str, delta: str) -> None:
        path = self._display_path(session_id, message_id)
        if path is None:
            return
        self._write_display(path, delta, consume=False)

    def finish_display(self, session_id: str, message_id: str, delta: str) -> str:
        path = self._display_path(session_id, message_id)
        if path is None:
            return delta
        return self._write_display(path, delta, consume=True)

    def _prompt_path(self, session_id: str) -> Path | None:
        if _SESSION_ID.fullmatch(session_id) is None:
            return None
        return self.root / "prompts" / session_id

    def _display_path(self, session_id: str, message_id: str) -> Path | None:
        if _SESSION_ID.fullmatch(session_id) is None or _SESSION_ID.fullmatch(message_id) is None:
            return None
        return self.root / "display" / session_id / message_id

    def _mkdir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(path, 0o700)

    def _write_display(self, path: Path, delta: str, *, consume: bool) -> str:
        self._mkdir(path.parent)
        for child in path.parent.iterdir():
            if child.name != path.name and child.is_file():
                child.unlink()
        self._mkdir(path.parent)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(delta)
            handle.flush()
            os.chmod(path, 0o600)
            handle.seek(0)
            text = handle.read()
        if consume:
            path.unlink(missing_ok=True)
        return text


def load_object(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigError("invalid JSON") from error
    if not isinstance(value, dict):
        raise ConfigError("invalid JSON")
    return cast(dict[str, object], value)


def event_from_payload(payload: Mapping[str, object]) -> HookEvent:
    stage = payload.get("stage")
    if not isinstance(stage, str) or stage not in _STAGES:
        raise ConfigError("invalid stage")
    prompt = _optional_str(payload, "prompt")
    session_id = _optional_str(payload, "session_id")
    if stage == "tool_call":
        tool = _required_tool(payload)
        arguments = _arguments(payload)
        return HookEvent(
            stage="tool_call",
            text=arguments,
            tool=tool,
            arguments=arguments,
            prompt=prompt,
            session_id=session_id,
        )
    text = _required_str(payload, "text")
    if stage == "input":
        return HookEvent(stage="input", text=text, prompt=prompt, session_id=session_id)
    if stage == "tool_result":
        return HookEvent(
            stage="tool_result",
            text=text,
            tool=_required_tool(payload),
            prompt=prompt,
            session_id=session_id,
        )
    return HookEvent(stage="output", text=text, prompt=prompt, session_id=session_id)


def refusal(stage: Stage) -> HookResponse:
    """A closed failure with no findings. Blocking stages exit 2."""

    return HookResponse(
        ok=False,
        decision="block",
        onward=_FAILURE_TEXT[stage],
        findings=(),
        exit_code=2 if stage in _FAIL_CLOSED else 0,
    )


def run_hook(event: HookEvent, *, guard: Guard, sessions: SessionStore) -> HookResponse:
    """Run one check. Stores the prompt only after an allowed input check."""

    prompt = _resolve_prompt(event, sessions)
    if event.stage in {"tool_call", "output"} and prompt is None:
        return refusal(event.stage)
    try:
        result = _check(event, guard, prompt)
    except Exception:
        print("jes: check failed", file=sys.stderr)
        return refusal(event.stage)
    if event.stage == "input" and result.ok and event.session_id is not None:
        sessions.put(event.session_id, result.onward)
    return HookResponse(
        ok=result.ok,
        decision="allow" if result.ok else "block",
        onward=result.onward,
        findings=_labels(result),
        exit_code=0,
    )


def _check(event: HookEvent, guard: Guard, prompt: str | None) -> ScanResult:
    if event.stage == "input":
        return guard.check_input(event.text)
    if event.stage == "tool_call":
        return guard.check_tool_call(
            cast(str, event.tool),
            cast(str, event.arguments),
            prompt=cast(str, prompt),
        )
    if event.stage == "tool_result":
        return guard.check_tool_result(event.text, name=cast(str, event.tool), prompt=prompt)
    return guard.check_output(event.text, prompt=cast(str, prompt))


def _resolve_prompt(event: HookEvent, sessions: SessionStore) -> str | None:
    if event.prompt is not None:
        return event.prompt
    if event.session_id is None:
        return None
    return sessions.get(event.session_id)


def _labels(result: ScanResult) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                finding.policy if finding.label == "violation" else finding.label
                for finding in result.findings
            }
        )
    )


def _threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("profile requires threshold")
    if not math.isfinite(float(value)):
        raise ConfigError("profile requires threshold")
    return float(value)


def _model_name(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConfigError("profile requires model")
    return value


def _hazard_categories(values: Mapping[str, object]) -> tuple[str, ...] | None:
    if "hazards" not in values:
        return None
    categories = _string_list(values["hazards"], field="hazards")
    if not categories:
        raise ConfigError("profile hazards must name at least one category")
    return categories


def _name_list(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _string_list(value, field=field)


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"profile {field} must be a list of strings")
    items = cast(list[object], value)
    names: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item or item.strip() != item:
            raise ConfigError(f"profile {field} must be a list of strings")
        names.append(item)
    return tuple(names)


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str):
        raise ConfigError(f"invalid {key}")
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = _optional_str(payload, key)
    if value is None:
        raise ConfigError(f"invalid {key}")
    return value


def _required_tool(payload: Mapping[str, object]) -> str:
    tool = _required_str(payload, "tool")
    try:
        return require_tool_name(tool)
    except PolicyError as error:
        raise ConfigError("invalid tool") from error


def _arguments(payload: Mapping[str, object]) -> str:
    if "arguments" in payload and payload["arguments"] is not None:
        value = payload["arguments"]
    elif "text" in payload:
        value = payload["text"]
    else:
        raise ConfigError("invalid arguments")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return freeze_arguments(cast(dict[str, object], value))
        except PolicyError as error:
            raise ConfigError("invalid arguments") from error
    raise ConfigError("invalid arguments")
