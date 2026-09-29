"""Claude Code hook adapter.

The normalized command stays free of Claude Code fields. This module only
translates that agent's stdin and stdout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from jes import Guard
from jes._engine.core import freeze_arguments, require_tool_name
from jes.errors import PolicyError
from jes.hook import ConfigError, HookEvent, HookResponse, SessionStore, refusal, run_hook
from jes.types import Stage

_EVENTS: dict[str, Stage] = {
    "UserPromptSubmit": "input",
    "PreToolUse": "tool_call",
    "PostToolUse": "tool_result",
    "MessageDisplay": "output",
}
_TEXT_FIELDS = ("stdout", "text", "content", "output")


def is_partial_display(payload: Mapping[str, object]) -> bool:
    """True when this MessageDisplay batch is not the end of the reply."""

    return payload.get("hook_event_name") == "MessageDisplay" and payload.get("final") is not True


def remember_partial(payload: Mapping[str, object], sessions: SessionStore) -> None:
    ids = _display_ids(payload)
    if ids is None:
        return
    sessions.append_display(ids[0], ids[1], _delta(payload))


def handle(
    payload: Mapping[str, object],
    *,
    guard: Guard,
    sessions: SessionStore,
) -> tuple[dict[str, object], int]:
    """Translate one Claude Code event into a decision JSON object and exit code."""

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return {}, 2
    stage = _EVENTS.get(event_name)
    if stage is None:
        return {}, 0
    try:
        event = _event(payload, stage, sessions)
    except ConfigError:
        return closed_failure(payload)
    response = run_hook(event, guard=guard, sessions=sessions)
    return _body(payload, response), _exit(event_name, response)


def closed_failure(payload: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """Refusal for an event we could not check. Tool results and replies still print JSON."""

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return {}, 2
    stage = _EVENTS.get(event_name)
    if stage is None:
        return {}, 2
    response = refusal(stage)
    return _body(payload, response), _exit(event_name, response)


def _event(payload: Mapping[str, object], stage: Stage, sessions: SessionStore) -> HookEvent:
    session_id = _optional_str(payload, "session_id")
    if stage == "input":
        return HookEvent(
            stage="input",
            text=_required_text(payload, "prompt"),
            session_id=session_id,
        )
    if stage == "tool_call":
        arguments = _tool_input(payload.get("tool_input"))
        return HookEvent(
            stage="tool_call",
            text=arguments,
            tool=_tool_name(payload),
            arguments=arguments,
            session_id=session_id,
        )
    if stage == "tool_result":
        return HookEvent(
            stage="tool_result",
            text=_tool_text(payload.get("tool_response")),
            tool=_tool_name(payload),
            session_id=session_id,
        )
    return HookEvent(
        stage="output",
        text=_reply_text(payload, sessions),
        session_id=session_id,
    )


def _reply_text(payload: Mapping[str, object], sessions: SessionStore) -> str:
    delta = _delta(payload)
    ids = _display_ids(payload)
    if ids is None:
        return delta
    return sessions.finish_display(ids[0], ids[1], delta)


def _body(payload: Mapping[str, object], response: HookResponse) -> dict[str, object]:
    event_name = payload.get("hook_event_name")
    if event_name == "UserPromptSubmit":
        if response.ok:
            return {}
        return {"decision": "block", "reason": response.onward}
    if event_name == "PreToolUse":
        if response.ok:
            return {}
        return _specific(
            "PreToolUse",
            {
                "permissionDecision": "deny",
                "permissionDecisionReason": response.onward,
            },
        )
    if event_name == "PostToolUse":
        if response.ok:
            return {}
        return {
            "decision": "block",
            "reason": response.onward,
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": _replaced_output(payload, response.onward),
            },
        }
    if event_name == "MessageDisplay" and not response.ok:
        return _specific("MessageDisplay", {"displayContent": response.onward})
    return {}


def _exit(event_name: str, response: HookResponse) -> int:
    if event_name in {"PostToolUse", "MessageDisplay"}:
        return 0
    return response.exit_code


def _specific(event_name: str, fields: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {"hookEventName": event_name}
    output.update(fields)
    return {"hookSpecificOutput": output}


def _replaced_output(payload: Mapping[str, object], onward: str) -> object:
    response = payload.get("tool_response")
    parsed = _as_dict(response)
    if parsed is not None and isinstance(parsed.get("stdout"), str):
        updated = dict(parsed)
        updated["stdout"] = onward
        return updated
    if payload.get("tool_name") == "Bash":
        updated = dict(parsed or {})
        updated["stdout"] = onward
        updated.setdefault("stderr", "")
        updated.setdefault("interrupted", False)
        updated.setdefault("isImage", False)
        return updated
    if parsed is not None:
        for key in ("text", "content", "output"):
            if isinstance(parsed.get(key), str):
                updated = dict(parsed)
                updated[key] = onward
                return updated
    return onward


def _tool_text(response: object) -> str:
    if isinstance(response, str):
        return response
    parsed = _as_dict(response)
    if parsed is not None:
        for key in _TEXT_FIELDS:
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    try:
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except TypeError as error:
        raise ConfigError("invalid tool result") from error


def _tool_input(value: object) -> str:
    if value is None:
        value = {}
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ConfigError("invalid arguments")
    try:
        return freeze_arguments(cast(dict[str, object], value))
    except PolicyError as error:
        raise ConfigError("invalid arguments") from error


def _tool_name(payload: Mapping[str, object]) -> str:
    name = payload.get("tool_name")
    if not isinstance(name, str):
        raise ConfigError("invalid tool")
    try:
        return require_tool_name(name)
    except PolicyError as error:
        raise ConfigError("invalid tool") from error


def _display_ids(payload: Mapping[str, object]) -> tuple[str, str] | None:
    session_id = payload.get("session_id")
    message_id = payload.get("message_id")
    if not isinstance(session_id, str) or not isinstance(message_id, str):
        return None
    return session_id, message_id


def _delta(payload: Mapping[str, object]) -> str:
    delta = payload.get("delta")
    if delta is None:
        return ""
    if not isinstance(delta, str):
        raise ConfigError("invalid delta")
    return delta


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"invalid {key}")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str):
        raise ConfigError(f"invalid {key}")
    return value


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None
