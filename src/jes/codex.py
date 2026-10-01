"""Codex hook adapter.

Codex uses Claude Code's event names. PostToolUse replaces the model-facing
result through ``decision`` and ``reason``. The reply is the Stop event's
``last_assistant_message``.
"""

from __future__ import annotations

from collections.abc import Mapping

from jes import Guard
from jes.claude import _body as claude_body
from jes.hook import HookEvent, HookResponse, SessionStore, refusal, run_hook
from jes.payload import ConfigError, optional_str, required_text, tool_input, tool_name, tool_text
from jes.types import Stage

_EVENTS: dict[str, Stage] = {
    "UserPromptSubmit": "input",
    "PreToolUse": "tool_call",
    "PostToolUse": "tool_result",
    "Stop": "output",
}


def prepare(
    payload: Mapping[str, object],
    sessions: SessionStore,
) -> tuple[dict[str, object], int] | None:
    """Skip a Stop that Codex already continued from a previous refusal."""

    del sessions
    if payload.get("hook_event_name") == "Stop" and payload.get("stop_hook_active") is True:
        return {}, 0
    return None


def handle(
    payload: Mapping[str, object],
    *,
    guard: Guard,
    sessions: SessionStore,
) -> tuple[dict[str, object], int]:
    """Translate one Codex event into a decision JSON object and exit code."""

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return {}, 2
    stage = _EVENTS.get(event_name)
    if stage is None:
        return {}, 0
    if event_name == "Stop" and not _assistant_text(payload):
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
    del sessions
    session_id = optional_str(payload, "session_id")
    if stage == "input":
        return HookEvent(
            stage="input",
            text=required_text(payload, "prompt"),
            session_id=session_id,
        )
    if stage == "tool_call":
        arguments = tool_input(payload.get("tool_input"))
        return HookEvent(
            stage="tool_call",
            text=arguments,
            tool=tool_name(payload),
            arguments=arguments,
            session_id=session_id,
        )
    if stage == "tool_result":
        return HookEvent(
            stage="tool_result",
            text=tool_text(payload.get("tool_response")),
            tool=tool_name(payload),
            session_id=session_id,
        )
    text = _assistant_text(payload)
    if text is None:
        raise ConfigError("invalid last_assistant_message")
    return HookEvent(stage="output", text=text, session_id=session_id)


def _body(payload: Mapping[str, object], response: HookResponse) -> dict[str, object]:
    event_name = payload.get("hook_event_name")
    if event_name in {"UserPromptSubmit", "PreToolUse"}:
        return claude_body(payload, response)
    if response.ok:
        return {}
    if event_name in {"PostToolUse", "Stop"}:
        return {"decision": "block", "reason": response.onward}
    return {}


def _exit(event_name: str, response: HookResponse) -> int:
    if event_name in {"PostToolUse", "Stop"}:
        return 0
    return response.exit_code


def _assistant_text(payload: Mapping[str, object]) -> str | None:
    value = payload.get("last_assistant_message")
    if isinstance(value, str) and value:
        return value
    return None
