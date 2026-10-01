"""Hermes shell-hook adapter.

Hermes sends ``hook_event_name`` plus ``tool_name``, ``tool_input``, and an
``extra`` object. ``pre_tool_call`` blocks with ``action`` and exit 2.
A blocked ``pre_llm_call`` prints ``context``; Hermes injects that text and
does not reject the user message. Transform events still run the check, but
their stdout is empty because Hermes shell hooks drop replacement JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from jes import Guard
from jes.hook import HookEvent, HookResponse, SessionStore, refusal, run_hook
from jes.payload import ConfigError, optional_str, tool_input, tool_name
from jes.types import Stage

_EVENTS: dict[str, Stage] = {
    "pre_llm_call": "input",
    "pre_tool_call": "tool_call",
    "transform_tool_result": "tool_result",
    "transform_llm_output": "output",
}


def prepare(
    payload: Mapping[str, object],
    sessions: SessionStore,
) -> tuple[dict[str, object], int] | None:
    del payload, sessions
    return None


def handle(
    payload: Mapping[str, object],
    *,
    guard: Guard,
    sessions: SessionStore,
) -> tuple[dict[str, object], int]:
    """Translate one Hermes event into a decision JSON object and exit code."""

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return {}, 2
    stage = _EVENTS.get(event_name)
    if stage is None:
        return {}, 0
    if event_name == "pre_llm_call" and _user_message(payload) is None:
        return {}, 0
    try:
        event = _event(payload, stage)
    except ConfigError:
        return closed_failure(payload)
    response = run_hook(event, guard=guard, sessions=sessions)
    return _body(event_name, response), _exit(event_name, response)


def closed_failure(payload: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """Refusal for an event we could not check."""

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return {}, 2
    stage = _EVENTS.get(event_name)
    if stage is None:
        return {}, 2
    response = refusal(stage)
    return _body(event_name, response), _exit(event_name, response)


def _event(payload: Mapping[str, object], stage: Stage) -> HookEvent:
    session_id = optional_str(payload, "session_id")
    if stage == "input":
        text = _user_message(payload)
        if text is None:
            raise ConfigError("invalid user_message")
        return HookEvent(stage="input", text=text, session_id=session_id)
    if stage == "output":
        return HookEvent(
            stage="output",
            text=_response_text(payload),
            session_id=session_id,
        )
    tool = tool_name(payload)
    if stage == "tool_call":
        arguments = tool_input(payload.get("tool_input"))
        return HookEvent(
            stage="tool_call",
            text=arguments,
            tool=tool,
            arguments=arguments,
            session_id=session_id,
        )
    return HookEvent(
        stage="tool_result",
        text=_result_text(payload),
        tool=tool,
        session_id=session_id,
    )


def _body(event_name: str, response: HookResponse) -> dict[str, object]:
    if response.ok:
        return {}
    if event_name == "pre_tool_call":
        return {"action": "block", "message": response.onward}
    if event_name == "pre_llm_call":
        return {"context": response.onward}
    return {}


def _exit(event_name: str, response: HookResponse) -> int:
    if event_name in {"transform_tool_result", "transform_llm_output"}:
        return 0
    if not response.ok:
        return 2
    return 0


def _extra(payload: Mapping[str, object]) -> Mapping[str, object]:
    value = payload.get("extra")
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _user_message(payload: Mapping[str, object]) -> str | None:
    value = _extra(payload).get("user_message", payload.get("user_message"))
    if isinstance(value, str):
        return value
    return None


def _result_text(payload: Mapping[str, object]) -> str:
    value = _extra(payload).get("result")
    if isinstance(value, str):
        return value
    if value is None:
        raise ConfigError("invalid result")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError as error:
        raise ConfigError("invalid result") from error


def _response_text(payload: Mapping[str, object]) -> str:
    value = _extra(payload).get("response_text")
    if isinstance(value, str):
        return value
    raise ConfigError("invalid response_text")
