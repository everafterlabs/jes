"""Each coding agent's hook protocol, translated to one jes check and back."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import ClassVar, Protocol, cast

from jes.agents.hook import (
    HookEvent,
    HookResponse,
    optional_text,
    refusal,
    required_text,
    run_hook,
    tool_arguments,
    tool_name,
    tool_text,
)
from jes.agents.sessions import SessionStore
from jes.errors import ConfigError, PolicyError
from jes.guard import Guard
from jes.types import Stage

Body = dict[str, object]
Payload = Mapping[str, object]


class Adapter(Protocol):
    """One agent's hook events, how to read them, and how to answer."""

    events: ClassVar[Mapping[str, Stage]]

    def skip(
        self, payload: Payload, event_name: str, sessions: SessionStore
    ) -> tuple[Body, int] | None: ...

    def event(self, payload: Payload, stage: Stage, sessions: SessionStore) -> HookEvent: ...

    def render(
        self, payload: Payload, event_name: str, response: HookResponse
    ) -> tuple[Body, int]: ...


def handle(
    adapter: Adapter,
    payload: Payload,
    *,
    open_guard: Callable[[], Guard],
    sessions: SessionStore,
) -> tuple[Body, int]:
    """Check one hook event. Anything that stops the check answers with a refusal."""

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str):
        return {}, 2
    stage = adapter.events.get(event_name)
    if stage is None:
        return {}, 0
    try:
        skipped = adapter.skip(payload, event_name, sessions)
        if skipped is not None:
            return skipped
        event = adapter.event(payload, stage, sessions)
    except ConfigError as error:
        print(f"jes: {error}", file=sys.stderr)
        return adapter.render(payload, event_name, refusal(stage))
    try:
        with open_guard() as guard:
            response = run_hook(event, guard=guard, sessions=sessions)
    except (ConfigError, PolicyError) as error:
        print(f"jes: {error}", file=sys.stderr)
        response = refusal(stage)
    return adapter.render(payload, event_name, response)


def _tool_event(payload: Payload, stage: Stage, session_id: str | None) -> HookEvent:
    if stage == "tool_call":
        return HookEvent(
            "tool_call",
            tool_arguments(payload.get("tool_input")),
            tool=tool_name(payload),
            session_id=session_id,
        )
    return HookEvent(
        "tool_result",
        tool_text(payload.get("tool_response")),
        tool=tool_name(payload),
        session_id=session_id,
    )


def _prompt_body(response: HookResponse) -> Body:
    return {} if response.ok else {"decision": "block", "reason": response.onward}


def _pre_tool_body(response: HookResponse) -> Body:
    if response.ok:
        return {}
    return _specific(
        "PreToolUse",
        {"permissionDecision": "deny", "permissionDecisionReason": response.onward},
    )


def _specific(event_name: str, fields: Body) -> Body:
    return {"hookSpecificOutput": {"hookEventName": event_name, **fields}}


class Claude:
    """Claude Code. A reply streams in MessageDisplay parts and is checked when final."""

    events: ClassVar[Mapping[str, Stage]] = {
        "UserPromptSubmit": "input",
        "PreToolUse": "tool_call",
        "PostToolUse": "tool_result",
        "MessageDisplay": "output",
    }

    def skip(
        self, payload: Payload, event_name: str, sessions: SessionStore
    ) -> tuple[Body, int] | None:
        if event_name != "MessageDisplay" or payload.get("final") is True:
            return None
        delta = _delta(payload)
        ids = _display_ids(payload)
        if ids is not None:
            sessions.append_display(*ids, delta)
        return {}, 0

    def event(self, payload: Payload, stage: Stage, sessions: SessionStore) -> HookEvent:
        session_id = optional_text(payload, "session_id")
        if stage == "input":
            return HookEvent("input", required_text(payload, "prompt"), session_id=session_id)
        if stage == "output":
            delta = _delta(payload)
            ids = _display_ids(payload)
            text = delta if ids is None else sessions.finish_display(*ids, delta)
            return HookEvent("output", text, session_id=session_id)
        return _tool_event(payload, stage, session_id)

    def render(self, payload: Payload, event_name: str, response: HookResponse) -> tuple[Body, int]:
        if event_name == "UserPromptSubmit":
            return _prompt_body(response), response.exit_code
        if event_name == "PreToolUse":
            return _pre_tool_body(response), response.exit_code
        if response.ok:
            return {}, 0
        if event_name == "PostToolUse":
            body: Body = {
                "decision": "block",
                "reason": response.onward,
                **_specific(
                    "PostToolUse", {"updatedToolOutput": _replaced(payload, response.onward)}
                ),
            }
            return body, 0
        return _specific("MessageDisplay", {"displayContent": response.onward}), 0


def _replaced(payload: Payload, onward: str) -> object:
    """The tool response with its text swapped for the refusal, in the shape Claude expects."""

    response = payload.get("tool_response")
    fields = cast(dict[str, object], response) if isinstance(response, dict) else None
    # The model reads stderr too, so a blocked shell result clears it along with stdout.
    if fields is not None and isinstance(fields.get("stdout"), str):
        updated = {**fields, "stdout": onward}
        if "stderr" in updated:
            updated["stderr"] = ""
        return updated
    if payload.get("tool_name") == "Bash":
        return {
            "interrupted": False,
            "isImage": False,
            **(fields or {}),
            "stdout": onward,
            "stderr": "",
        }
    if fields is not None:
        for key in ("text", "content", "output"):
            if isinstance(fields.get(key), str):
                return {**fields, key: onward}
    return onward


def _display_ids(payload: Payload) -> tuple[str, str] | None:
    session_id, message_id = payload.get("session_id"), payload.get("message_id")
    if isinstance(session_id, str) and isinstance(message_id, str):
        return session_id, message_id
    return None


def _delta(payload: Payload) -> str:
    return optional_text(payload, "delta") or ""


class Codex:
    """Codex. The reply is the Stop event's last assistant message."""

    events: ClassVar[Mapping[str, Stage]] = {
        "UserPromptSubmit": "input",
        "PreToolUse": "tool_call",
        "PostToolUse": "tool_result",
        "Stop": "output",
    }

    def skip(
        self, payload: Payload, event_name: str, sessions: SessionStore
    ) -> tuple[Body, int] | None:
        if event_name != "Stop":
            return None
        # A Stop that Codex already continued after a refusal, or one with no reply.
        if payload.get("stop_hook_active") is True or not _last_message(payload):
            return {}, 0
        return None

    def event(self, payload: Payload, stage: Stage, sessions: SessionStore) -> HookEvent:
        session_id = optional_text(payload, "session_id")
        if stage == "input":
            return HookEvent("input", required_text(payload, "prompt"), session_id=session_id)
        if stage == "output":
            return HookEvent("output", _last_message(payload), session_id=session_id)
        return _tool_event(payload, stage, session_id)

    def render(self, payload: Payload, event_name: str, response: HookResponse) -> tuple[Body, int]:
        if event_name == "UserPromptSubmit":
            return _prompt_body(response), response.exit_code
        if event_name == "PreToolUse":
            return _pre_tool_body(response), response.exit_code
        # PostToolUse replaces the model-facing result; Stop continues the turn with the reason.
        return _prompt_body(response), 0


def _last_message(payload: Payload) -> str:
    value = payload.get("last_assistant_message")
    return value if isinstance(value, str) else ""


class Hermes:
    """Hermes shell hooks. They can block a tool call and add context to a prompt.

    Hermes drops replacement JSON from its transform events, so jes does not check them.
    """

    events: ClassVar[Mapping[str, Stage]] = {"pre_llm_call": "input", "pre_tool_call": "tool_call"}

    def skip(
        self, payload: Payload, event_name: str, sessions: SessionStore
    ) -> tuple[Body, int] | None:
        if event_name == "pre_llm_call" and _user_message(payload) is None:
            return {}, 0
        return None

    def event(self, payload: Payload, stage: Stage, sessions: SessionStore) -> HookEvent:
        session_id = optional_text(payload, "session_id")
        if stage == "input":
            return HookEvent("input", cast(str, _user_message(payload)), session_id=session_id)
        return _tool_event(payload, stage, session_id)

    def render(self, payload: Payload, event_name: str, response: HookResponse) -> tuple[Body, int]:
        if response.ok:
            return {}, 0
        if event_name == "pre_tool_call":
            return {"action": "block", "message": response.onward}, 2
        return {"context": response.onward}, 2


def _user_message(payload: Payload) -> str | None:
    extra = payload.get("extra")
    fields = cast(dict[str, object], extra) if isinstance(extra, dict) else {}
    value = fields.get("user_message", payload.get("user_message"))
    return value if isinstance(value, str) else None


ADAPTERS: dict[str, Adapter] = {
    "claude-hook": Claude(),
    "codex-hook": Codex(),
    "hermes-hook": Hermes(),
}

__all__ = ["ADAPTERS", "Adapter", "Claude", "Codex", "Hermes", "handle"]
