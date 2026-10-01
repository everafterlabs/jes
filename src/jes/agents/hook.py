"""One check per hook call, in a shape every coding agent adapter shares.

``jes hook`` speaks this shape directly as JSON: ``stage``, ``text``, ``tool``,
``arguments``, ``prompt``, and ``session_id`` in; ``ok``, ``decision``, ``onward``, and
``findings`` out. A check that cannot run refuses: input and tool calls fail closed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from jes.agents.sessions import SessionStore
from jes.errors import ConfigError, PolicyError
from jes.guard import Guard, freeze_arguments
from jes.policies.transforms import valid_tool_name
from jes.result import Result, finding_name, refusal as refusal_text
from jes.types import Stage

_STAGES: frozenset[Stage] = frozenset({"input", "tool_call", "tool_result", "output"})
# A failed check of these stages exits 2, the code most agents read as "block".
_FAIL_CLOSED: frozenset[Stage] = frozenset({"input", "tool_call"})
_TEXT_FIELDS = ("text", "content", "output")


@dataclass(frozen=True, slots=True)
class HookEvent:
    """One normalized check. ``text`` is the tool arguments for a tool call."""

    stage: Stage
    text: str
    tool: str | None = None
    prompt: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class HookResponse:
    """What the check decided, plus the process exit code."""

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


def refusal(stage: Stage) -> HookResponse:
    """The response for a check that could not run."""

    return HookResponse(
        ok=False,
        decision="block",
        onward=refusal_text(stage, ()),
        findings=(),
        exit_code=2 if stage in _FAIL_CLOSED else 0,
    )


def run_hook(event: HookEvent, *, guard: Guard, sessions: SessionStore) -> HookResponse:
    """Run one check. An allowed input becomes the session's prompt."""

    prompt = event.prompt
    if prompt is None and event.session_id is not None:
        prompt = sessions.prompt(event.session_id)
    if event.stage in ("tool_call", "output") and prompt is None:
        # Without the user's request there is nothing to judge a tool call or reply against.
        return refusal(event.stage)
    try:
        result = _check(event, guard, prompt)
    except Exception as error:
        print(f"jes: check failed ({type(error).__name__})", file=sys.stderr)
        return refusal(event.stage)
    if event.stage == "input" and result.ok and event.session_id is not None:
        sessions.save_prompt(event.session_id, result.onward)
    return HookResponse(
        ok=result.ok,
        decision=result.decision if result.ok else "block",
        onward=result.onward,
        findings=tuple(sorted({finding_name(item) for item in result.findings})),
        exit_code=0,
    )


def _check(event: HookEvent, guard: Guard, prompt: str | None) -> Result:
    if event.stage == "input":
        return guard.check_input(event.text)
    if event.stage == "tool_result":
        return guard.check_tool_result(event.text, name=cast(str, event.tool), prompt=prompt)
    request = cast(str, prompt)
    if event.stage == "tool_call":
        return guard.check_tool_call(cast(str, event.tool), event.text, prompt=request)
    return guard.check_output(event.text, prompt=request)


def load_object(raw: str) -> dict[str, object]:
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError:
        raise ConfigError("invalid JSON") from None
    if not isinstance(value, dict):
        raise ConfigError("invalid JSON")
    return cast(dict[str, object], value)


def event_from_payload(payload: Mapping[str, object]) -> HookEvent:
    """A HookEvent from the JSON ``jes hook`` reads."""

    stage = payload.get("stage")
    if stage not in _STAGES:
        raise ConfigError("invalid stage")
    checked: Stage = stage
    prompt = optional_text(payload, "prompt")
    session_id = optional_text(payload, "session_id")
    if checked == "tool_call":
        raw = payload.get("arguments")
        if raw is None:
            raw = payload.get("text")
        if raw is None:
            raise ConfigError("invalid arguments")
        return HookEvent(
            checked, tool_arguments(raw), tool_name(payload, "tool"), prompt, session_id
        )
    text = optional_text(payload, "text")
    if text is None:
        raise ConfigError("invalid text")
    tool = tool_name(payload, "tool") if checked == "tool_result" else None
    return HookEvent(checked, text, tool, prompt, session_id)


def optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"invalid {key}")
    return value


def required_text(payload: Mapping[str, object], key: str) -> str:
    value = optional_text(payload, key)
    if value is None:
        raise ConfigError(f"invalid {key}")
    return value


def tool_name(payload: Mapping[str, object], key: str = "tool_name") -> str:
    name = payload.get(key)
    if not valid_tool_name(name):
        raise ConfigError("invalid tool")
    return cast(str, name)


def tool_arguments(value: object) -> str:
    """Tool arguments as one string: as given, or an object as canonical JSON."""

    if value is None:
        value = {}
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ConfigError("invalid arguments")
    try:
        return freeze_arguments(cast(dict[str, object], value))
    except PolicyError:
        raise ConfigError("invalid arguments") from None


def tool_text(response: object) -> str:
    """The text a tool result shows the model. A shell result is its stdout plus its stderr."""

    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        fields = cast(dict[str, object], response)
        stdout = fields.get("stdout")
        if isinstance(stdout, str):
            stderr = fields.get("stderr")
            streams = (stdout, stderr if isinstance(stderr, str) else "")
            return "\n".join(stream for stream in streams if stream)
        for key in _TEXT_FIELDS:
            value = fields.get(key)
            if isinstance(value, str):
                return value
    try:
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        raise ConfigError("invalid tool result") from None


__all__ = [
    "HookEvent",
    "HookResponse",
    "event_from_payload",
    "load_object",
    "optional_text",
    "refusal",
    "required_text",
    "run_hook",
    "tool_arguments",
    "tool_name",
    "tool_text",
]
