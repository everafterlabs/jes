"""One jes middleware for LangChain agents (lessons 13 and 15).

Three hooks cover the places untrusted text meets an agent:

- ``before_model``: the user's message, once per turn. A redacted message
  (PII, secrets) replaces the original, so the model never sees the raw text.
- ``wrap_tool_call``: the tool call before it runs, then the tool's result.
- ``wrap_model_call``: the final reply, before the user sees it.

A blocked step is replaced by ``result.onward``. Nothing blocked runs or
reaches the model.
"""

from __future__ import annotations

from collections.abc import Callable, Container
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, hook_config
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from examples._common import Check
from jes import Guard


class JesMiddleware(AgentMiddleware):
    """Guard one agent. ``checks`` collects every result; ``ran`` every tool that ran."""

    def __init__(
        self,
        guard: Guard,
        *,
        label: str = "",
        subagents: Container[str] | None = None,
        checks: list[Check] | None = None,
        ran: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.guard = guard
        self.label = f"{label} " if label else ""
        # Deep Agents only: the ``task`` subagents allowed to run.
        self.subagents = subagents
        # Pass the same lists to several middlewares to log one agent tree in order.
        self.checks: list[Check] = [] if checks is None else checks
        self.ran: list[str] = [] if ran is None else ran
        # This turn's checked input. Later checks pass it as prompt= so they
        # judge against what the user asked, with the same redactions.
        self.incoming: Any = None

    @property
    def name(self) -> str:
        return f"JesMiddleware[{self.label.strip() or 'agent'}]"

    def _record(self, stage: str, result: Any) -> Any:
        self.checks.append(Check(f"{self.label}{stage}", result))
        return result

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        del runtime
        last = state["messages"][-1]
        # Only a fresh user turn. Later model calls in the same turn are the tool loop.
        if not isinstance(last, HumanMessage):
            return None
        checked = self.incoming = self._record("input", self.guard.check_input(last.text))
        if not checked.ok:
            return {"messages": [AIMessage(checked.onward)], "jump_to": "end"}
        if checked.onward != last.text:
            # Same id, so this replaces the raw message in state.
            return {"messages": [HumanMessage(checked.onward, id=last.id)]}
        return None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        response = handler(request)
        message = response.result[-1]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return response
        checked = self._record(
            "output", self.guard.check_output(message.text, prompt=self._prompt(request.messages))
        )
        if checked.ok and checked.onward == message.text:
            return response
        return ModelResponse(result=[AIMessage(content=checked.onward, id=message.id)])

    def wrap_tool_call(
        self, request: Any, handler: Callable[[Any], ToolMessage | Command[Any]]
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = str(call["name"])
        prompt = self._prompt(request.state["messages"])

        def refuse(text: str) -> ToolMessage:
            return ToolMessage(content=text, tool_call_id=call["id"], name=name)

        checked = self._record(
            f"tool_call {name}", self.guard.check_tool_call(name, call["args"], prompt=prompt)
        )
        if not checked.ok:
            return refuse(checked.onward)
        subagent = call["args"].get("subagent_type")
        if name == "task" and self.subagents is not None and subagent not in self.subagents:
            return refuse("Refused: that subagent is not guarded.")

        self.ran.append(name)
        outcome = handler(request)
        # Deep Agents' task tool returns a Command; its last message is the report.
        message = outcome.update["messages"][-1] if isinstance(outcome, Command) else outcome
        if not isinstance(message, ToolMessage):
            return refuse("Tool result refused: unexpected type.")
        result = self._record(
            f"tool_result {name}",
            self.guard.check_tool_result(message.text, name=name, prompt=prompt),
        )
        if not result.ok or result.onward != message.text:
            # Blocked or redacted: the model sees onward, never the raw result.
            return refuse(result.onward)
        # A Command could carry more state than the report; pass on only what was checked.
        return Command(update={"messages": [message]}) if isinstance(outcome, Command) else outcome

    def _prompt(self, messages: list[Any]) -> Any:
        """This turn's checked input, else the latest user message."""

        if self.incoming is not None:
            return self.incoming
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.text
        return ""
