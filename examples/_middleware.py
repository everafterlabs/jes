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
from jes.types import InputResult, ScanResult


class JesMiddleware(AgentMiddleware):
    """Guard one agent and log every check and tool call it makes."""

    def __init__(
        self,
        guard: Guard,
        *,
        label: str = "",
        subagents: Container[str] | None = None,
        checks: list[Check] | None = None,
        tools_called: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.guard = guard
        self.label = label
        # Deep Agents only: the ``task`` subagents allowed to run.
        self.subagents = subagents
        # Share these lists between middlewares to log a whole agent tree in order.
        self.checks: list[Check] = [] if checks is None else checks
        self.tools_called: list[str] = [] if tools_called is None else tools_called
        # Later checks pass this turn's checked input as prompt=, so they judge
        # against what the user asked, with the same redactions.
        self.incoming: InputResult | None = None

    @property
    def name(self) -> str:
        return f"JesMiddleware[{self.label or 'agent'}]"

    def _record(self, stage: str, result: ScanResult) -> None:
        self.checks.append(Check(f"{self.label} {stage}".strip(), result))

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        del runtime  # Unused, but LangChain passes it by name.
        last = state["messages"][-1]
        # Later model calls in the same turn follow a tool result, not the user.
        if not isinstance(last, HumanMessage):
            return None
        incoming = self.guard.check_input(last.text)
        self.incoming = incoming
        self._record("input", incoming)
        if not incoming.ok:
            return {"messages": [AIMessage(incoming.onward)], "jump_to": "end"}
        if incoming.onward != last.text:
            # Same id, so this replaces the raw message in state.
            return {"messages": [HumanMessage(incoming.onward, id=last.id)]}
        return None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        response = handler(request)
        message = response.result[-1]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return response
        outgoing = self.guard.check_output(message.text, prompt=self._prompt(request.messages))
        self._record("output", outgoing)
        if outgoing.ok and outgoing.onward == message.text:
            return response
        return ModelResponse(result=[AIMessage(content=outgoing.onward, id=message.id)])

    def wrap_tool_call(
        self, request: Any, handler: Callable[[Any], ToolMessage | Command[Any]]
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = str(call["name"])
        prompt = self._prompt(request.state["messages"])

        def refuse(text: str) -> ToolMessage:
            return ToolMessage(content=text, tool_call_id=call["id"], name=name)

        checked = self.guard.check_tool_call(name, call["args"], prompt=prompt)
        self._record(f"tool_call {name}", checked)
        if not checked.ok:
            return refuse(checked.onward)
        subagent = call["args"].get("subagent_type")
        if name == "task" and self.subagents is not None and subagent not in self.subagents:
            return refuse("Refused: that subagent is not guarded.")

        self.tools_called.append(name)
        outcome = handler(request)
        # Deep Agents' task tool returns a Command; its last message is the report.
        message = outcome.update["messages"][-1] if isinstance(outcome, Command) else outcome
        if not isinstance(message, ToolMessage):
            return refuse("Tool result refused: unexpected type.")
        result = self.guard.check_tool_result(message.text, name=name, prompt=prompt)
        self._record(f"tool_result {name}", result)
        if not result.ok or result.onward != message.text:
            # Blocked or redacted: the model gets onward instead of the raw result.
            return refuse(result.onward)
        # A Command could carry more state than the report; pass on only what was checked.
        return Command(update={"messages": [message]}) if isinstance(outcome, Command) else outcome

    def _prompt(self, messages: list[Any]) -> InputResult | str:
        """This turn's checked input, else the latest user message."""

        if self.incoming is not None:
            return self.incoming
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.text
        return ""
