"""Offline stand-ins for ``--mock`` runs and the test suite.

Nothing here is needed to use jes. The lessons stay readable because every
scripted model and fixed score lives in this one file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from jes.judge import render_state
from jes.questions import Answer, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import State

_ZERO = YesNoAnswer(0.0, "probability")


def mock_jev(scores: Mapping[str, float | Answer], when: str | None = None) -> Any:
    """A decision model that returns fixed scores instead of judging.

    Keys are question ids such as ``"injection.violation"``,
    ``"indirect_injection.violation"`` or ``"toxicity.insult"``. A key may be
    scoped to one stage, for example ``"output:any"``. Every other question
    scores 0. A float is a yes/no probability; pass an ``Answer`` such as
    ``ChoiceAnswer`` for other question types.

    With ``when``, the scores apply only to text containing it; other text
    scores 0. That lets one guard allow a benign text and block an attack.
    """

    hit = FakeBackend(
        answers={
            key: YesNoAnswer(value, "probability") if isinstance(value, int | float) else value
            for key, value in scores.items()
        },
        default_answer=_ZERO,
        max_units=100_000,
    )
    if when is None:
        return hit
    return _When(when, hit, FakeBackend(default_answer=_ZERO, max_units=100_000))


class _When:
    """Route each check to ``hit`` if its text contains ``marker``, else ``miss``."""

    def __init__(self, marker: str, hit: FakeBackend, miss: FakeBackend) -> None:
        self._marker, self._hit, self._miss = json.dumps(marker)[1:-1], hit, miss

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hit, name)

    def _pick(self, state: State) -> FakeBackend:
        # render_state is JSON, so compare against the JSON-escaped marker.
        return self._hit if self._marker in render_state(state) else self._miss

    def decide(self, state: State, questions: Any, request: Any) -> Any:
        return self._pick(state).decide(state, questions, request)

    async def adecide(self, state: State, questions: Any, request: Any) -> Any:
        return await self._pick(state).adecide(state, questions, request)


# --- LangChain -------------------------------------------------------------
# Imported lazily so the basics lessons run without LangChain installed.


def tool_call(name: str, args: Mapping[str, Any], call_id: str = "call_1") -> Any:
    """An assistant message that asks for one tool call."""

    from langchain.messages import AIMessage

    return AIMessage(content="", tool_calls=[{"name": name, "args": dict(args), "id": call_id}])


def scripted_chat(replies: Sequence[Any]) -> Any:
    """A chat model that returns ``replies`` in order, ignoring its input.

    Each reply is a string (a final answer) or a message from ``tool_call``.
    """

    from langchain.messages import AIMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ScriptedChatModel(BaseChatModel):
        script: list[Any]
        cursor: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
            del tools, kwargs
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            del messages, stop, run_manager, kwargs
            reply = self.script[min(self.cursor, len(self.script) - 1)]
            self.cursor += 1
            message = AIMessage(content=reply) if isinstance(reply, str) else reply
            return ChatResult(generations=[ChatGeneration(message=message)])

    return ScriptedChatModel(script=list(replies))


# --- OpenAI SDK, Agents SDK, Ollama ---------------------------------------
# Lessons 11, 12 and 16. Each scripted model returns its replies in order and
# ignores its input. A reply is a string (a final answer) or a tool call.


def mock_search(query: str) -> str:
    """Search results without the network: one ordinary page about ``query``."""

    return f"Example page (https://example.com)\nAn ordinary page about {query}."


def openai_call(name: str, args: Mapping[str, Any], call_id: str = "call_1") -> Any:
    """A Responses API output item that asks for one function call."""

    from openai.types.responses import ResponseFunctionToolCall

    return ResponseFunctionToolCall(
        type="function_call", call_id=call_id, name=name, arguments=json.dumps(dict(args))
    )


def _openai_output(reply: Any) -> list[Any]:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    if not isinstance(reply, str):
        return [reply]
    text = ResponseOutputText(type="output_text", text=reply, annotations=[])
    return [
        ResponseOutputMessage(
            id="msg_1", type="message", role="assistant", status="completed", content=[text]
        )
    ]


def scripted_openai(replies: Sequence[Any]) -> Any:
    """An ``OpenAI`` client whose ``responses.create`` returns ``replies``.

    Each reply is a string or an item from ``openai_call``.
    """

    from openai.types.responses import Response

    script = iter(replies)

    def create(**kwargs: Any) -> Response:
        del kwargs
        return Response.model_construct(output=_openai_output(next(script)))

    return SimpleNamespace(responses=SimpleNamespace(create=create))


def scripted_agent_model(replies: Sequence[Any]) -> Any:
    """An Agents SDK ``Model`` that returns ``replies``.

    Each reply is a string or an item from ``openai_call``.
    """

    from agents import Model, ModelResponse, Usage

    class ScriptedModel(Model):
        def __init__(self) -> None:
            self.script = iter(replies)

        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            del args, kwargs
            output = _openai_output(next(self.script))
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    return ScriptedModel()


def ollama_call(name: str, args: Mapping[str, Any]) -> Any:
    """An ollama assistant message that asks for one tool call."""

    from ollama import Message

    function = Message.ToolCall.Function(name=name, arguments=dict(args))
    return Message(role="assistant", content="", tool_calls=[Message.ToolCall(function=function)])


def scripted_ollama(replies: Sequence[Any]) -> Any:
    """An ``ollama.Client`` whose ``chat`` returns ``replies``.

    Each reply is a string or a message from ``ollama_call``.
    """

    from ollama import ChatResponse, Message

    script = iter(replies)

    def chat(**kwargs: Any) -> ChatResponse:
        del kwargs
        reply = next(script)
        message = Message(role="assistant", content=reply) if isinstance(reply, str) else reply
        return ChatResponse(message=message)

    return SimpleNamespace(chat=chat)
