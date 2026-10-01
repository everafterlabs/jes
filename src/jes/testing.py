"""A deterministic backend for testing policies and guards without a network."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping

from jes.backend import Reply, Request, render_state
from jes.errors import BackendError
from jes.questions import Answer, Question
from jes.types import State

Rule = Callable[[Request, str], Answer | None]


class FakeBackend:
    """Answers each question from fixed answers, a rule, or a default, and records requests.

    Question ids arrive as ``"policy.question"``. An answer under the full id wins over one
    under the bare question id. ``rule(request, question_id)`` runs first and can answer
    from the text being judged. A question with no answer raises ``BackendError``.
    """

    name = "fake"

    def __init__(
        self,
        answers: Mapping[str, Answer] | None = None,
        *,
        default: Answer | None = None,
        rule: Rule | None = None,
        model: str = "fake",
        max_request_bytes: int | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.model = model
        self.requests: list[Request] = []
        self._answers = dict(answers or {})
        self._default = default
        self._rule = rule
        self._max_request_bytes = max_request_bytes
        self._delay_s = delay_s
        self._lock = threading.Lock()

    def answer(self, question_id: str, answer: Answer) -> None:
        """Set the answer for a full or bare question id."""

        self._answers[question_id] = answer

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        if self._max_request_bytes is None:
            return None
        used = len(render_state(state).encode("utf-8")) + sum(map(len, questions))
        return self._max_request_bytes - used

    def decide(self, request: Request) -> Reply:
        if self._delay_s:
            threading.Event().wait(self._delay_s)
        return self._reply(request)

    async def adecide(self, request: Request) -> Reply:
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._reply(request)

    def _reply(self, request: Request) -> Reply:
        with self._lock:
            self.requests.append(request)
        answers: dict[str, Answer] = {}
        for question_id in request.questions:
            answer = self._lookup(request, question_id)
            if answer is None:
                raise BackendError(self.name, "no_answer", question_ids=(question_id,))
            answers[question_id] = answer
        return Reply(answers, input_tokens=len(render_state(request.state)), output_tokens=1)

    def _lookup(self, request: Request, question_id: str) -> Answer | None:
        if self._rule is not None:
            ruled = self._rule(request, question_id)
            if ruled is not None:
                return ruled
        bare = question_id.rsplit(".", 1)[-1]
        return self._answers.get(question_id, self._answers.get(bare, self._default))


__all__ = ["FakeBackend", "Rule"]
