"""Run the jes command the way an agent does: JSON on stdin, JSON and an exit code out."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from jes.agents.cli import main
from jes.backend import Reply, Request
from jes.errors import BackendError
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend

SAFE = YesNoAnswer(0.0)
UNSAFE = YesNoAnswer(0.99)


def allow() -> FakeBackend:
    return FakeBackend(default=SAFE)


def block(question_id: str) -> FakeBackend:
    """Unsafe on one question, such as ``injection.violation`` or ``hazards.S1``."""

    return FakeBackend({question_id: UNSAFE}, default=SAFE)


class Down(FakeBackend):
    """Counts its calls, then fails like a backend that is down."""

    def __init__(self) -> None:
        super().__init__(default=SAFE)
        self.decisions = 0

    def decide(self, request: Request) -> Reply:
        self.decisions += 1
        raise BackendError(self.name, "down")


def run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    payload: object,
    model: FakeBackend | None = None,
) -> tuple[int, object, str]:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(sys, "stdin", io.StringIO(raw))
    code = main(argv, model=model)
    captured = capsys.readouterr()
    body: object = json.loads(captured.out) if captured.out.strip() else None
    return code, body, captured.err


def argv(command: str, sessions: Path) -> list[str]:
    return [command, "--session-dir", str(sessions)]
