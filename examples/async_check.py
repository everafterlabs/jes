"""The same input check on AsyncGuard."""

import asyncio

from jes import AsyncGuard
from jes.policies import injection
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend
from jes.types import InputResult


async def check() -> InputResult:
    guard = AsyncGuard(
        [injection(threshold=0.72)],
        backend=FakeBackend(answers={"violation": YesNoAnswer(0.1, "probability")}),
    )
    result = await guard.check_input("Please summarize the notes.")
    # Send result.onward to the model.
    return result


def main() -> InputResult:
    return asyncio.run(check())
