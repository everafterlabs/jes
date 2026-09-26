"""Topics always take a threshold. Toxicity scores here come from FakeBackend."""

from jes import Guard
from jes.policies import topics, toxicity
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend
from jes.types import InputResult

_ZERO = YesNoAnswer(0.0, "probability")


def main() -> tuple[InputResult, InputResult]:
    topic_result = Guard(
        [topics(["medical advice"], threshold=0.70)],
        backend=FakeBackend(answers={"topic_0": YesNoAnswer(0.8, "probability")}),
    ).check_input("What dose should I take?")
    toxic = Guard(
        [toxicity(threshold=0.70)],
        backend=FakeBackend(
            default_answer=_ZERO,
            answers={"insult": YesNoAnswer(0.9, "probability")},
        ),
    ).check_input("ordinary sentence")
    return topic_result, toxic
