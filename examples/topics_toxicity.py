"""Topics always take a threshold. Toxicity scores here come from FakeBackend."""

from jes import Guard
from jes.policies import topics, toxicity
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend
from jes.types import InputResult

_LABELS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)


def main() -> tuple[InputResult, InputResult]:
    topic_backend = FakeBackend(answers={"topic_0": YesNoAnswer(0.8, "probability")})
    topic_result = Guard(
        [topics(["medical advice"], threshold=0.70)],
        backend=topic_backend,
    ).check_input("What dose should I take?")
    toxicity_backend = FakeBackend()
    for label in _LABELS:
        toxicity_backend.register_answer(label, YesNoAnswer(0.0, "probability"))
    toxicity_backend.register_answer("insult", YesNoAnswer(0.9, "probability"))
    toxic = Guard(
        [toxicity(threshold=0.70)],
        backend=toxicity_backend,
    ).check_input("ordinary sentence")
    return topic_result, toxic
