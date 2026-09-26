"""Custom questions through judge(). Thresholds are application choices."""

from jes import Guard
from jes.policies import judge
from jes.questions import Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import InputResult


def main() -> tuple[InputResult, InputResult, InputResult]:
    refund = Guard(
        [
            judge(
                "refund",
                YesNo("The text asks for money back."),
                threshold=0.80,
                stages=("input",),
            )
        ],
        backend=FakeBackend(answers={"violation": YesNoAnswer(0.9, "probability")}),
    ).check_input("Please refund the order.")
    route = Guard(
        [
            judge(
                "route",
                Choice(
                    "Which team should handle this?",
                    {"billing": "Money problems", "other": "Anything else"},
                ),
                threshold=0.50,
                violating=["billing"],
                stages=("input",),
            )
        ],
        backend=FakeBackend(
            answers={"violation": ChoiceAnswer({"billing": 0.7, "other": 0.3}, "probability")},
        ),
    ).check_input("I was charged twice.")
    severity = Guard(
        [
            judge(
                "severity",
                Score("How severe is this?", ("low", "mid", "high")),
                threshold=0.50,
                violation_level=2,
                stages=("input",),
            )
        ],
        backend=FakeBackend(
            answers={"violation": ScoreAnswer((0.1, 0.1, 0.8), "probability")},
        ),
    ).check_input("The service is down.")
    return refund, route, severity
