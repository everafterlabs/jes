"""Lesson 07: Ask your own question with judge().

judge() turns a question into a policy: a yes/no, a pick-one choice, or a
graded score. Prints each of the three messages as blocked, with its score.

Run:   uv run --group examples python -m examples.07_custom_questions
Mock:  add --mock to run offline, no keys needed.
Needs: TYPESAFE_API_KEY
"""

from examples._backend import Check, decision_model, parse_mock, show
from jes import Guard
from jes.policies import judge
from jes.questions import Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo
from jes.types import InputResult

THRESHOLD = 0.72  # An application choice; lesson 01 explains it.

# Offline answers for --mock, one per question type. "probability" says the
# numbers are probabilities (each answer's values sum to 1).
MOCK_SCORES = {
    "refund.violation": 0.9,
    "route.violation": ChoiceAnswer({"billing": 0.8, "other": 0.2}, "probability"),
    "severity.violation": ScoreAnswer((0.1, 0.1, 0.8), "probability"),
}


def main(*, mock: bool = False) -> tuple[InputResult, InputResult, InputResult]:
    # judge(name, question) makes a policy. A yes/no blocks when P(yes) >= THRESHOLD.
    refund = judge("refund", YesNo("The text asks for money back."), threshold=THRESHOLD)
    # violating= lists the choices that count as a violation; their summed
    # probability is compared with THRESHOLD.
    route = judge(
        "route",
        Choice(
            "Which team should handle this?", {"billing": "Money problems", "other": "Anything"}
        ),
        threshold=THRESHOLD,
        violating=["billing"],
    )
    # violation_level=2 means level index 2 ("high") and above is a violation;
    # the summed probability of those levels is compared with THRESHOLD.
    severity = judge(
        "severity",
        Score("How severe is this?", ("low", "mid", "high")),
        threshold=THRESHOLD,
        violation_level=2,
    )

    model = decision_model(mock, MOCK_SCORES)
    asks_refund = Guard([refund], model=model).check_input("Please refund the order.")
    billing = Guard([route], model=model).check_input("I was charged twice.")
    outage = Guard([severity], model=model).check_input("The service is down for everyone.")
    return asks_refund, billing, outage


if __name__ == "__main__":
    for name, result in zip(("refund", "route", "severity"), main(mock=parse_mock()), strict=True):
        show(name, [Check("input", result)])
