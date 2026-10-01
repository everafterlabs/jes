"""Lesson 07 (Jev): Ask your own question with judge().

judge() turns a question into a policy: a yes/no, a pick-one choice, or a
graded score.

Run:   uv run --group examples python -m examples.07_custom_questions.jev
Needs: TYPESAFE_API_KEY
"""

from examples._common import print_check, require_env
from jes import Guard
from jes.policies import judge
from jes.questions import Choice, Score, YesNo

MODEL = "jev-latest"
THRESHOLD = 0.72


def main() -> None:
    # Blocks when P(yes) >= THRESHOLD.
    refund = judge("refund", YesNo("The text asks for money back."), threshold=THRESHOLD)
    # Blocks when the summed probability of the violating choices >= THRESHOLD.
    route = judge(
        "route",
        Choice(
            "Which team should handle this?", {"billing": "Money problems", "other": "Anything"}
        ),
        threshold=THRESHOLD,
        violating=["billing"],
    )
    # Blocks when the summed probability of level 2 ("high") and above >= THRESHOLD.
    severity = judge(
        "severity",
        Score("How severe is this?", ("low", "mid", "high")),
        threshold=THRESHOLD,
        violation_level=2,
    )

    print("== custom questions")
    print_check("refund", Guard([refund], model=MODEL).check_input("Please refund the order."))
    print_check("route", Guard([route], model=MODEL).check_input("I was charged twice."))
    print_check(
        "severity", Guard([severity], model=MODEL).check_input("The service is down for everyone.")
    )


if __name__ == "__main__":
    require_env("TYPESAFE_API_KEY")
    main()
