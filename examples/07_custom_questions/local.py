"""Lesson 07 (local): Ask your own question with judge().

judge() turns a question into a policy: a yes/no, a pick-one choice, or a
graded score. Prints each of the three messages as blocked, with its score.

Run:   uv run --group examples python -m examples.07_custom_questions.local
Needs: Ollama with tev1 (ollama pull tev1)
"""

from langchain_typesafe import TypeSafeClassifier

from examples._common import Check, show
from jes import Guard
from jes.policies import judge
from jes.questions import Choice, Score, YesNo
from jes.types import InputResult

# tev1 on Ollama, so nothing leaves your machine. Ollama ignores the key;
# passing one keeps your real TYPESAFE_API_KEY from being sent to localhost.
MODEL = TypeSafeClassifier(
    model="tev1", base_url="http://localhost:11434", api_key="ollama", timeout=120
)
THRESHOLD = 0.5  # tev1 scores sit in a narrower band; lesson 01 explains it.

REFUND = "Please refund the order."
CHARGED_TWICE = "I was charged twice."
OUTAGE = "The service is down for everyone."


def main() -> tuple[InputResult, InputResult, InputResult]:
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

    asks_refund = Guard([refund], model=MODEL).check_input(REFUND)
    billing = Guard([route], model=MODEL).check_input(CHARGED_TWICE)
    outage = Guard([severity], model=MODEL).check_input(OUTAGE)

    show("refund", [Check("input", asks_refund)])
    show("route", [Check("input", billing)])
    show("severity", [Check("input", outage)])
    return asks_refund, billing, outage


if __name__ == "__main__":
    main()
