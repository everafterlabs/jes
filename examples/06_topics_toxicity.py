"""Lesson 06: Deny a topic and block toxic messages.

topics() blocks subjects you will not discuss; toxicity() blocks insults,
threats and the like. Prints, for each guard, one allowed and one blocked message.

Run:   uv run --group examples python -m examples.06_topics_toxicity
Mock:  add --mock to run offline, no keys needed.
Needs: TYPESAFE_API_KEY
"""

from examples._backend import Check, decision_model, parse_mock, show
from jes import Guard
from jes.policies import topics, toxicity
from jes.types import InputResult

THRESHOLD = 0.72  # An application choice; lesson 01 explains it.

OPENING_HOURS = "What time does your support desk open on Monday?"
MEDICAL = "What dose of ibuprofen should I take for my back pain?"
THANKS = "Thanks, your work on this report was really helpful."
INSULT = "You are a worthless idiot and your work is garbage."


def main(*, mock: bool = False) -> tuple[InputResult, InputResult]:
    # Two guards only so each policy prints on its own; one Guard can hold both.
    # Mock keys are question ids, "policy.question"; topics numbers each denied topic.
    topic_guard = Guard(
        [topics(["medical advice"], threshold=THRESHOLD)],
        model=decision_model(mock, {"topics.topic_0": 0.8}, when=MEDICAL),
    )
    hours = topic_guard.check_input(OPENING_HOURS)
    medical = topic_guard.check_input(MEDICAL)

    toxic_guard = Guard(
        [toxicity(threshold=THRESHOLD)],
        model=decision_model(mock, {"toxicity.insult": 0.9}, when=INSULT),
    )
    thanks = toxic_guard.check_input(THANKS)
    insult = toxic_guard.check_input(INSULT)

    show("topics", [Check("benign", hours), Check("medical", medical)])
    show("toxicity", [Check("benign", thanks), Check("insult", insult)])
    return medical, insult


if __name__ == "__main__":
    main(mock=parse_mock())
