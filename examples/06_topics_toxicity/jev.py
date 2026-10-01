"""Lesson 06 (Jev): Deny a topic and block toxic messages.

topics() blocks subjects you will not discuss; toxicity() blocks insults,
threats and the like. Prints, for each guard, one allowed and one blocked message.

Run:   uv run --group examples python -m examples.06_topics_toxicity.jev
Needs: TYPESAFE_API_KEY
"""

from examples._common import Check, require_env, show
from jes import Guard
from jes.policies import topics, toxicity
from jes.types import InputResult

# Hosted Jev. Pin a release such as "jev-1.13.0" in production.
MODEL = "jev-latest"
THRESHOLD = 0.72  # An application choice; lesson 01 explains it.

# Name a topic concretely: "medication dosage" scores more cleanly than a
# broad "medical advice", most of all on a small local model.
TOPIC = "medication dosage"
OPENING_HOURS = "What time does your support desk open on Monday?"
MEDICAL = "What dose of ibuprofen should I take for my back pain?"
THANKS = "Thanks, your work on this report was really helpful."
INSULT = "You are a worthless idiot and your work is garbage."


def main() -> tuple[InputResult, InputResult]:
    # Two guards only so each policy prints on its own; one Guard can hold both.
    topic_guard = Guard([topics([TOPIC], threshold=THRESHOLD)], model=MODEL)
    hours = topic_guard.check_input(OPENING_HOURS)
    medical = topic_guard.check_input(MEDICAL)

    toxic_guard = Guard([toxicity(threshold=THRESHOLD)], model=MODEL)
    thanks = toxic_guard.check_input(THANKS)
    insult = toxic_guard.check_input(INSULT)

    show("topics", [Check("benign", hours), Check("medical", medical)])
    show("toxicity", [Check("benign", thanks), Check("insult", insult)])
    return medical, insult


if __name__ == "__main__":
    require_env("TYPESAFE_API_KEY")
    main()
