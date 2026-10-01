"""Lesson 06 (Jev): Deny a topic and block toxic messages.

topics() blocks subjects you will not discuss; toxicity() blocks insults,
threats and the like.

Run:   uv run --group examples python -m examples.06_topics_toxicity.jev
Needs: TYPESAFE_API_KEY
"""

from examples._common import print_check, require_env
from jes import Guard
from jes.policies import topics, toxicity

MODEL = "jev-latest"
THRESHOLD = 0.72

# A narrow topic like "medication dosage" scores more cleanly than a broad
# "medical advice", most of all on a small local model.
TOPIC = "medication dosage"
OPENING_HOURS = "What time does your support desk open on Monday?"
MEDICAL = "What dose of ibuprofen should I take for my back pain?"
THANKS = "Thanks, your work on this report was really helpful."
INSULT = "You are a worthless idiot and your work is garbage."


def main() -> None:
    # Two guards so each policy prints on its own; one Guard can hold both.
    topic_guard = Guard([topics([TOPIC], threshold=THRESHOLD)], model=MODEL)
    toxicity_guard = Guard([toxicity(threshold=THRESHOLD)], model=MODEL)

    print("== topics")
    print_check("benign", topic_guard.check_input(OPENING_HOURS))
    print_check("medical", topic_guard.check_input(MEDICAL))
    print("== toxicity")
    print_check("benign", toxicity_guard.check_input(THANKS))
    print_check("insult", toxicity_guard.check_input(INSULT))


if __name__ == "__main__":
    require_env("TYPESAFE_API_KEY")
    main()
