"""Lesson 08: Ready-made recipes.

jes.recipes are prebuilt policies from the catalog (see docs/recipes.md):
sentiment, competitors, malicious URLs and factual consistency. Prints each
decision and score; competitors() is local and rewrites the text with no model call.

Run:   uv run --group examples python -m examples.08_recipes
Mock:  add --mock to run offline, no keys needed.
Needs: TYPESAFE_API_KEY
"""

from examples._backend import Check, decision_model, parse_mock, show
from jes import Guard
from jes.recipes import competitors, factual_consistency, malicious_urls, sentiment
from jes.types import InputResult, ScanResult

THRESHOLD = 0.72  # An application choice; lesson 01 explains it.


def main(*, mock: bool = False) -> tuple[InputResult, InputResult, InputResult, ScanResult]:
    hostile = Guard(
        [sentiment(threshold=THRESHOLD)],
        model=decision_model(mock, {"sentiment.violation": 0.88}),
    ).check_input("This is unacceptable.")

    # A local policy needs no model. It is allowed, and onward says "[REDACTED]".
    redacted = Guard([competitors(["Acme"])]).check_input("Call Acme today.")

    # Each URL in the text is judged on its own. The mock scores every URL 0
    # (safe); live, Jev reads each URL, so its scores may differ.
    linked = Guard(
        [malicious_urls(threshold=THRESHOLD)],
        model=decision_model(mock),
    ).check_input("See https://one.example and https://two.example")

    # Is the answer backed by the sources it was given?
    inconsistent = Guard(
        [factual_consistency(threshold=THRESHOLD)],
        model=decision_model(mock, {"factual_consistency.violation": 0.9}),
    ).check_output(
        "The total is nine.",
        prompt="What is the total?",
        sources=["The total is four."],
    )
    return hostile, redacted, linked, inconsistent


if __name__ == "__main__":
    hostile, redacted, linked, inconsistent = main(mock=parse_mock())
    show("sentiment", [Check("input", hostile)])
    show("competitors", [Check("input", redacted)])
    print(f"    onward: {redacted.onward}")  # Allowed, so show() skips it; this is the rewrite.
    show("malicious_urls", [Check("input", linked)])
    show("factual_consistency", [Check("output", inconsistent)])
