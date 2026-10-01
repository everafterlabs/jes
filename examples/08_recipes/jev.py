"""Lesson 08 (Jev): Ready-made recipes.

jes.recipes are prebuilt policies from the catalog (see docs/recipes.md):
sentiment, competitors, malicious URLs and factual consistency. Prints each
decision and score; competitors() is local and rewrites the text with no model call.

Run:   uv run --group examples python -m examples.08_recipes.jev
Needs: TYPESAFE_API_KEY
"""

from examples._common import Check, require_env, show
from jes import Guard
from jes.recipes import competitors, factual_consistency, malicious_urls, sentiment
from jes.types import InputResult, ScanResult

# Hosted Jev. Pin a release such as "jev-1.13.0" in production.
MODEL = "jev-latest"
THRESHOLD = 0.72  # An application choice; lesson 01 explains it.


def main() -> tuple[InputResult, InputResult, InputResult, ScanResult]:
    hostile = Guard([sentiment(threshold=THRESHOLD)], model=MODEL).check_input(
        "This is unacceptable."
    )

    # A local policy needs no model. It is allowed, and onward says "[REDACTED]".
    redacted = Guard([competitors(["Acme"])]).check_input("Call Acme today.")

    # Each URL in the text is judged on its own.
    linked = Guard([malicious_urls(threshold=THRESHOLD)], model=MODEL).check_input(
        "See https://one.example and https://two.example"
    )

    # Is the answer backed by the sources it was given?
    inconsistent = Guard([factual_consistency(threshold=THRESHOLD)], model=MODEL).check_output(
        "The total is nine.",
        prompt="What is the total?",
        sources=["The total is four."],
    )

    show("sentiment", [Check("input", hostile)])
    show("competitors", [Check("input", redacted)])
    print(f"    onward: {redacted.onward}")  # Allowed, so show() skips it; this is the rewrite.
    show("malicious_urls", [Check("input", linked)])
    show("factual_consistency", [Check("output", inconsistent)])
    return hostile, redacted, linked, inconsistent


if __name__ == "__main__":
    require_env("TYPESAFE_API_KEY")
    main()
