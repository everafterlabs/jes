"""Lesson 08 (local): Ready-made recipes.

jes.recipes are prebuilt policies from the catalog (see docs/recipes.md):
sentiment, competitors, malicious URLs and factual consistency. Prints each
decision and score; competitors() is local and rewrites the text with no model call.

Run:   uv run --group examples python -m examples.08_recipes.local
Needs: Ollama with tev1 (ollama pull tev1)
"""

from langchain_typesafe import TypeSafeClassifier

from examples._common import Check, show
from jes import Guard
from jes.recipes import competitors, factual_consistency, malicious_urls, sentiment
from jes.types import InputResult, ScanResult

# tev1 on Ollama, so nothing leaves your machine. Ollama ignores the key;
# passing one keeps your real TYPESAFE_API_KEY from being sent to localhost.
MODEL = TypeSafeClassifier(
    model="tev1", base_url="http://localhost:11434", api_key="ollama", timeout=120
)
THRESHOLD = 0.5  # tev1 scores sit in a narrower band; lesson 01 explains it.


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
    main()
