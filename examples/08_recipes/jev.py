"""Lesson 08 (Jev): Ready-made recipes.

jes.recipes are prebuilt policies from the catalog (see docs/recipes.md).
competitors() is local: it rewrites the text without a model call.

Run:   uv run --group examples python -m examples.08_recipes.jev
Needs: TYPESAFE_API_KEY
"""

from examples._common import print_check, require_env
from jes import Guard
from jes.recipes import competitors, factual_consistency, malicious_urls, sentiment

MODEL = "jev-latest"
THRESHOLD = 0.72


def main() -> None:
    sentiment_guard = Guard([sentiment(threshold=THRESHOLD)], model=MODEL)
    competitors_guard = Guard([competitors(["Acme"])])
    # Judges each URL in the text separately.
    urls_guard = Guard([malicious_urls(threshold=THRESHOLD)], model=MODEL)
    # Is the answer backed by its sources?
    facts_guard = Guard([factual_consistency(threshold=THRESHOLD)], model=MODEL)

    print("== recipes")
    print_check("sentiment", sentiment_guard.check_input("This is unacceptable."))
    print_check("competitors", competitors_guard.check_input("Call Acme today."))
    print_check(
        "malicious_urls",
        urls_guard.check_input("See https://one.example and https://two.example"),
    )
    print_check(
        "factual_consistency",
        facts_guard.check_output(
            "The total is nine.", prompt="What is the total?", sources=["The total is four."]
        ),
    )


if __name__ == "__main__":
    require_env("TYPESAFE_API_KEY")
    main()
