"""One judgment recipe and one transform recipe, plus URL items and sources."""

from jes import Guard
from jes.questions import YesNoAnswer
from jes.recipes import competitors, factual_consistency, malicious_urls, sentiment
from jes.testing import FakeBackend
from jes.types import InputResult, ScanResult


def main() -> tuple[InputResult, InputResult, InputResult, ScanResult]:
    hostile = Guard(
        [sentiment(threshold=0.70)],
        backend=FakeBackend(answers={"violation": YesNoAnswer(0.88, "probability")}),
    ).check_input("This is unacceptable.")
    redacted = Guard(
        [competitors(["Acme"])],
        backend=FakeBackend(),
    ).check_input("Call Acme today.")
    # Keep redacted.onward. hostile.onward names the sentiment block.
    urls = Guard(
        [malicious_urls(threshold=0.80)],
        backend=FakeBackend(answers={"violation": YesNoAnswer(0.1, "probability")}),
    )
    linked = urls.check_input("See https://one.example and https://two.example")
    prompt = Guard([], backend=FakeBackend()).check_input("What is the total?")
    inconsistent = Guard(
        [factual_consistency(threshold=0.60)],
        backend=FakeBackend(answers={"violation": YesNoAnswer(0.9, "probability")}),
    ).check_output(
        "The total is nine.",
        prompt=prompt,
        sources=["The total is four."],
    )
    # inconsistent.onward names the factual-consistency block.
    return hostile, redacted, linked, inconsistent
