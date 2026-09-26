from __future__ import annotations

import pytest

from jes import Guard
from jes.errors import DeadlineExceeded, PolicyError
from jes.policies import CallContext
from jes.questions import ChoiceAnswer, YesNoAnswer
from jes.recipes import (
    bias,
    code,
    competitors,
    emotions,
    factual_consistency,
    gibberish,
    json_check,
    language,
    language_same,
    malicious_urls,
    reading_time,
    refusal,
    refusal_phrases,
    relevance,
    sentiment,
)
from jes.recipes.prompts import SENTIMENT_V1
from jes.testing import FakeBackend


def _yes() -> FakeBackend:
    backend = FakeBackend(
        max_units=100_000,
        answers={"violation": YesNoAnswer(0.9, "probability")},
    )
    return backend


def test_snapshot_and_required_threshold() -> None:
    policy = sentiment(threshold=0.5)
    assert policy.questions(None)["violation"].instructions == SENTIMENT_V1
    assert policy.version == "sentiment.v1"
    with pytest.raises(TypeError):
        sentiment()  # type: ignore[call-arg]
    with pytest.raises(PolicyError):
        gibberish(threshold=0.5, version="v2")
    with pytest.raises(PolicyError):
        emotions((), threshold=0.5)
    with pytest.raises(PolicyError):
        language((), threshold=0.5)
    with pytest.raises(PolicyError):
        code("ban", (), threshold=0.5)
    with pytest.raises(PolicyError):
        competitors(())


def test_yesno_recipes_block_on_a_high_score() -> None:
    backend = _yes()
    assert Guard([sentiment(threshold=0.5)], backend=backend).check_input("x").decision == "block"
    assert Guard([gibberish(threshold=0.5)], backend=backend).check_input("x").decision == "block"
    biased = Guard([bias(threshold=0.5)], backend=backend).check_output("x", prompt="p")
    assert biased.decision == "block"
    refused = Guard([refusal(threshold=0.5)], backend=backend).check_output("x", prompt="p")
    assert refused.decision == "block"
    relevant = relevance(threshold=0.5)
    assert relevant.whole_text is True
    assert relevant.context == "required"
    assert Guard([relevant], backend=backend).check_output("x", prompt="p").decision == "block"
    consistent = factual_consistency(threshold=0.5)
    assert consistent.sources is True
    assert consistent.whole_text is True
    checked = Guard([consistent], backend=backend).check_output(
        "x",
        prompt="p",
        sources=("a source",),
    )
    assert checked.decision == "block"
    same = language_same(threshold=0.5)
    assert same.context == "required"
    assert Guard([same], backend=backend).check_output("x", prompt="p").decision == "block"


def test_emotions_language_and_code() -> None:
    emotions_backend = FakeBackend(
        max_units=100_000,
        answers={"anger": YesNoAnswer(0.8, "probability")},
    )
    result = Guard([emotions(("anger",), threshold=0.5)], backend=emotions_backend).check_input("x")
    assert result.decision == "block"

    language_backend = FakeBackend(
        max_units=100_000,
        answers={"violation": ChoiceAnswer({"en": 0.2, "other": 0.8}, "probability")},
    )
    detected = Guard([language(("en",), threshold=0.5)], backend=language_backend).check_input("x")
    assert detected.decision == "block"

    code_backend = FakeBackend(
        max_units=100_000,
        answers={
            "violation": ChoiceAnswer(
                {
                    "python": 0.0,
                    "javascript": 0.0,
                    "sql": 1.0,
                    "shell": 0.0,
                    "html": 0.0,
                    "other": 0.0,
                    "not_code": 0.0,
                },
                "probability",
            )
        },
    )
    banned = code("ban", ("sql",), threshold=0.5)
    assert Guard([banned], backend=code_backend).check_input("x").decision == "block"


def test_malicious_urls_are_separate_items() -> None:
    backend = _yes()
    text = "see https://example.com/a and https://example.com/a"
    result = Guard([malicious_urls(threshold=0.5)], backend=backend).check_input(text)
    assert result.decision == "block"
    assert len(backend.calls) == 2
    assert all(call[0].text == "https://example.com/a" for call in backend.calls)

    crowded = " ".join(f"https://example.com/{index}" for index in range(21))
    quiet = FakeBackend(max_units=100_000, answers={"violation": YesNoAnswer(0.1, "probability")})
    overflow = Guard([malicious_urls(threshold=0.5)], backend=quiet).check_input(crowded)
    assert overflow.decision == "block"
    assert any(item.label == "too_many_urls" for item in overflow.findings)
    assert quiet.calls == []


def test_phrase_and_reading_time_and_json() -> None:
    redacted = Guard([competitors(("Acme",))]).check_input("See Acme today")
    assert "Acme" not in redacted.sanitized
    assert redacted.decision == "allow"

    refused = Guard([refusal_phrases()]).check_input("I cannot help with that request")
    assert refused.decision == "block"

    short = Guard([reading_time(1)]).check_input("one two")
    assert short.decision == "allow"
    long = " ".join(["word"] * 201)
    blocked = Guard([reading_time(1)]).check_input(long)
    assert blocked.decision == "block"
    trimmed = Guard([reading_time(1, mode="truncate")]).check_input(long)
    assert trimmed.decision == "allow"
    assert len(trimmed.sanitized.split()) == 200

    valid = Guard([json_check()]).check_output('prefix {"a": 1}', prompt="p")
    assert valid.decision == "allow"
    missing = Guard([json_check(required_elements=2)]).check_output('{"a": 1}', prompt="p")
    assert missing.decision == "block"
    invalid = Guard([json_check()]).check_output("no object", prompt="p")
    assert invalid.decision == "block"
    with pytest.raises(PolicyError):
        json_check(repair=True)
    with pytest.raises(PolicyError):
        reading_time(0)
    with pytest.raises(PolicyError):
        reading_time(1, mode="shrink")
    with pytest.raises(PolicyError):
        language(("other",), threshold=0.5)
    with pytest.raises(PolicyError):
        language(("en", "en"), threshold=0.5)
    with pytest.raises(PolicyError):
        code("inspect", ("python",), threshold=0.5)
    array = Guard([json_check(required_elements=2)]).check_output("[1, 2]", prompt="p")
    assert array.decision == "allow"
    quoted = Guard([json_check()]).check_output('{"a": "{"}', prompt="p")
    assert quoted.decision == "allow"
    broken = Guard([json_check()]).check_output("{]", prompt="p")
    assert broken.decision == "block"
    with pytest.raises(DeadlineExceeded):
        reading_time(1).apply("one", CallContext("input", "input", "subject", None, 0.0))


def test_json_repair_rewrites_a_broken_object(monkeypatch: pytest.MonkeyPatch) -> None:
    import jes.recipes.transforms as transforms

    class FakeRepair:
        @staticmethod
        def loads(text: str) -> dict[str, int]:
            del text
            return {"a": 1}

    monkeypatch.setattr(transforms.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(transforms.importlib, "import_module", lambda _name: FakeRepair())
    repaired = Guard([json_check(repair=True)]).check_output("prefix {broken", prompt="p")
    assert repaired.sanitized == 'prefix {"a":1}'
    assert repaired.decision == "allow"

    class Boom:
        @staticmethod
        def loads(text: str) -> str:
            del text
            raise ValueError("nope")

    monkeypatch.setattr(transforms.importlib, "import_module", lambda _name: Boom())
    failed = Guard([json_check(repair=True)]).check_output("{broken", prompt="p")
    assert failed.decision == "block"
