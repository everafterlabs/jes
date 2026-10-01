"""The recipe catalog."""

from __future__ import annotations

import ast
import sys
import time

import pytest

from jes import recipes
from jes.errors import PolicyError
from jes.policies import TransformContext, TransformOutcome
from jes.questions import Choice, ChoiceAnswer
from jes.text.textmap import TextMap
from jes.types import Span

OUTPUT = TransformContext(stage="output", origin="output", target="text")


def _apply(outcome: TransformOutcome, text: str) -> str:
    return TextMap.identity(len(text)).apply(text, outcome.edits)[0]


def test_yes_no_recipes() -> None:
    for factory in (recipes.sentiment, recipes.gibberish):
        assert factory(threshold=0.5).stages == {"input", "output"}
    for factory in (recipes.bias, recipes.refusal):
        assert factory(threshold=0.5).stages == {"output"}
    for factory in (recipes.relevance, recipes.language_same, recipes.factual_consistency):
        judgment = factory(threshold=0.5)
        assert judgment.stages == {"output"}
        assert judgment.context == "required"
        assert judgment.whole_text
    assert recipes.factual_consistency(threshold=0.5).sources


def test_emotions() -> None:
    assert list(recipes.emotions(threshold=0.5).questions) == list(recipes.EMOTIONS)
    assert list(recipes.emotions(["joy"], threshold=0.5).questions) == ["joy"]
    with pytest.raises(PolicyError, match="at least one emotion"):
        recipes.emotions([], threshold=0.5)
    with pytest.raises(PolicyError, match="emotion"):
        recipes.emotions(["bad emotion"], threshold=0.5)


def test_language() -> None:
    judgment = recipes.language(["en", "fr"], threshold=0.5)
    question = judgment.questions["violation"]
    assert isinstance(question, Choice)
    assert list(question.options) == ["en", "fr", "other"]
    answer = ChoiceAnswer({"en": 0.1, "fr": 0.1, "other": 0.8})
    assert judgment.evaluate({"violation": answer}) == [("violation", 0.8, "block")]
    for bad in ([], ["en", "en"], ["other"]):
        with pytest.raises(PolicyError, match="unique codes"):
            recipes.language(bad, threshold=0.5)


def test_code() -> None:
    banned = recipes.code("ban", ["sql"], threshold=0.5)
    assert banned.violating == {"violation": {"sql"}}
    allowed = recipes.code("allow", ["python"], threshold=0.5)
    assert allowed.violating["violation"] == set(recipes.CODE_LANGUAGES) - {"python"}
    question = allowed.questions["violation"]
    assert isinstance(question, Choice) and "not_code" in question.options
    with pytest.raises(PolicyError, match="ban or allow"):
        recipes.code("deny", ["sql"], threshold=0.5)  # type: ignore[arg-type]
    with pytest.raises(PolicyError, match="unique languages"):
        recipes.code("ban", [], threshold=0.5)
    with pytest.raises(PolicyError, match="unknown code language"):
        recipes.code("ban", ["cobol"], threshold=0.5)
    with pytest.raises(PolicyError, match="leaves nothing"):
        recipes.code("allow", recipes.CODE_LANGUAGES, threshold=0.5)


def test_malicious_urls_judges_each_url() -> None:
    judgment = recipes.malicious_urls(threshold=0.5, max_urls=3)
    assert judgment.max_items == 3
    assert "tool_result" in judgment.stages
    assert judgment.items is not None
    items = list(judgment.items("see http://a.example/x and https://b.example, not ftp://c"))
    assert [item.text for item in items] == ["http://a.example/x", "https://b.example,"]
    assert items[0].span == Span(4, 22)
    # Browsers accept any case in the scheme, so an uppercase one must not slip past.
    assert [item.text for item in judgment.items("go to HTTPS://evil.example/a")] == [
        "HTTPS://evil.example/a"
    ]


def test_competitors_and_refusal_phrases() -> None:
    competitors = recipes.competitors(["Acme"])
    text = "Try ACME or \uff21cme instead"
    assert _apply(competitors.apply(text, OUTPUT), text) == "Try [REDACTED] or [REDACTED] instead"
    with pytest.raises(PolicyError, match="at least one name"):
        recipes.competitors([])
    refusal = recipes.refusal_phrases()
    assert refusal.stages == {"output"}
    assert refusal.apply("Sorry, I can't assist with that.", OUTPUT).findings


def test_reading_time() -> None:
    text = " ".join(["word"] * 500)
    blocked = recipes.reading_time(2).apply(text, OUTPUT)
    assert [(item.label, item.action) for item in blocked.findings] == [("reading_time", "block")]
    cut = recipes.reading_time(1, mode="truncate").apply(text, OUTPUT)
    assert _apply(cut, text) == " ".join(["word"] * 200)
    assert recipes.reading_time(5).apply(text, OUTPUT) == TransformOutcome()
    for bad in (0, True, 0.001):
        with pytest.raises(PolicyError, match=r"max_minutes|fewer than one word"):
            recipes.reading_time(bad)
    with pytest.raises(PolicyError, match="block or truncate"):
        recipes.reading_time(1, mode="cut")  # type: ignore[arg-type]


def test_json_check_finds_the_first_json_value() -> None:
    check = recipes.json_check(2)
    assert check.apply('Here: {"a": 1, "b": 2} done', OUTPUT) == TransformOutcome()
    assert check.apply('Here: {"a": 1, "b"} and [1, 2]', OUTPUT) == TransformOutcome()
    small = check.apply('Here: {"a": 1}', OUTPUT)
    assert small.findings[0].spans == (Span(6, 14),)
    missing = check.apply("no json at all", OUTPUT)
    assert [(item.label, item.action) for item in missing.findings] == [("json_check", "block")]
    assert recipes.json_check().apply('"just a string"', OUTPUT).findings


def test_json_check_survives_hostile_input() -> None:
    # 1.x re-scanned from every bracket, quadratic on unclosed brackets.
    started = time.perf_counter()
    assert recipes.json_check().apply("[" * 200_000, OUTPUT).findings
    # Nesting past the recursion limit is not JSON jes can read, and must not crash the check.
    assert recipes.json_check().apply("[" * 100_000 + "]" * 100_000, OUTPUT).findings
    assert time.perf_counter() - started < 2
    nested = recipes.json_check(2).apply("[" * 500 + "]" * 500, OUTPUT)
    assert nested.findings[0].spans == (Span(0, 1_000),)


class _Repair:
    """Stands in for json_repair: reads Python-style literals."""

    @staticmethod
    def loads(text: str) -> object:
        return ast.literal_eval(text)


def test_json_check_repairs() -> None:
    check = recipes.JsonCheck(name="json_check", required_elements=0, repair=_Repair)
    text = "Result: {'a': 1, 'b': [2, 3],}"
    outcome = check.apply(text, OUTPUT)
    assert _apply(outcome, text) == 'Result: {"a":1,"b":[2,3]}'
    assert check.apply('Result: {"a":1}', OUTPUT) == TransformOutcome()
    assert check.apply("nothing here", OUTPUT).findings
    assert check.apply("Result: [unclosed", OUTPUT).findings


def test_json_check_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PolicyError, match="zero or more"):
        recipes.json_check(-1)
    monkeypatch.setitem(sys.modules, "json_repair", _Repair)
    assert recipes.json_check(repair=True).repair is _Repair
    monkeypatch.setitem(sys.modules, "json_repair", None)
    with pytest.raises(PolicyError, match=r"jes\[json\]"):
        recipes.json_check(repair=True)


def test_json_repair_failures_block() -> None:
    class Broken:
        @staticmethod
        def loads(text: str) -> object:
            raise ValueError("cannot repair")

    class Scalar:
        @staticmethod
        def loads(text: str) -> object:
            return 5

    for module in (Broken, Scalar):
        check = recipes.JsonCheck(name="json_check", required_elements=0, repair=module)
        assert check.apply("value: [oops", OUTPUT).findings
