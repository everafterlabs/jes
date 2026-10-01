"""The local transforms."""

from __future__ import annotations

import dataclasses
import sys
import time

import pytest

from jes.errors import DeadlineExceeded, PolicyError
from jes.policies import (
    EXPLOIT_TERMS,
    TransformContext,
    TransformOutcome,
    allowed_tools,
    invisible_text,
    regex,
    substrings,
    token_limit,
)
from jes.text.textmap import TextMap
from jes.types import Span, Stage

INPUT = TransformContext(stage="input", origin="input", target="text")


def _apply(outcome: TransformOutcome, text: str) -> str:
    return TextMap.identity(len(text)).apply(text, outcome.edits)[0]


def test_invisible_text_targeted_keeps_joiners_and_registered_variations() -> None:
    text = "ig\u200bnore\u202e me \u2764\ufe0f \U0001f468\u200d\U0001f469 a\ufe0f\u0007\t\n"
    outcome = invisible_text().apply(text, INPUT)
    cleaned = _apply(outcome, text)
    assert cleaned == "ignore me \u2764\ufe0f \U0001f468\u200d\U0001f469 a\t\n"
    (finding,) = outcome.findings
    assert (finding.label, finding.action) == ("invisible_text", "redact")


def test_invisible_text_all_mode_and_block() -> None:
    text = "a\u200db\ue000c\u2060d"
    assert _apply(invisible_text("all").apply(text, INPUT), text) == "abcd"
    blocked = invisible_text(block=True).apply(text, INPUT)
    assert blocked.edits == ()
    assert [(item.label, item.action) for item in blocked.findings] == [("invisible_text", "block")]
    assert invisible_text().apply("plain text", INPUT) == TransformOutcome()
    assert invisible_text("all").apply("plain", INPUT) == TransformOutcome()
    with pytest.raises(PolicyError, match="targeted or all"):
        invisible_text("some")  # type: ignore[arg-type]


def test_regex_blocks_redacts_requires_and_folds() -> None:
    text = "card 4111-1111 and \uff33\uff25\uff23\uff32\uff25\uff34"
    blocked = regex([r"\d{4}-\d{4}"]).apply(text, INPUT)
    assert blocked.findings[0].spans == (Span(5, 14),)
    redacted = regex([r"\d{4}-\d{4}"], action="redact").apply(text, INPUT)
    assert _apply(redacted, text) == "card [REDACTED] and \uff33\uff25\uff23\uff32\uff25\uff34"
    folded = regex(["secret"], fold=True).apply(text, INPUT)
    assert folded.findings[0].spans == (Span(19, 25),)
    assert regex(["secret"]).apply(text, INPUT) == TransformOutcome()
    assert (
        regex([r"card .*"], match="fullmatch", require=True).apply(text, INPUT)
        == TransformOutcome()
    )
    missing = regex([r"\d+"], match="fullmatch", require=True).apply(text, INPUT)
    assert [(item.label, item.action) for item in missing.findings] == [("regex", "block")]
    assert regex(["x*"]).apply("abc", INPUT) == TransformOutcome()


def test_regex_timeouts_block() -> None:
    class Slow:
        """A compiled pattern that runs out of time, as the regex module reports it."""

        def finditer(self, text: str, timeout: float) -> object:
            raise TimeoutError

        def fullmatch(self, text: str, timeout: float) -> object:
            raise TimeoutError

    for match in ("search", "fullmatch"):
        slow = dataclasses.replace(regex(["x"], match=match), patterns=(Slow(),))  # type: ignore[arg-type]
        outcome = slow.apply("text", INPUT)
        assert [(item.label, item.action) for item in outcome.findings] == [
            ("regex_timeout", "block")
        ]
    past = TransformContext(
        stage="input", origin="input", target="text", deadline=time.monotonic() - 1
    )
    with pytest.raises(DeadlineExceeded):
        regex(["x"]).apply("text", past)
    soon = TransformContext(
        stage="input", origin="input", target="text", deadline=time.monotonic() + 60
    )
    assert regex(["x"]).apply("x", soon).findings


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"patterns": []}, "at least one pattern"),
        ({"patterns": ["("]}, "invalid regex"),
        ({"action": "warn"}, "block or redact"),
        ({"match": "prefix"}, "search or fullmatch"),
        ({"timeout_ms": 0}, "timeout_ms"),
        ({"timeout_ms": True}, "timeout_ms"),
        ({"name": "bad name"}, "policy name"),
    ],
)
def test_regex_settings(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {"patterns": ["x"], **kwargs}
    with pytest.raises(PolicyError, match=message):
        regex(**arguments)  # type: ignore[arg-type]


def test_substrings_fold_lookalikes_and_respect_whole_words() -> None:
    text = "Use \u0440aypal (Cyrillic \u0440) or PAY PAL, not paypals"
    policy = substrings(["paypal"])
    spans = policy.apply(text, INPUT).findings[0].spans
    assert [text[span.start : span.end] for span in spans] == ["\u0440aypal", "paypal"]
    whole = substrings(["paypal"], whole_words=True).apply(text, INPUT)
    assert [text[span.start : span.end] for span in whole.findings[0].spans] == ["\u0440aypal"]
    exact = substrings(["paypal"], fold=False).apply(text, INPUT)
    assert len(exact.findings[0].spans) == 1
    redacted = substrings(["PAY PAL"], action="redact").apply(text, INPUT)
    assert "[REDACTED]" in _apply(redacted, text)
    assert substrings(["absent"]).apply(text, INPUT) == TransformOutcome()
    assert "ransomware" in EXPLOIT_TERMS


def test_substrings_settings() -> None:
    with pytest.raises(PolicyError, match="non-empty terms"):
        substrings([])
    with pytest.raises(PolicyError, match="non-empty terms"):
        substrings(["ok", ""])
    with pytest.raises(PolicyError, match="folds to nothing"):
        substrings(["\u200b"])
    with pytest.raises(PolicyError, match="block or redact"):
        substrings(["x"], action="flag")  # type: ignore[arg-type]


def test_token_limit_blocks_or_truncates() -> None:
    text = "one two three four five six seven eight"
    blocked = token_limit(3).apply(text, INPUT)
    assert [(item.label, item.action) for item in blocked.findings] == [("token_limit", "block")]
    truncated = token_limit(3, mode="truncate").apply(text, INPUT)
    assert _apply(truncated, text) == "one two three"
    assert token_limit(100).apply(text, INPUT) == TransformOutcome()
    # Special-token text is counted as ordinary text instead of raising.
    assert token_limit(100).apply("<|endoftext|>", INPUT) == TransformOutcome()


def test_token_limit_rewrites_when_decoding_does_not_round_trip() -> None:
    class Encoding:
        def encode(self, text: str, disallowed_special: object = ()) -> list[str]:
            return list(text)

        def decode(self, tokens: list[str]) -> str:
            return "".join(tokens).upper()

    policy = token_limit(2, mode="truncate")
    rewritten = type(policy)(policy.name, policy.stages, 2, "truncate", Encoding())
    assert _apply(rewritten.apply("abcd", INPUT), "abcd") == "AB"


def test_token_limit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in (0, True):
        with pytest.raises(PolicyError, match="positive limit"):
            token_limit(bad)
    with pytest.raises(PolicyError, match="block or truncate"):
        token_limit(5, mode="cut")  # type: ignore[arg-type]
    with pytest.raises(PolicyError, match="unknown token encoding"):
        token_limit(5, encoding="nope")
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(PolicyError, match=r"jes\[tokens\]"):
        token_limit(5)


def test_regex_needs_its_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "regex", None)
    with pytest.raises(PolicyError, match=r"jes\[regex\]"):
        regex(["x"])


@pytest.mark.parametrize("stage", ["tool_call"])
def test_allowed_tools(stage: Stage) -> None:
    policy = allowed_tools(["Read", "mcp__search__query"])
    allowed = TransformContext(stage=stage, origin=stage, target="text", tool="Read")
    other = TransformContext(stage=stage, origin=stage, target="text", tool="Bash")
    assert policy.apply("{}", allowed) == TransformOutcome()
    assert [(item.label, item.action) for item in policy.apply("{}", other).findings] == [
        ("tool_name", "block")
    ]
    assert policy.stages == {"tool_call"}
    for bad in ([], [""], [" Read"], ["Re\nad"], ["x" * 257]):
        with pytest.raises(PolicyError, match="tool names"):
            allowed_tools(bad)
