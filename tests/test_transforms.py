from __future__ import annotations

import sys
import time

import pytest

from jes import Guard
from jes.errors import DeadlineExceeded, PolicyError
from jes.policies import EXPLOIT_TERMS, CallContext, invisible_text, regex, substrings, token_limit
from jes.policies._fold import fold, nfkc
from jes.policies._protocols import apply_transform_edits
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend
from jes.types import Span


def _call(*, deadline: float | None = None) -> CallContext:
    return CallContext(
        call_stage="input",
        origin_stage="input",
        target="subject",
        redactions=None,
        deadline=deadline,
    )


def test_invisible_targeted_removes_evasion_and_keeps_legitimate() -> None:
    policy = invisible_text()
    heart = "\u2764\ufe0f"
    persian = "می\u200cشود"
    hindi = "नमस्ते"
    family = "👨\u200d👩\u200d👧"
    text = (
        f"A\uff21 {_cyrillic()} soft\u00adhyphen "
        f"tag\U000e0061 bidi\u202e mark\u200e hangul\u3164 "
        f"nul\x00 c1\x81 ansi\x1b[31m vs\ufe00 {heart} {persian} {hindi} "
        f"{family} keep\t\n\r"
    )
    outcome = policy.apply(text, _call())
    assert "\u00ad" not in outcome.text
    assert "\u202e" not in outcome.text
    assert "\u200e" not in outcome.text
    assert "\u3164" not in outcome.text
    assert "\x00" not in outcome.text
    assert "\x81" not in outcome.text
    assert "\x1b" not in outcome.text
    assert "\ufe00" not in outcome.text
    assert "\U000e0061" not in outcome.text
    assert heart in outcome.text
    assert "\u200c" in outcome.text
    assert hindi in outcome.text
    assert "\u200d" in outcome.text
    assert "\t" in outcome.text and "\n" in outcome.text and "\r" in outcome.text
    assert outcome.findings[0].action == "redact"
    assert apply_transform_edits(text, outcome.edits) == outcome.text


def _cyrillic() -> str:
    return "\u0430"


def test_invisible_all_drops_joiners() -> None:
    text = "👨\u200d👩"
    targeted = invisible_text().apply(text, _call())
    everything = invisible_text(mode="all").apply(text, _call())
    assert "\u200d" in targeted.text
    assert "\u200d" not in everything.text


def test_invisible_block_and_empty() -> None:
    empty = invisible_text().apply("", _call())
    assert empty.text == ""
    assert empty.findings == ()
    blocked = invisible_text(block=True).apply("a\x00b", _call())
    assert blocked.text == "a\x00b"
    assert blocked.findings[0].action == "block"
    assert blocked.edits == ()


def test_substrings_catches_folding_evasions() -> None:
    policy = substrings(["ransomware"])
    cases = [
        "ransomware",
        "RANSOMWARE",
        "\uff52ansomware",
        "ransomw" + "\u0430" + "re",
        "ransom\u00adware",
        "ransom\U000e0020ware",
        "ransom\u202eware",
        "ransom\u200eware",
        "ransom\u3164ware",
        "r\ufe00ansomware",
        "".join(char + "\ufe00" for char in "ransomware"),
    ]
    for text in cases:
        outcome = policy.apply(text, _call())
        assert outcome.findings, repr(text)
        assert outcome.findings[0].action == "block"


def test_controls_are_stripped_before_substring_detect() -> None:
    normalizer = invisible_text()
    detector = substrings(["ransomware"])
    cases = ["ransom\x00ware", "ransom\x81ware", "ransom\x1bware"]
    for text in cases:
        cleaned = normalizer.apply(text, _call())
        outcome = detector.apply(cleaned.text, _call())
        assert outcome.findings, repr(text)


def test_substrings_redact_maps_to_original() -> None:
    policy = substrings(["ban"], action="redact")
    text = "xxb\u00adanxx"
    outcome = policy.apply(text, _call())
    assert "[REDACTED]" in outcome.text
    assert "ban" not in outcome.text.casefold()
    assert apply_transform_edits(text, outcome.edits) == outcome.text
    span = outcome.findings[0].spans[0]
    assert text[span.start : span.end].replace("\u00ad", "") == "ban"


def test_substrings_whole_word_and_no_match() -> None:
    whole = substrings(["ban"], whole_words=True)
    assert whole.apply("ban", _call()).findings
    assert whole.apply("banner", _call()).findings == ()
    assert substrings(["ban"]).apply("banner", _call()).findings
    assert substrings(["ban"]).apply("safe", _call()).findings == ()
    assert substrings(["ban"]).apply("", _call()).findings == ()
    overlapped = substrings(["ran", "ansom"], action="redact").apply("ransom", _call())
    assert overlapped.text == "[REDACTED]"
    with pytest.raises(PolicyError):
        substrings(["\u00ad"])


def test_regex_require_redact_fullmatch_and_timeout() -> None:
    required = regex([r"ok-\d+"], require=True)
    assert required.apply("ok-12", _call()).findings == ()
    missing = required.apply("nope", _call())
    assert missing.findings[0].action == "block"

    redacting = regex([r"secret"], action="redact")
    outcome = redacting.apply("a secret here", _call())
    assert outcome.text == "a [REDACTED] here"

    full = regex([r"only"], match="fullmatch")
    assert full.apply("only", _call()).findings
    assert full.apply("not only", _call()).findings == ()

    folded = regex([r"ban"], fold=True)
    assert folded.apply("b\u00adAN", _call()).findings

    started = time.monotonic()
    timed = regex([r"(x+x+)+y"], timeout_ms=20).apply("x" * 30, _call())
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    if timed.findings:
        assert timed.findings[0].action == "block"

    near = time.monotonic() + 0.05
    bounded = regex([r"(x+x+)+y"], timeout_ms=5000).apply("x" * 40, _call(deadline=near))
    assert time.monotonic() - near < 1.0
    if bounded.findings:
        assert bounded.findings[0].action == "block"


def test_token_limit_block_and_truncate() -> None:
    text = "one two three four five"
    blocked = token_limit(2).apply(text, _call())
    assert blocked.text == text
    assert blocked.findings[0].action == "block"
    truncated = token_limit(2, mode="truncate").apply(text, _call())
    assert truncated.findings[0].action == "flag"
    assert len(truncated.text) < len(text)
    assert apply_transform_edits(text, truncated.edits) == truncated.text
    assert token_limit(8).apply(text, _call()).findings == ()
    assert token_limit(8).apply("", _call()).findings == ()


def test_factories_validate_and_guard_round_trip() -> None:
    with pytest.raises(PolicyError):
        invisible_text(mode="other")  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        regex([])
    with pytest.raises(PolicyError):
        regex(["("])
    with pytest.raises(PolicyError):
        regex(["x"], timeout_ms=0)
    with pytest.raises(PolicyError):
        substrings([])
    with pytest.raises(PolicyError):
        substrings([""])
    with pytest.raises(PolicyError):
        token_limit(0)
    with pytest.raises(PolicyError):
        token_limit(4, encoding="missing-encoding")
    with pytest.raises(PolicyError):
        regex(["x"], action="flag")  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        regex(["x"], match="span")  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        substrings(["x"], action="flag")  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        token_limit(4, mode="clip")  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        invisible_text(stages=())
    with pytest.raises(DeadlineExceeded):
        invisible_text().apply("x", _call(deadline=time.monotonic() - 1))
    assert substrings(["Ban"], fold=False).apply("ban", _call()).findings == ()
    assert substrings(["Ban"], fold=False).apply("Ban", _call()).findings
    private = invisible_text(mode="all").apply("x\ue000y", _call())
    assert "\ue000" not in private.text
    composed = nfkc("e\u0301")
    assert composed.text == "\u00e9"
    empty_fold = fold("")
    assert empty_fold.text == ""
    assert empty_fold.origin(0, 0) == Span(0, 0)

    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    from tests.helpers import yesno_policy

    guard = Guard(
        [
            invisible_text(),
            substrings(["ransomware"]),
            token_limit(64),
            yesno_policy(),
        ],
        model=backend,
    )
    allowed = guard.check_input("hello")
    assert allowed.ok
    blocked = guard.check_input("please discuss ransomware here")
    assert blocked.decision == "block"
    assert any(finding.label == "substrings" for finding in blocked.findings)
    smuggled = guard.check_input("please discuss ransom\x00ware here")
    assert smuggled.decision == "block"
    assert any(finding.label == "substrings" for finding in smuggled.findings)


def test_exploit_terms_are_opt_in_phrases() -> None:
    assert "ransomware" in EXPLOIT_TERMS
    assert "code injection" in EXPLOIT_TERMS
    outcome = substrings(EXPLOIT_TERMS).apply("talk about ransomware", _call())
    assert outcome.findings


def test_nfkc_and_fold_maps() -> None:
    from jes.policies._fold import MappedText, _prefix_normalize

    text = "\uff21\u00ad\u0430"
    mapped = nfkc(text)
    assert mapped.text == "A\u00ad\u0430"
    assert mapped.origin(0, 1) == Span(0, 1)
    folded = fold(text)
    assert "a" in folded.text
    assert folded.origin(0, 1).end <= len(text)
    assert len(folded.spans) == len(folded.text)
    composed = _prefix_normalize("e\u0301", "NFKC")
    assert composed.text == "\u00e9"
    assert composed.origin(0, 1) == Span(0, 2)
    empty = _prefix_normalize("", "NFKC")
    assert empty.text == ""
    assert empty.origin(0, 0) == Span(0, 0)
    with pytest.raises(ValueError):
        MappedText("ab", (Span(0, 1),))
    with pytest.raises(ValueError):
        fold("x").origin(-1, 0)
    with pytest.raises(ValueError):
        fold("x").origin(0, 2)


def test_missing_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def blocked(name: str, *args: object, **kwargs: object):
        if name in {"regex", "tiktoken"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    monkeypatch.delitem(sys.modules, "regex", raising=False)
    monkeypatch.delitem(sys.modules, "tiktoken", raising=False)
    from jes.policies import transforms as module

    with pytest.raises(PolicyError, match="jes\\[regex\\]"):
        module.regex(["x"])
    with pytest.raises(PolicyError, match="jes\\[tokens\\]"):
        module.token_limit(8)
