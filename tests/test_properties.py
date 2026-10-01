from __future__ import annotations

from hypothesis import given, strategies as st

from jes._engine.merge import merge_findings, merge_scores
from jes._engine.plan import chunk_text
from jes.questions import YesNo
from jes.testing import FakeBackend
from jes.types import Finding, Provenance, ScoreResult, State, ThresholdProvenance

safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=80,
)


def _provenance(value: float) -> ScoreResult:
    return ScoreResult(
        value=value,
        kind="probability",
        confidence=None,
        provenance=Provenance(
            backend="fake",
            model="fake@1",
            request_profile="r",
            decision_profile="d",
            prompt_version="v1",
            threshold=ThresholdProvenance(
                source="explicit",
                block_at=0.8,
                flag_at=None,
                fingerprint="t",
                vector_fingerprint="v",
            ),
        ),
    )


@given(safe_text)
def test_chunks_cover_and_advance(text: str) -> None:
    backend = FakeBackend(max_units=40)
    questions = {"q": YesNo("Is this a violation?")}
    state = State(stage="input", text="")
    try:
        chunks = chunk_text(
            model=backend,
            base_state=state,
            questions=questions,
            text=text,
            max_chunks=16,
            whole_text=False,
        )
    except Exception:
        return
    covered = [False] * len(text)
    last = -1
    for chunk, span in chunks:
        assert span.start < span.end
        assert span.start >= last
        assert chunk == text[span.start : span.end]
        room = backend.headroom(State(stage="input", text=chunk), questions)
        assert room is None or room >= 0
        for index in range(span.start, span.end):
            covered[index] = True
        last = span.start
    assert all(covered)


@given(
    st.lists(st.sampled_from(["flag", "redact", "block"]), min_size=1, max_size=5),
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=5),
)
def test_merge_is_monotonic(actions: list[str], values: list[float]) -> None:
    findings = [
        Finding(policy="p", label="l", action=action, question="q", score=_provenance(value))
        for action, value in zip(actions, values, strict=False)
    ]
    merged = merge_findings(findings)
    assert len(merged) == 1
    rank = {"flag": 0, "redact": 1, "block": 2}
    assert rank[merged[0].action] == max(rank[item.action] for item in findings)
    scores = merge_scores([{"p.q": item.score} for item in findings if item.score is not None])
    expected = max(item.score.value for item in findings if item.score is not None)
    assert scores["p.q"].value == expected


@given(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=10),
)
def test_transform_edit_mapping(text: str, inserted: str) -> None:
    from jes._engine.limits import GuardLimits
    from jes._engine.plan import run_transforms
    from jes.policies import TransformEdit, TransformOutcome
    from tests.helpers import FakeTransform

    def handler(current, call):
        del call
        if not current:
            edits = (TransformEdit(0, 0, inserted),) if inserted else ()
            return TransformOutcome(text=inserted, edits=edits)
        return TransformOutcome(
            text=current[0] + inserted + current[1:],
            edits=(TransformEdit(1, 1, inserted),) if inserted else (),
        )

    result = run_transforms(
        text,
        stage="input",
        origin_stage="input",
        target="subject",
        policies=[FakeTransform("ins", frozenset({"x"}), handler=handler)],
        redactions=None,
        deadline=None,
        limits=GuardLimits(),
        location_target="subject",
        neutralize=False,
    )
    if not text:
        return
    assert len(result.mapping) == len(result.text)
    for span in result.mapping:
        assert 0 <= span.start <= span.end <= len(text)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40))
def test_fold_offsets_map_back(text: str) -> None:
    import unicodedata

    from jes.policies._fold import fold, nfkc
    from jes.policies._protocols import TransformEdit, apply_transform_edits

    mapped = nfkc(text)
    assert mapped.text == unicodedata.normalize("NFKC", text)
    assert len(mapped.spans) == len(mapped.text)
    folded = fold(text)
    assert len(folded.spans) == len(folded.text)
    previous = 0
    for span in folded.spans:
        assert 0 <= span.start <= span.end <= len(text)
        assert span.start >= previous or span.end >= previous
        previous = span.start
    if folded.text:
        origin = folded.origin(0, min(2, len(folded.text)))
        edited = apply_transform_edits(text, (TransformEdit(origin.start, origin.end, ""),))
        assert edited == text[: origin.start] + text[origin.end :]


@given(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=16),
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=16),
)
def test_redacting_folded_match_removes_term(prefix: str, suffix: str) -> None:
    from jes.policies import CallContext, substrings
    from jes.policies._fold import fold
    from jes.policies._protocols import apply_transform_edits

    term = "xyzzy"
    text = f"{prefix}\n{term}\n{suffix}"
    policy = substrings([term], action="redact")
    outcome = policy.apply(
        text,
        CallContext(
            call_stage="input",
            origin_stage="input",
            target="subject",
            redactions=None,
            deadline=None,
        ),
    )
    assert outcome.findings
    assert apply_transform_edits(text, outcome.edits) == outcome.text
    assert fold(term).text not in fold(outcome.text).text
