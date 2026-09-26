"""Pure System One and LiteLLM request builders and response parsers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from jes.errors import BackendError
from jes.questions import (
    Answer,
    Choice,
    ChoiceAnswer,
    Question,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
    validate_answer,
)
from jes.types import State

from ._tokens import label_variants

ScoreKindName = Literal["probability", "label", "verbalized"]


def dumps(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def render_state(state: State) -> str:
    if (
        state.prompt is None
        and state.question is None
        and not state.sources
        and not state.history
    ):
        return state.text
    payload = {
        "history": [{"role": message.role, "text": message.text} for message in state.history],
        "prompt": state.prompt,
        "question": state.question,
        "sources": list(state.sources),
        "text": state.text,
    }
    return dumps(payload).decode("utf-8")


def encode_question(question: Question) -> dict[str, object]:
    if isinstance(question, YesNo):
        return {
            "criteria": {
                "false": question.false or "The statement is false.",
                "true": question.true or "The statement is true.",
            },
            "instructions": question.instructions,
            "type": "noul",
        }
    if isinstance(question, Choice):
        return {
            "criteria": {key: (value or key) for key, value in question.options.items()},
            "instructions": question.instructions,
            "type": "choice",
        }
    return {
        "criteria": list(question.levels),
        "instructions": question.instructions,
        "type": "score",
    }


def build_system_one_request(
    model: str,
    state: State,
    questions: Mapping[str, Question],
) -> dict[str, object]:
    return {
        "model": model,
        "questions": {key: encode_question(value) for key, value in questions.items()},
        "state": render_state(state),
    }


def serialized_system_one_request(
    model: str,
    state: State,
    questions: Mapping[str, Question],
) -> bytes:
    return dumps(build_system_one_request(model, state, questions))


def _as_mapping(value: object, *, backend: str, reason: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BackendError(backend, reason)
    return cast(dict[str, Any], value)


def _unit(value: object, *, backend: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BackendError(backend, "malformed_answer")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise BackendError(backend, "malformed_answer")
    del field
    return number


def _yes_no_score(payload: Mapping[str, object], *, backend: str) -> float:
    if "noul" in payload:
        return _unit(payload["noul"], backend=backend, field="noul")
    if "score" in payload:
        return _unit(payload["score"], backend=backend, field="score")
    raise BackendError(backend, "malformed_answer")


def _choice_scores(payload: Mapping[str, object], *, backend: str) -> dict[str, float]:
    raw = payload.get("probabilities", payload.get("scores"))
    if not isinstance(raw, dict):
        raise BackendError(backend, "malformed_answer")
    scores = cast(dict[str, object], raw)
    return {
        str(key): _unit(value, backend=backend, field=str(key)) for key, value in scores.items()
    }


def _score_levels(
    payload: Mapping[str, object],
    *,
    backend: str,
    width: int,
) -> tuple[float, ...]:
    raw = payload.get("probabilities")
    if isinstance(raw, dict):
        scores = cast(dict[str, object], raw)
        expected = {str(index) for index in range(width)}
        if set(scores) != expected:
            raise BackendError(backend, "malformed_answer")
        return tuple(
            _unit(scores[str(index)], backend=backend, field=str(index)) for index in range(width)
        )
    listed = payload.get("scores")
    if not isinstance(listed, list):
        raise BackendError(backend, "malformed_answer")
    values = cast(list[object], listed)
    return tuple(_unit(value, backend=backend, field="level") for value in values)


def _confidence(value: object, *, backend: str) -> float | None:
    if value is None:
        return None
    return _unit(value, backend=backend, field="confidence")


def _parse_one_answer(
    question: Question,
    raw: object,
    *,
    backend: str,
    kind: ScoreKindName,
) -> Answer:
    payload = _as_mapping(raw, backend=backend, reason="malformed_answer")
    confidence = _confidence(payload.get("confidence"), backend=backend)
    if isinstance(question, YesNo):
        answer = YesNoAnswer(_yes_no_score(payload, backend=backend), kind, confidence)
    elif isinstance(question, Choice):
        answer = ChoiceAnswer(_choice_scores(payload, backend=backend), kind, confidence)
    else:
        answer = ScoreAnswer(
            _score_levels(payload, backend=backend, width=len(question.levels)),
            kind,
            confidence,
        )
    validate_answer(question, answer)
    return answer


def parse_system_one_response(
    body: bytes,
    questions: Mapping[str, Question],
    *,
    backend: str,
    kind: ScoreKindName = "probability",
) -> tuple[dict[str, Answer], int | None, int | None]:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BackendError(backend, "malformed_answer") from None
    payload = _as_mapping(parsed, backend=backend, reason="malformed_answer")
    raw_answers = payload.get("answers")
    answers_map = _as_mapping(raw_answers, backend=backend, reason="malformed_answer")
    missing = set(questions) - set(answers_map)
    extra = set(answers_map) - set(questions)
    if missing or extra:
        raise BackendError(backend, "malformed_answer", question_ids=missing or extra)
    answers = {
        key: _parse_one_answer(question, answers_map[key], backend=backend, kind=kind)
        for key, question in questions.items()
    }
    usage = payload.get("usage")
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_tokens, output_tokens = usage_from_mapping(usage)
    return answers, input_tokens, output_tokens


def request_hash(model: str, state: State, questions: Mapping[str, Question]) -> str:
    return hashlib.sha256(serialized_system_one_request(model, state, questions)).hexdigest()


def choose_delimiter(seed: str, fields: Sequence[str]) -> str:
    counter = 0
    while True:
        delimiter = f"JESv1_{seed[:16]}_{counter:x}"
        if all(delimiter not in field for field in fields):
            return delimiter
        counter += 1


def question_candidates(question: Question) -> tuple[str, ...]:
    if isinstance(question, YesNo):
        return ("true", "false")
    if isinstance(question, Choice):
        return tuple(question.options)
    return question.levels


def litellm_messages(
    model: str,
    state: State,
    questions: Mapping[str, Question],
    *,
    mode: Literal["logprobs", "verbalized"],
) -> tuple[list[dict[str, str]], str]:
    state_text = render_state(state)
    question_block = dumps(
        {key: encode_question(value) for key, value in questions.items()}
    ).decode("utf-8")
    seed = request_hash(model, state, questions)
    delimiter = choose_delimiter(seed, (state_text, question_block, model))
    if mode == "logprobs":
        format_text = (
            "Answer each question with exactly one candidate token, one question per line, "
            f"in the listed order. Do not write the boundary {delimiter} in an answer."
        )
    else:
        format_text = (
            f"Write only a JSON object between {delimiter} markers. The object is "
            '{"answers":{"<id>":{"score":0.0}}} or {"answers":{"<id>":{"scores":{...}}}}.'
        )
    system = (
        f"{format_text}\nQuestions:\n{question_block}\nBoundary:{delimiter}"
    )
    user = f"{delimiter}\n{state_text}\n{delimiter}"
    return (
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        delimiter,
    )


def parse_verbalized_content(
    content: str,
    questions: Mapping[str, Question],
    *,
    backend: str,
    delimiter: str,
) -> dict[str, Answer]:
    text = content.strip()
    if delimiter in text:
        parts = text.split(delimiter)
        if len(parts) < 3:
            raise BackendError(backend, "malformed_answer")
        text = parts[1].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise BackendError(backend, "malformed_answer") from None
    payload = _as_mapping(parsed, backend=backend, reason="malformed_answer")
    raw_answers = payload.get("answers", payload)
    body = dumps({"answers": raw_answers})
    answers, _, _ = parse_system_one_response(
        body,
        questions,
        backend=backend,
        kind="verbalized",
    )
    return answers


def _top_mass(top: Sequence[Mapping[str, Any]], label: str) -> float:
    variants = set(label_variants(label))
    total = 0.0
    for item in top:
        token = item.get("token")
        logprob = item.get("logprob")
        if token not in variants:
            continue
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            continue
        total += math.exp(float(logprob))
    return total


def parse_logprob_choice(
    raw: object,
    questions: Mapping[str, Question],
    *,
    backend: str,
) -> tuple[dict[str, Answer], int | None, int | None]:
    raw_obj: object = raw
    if isinstance(raw_obj, dict):
        source: object = cast(dict[str, object], raw_obj)
    else:
        source = getattr(raw_obj, "__dict__", None)
    payload = _as_mapping(source, backend=backend, reason="malformed_answer")
    choices_raw = payload.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        raise BackendError(backend, "malformed_answer")
    first_obj: object = cast(list[object], choices_raw)[0]
    if isinstance(first_obj, dict):
        first_source: object = cast(dict[str, object], first_obj)
    else:
        first_source = getattr(first_obj, "__dict__", {})
    choice = _as_mapping(first_source, backend=backend, reason="malformed_answer")
    logprobs_obj: object = choice.get("logprobs")
    if isinstance(logprobs_obj, dict):
        logprob_map = _as_mapping(
            cast(dict[str, object], logprobs_obj),
            backend=backend,
            reason="malformed_answer",
        )
    else:
        logprob_map = {}
    content = logprob_map.get("content")
    if not isinstance(content, list):
        raise BackendError(backend, "malformed_answer")
    tokens: list[Mapping[str, Any]] = []
    for item_obj in cast(list[object], content):
        if not isinstance(item_obj, dict):
            continue
        item = cast(dict[str, Any], item_obj)
        token = item.get("token")
        if not isinstance(token, str) or not token.strip():
            continue
        tokens.append(item)
    if len(tokens) < len(questions):
        raise BackendError(backend, "malformed_answer", question_ids=questions)
    answers: dict[str, Answer] = {}
    for index, (question_id, question) in enumerate(questions.items()):
        item = tokens[index]
        top = item.get("top_logprobs")
        if not isinstance(top, list) or not top:
            raise BackendError(backend, "malformed_answer", question_ids=(question_id,))
        typed_top = [
            cast(dict[str, Any], entry)
            for entry in cast(list[object], top)
            if isinstance(entry, dict)
        ]
        candidates = question_candidates(question)
        masses = {label: _top_mass(typed_top, label) for label in candidates}
        if any(mass <= 0.0 for mass in masses.values()):
            raise BackendError(backend, "missing_logprob_candidate", question_ids=(question_id,))
        total = sum(masses.values())
        if total < 0.5:
            raise BackendError(backend, "low_logprob_mass", question_ids=(question_id,))
        normalized = {label: mass / total for label, mass in masses.items()}
        if isinstance(question, YesNo):
            answers[question_id] = YesNoAnswer(
                normalized["true"],
                "probability",
                min(total, 1.0),
            )
        elif isinstance(question, Choice):
            answers[question_id] = ChoiceAnswer(normalized, "probability", min(total, 1.0))
        else:
            answers[question_id] = ScoreAnswer(
                tuple(normalized[level] for level in question.levels),
                "probability",
                min(total, 1.0),
            )
        validate_answer(question, answers[question_id])
    input_tokens, output_tokens = usage_from_mapping(payload.get("usage"))
    return answers, input_tokens, output_tokens


def usage_from_mapping(raw: object) -> tuple[int | None, int | None]:
    if not isinstance(raw, dict):
        return None, None
    usage = cast(dict[str, object], raw)
    raw_in = usage.get("prompt_tokens")
    if raw_in is None:
        raw_in = usage.get("input_tokens")
    raw_out = usage.get("completion_tokens")
    if raw_out is None:
        raw_out = usage.get("output_tokens")
    input_tokens = raw_in if isinstance(raw_in, int) and not isinstance(raw_in, bool) else None
    output_tokens = raw_out if isinstance(raw_out, int) and not isinstance(raw_out, bool) else None
    return input_tokens, output_tokens


def message_content(raw: object) -> str:
    value: object = raw
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        payload = cast(dict[str, object], value)
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = cast(list[object], choices)[0]
            if isinstance(first, dict):
                first_map = cast(dict[str, object], first)
                message = first_map.get("message")
                if isinstance(message, dict):
                    content = cast(dict[str, object], message).get("content")
                    if isinstance(content, str):
                        return content
                text = first_map.get("text")
                if isinstance(text, str):
                    return text
        direct = payload.get("content")
        if isinstance(direct, str):
            return direct
    choices_attr: object = getattr(raw, "choices", None)
    if isinstance(choices_attr, list) and choices_attr:
        first_choice: object = cast(list[object], choices_attr)[0]
        message_obj: object = getattr(first_choice, "message", None)
        value = getattr(message_obj, "content", None)
        if isinstance(value, str):
            return value
    raise BackendError("litellm", "malformed_answer")


__all__ = [
    "build_system_one_request",
    "choose_delimiter",
    "dumps",
    "encode_question",
    "litellm_messages",
    "message_content",
    "parse_logprob_choice",
    "parse_system_one_response",
    "parse_verbalized_content",
    "question_candidates",
    "render_state",
    "request_hash",
    "serialized_system_one_request",
    "usage_from_mapping",
]
