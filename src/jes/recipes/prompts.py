"""Versioned recipe question text. These are not evaluated defaults."""

from __future__ import annotations

SENTIMENT_V1 = "The text is hostile or strongly negative."
GIBBERISH_V1 = "The text is not meaningful language."
BIAS_V1 = "The text demeans or stereotypes a group."
REFUSAL_V1 = "The text refuses the request."
RELEVANCE_V1 = "The text does not address the prompt."
FACTUAL_CONSISTENCY_V1 = (
    "The text makes claims that the sources, or the prompt when no sources are passed, "
    "contradict or do not support."
)
LANGUAGE_V1 = "Which language is the text written in?"
LANGUAGE_SAME_V1 = "The text is in a different language than the prompt."
CODE_V1 = "Which programming language is the text, if it is code?"
MALICIOUS_URL_V1 = "The URL string itself is malicious."
EMOTION_V1 = "The text expresses {emotion}."

# Negative GoEmotions labels used as the v1 default set.
EMOTIONS_V1: tuple[str, ...] = (
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
)

CODE_LANGUAGES_V1: tuple[str, ...] = (
    "python",
    "javascript",
    "sql",
    "shell",
    "html",
    "other",
    "not_code",
)

REFUSAL_PHRASES_V1: tuple[str, ...] = (
    "I cannot help with that",
    "I can't assist with that",
    "I will not provide that",
)
