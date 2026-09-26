"""Token and byte counters for exact and conservative backend budgets."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from typing import Any

from jes.errors import PolicyError

Tokenize = Callable[[str], Sequence[str]]


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def label_variants(label: str) -> tuple[str, str]:
    return (label, f" {label}")


def default_label_tokenize(text: str) -> list[str]:
    """Treat a leading-space identifier as one token; otherwise split on spaces."""

    if text.startswith(" ") and text[1:] and " " not in text[1:]:
        return [text]
    if text and " " not in text:
        return [text]
    return [part for part in text.split(" ") if part]


def assert_single_token_labels(labels: Sequence[str], tokenize: Tokenize) -> None:
    for label in labels:
        for variant in label_variants(label):
            pieces = list(tokenize(variant))
            if len(pieces) != 1:
                raise PolicyError("logprob candidate is not a single token")


def load_tokenizer(revision: str | None) -> Tokenize | None:
    if revision is None:
        return None
    if importlib.util.find_spec("tokenizers") is None:
        return None
    module: Any = importlib.import_module("tokenizers")
    try:
        tokenizer = module.Tokenizer.from_pretrained(revision)
    except Exception:
        raise PolicyError("pinned tokenizer could not be loaded") from None

    def tokenize(text: str) -> list[str]:
        encoded = tokenizer.encode(text)
        return list(encoded.tokens)

    return tokenize


__all__ = [
    "Tokenize",
    "assert_single_token_labels",
    "default_label_tokenize",
    "label_variants",
    "load_tokenizer",
    "utf8_bytes",
]
