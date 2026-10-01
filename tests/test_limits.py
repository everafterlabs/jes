"""Guard limits and byte counting."""

from __future__ import annotations

import time

import pytest

from jes.errors import PolicyError
from jes.limits import LimitExceeded, Limits, require_size, utf8_size


def test_limits_must_be_positive_integers() -> None:
    assert Limits().max_input_bytes == 1_048_576
    for bad in (0, -1, True, 1.5):
        with pytest.raises(PolicyError, match="max_chunks must be a positive integer"):
            Limits(max_chunks=bad)  # type: ignore[arg-type]


def test_utf8_size_counts_encoded_bytes() -> None:
    assert utf8_size("") == 0
    assert utf8_size("abc") == 3
    assert utf8_size("é") == 2
    assert utf8_size("€") == 3
    assert utf8_size("😀") == 4


def test_lone_surrogates_block() -> None:
    with pytest.raises(LimitExceeded) as raised:
        utf8_size("a\ud800b")
    assert raised.value.label == "invalid_unicode"


def test_require_size_raises_with_the_given_label() -> None:
    assert require_size("abcd", 4, "input_too_long") == 4
    with pytest.raises(LimitExceeded) as raised:
        require_size("abcde", 4, "input_too_long")
    assert raised.value.label == "input_too_long"


def test_sizing_a_megabyte_is_fast() -> None:
    text = "é" * 500_000
    started = time.perf_counter()
    for _ in range(10):
        utf8_size(text)
    assert time.perf_counter() - started < 0.1
