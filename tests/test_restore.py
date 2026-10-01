"""Placeholder escaping, URL detection and origins, token restoration, and local markers."""

from __future__ import annotations

import sys

import pytest

from jes.engine.restore import (
    LocalMarkers,
    canonical_origin,
    escape_placeholders,
    parse_origin,
    restore_tokens,
    url_spans,
)
from jes.errors import PolicyError
from jes.redactions import Redactions
from jes.text.textmap import TextMap


def _escaped(text: str) -> str:
    return TextMap.identity(len(text)).apply(text, escape_placeholders(text))[0]


def test_placeholder_shaped_strings_are_escaped_once() -> None:
    text = "a [JES_PII_" + "A" * 22 + "] b [JES_whatever] c [JES_LITERAL_x] [jes_lower]"
    escaped = _escaped(text)
    assert escaped == (
        "a [JES_LITERAL_PII_"
        + "A" * 22
        + "] b [JES_LITERAL_whatever] c [JES_LITERAL_x] [jes_lower]"
    )
    assert _escaped(escaped) == escaped


@pytest.mark.parametrize(
    ("text", "urls"),
    [
        ("see https://a.example/x?q=1 now", ["https://a.example/x?q=1"]),
        ("![img](http://evil.example/p.png)", ["http://evil.example/p.png)"]),
        ("mail mailto:x@evil.example?body=hi", ["mailto:x@evil.example?body=hi"]),
        ("<a href='javascript:alert(1)'>", ["javascript:alert(1)"]),
        ("data:text/html,hi", ["data:text/html,hi"]),
        ("go to //evil.example/x", ["//evil.example/x"]),
        ("www.evil.example/a", ["www.evil.example/a"]),
        ("ftp://files.example/a", ["ftp://files.example/a"]),
        ("Email:[JES_PII_x] and Note: plain text", []),
        ("// a code comment", ["//"]),
    ],
)
def test_url_spans(text: str, urls: list[str]) -> None:
    assert [text[start:end] for start, end in url_spans(text)] == urls


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://App.Example.com/path", ("https", "app.example.com", 443)),
        ("http://app.example.com.:8080/", ("http", "app.example.com", 8080)),
        ("HTTPS://app.example.com", ("https", "app.example.com", 443)),
        ("http://127.0.0.1/x", ("http", "127.0.0.1", 80)),
        ("https://[::1]:8443/", ("https", "[::1]", 8443)),
        ("https://b\u00fccher.example/", ("https", "xn--bcher-kva.example", 443)),
        ("https://user:pw@app.example.com/", None),
        ("https://app.example.com\\@evil.example/", None),
        ("https://app%2Eexample.com/", None),
        ("https://app\u3002example.com/", None),
        ("https://app.example.com:99999/", None),
        ("https://app.example.com:port/", None),
        ("https://[fe80::1%25eth0]/", None),
        ("https://[::1/", None),
        ("http:/app.example.com", None),
        ("ftp://app.example.com/", None),
        ("javascript:alert(1)", None),
        ("https:///path-only", None),
        ("https://./", None),
        ("https://bad_host.example/", None),
    ],
)
def test_canonical_origin(url: str, origin: tuple[str, str, int] | None) -> None:
    assert canonical_origin(url) == origin


def test_parse_origin_accepts_bare_origins_only() -> None:
    assert parse_origin("https://app.example.com") == ("https", "app.example.com", 443)
    assert parse_origin("https://app.example.com/") == ("https", "app.example.com", 443)
    for bad in (
        "https://app.example.com/x",
        "https://a.example/?q",
        "https://a.example/#f",
        "a.example",
    ):
        with pytest.raises(PolicyError, match="restore_origins"):
            parse_origin(bad)


def test_origins_need_the_pii_extra_for_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "idna", None)
    assert parse_origin("http://127.0.0.1") == ("http", "127.0.0.1", 80)
    with pytest.raises(PolicyError, match=r"jes\[pii\]"):
        parse_origin("https://app.example.com")
    # A check never raises for it: without idna a name matches no origin.
    assert canonical_origin("https://app.example.com/") is None
    store, email, _name = _store()
    for origins in (frozenset(), frozenset({parse_origin("http://127.0.0.1")})):
        restored, findings = restore_tokens(
            f"See https://evil.example/?u={email}",
            lookup=store._value,
            authorized=[email],
            origins=origins,
            on_url="block",
        )
        assert email in restored
        assert [(item.label, item.action) for item in findings] == [("placeholder_in_url", "block")]


def _store() -> tuple[Redactions, str, str]:
    store = Redactions()
    email, name = store._token("ada@example.com"), store._token("Ada")
    store._commit({email: "ada@example.com", name: "Ada"})
    return store, email, name


def test_only_authorized_tokens_in_the_store_are_restored() -> None:
    store, email, name = _store()
    unknown = Redactions()._token("x")
    text = f"Hi {name}, we wrote to {email}. {unknown}"
    restored, findings = restore_tokens(
        text, lookup=store._value, authorized=[email, unknown], origins=frozenset(), on_url="block"
    )
    assert restored == f"Hi {name}, we wrote to ada@example.com. {unknown}"
    assert findings == []


def test_tokens_in_urls_restore_only_for_allowed_origins() -> None:
    store, email, _name = _store()
    text = f"Open https://evil.example/?u={email} or https://app.example.com/u/{email}"
    restored, findings = restore_tokens(
        text,
        lookup=store._value,
        authorized=[email],
        origins=frozenset({parse_origin("https://app.example.com")}),
        on_url="block",
    )
    assert (
        restored
        == f"Open https://evil.example/?u={email} or https://app.example.com/u/ada@example.com"
    )
    assert [(item.label, item.action) for item in findings] == [("placeholder_in_url", "block")]

    _restored, findings = restore_tokens(
        text, lookup=store._value, authorized=[email], origins=frozenset(), on_url="allow"
    )
    assert {(item.label, item.action) for item in findings} == {("placeholder_in_url", "flag")}


def test_local_markers_round_trip() -> None:
    markers = LocalMarkers()
    first = markers.marker("ada@example.com")
    assert markers.marker("ada@example.com") == first
    second = markers.marker("bob@example.com")
    assert len(markers) == 2
    assert first.startswith("[JES_LOCAL_") and first != second
    assert LocalMarkers().marker("ada@example.com") != first
    text = f"to {first} and {second}, not {first[:-3]}"
    assert markers.restore(text) == f"to ada@example.com and bob@example.com, not {first[:-3]}"
    assert LocalMarkers().restore(text) == text
