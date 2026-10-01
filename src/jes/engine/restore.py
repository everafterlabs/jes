"""Putting values back into a complete reply, and the URL rule that guards it.

Only plain text is restored. A token inside a URL restores only when the URL is an
absolute http or https URL whose origin the caller allowed. Otherwise a reply could
carry a value to someone else's server in a link or an image.
"""

from __future__ import annotations

import ipaddress
import re
import secrets
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeAlias
from urllib.parse import urlsplit

from jes.errors import PolicyError
from jes.redactions import TOKEN_RE
from jes.text.textmap import Edit
from jes.types import Finding

Origin: TypeAlias = tuple[str, str, int]
UrlMode: TypeAlias = Literal["block", "allow"]

# Anything shaped like a jes placeholder. Escaped ones are left alone, so escaping is idempotent.
PLACEHOLDER_RE = re.compile(r"\[JES_(?!LITERAL_)[^\]\r\n]{1,256}\]")
_ESCAPED_PREFIX = "[JES_LITERAL_"

_URL_START = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://"
    r"|(?<![A-Za-z0-9+.-])(?:mailto|data|javascript|vbscript|file|tel|sms|blob):"
    r"|(?<![:/])//"
    r"|(?<![A-Za-z0-9.-])www\.",
    re.IGNORECASE,
)
_URL_STOP = re.compile(r"[\s<>\"'`\x00-\x1f\x7f-\x9f]")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def escape_placeholders(text: str) -> list[Edit]:
    """Edits that defuse placeholder-shaped strings in incoming text so they never restore."""

    return [
        Edit(match.start(), match.start() + len("[JES_"), _ESCAPED_PREFIX)
        for match in PLACEHOLDER_RE.finditer(text)
    ]


def url_spans(text: str) -> list[tuple[int, int]]:
    """Every range that starts like a URL and runs to the next space or delimiter."""

    spans: list[tuple[int, int]] = []
    for match in _URL_START.finditer(text):
        stop = _URL_STOP.search(text, match.end())
        spans.append((match.start(), len(text) if stop is None else stop.start()))
    return spans


def canonical_origin(url: str) -> Origin | None:
    """``(scheme, host, port)`` for an absolute http or https URL, else None.

    Anything a browser and this parser might read differently counts as no origin:
    credentials, backslashes, percent-encoded or scoped hosts, and bad ports.
    """

    if "\\" in url or "\u3002" in url or "\uff0e" in url or "\uff61" in url:
        return None
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not url[len(scheme) :].startswith("://"):
        return None
    host = parsed.hostname
    if host is None or parsed.username is not None or parsed.password is not None:
        return None
    if "%" in parsed.netloc:
        return None
    host = host.rstrip(".")
    if not host:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        canonical = _idna(host)
        if canonical is None:
            return None
    else:
        canonical = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return scheme, canonical, _DEFAULT_PORTS[scheme] if port is None else port


def parse_origin(value: str) -> Origin:
    """An origin from configuration, such as ``https://app.example.com``."""

    parsed = urlsplit(value)
    origin = canonical_origin(value)
    if origin is None and _load_idna() is None:
        raise PolicyError("restore_origins with host names requires the jes[pii] extra")
    if origin is None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise PolicyError("restore_origins entries must be http or https origins")
    return origin


def _load_idna() -> Any:
    try:
        import idna
    except ImportError:
        return None
    return idna


def _idna(host: str) -> str | None:
    # Without idna a name cannot be compared safely, so it matches no origin.
    idna = _load_idna()
    if idna is None:
        return None
    try:
        return idna.encode(host, uts46=False, std3_rules=True).decode("ascii").lower()
    except (idna.IDNAError, UnicodeError, ValueError):
        return None


def restore_tokens(
    text: str,
    *,
    lookup: Callable[[str], str | None],
    authorized: Iterable[str],
    origins: frozenset[Origin],
    on_url: UrlMode,
) -> tuple[str, list[Finding]]:
    """Replace authorized tokens with their values, except inside URLs to other origins."""

    allowed = frozenset(authorized)
    edits: list[Edit] = []
    findings: list[Finding] = []
    urls: list[tuple[int, int]] | None = None
    for match in TOKEN_RE.finditer(text):
        token = match.group()
        value = lookup(token) if token in allowed else None
        if value is None:
            continue
        if urls is None:
            urls = url_spans(text)
        containers = [text[start:end] for start, end in urls if start <= match.start() < end]
        if any(canonical_origin(url) not in origins for url in containers):
            action = "block" if on_url == "block" else "flag"
            findings.append(Finding("jes", "placeholder_in_url", action))
            continue
        edits.append(Edit(match.start(), match.end(), value))
    return _apply(text, edits), findings


class LocalMarkers:
    """Stand-ins that keep a reply's own personal data away from judges, then go back."""

    __slots__ = ("_markers", "_pattern", "_prefix", "_values")

    def __init__(self) -> None:
        # A random prefix, so a reply cannot spell a marker in advance.
        self._prefix = f"[JES_LOCAL_{secrets.token_hex(6)}_"
        self._pattern = re.compile(re.escape(self._prefix) + r"[0-9]+\]")
        self._markers: dict[str, str] = {}
        self._values: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._values)

    def marker(self, value: str) -> str:
        marker = self._markers.get(value)
        if marker is None:
            marker = f"{self._prefix}{len(self._markers)}]"
            self._markers[value] = marker
            self._values[marker] = value
        return marker

    def restore(self, text: str) -> str:
        """Put values back. A marker that a later edit cut apart stays as it is."""

        if not self._values:
            return text
        return self._pattern.sub(lambda match: self._values.get(match.group(), match.group()), text)


def _apply(text: str, edits: list[Edit]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in edits:
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


__all__ = [
    "PLACEHOLDER_RE",
    "LocalMarkers",
    "Origin",
    "UrlMode",
    "canonical_origin",
    "escape_placeholders",
    "parse_origin",
    "restore_tokens",
    "url_spans",
]
