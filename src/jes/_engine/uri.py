"""Strict complete-reply URI candidate and origin canonicalization."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from jes.errors import PolicyError
from jes.types import Span

_SCHEME_START = re.compile(r"(?:[A-Za-z][A-Za-z0-9+.-]*:|//|www\.)")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_DELIMITERS = frozenset("<>\"'`")


def _is_stop(char: str) -> bool:
    code = ord(char)
    if char in _DELIMITERS:
        return True
    if code <= 0x1F or 0x7F <= code <= 0x9F:
        return True
    return char.isspace()


def uri_candidates(text: str) -> tuple[Span, ...]:
    """Return every syntactically started URI range, including nested starts."""

    spans: list[Span] = []
    for match in _SCHEME_START.finditer(text):
        if match.group().startswith("//") and match.start() > 0 and text[match.start() - 1] == ":":
            continue
        end = match.start()
        while end < len(text) and not _is_stop(text[end]):
            end += 1
        if end > match.start():
            spans.append(Span(match.start(), end))
    return tuple(spans)


def containing_candidates(span: Span, candidates: tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if candidate.start <= span.start and span.end <= candidate.end
    )


def _idna_host(host: str) -> str:
    try:
        import idna
    except ImportError:
        raise PolicyError("origin canonicalization requires the jes[pii] extra") from None
    return idna.encode(host, uts46=False, std3_rules=True, transitional=False).decode("ascii")


def canonical_origin(value: str, *, construction: bool = False) -> tuple[str, str, int] | None:
    """Return (scheme, host, port) or None when the candidate cannot restore."""

    if not value or "\\" in value or "\u3002" in value or "\uff0e" in value:
        return None
    if value.startswith("//") or value.startswith("www."):
        return None
    parsed = urlsplit(value, allow_fragments=True)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if construction and (parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        return None
    host = parsed.hostname
    if host is None or "%" in (parsed.netloc or "") or "%" in host:
        return None
    if "%" in host or host.endswith("%") or "%" in parsed.netloc:
        return None
    if "%" in parsed.netloc.split("@")[-1]:
        return None
    if zone_id(host):
        return None
    host = host.rstrip(".")
    if not host:
        return None
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.scope_id:
                return None
            canonical_host = f"[{ip.compressed}]"
        else:
            canonical_host = str(ipaddress.IPv4Address(int(ip)))
    except ValueError:
        try:
            canonical_host = _idna_host(host).lower()
        except Exception:
            return None
    port = parsed.port if parsed.port is not None else _DEFAULT_PORTS[scheme]
    return scheme, canonical_host, port


def zone_id(host: str) -> bool:
    return "%" in host or "%25" in host


def parse_restore_origin(value: str) -> tuple[str, str, int]:
    origin = canonical_origin(value, construction=True)
    if origin is None:
        raise PolicyError("restore_origins entries must be absolute http or https origins")
    return origin


__all__ = [
    "canonical_origin",
    "containing_candidates",
    "parse_restore_origin",
    "uri_candidates",
]
