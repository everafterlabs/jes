"""Deadline-aware httpx helpers with capped success and error bodies."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import httpx

from jes.errors import BackendError, DeadlineExceeded

logger = logging.getLogger("jes")

_CONNECT_SHARE = 0.25


def remaining_timeout(deadline: float | None, fallback: float) -> float:
    """Return a finite timeout derived from the remaining monotonic deadline."""

    if deadline is None:
        if fallback <= 0 or not math.isfinite(fallback):
            raise DeadlineExceeded("http")
        return fallback
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not math.isfinite(remaining):
        raise DeadlineExceeded("http")
    return min(fallback, remaining)


def http_timeout(deadline: float | None, fallback: float) -> httpx.Timeout:
    total = remaining_timeout(deadline, fallback)
    connect = min(total, max(total * _CONNECT_SHARE, 0.001))
    return httpx.Timeout(total, connect=connect)


def read_capped(
    response: httpx.Response,
    max_bytes: int,
    *,
    backend: str,
) -> bytes:
    """Read a response body and stop before buffering more than `max_bytes`."""

    chunks: list[bytes] = []
    total = 0
    iterator: Iterator[bytes] = response.iter_bytes()
    for chunk in iterator:
        total += len(chunk)
        if total > max_bytes:
            response.close()
            raise BackendError(backend, "response_too_large", status_code=response.status_code)
        chunks.append(chunk)
    return b"".join(chunks)


async def aread_capped(
    response: httpx.Response,
    max_bytes: int,
    *,
    backend: str,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    iterator: AsyncIterator[bytes] = response.aiter_bytes()
    async for chunk in iterator:
        total += len(chunk)
        if total > max_bytes:
            await response.aclose()
            raise BackendError(backend, "response_too_large", status_code=response.status_code)
        chunks.append(chunk)
    return b"".join(chunks)


def request_capped(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    backend: str,
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout,
    max_bytes: int,
) -> tuple[int, bytes]:
    try:
        with client.stream(
            method,
            url,
            content=content,
            headers=dict(headers or {}),
            timeout=timeout,
        ) as response:
            body = read_capped(response, max_bytes, backend=backend)
            return response.status_code, body
    except DeadlineExceeded:
        raise
    except BackendError:
        raise
    except httpx.TimeoutException:
        raise DeadlineExceeded(backend) from None
    except httpx.HTTPError:
        raise BackendError(backend, "connection_error") from None


async def arequest_capped(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    backend: str,
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout,
    max_bytes: int,
) -> tuple[int, bytes]:
    try:
        async with client.stream(
            method,
            url,
            content=content,
            headers=dict(headers or {}),
            timeout=timeout,
        ) as response:
            body = await aread_capped(response, max_bytes, backend=backend)
            return response.status_code, body
    except DeadlineExceeded:
        raise
    except BackendError:
        raise
    except httpx.TimeoutException:
        raise DeadlineExceeded(backend) from None
    except httpx.HTTPError:
        raise BackendError(backend, "connection_error") from None


def retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def status_reason(status_code: int) -> str:
    if status_code == 422:
        return "unprocessable"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "server_error"
    return "http_error"


def log_http(event: str, *, backend: str, status_code: int | None = None) -> None:
    extra: dict[str, Any] = {"backend": backend, "event": event}
    if status_code is not None:
        extra["status"] = status_code
    logger.debug("backend http", extra=extra)


__all__ = [
    "aread_capped",
    "arequest_capped",
    "http_timeout",
    "log_http",
    "read_capped",
    "remaining_timeout",
    "request_capped",
    "retryable_status",
    "status_reason",
]
