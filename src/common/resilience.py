"""
Shared exponential-backoff retry wrapper (NFR-8). Reused by Step 16's router
call, Step 20's Tavily call, Step 22's generation calls — build once here.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from config.settings import settings

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetriesExhaustedError(RuntimeError):
    pass


def _default_is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError))

async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_retries: int | None = None,
    base_delay_s: float = 0.25,
    is_retryable: Callable[[Exception], bool] | None = None,
) -> T:
    """
    Exponential backoff with jitter. Raises RetriesExhaustedError after
    (max_retries + 1) total attempts if every attempt fails.
    Defaults max_retries to settings.max_retries (currently 2, NFR-8).
    """
    max_retries = max_retries if max_retries is not None else settings.max_retries
    check = is_retryable or _default_is_retryable
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries or not check(exc):
                break
            delay = base_delay_s * (2 ** attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)

    raise RetriesExhaustedError(f"All {max_retries + 1} attempts failed") from last_exc