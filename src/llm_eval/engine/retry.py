"""Exponential backoff with jitter for API calls."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from rich.console import Console

T = TypeVar("T")

console = Console(stderr=True)


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    label: str = "request",
) -> T:
    """Execute an async function with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            total_delay = delay + jitter
            console.print(
                f"  [yellow]⟳ {label} attempt {attempt + 1} failed: {e}. "
                f"Retrying in {total_delay:.1f}s...[/yellow]"
            )
            await asyncio.sleep(total_delay)
    raise RuntimeError("Unreachable")  # pragma: no cover
