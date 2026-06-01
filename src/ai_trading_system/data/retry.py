"""Retry helpers for free public data sources."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


def retry(
    attempts: int = 3,
    initial_delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a function with exponential backoff and structured logging."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if initial_delay < 0:
        raise ValueError("initial_delay must be >= 0")
    if backoff < 1:
        raise ValueError("backoff must be >= 1")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - retry loop intentionally catches
                    last_error = exc
                    if attempt == attempts:
                        LOGGER.exception(
                            "Data operation failed after retries",
                            extra={"function": func.__name__, "attempt": attempt, "attempts": attempts},
                        )
                        break
                    LOGGER.warning(
                        "Data operation failed; retrying",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "attempts": attempts,
                            "delay_seconds": delay,
                            "error": str(exc),
                        },
                    )
                    if delay:
                        time.sleep(delay)
                    delay *= backoff
            if last_error is None:  # defensive; should never happen
                raise RuntimeError("retry wrapper exited without result or error")
            raise last_error

        return wrapper

    return decorator
