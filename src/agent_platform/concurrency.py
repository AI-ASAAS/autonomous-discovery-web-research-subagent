"""Bounded-wait helpers for blocking, network-facing operations.

The external discovery sub-agent is a synchronous, best-effort call.  A slow
provider must never stall the conversational turn, so it runs on a shared
worker pool and the caller waits at most ``timeout_seconds``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-platform-io")


class OperationTimeoutError(TimeoutError):
    """Raised when a blocking operation exceeds its deadline."""


def run_with_timeout(
    fn: Callable[..., T], timeout_seconds: float, *args: Any, **kwargs: Any
) -> T:
    """Run ``fn`` on the shared pool and give up after ``timeout_seconds``.

    The worker thread is not interrupted (Python cannot do that); it finishes
    in the background.  Callers get ``OperationTimeoutError`` instead of
    hanging, which is what the user-facing layers convert into a polite
    "external search unavailable" response.
    """
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=max(0.001, float(timeout_seconds)))
    except FuturesTimeoutError as exc:
        future.cancel()
        raise OperationTimeoutError(
            f"operation exceeded {timeout_seconds:g}s"
        ) from exc
