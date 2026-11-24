"""Production plumbing: structured logs, request correlation and abuse controls.

Everything here is opt-in so the offline/local workflow and the test-suite
behave exactly as before:

* ``API_KEY``             — empty disables authentication
* ``RATE_LIMIT_PER_MINUTE`` — 0 disables the limiter
* ``LOG_JSON``            — 0 keeps human-readable logs

The middleware never touches the response body, so Server-Sent Events streams
pass through untouched.  It only stamps headers and records timing.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("agent_platform.access")

REQUEST_ID_HEADER = "X-Request-ID"
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

#: Endpoints that must stay reachable without credentials (probes and docs).
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {"/", "/health", "/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json"}
)

_HANDLER_MARK = "_agent_platform_handler"


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record (``-`` outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = request_id_var.get()
        return True


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line — the format every log shipper expects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Install exactly one root handler; safe to call from every entry point."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARK, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonLogFormatter() if json_logs else PlainFormatter())

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper() if level else "INFO", logging.INFO))


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request id + access log + security headers + optional auth/rate limit."""

    def __init__(
        self,
        app: Any,
        *,
        api_key: str = "",
        rate_limit_per_minute: int = 0,
        log_requests: bool = True,
    ) -> None:
        super().__init__(app)
        self.api_key = api_key or ""
        self.rate_limit = max(0, int(rate_limit_per_minute or 0))
        self.log_requests = log_requests
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    # -- Starlette hook --------------------------------------------------
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        request_id = (request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex)[:64]
        token = request_id_var.set(request_id)
        try:
            response = self._gate(request)
            if response is None:
                response = await call_next(request)
            return self._decorate(request, response, request_id, started)
        finally:
            request_id_var.reset(token)

    # -- gates -----------------------------------------------------------
    def _gate(self, request: Request) -> Response | None:
        # CORS preflight is answered by CORSMiddleware before reaching us.
        if request.method == "OPTIONS":
            return None
        if not self._authorized(request):
            return JSONResponse(
                {"detail": "invalid or missing API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if self._rate_limited(request):
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return None

    def _authorized(self, request: Request) -> bool:
        if not self.api_key:
            return True
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/health"):
            return True
        provided = request.headers.get("X-API-Key", "")
        if not provided:
            header = request.headers.get("Authorization", "")
            if header.lower().startswith("bearer "):
                provided = header[7:].strip()
        try:
            return bool(provided) and hmac.compare_digest(provided, self.api_key)
        except TypeError:  # non-ASCII input can never match an ASCII key
            return False

    def _rate_limited(self, request: Request) -> bool:
        if self.rate_limit <= 0 or request.url.path.startswith("/health"):
            return False
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._hits.setdefault(client, [])
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) >= self.rate_limit:
                return True
            bucket.append(now)
            if len(self._hits) > 4096:  # keep the table bounded under abuse
                for key in [key for key, value in self._hits.items() if not value]:
                    del self._hits[key]
        return False

    # -- response --------------------------------------------------------
    def _decorate(
        self,
        request: Request,
        response: Response,
        request_id: str,
        started: float,
    ) -> Response:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if self.log_requests:
            logger.info(
                "%s %s -> %s in %.1f ms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response
