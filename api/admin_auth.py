"""Admin-key guard for the discovery API.

The discovery app's write and system endpoints had no authentication.  When
``ADMIN_API_KEY`` is set in the environment, this middleware requires the
``X-Admin-Key`` header (constant-time comparison) on:

* every method under ``/api/system/*`` and ``/api/agents/*``
* ``POST /api/grants/{id}/publish``
* ``GET /api/grants`` (the full list)

Public routes — including ``GET /api/grants/top10`` and the SPA — are never
challenged.  ``OPTIONS`` requests pass through so CORS preflight keeps working.
When ``ADMIN_API_KEY`` is unset the middleware allows everything and logs one
warning at startup.
"""

from __future__ import annotations

import logging
import os
import re
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_GRANT_PUBLISH = re.compile(r"^/api/grants/[^/]+/publish$")

_warned_missing_key = False


def is_admin_route(method: str, path: str) -> bool:
    """Return True when ``method path`` must carry the admin key."""
    method = method.upper()
    normalized = path.rstrip("/") or "/"
    if normalized == "/api/system" or normalized.startswith("/api/system/"):
        return True
    if normalized == "/api/agents" or normalized.startswith("/api/agents/"):
        return True
    if method == "POST" and _GRANT_PUBLISH.fullmatch(normalized):
        return True
    if method == "GET" and normalized == "/api/grants":
        return True
    return False


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """Enforce ``X-Admin-Key`` on administrative discovery endpoints."""

    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        global _warned_missing_key
        if not _admin_key() and not _warned_missing_key:
            logger.warning(
                "ADMIN_API_KEY is not set: administrative endpoints are "
                "unauthenticated. Set it before exposing the service."
            )
            _warned_missing_key = True

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        # Read the key at request time so tests and key rotation work without
        # rebuilding the ASGI stack.
        expected = _admin_key()
        if not expected:
            return await call_next(request)
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        if not is_admin_route(request.method, request.url.path):
            return await call_next(request)
        provided = request.headers.get("X-Admin-Key", "")
        if not secrets.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        ):
            return JSONResponse(
                {"message": "Admin API key required."}, status_code=401
            )
        return await call_next(request)


def _admin_key() -> str:
    return os.getenv("ADMIN_API_KEY", "").strip()
