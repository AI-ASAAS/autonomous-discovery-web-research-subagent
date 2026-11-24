"""FastAPI boundary for the scholarship orchestrator.

The HTTP layer is intentionally thin: every decision (routing, matching,
reflection, guardrails) lives in the orchestrator and its sub-agents.  The
endpoints only validate input, resolve a session profile and serialize the
typed results for the existing frontends.

Endpoints
---------
``GET  /health``                      service + data readiness (always 200)
``GET  /health/live``                 liveness probe (no I/O)
``GET  /health/ready``                readiness probe (503 when unusable)
``GET  /api/grants/top10``            verified grants from the discovery DB
``POST /api/session``                 issue an unguessable session id
``GET  /api/profile/{session_id}``    stored profile + completeness
``PUT  /api/profile/{session_id}``    replace/merge the stored profile
``POST /api/chat``                    one conversational turn
``POST /api/chat/stream``             same turn streamed as Server-Sent Events
``POST /api/documents``               reflection document generation
``POST /api/match``                   ranked grant matching
``POST /api/plan``                    application plan for the top grants
``POST /api/advisor``                 profile strengthening report
``POST /api/session/{id}/reset``      clear a session (memory + artifacts)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import MISSING_REPOS, settings
from .observability import RequestContextMiddleware, configure_logging
from .repos import repos_present, setup_repo_paths

# The adapters import modules from both repositories at import time, so the
# repository roots (and the absolute grants DB URL) must be registered first.
setup_repo_paths()

from . import llm  # noqa: E402
from .advisor import build_advisor_report  # noqa: E402
from .document_agent import DocumentAgent  # noqa: E402
from .extraction import merge_profile  # noqa: E402
from .graph.service import AgentService  # noqa: E402
from .knowledge import GrantRepository  # noqa: E402
from .matching import match_grants  # noqa: E402
from .memory import MemoryStore  # noqa: E402
from .orchestrator import Orchestrator  # noqa: E402
from .planning import build_plan, plan_summary  # noqa: E402
from .schemas import (  # noqa: E402
    AdvisorRequest,
    ChatRequest,
    ChatResponse,
    DocumentRequest,
    DocumentResult,
    GrantCard,
    HealthResponse,
    MatchReport,
    MatchRequest,
    PlanRequest,
    StudentProfile,
)

logger = logging.getLogger(__name__)

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22,64}$")


class InvalidSessionIdError(Exception):
    """Raised when a caller-supplied session id fails strict validation."""


# ---------------------------------------------------------------------------
# Runtime (single shared instance per process, injectable in tests)
# ---------------------------------------------------------------------------


class Runtime:
    def __init__(
        self,
        *,
        memory_path: Path | str | None = None,
        grants_path: Path | str | None = None,
    ) -> None:
        setup_repo_paths()
        self.memory = MemoryStore(memory_path)
        self.repository = GrantRepository(grants_path)
        self.documents = DocumentAgent()
        self.orchestrator = Orchestrator(
            memory=self.memory,
            repository=self.repository,
            documents=self.documents,
        )
        # The agent engine wraps the orchestrator: graph when an LLM is
        # configured, deterministic legacy otherwise.
        self.agent = AgentService(
            memory=self.memory,
            repository=self.repository,
            documents=self.documents,
            orchestrator=self.orchestrator,
        )

    def close(self) -> None:
        self.memory.close()

    def maintenance(self) -> dict[str, int]:
        """Age out stale sessions; run at startup and on a fixed interval."""
        removed = self.memory.purge_older_than(settings.session_ttl_days)
        if any(removed.values()):
            logger.info("Purged expired sessions: %s", removed)
        return removed


_runtime_lock = threading.Lock()
_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime()
        return _runtime


def reset_runtime_for_tests(runtime: Runtime | None = None) -> None:
    """Replace the shared runtime (used by the test-suite only)."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None and _runtime is not runtime:
            _runtime.close()
        _runtime = runtime


def _release_runtime(runtime: Runtime | None) -> None:
    """Close an app-owned runtime without masking the shutdown path."""
    if runtime is None:
        return
    try:
        runtime.close()
    except Exception:  # pragma: no cover - close must never break shutdown
        logger.exception("Error while closing the runtime")
    global _runtime
    with _runtime_lock:
        if _runtime is runtime:
            _runtime = None
    logger.info("Runtime closed")


async def _maintenance_loop(runtime: Runtime) -> None:
    interval = max(60, settings.maintenance_interval_seconds)
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(runtime.maintenance)
        except Exception:  # pragma: no cover - background safety net
            logger.exception("Session maintenance cycle failed")


def _log_startup(runtime: Runtime) -> None:
    """One loud, secret-free summary of the effective configuration."""
    logger.info(
        "Starting %s %s (engine=%s, provider=%s, grants_db=%s, sessions_db=%s)",
        settings.app_name,
        settings.app_version,
        runtime.agent.engine,
        llm.provider_label(),
        settings.grants_db_path,
        settings.sessions_db_path,
    )
    if MISSING_REPOS:
        logger.error(
            "Repository sources are missing: %s. Grant features are degraded.",
            ", ".join(MISSING_REPOS),
        )
    if settings.agent_engine == "legacy":
        logger.warning("AGENT_ENGINE=legacy: the LLM supervisor/tool loop is off.")
    elif not runtime.agent.graph_enabled:
        logger.warning(
            "AGENT_ENGINE=%s but no usable LLM provider is configured; requests "
            "will use the deterministic engine. Set LLM_API_KEY in .env.",
            settings.agent_engine,
        )
    if not settings.api_key:
        logger.warning(
            "API_KEY is not set: every endpoint is unauthenticated. Set it before "
            "exposing the service publicly."
        )
    if settings.rate_limit_per_minute <= 0:
        logger.warning("RATE_LIMIT_PER_MINUTE=0: request rate limiting is disabled.")
    try:
        grants = runtime.repository.count()
    except Exception:  # pragma: no cover - unreadable database
        logger.exception(
            "Could not read the grants database at %s", settings.grants_db_path
        )
    else:
        if grants <= 0:
            logger.error(
                "The grants database is empty (%s). Run `make seed`.",
                settings.grants_db_path,
            )
    logger.info("Session store stats: %s", runtime.memory.stats())


def _health_payload(runtime: Runtime) -> HealthResponse:
    try:
        grants = runtime.repository.count()
    except Exception:  # pragma: no cover - unreadable database
        logger.exception("Could not count grants")
        grants = 0
    configured = llm.is_configured()
    available = not llm.circuit.is_open()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        mode="llm" if configured else "deterministic",
        llm_provider=llm.provider_label() if configured else "deterministic",
        agent_engine=runtime.agent.engine,
        agent_ready=runtime.agent.graph_enabled,
        external_search_configured=settings.external_search_configured,
        grants_available=grants,
        repos_present=repos_present(),
        llm_available=available,
        llm_retry_after_seconds=(
            0.0 if available else llm.circuit.retry_after_seconds()
        ),
        llm_config_error=llm.circuit.config_error(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_override(
    profile: StudentProfile, override: StudentProfile | None
) -> StudentProfile:
    """Merge the non-empty fields of an explicit profile into the session one."""
    if override is None:
        return profile
    patch = {
        key: value
        for key, value in override.model_dump().items()
        if key != "student_id" and value not in (None, "", [])
    }
    return merge_profile(profile, patch) if patch else profile


def _profile_for(
    runtime: Runtime, session_id: str, override: StudentProfile | None
) -> StudentProfile:
    profile = runtime.memory.load_profile(session_id)
    if override is None:
        return profile
    merged = _merge_override(profile, override)
    runtime.memory.save_profile(session_id, merged)
    return merged


def _require_complete(profile: StudentProfile) -> None:
    missing = profile.missing_required()
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "incomplete_profile",
                "message": "Complete the profile before running this operation.",
                "missing_fields": missing,
            },
        )


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _resolve_grant(
    runtime: Runtime, grant_id: str | None, grant_title: str | None
) -> GrantCard | None:
    if grant_id:
        card = runtime.repository.get(grant_id)
        if card is not None:
            return card
    if grant_title:
        card = runtime.repository.find_by_title(grant_title)
        if card is not None:
            return card
    return None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    runtime: Runtime | None = None,
    *,
    api_key: str | None = None,
    rate_limit_per_minute: int | None = None,
    strict_session_ids: bool | None = None,
    log_requests: bool = True,
) -> FastAPI:
    configure_logging(settings.log_level, settings.log_json)
    strict_ids = (
        settings.strict_session_ids
        if strict_session_ids is None
        else strict_session_ids
    )

    holder: dict[str, Any] = {"runtime": runtime, "owned": False}

    def rt() -> Runtime:
        if holder["runtime"] is None:
            holder["runtime"] = get_runtime()
            holder["owned"] = True
        return holder["runtime"]  # type: ignore[return-value]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service = rt()
        _log_startup(service)
        service.maintenance()
        task = asyncio.create_task(_maintenance_loop(service))
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if holder["owned"]:
                _release_runtime(holder["runtime"])

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Stateful multi-agent scholarship platform: conversational "
            "orchestrator, verified-grant matching, reflection documents and "
            "profile advisory."
        ),
        lifespan=lifespan,
    )
    # Middleware runs outermost-first: add the CORS layer last so preflight
    # requests are answered before the auth/rate-limit gate.
    app.add_middleware(
        RequestContextMiddleware,
        api_key=settings.api_key if api_key is None else api_key,
        rate_limit_per_minute=(
            settings.rate_limit_per_minute
            if rate_limit_per_minute is None
            else rate_limit_per_minute
        ),
        log_requests=log_requests,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    app.state.runtime_holder = holder

    @app.exception_handler(InvalidSessionIdError)
    async def _invalid_session_id(
        _: Request, exc: InvalidSessionIdError
    ) -> JSONResponse:
        return JSONResponse({"message": str(exc)}, status_code=422)

    def require_valid_session(session_id: str) -> None:
        if strict_ids and not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise InvalidSessionIdError(
                "Invalid session id. Request one from POST /api/session."
            )

    # -- meta -----------------------------------------------------------
    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "message": "API is running. Open /docs for interactive testing.",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "chat": "/api/chat",
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
        }

    @app.get("/health/live")
    def health_live() -> dict[str, str]:
        """Liveness: the process is up. Deliberately free of I/O."""
        return {"status": "ok"}

    @app.get("/health/ready", response_model=HealthResponse)
    def health_ready(response: Response) -> HealthResponse:
        """Readiness: dependencies usable. 503 makes load balancers hold traffic."""
        payload = _health_payload(rt())
        if payload.grants_available <= 0 or not payload.repos_present:
            response.status_code = 503
            payload.status = "degraded"
        return payload

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Backwards-compatible readiness summary (always HTTP 200)."""
        return _health_payload(rt())

    # -- grants ---------------------------------------------------------
    @app.get("/api/grants/top10")
    def top_grants(limit: int = 10) -> dict[str, Any]:
        service = rt()
        limit = max(1, min(limit, 50))
        cards = service.repository.list_published(limit=limit)
        return {
            "count": len(cards),
            "grants": [card.model_dump(mode="json") for card in cards],
        }

    # -- profile --------------------------------------------------------
    @app.post("/api/session")
    def create_session() -> dict[str, str]:
        """Issue a fresh, unguessable session id (recommended in production)."""
        return {"session_id": secrets.token_urlsafe(24)}

    @app.get("/api/profile/{session_id}")
    def get_profile(session_id: str) -> dict[str, Any]:
        require_valid_session(session_id)
        service = rt()
        profile = service.memory.load_profile(session_id)
        return {
            "session_id": session_id,
            "profile": profile.model_dump(mode="json"),
            "missing_fields": profile.missing_required(),
            "completeness": profile.completeness(),
            "message_count": service.memory.message_count(session_id),
        }

    @app.put("/api/profile/{session_id}")
    def put_profile(session_id: str, body: StudentProfile) -> dict[str, Any]:
        require_valid_session(session_id)
        service = rt()
        stored = service.memory.load_profile(session_id)
        profile = _merge_override(stored, body).model_copy(
            update={"student_id": session_id}
        )
        service.memory.save_profile(session_id, profile)
        return {
            "session_id": session_id,
            "profile": profile.model_dump(mode="json"),
            "missing_fields": profile.missing_required(),
            "completeness": profile.completeness(),
        }

    # -- conversation ---------------------------------------------------
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        require_valid_session(body.session_id)
        service = rt()
        return service.agent.chat(
            body.session_id,
            body.message,
            profile_override=body.profile,
        )

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        require_valid_session(body.session_id)
        service = rt()
        if body.profile is not None:
            _profile_for(service, body.session_id, body.profile)

        def events() -> Iterator[str]:
            try:
                for kind, payload in service.agent.stream(
                    body.session_id, body.message
                ):
                    if kind == "step":
                        yield _sse("step", payload.model_dump(mode="json"))
                    elif kind == "final":
                        yield _sse("final", payload.model_dump(mode="json"))
                    elif kind == "error":
                        yield _sse("error", {"message": str(payload)})
                yield _sse("end", {"session_id": body.session_id})
            except Exception:  # pragma: no cover - stream transport failure
                logger.exception("Streaming chat failed")
                yield _sse("error", {"message": "internal error"})
                yield _sse("end", {"session_id": body.session_id})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers=_sse_headers()
        )

    # -- documents ------------------------------------------------------
    @app.post("/api/documents", response_model=DocumentResult)
    def documents(body: DocumentRequest) -> DocumentResult:
        require_valid_session(body.session_id)
        service = rt()
        profile = _profile_for(service, body.session_id, body.profile)
        _require_complete(profile)

        grant = _resolve_grant(service, body.grant_id, body.grant_title)
        if grant is None:
            cards = service.repository.list_published(limit=50)
            report = match_grants(profile, cards, limit=1)
            if not report.results:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "grant_not_found",
                        "message": "No verified grant matched the request.",
                    },
                )
            grant = next(
                (card for card in cards if card.grant_id == report.results[0].grant_id),
                None,
            )
        if grant is None:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail={"error": "grant_not_found"})

        result = service.documents.generate(
            profile, grant, body.document_type, language=body.language
        )
        service.memory.save_artifact(
            body.session_id, "document", result.model_dump(mode="json")
        )
        return result

    # -- matching / planning / advisor ---------------------------------
    @app.post("/api/match", response_model=MatchReport)
    def match(body: MatchRequest) -> MatchReport:
        require_valid_session(body.session_id)
        service = rt()
        profile = _profile_for(service, body.session_id, body.profile)
        _require_complete(profile)
        cards = service.repository.list_published(limit=50)
        report = match_grants(profile, cards, limit=body.limit)
        service.memory.save_artifact(
            body.session_id, "match", report.model_dump(mode="json")
        )
        return report

    @app.post("/api/plan")
    def plan(body: PlanRequest) -> dict[str, Any]:
        require_valid_session(body.session_id)
        service = rt()
        profile = _profile_for(service, body.session_id, body.profile)
        _require_complete(profile)
        cards = service.repository.list_published(limit=50)
        report = match_grants(profile, cards, limit=10)
        items = build_plan(profile, report, cards, top_n=body.top_n)
        service.memory.save_artifact(
            body.session_id, "plan", {"summary": plan_summary(items), "plans": items}
        )
        return {
            "session_id": body.session_id,
            "summary": plan_summary(items),
            "plans": items,
            "matches": [result.model_dump(mode="json") for result in report.results],
        }

    @app.post("/api/advisor")
    def advisor(body: AdvisorRequest) -> dict[str, Any]:
        require_valid_session(body.session_id)
        service = rt()
        profile = _profile_for(service, body.session_id, body.profile)
        cards = service.repository.list_published(limit=50)
        report = None
        if not profile.missing_required():
            report = match_grants(profile, cards, limit=10)
        result = build_advisor_report(profile, report)
        service.memory.save_artifact(
            body.session_id, "advisor", result.model_dump(mode="json")
        )
        return {
            "session_id": body.session_id,
            "missing_fields": profile.missing_required(),
            "advisor": result.model_dump(mode="json"),
        }

    # -- session --------------------------------------------------------
    @app.post("/api/session/{session_id}/reset")
    def reset_session(session_id: str) -> dict[str, Any]:
        require_valid_session(session_id)
        service = rt()
        service.memory.reset(session_id)
        return {"session_id": session_id, "reset": True}

    return app


app = create_app()
