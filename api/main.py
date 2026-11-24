"""FastAPI HTTP boundary for Autonomous Discovery & Web Research Sub-Agent.

Follows ASAAS Backend Foundation patterns:
- Lifespan-managed database initialization.
- Standardized {"message": "..."} error responses for the ASAAS frontend.
- Blocking network/agent operations offloaded cleanly via asyncio.to_thread.
- Read-only system readiness status without exposing secrets.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from starlette.exceptions import HTTPException as StarletteHTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.admin_auth import AdminAuthMiddleware
from api.schemas import (
    ExternalSearchRequest,
    GrantCountsResponse,
    GrantListResponse,
    GrantResponse,
    IngestRequest,
    SystemStatusResponse,
    TelegramProcessRequest,
    UpdateKeysRequest,
    UpdateLogResponse,
)
from core.config import ENV_PATH, settings
from core.db import get_db, init_db
from core.models import Grant, GrantStatus, UpdateLog
from core.repository import get_grant_by_id, get_grant_by_url, get_top_10_grants, publish
from external_search_sub_agent import search_external_scholarships
from ingestion_sub_agent import ingest
from integrations.telegram_watcher import (
    extract_urls_from_text,
    filter_official_urls,
    process_telegram_posts,
)
from monitor_service import run_monitor_cycle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Auto-seed verified Top 10 grants on fresh cloud deployments (empty DB)
    with get_db() as db:
        if db.query(Grant).count() == 0:
            from seed.seed_db import seed_database
            logger.info("Empty database detected on startup — seeding verified Top 10 grants...")
            seed_database(publish_all=True)
    yield


app = FastAPI(
    title="Autonomous Discovery & Web Research Sub-Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Outermost layer: admin checks run before CORS, while OPTIONS passes through.
app.add_middleware(AdminAuthMiddleware)


# ---------------------------------------------------------------------------
# ASAAS Standardized Error Handlers ({"message": "..."})
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        {"message": str(error.detail)},
        status_code=error.status_code,
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"][1:])
    msg = f"{field}: {first['msg']}" if field else first["msg"]
    return JSONResponse({"message": msg}, status_code=422)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_counts(db) -> GrantCountsResponse:
    all_grants = db.query(Grant).all()
    published_c = sum(1 for g in all_grants if g.status == GrantStatus.PUBLISHED)
    draft_c = sum(1 for g in all_grants if g.status == GrantStatus.DRAFT)
    expired_c = sum(1 for g in all_grants if g.status == GrantStatus.EXPIRED)
    logs_c = db.query(UpdateLog).count()
    return GrantCountsResponse(
        total=len(all_grants),
        published=published_c,
        draft=draft_c,
        expired=expired_c,
        update_logs=logs_c,
    )


# ---------------------------------------------------------------------------
# System Status & Manual API Keys Configuration
# ---------------------------------------------------------------------------

def _mask_key(value: Optional[str]) -> Optional[str]:
    """Return a safe masked preview (never the full secret key)."""
    if not value or not value.strip():
        return None
    v = value.strip()
    if len(v) <= 8:
        return "••••"
    return f"{v[:4]}••••{v[-4:]}"


def _persist_env_key(key: str, value: str) -> None:
    """Safely persist a single key to local .env without touching git-tracked files."""
    from dotenv import set_key

    if not ENV_PATH.exists():
        ENV_PATH.write_text("", encoding="utf-8")
    set_key(str(ENV_PATH), key, value, quote_mode="never")


def _build_system_status() -> SystemStatusResponse:
    search_provider = None
    if settings.TAVILY_API_KEY:
        search_provider = "Tavily"
    elif settings.SERPER_API_KEY:
        search_provider = "Serper"

    with get_db() as db:
        counts = _get_counts(db)

    return SystemStatusResponse(
        llm_configured=bool(settings.LLM_API_KEY),
        llm_model=settings.LLM_MODEL,
        llm_key_hint=_mask_key(settings.LLM_API_KEY),
        search_configured=bool(search_provider),
        search_provider=search_provider,
        tavily_key_hint=_mask_key(settings.TAVILY_API_KEY),
        serper_key_hint=_mask_key(settings.SERPER_API_KEY),
        telegram_configured=bool(settings.TELEGRAM_API_ID and settings.TELEGRAM_API_HASH),
        telegram_channels=settings.TELEGRAM_CHANNELS_RAW or "",
        counts=counts,
    )


@app.get("/api/system/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    return _build_system_status()


@app.post("/api/system/keys", response_model=SystemStatusResponse)
def api_update_keys(payload: UpdateKeysRequest) -> SystemStatusResponse:
    """Manually configure API keys from the UI: applies immediately in memory and saves to .env."""
    if payload.llm_api_key and payload.llm_api_key.strip().startswith("tvly-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="مفتاح tvly- مخصّص لمحرك بحث Tavily وليس للذكاء الاصطناعي؛ ضعه في حقل Tavily وأدخل مفتاح Gemini أو Groq أو OpenRouter أو OpenAI في حقل LLM.",
        )

    field_map = {
        "llm_api_key": ("LLM_API_KEY", "LLM_API_KEY"),
        "llm_model": ("LLM_MODEL", "LLM_MODEL"),
        "tavily_api_key": ("TAVILY_API_KEY", "TAVILY_API_KEY"),
        "serper_api_key": ("SERPER_API_KEY", "SERPER_API_KEY"),
        "telegram_api_id": ("TELEGRAM_API_ID", "TELEGRAM_API_ID"),
        "telegram_api_hash": ("TELEGRAM_API_HASH", "TELEGRAM_API_HASH"),
        "telegram_channels": ("TELEGRAM_CHANNELS_RAW", "TELEGRAM_CHANNELS"),
    }

    for field_name, (attr_name, env_key) in field_map.items():
        val = getattr(payload, field_name)
        if val is None:
            continue
        cleaned = val.strip()
        # Allow empty string to clear a key (except llm_model which falls back to default)
        if field_name == "llm_model" and not cleaned:
            continue
        setattr(settings, attr_name, cleaned or None if field_name != "telegram_channels" else cleaned)
        _persist_env_key(env_key, cleaned)

    return _build_system_status()


# ---------------------------------------------------------------------------
# Task 1: Top 10 & Grant Management
# ---------------------------------------------------------------------------

@app.get("/api/grants/top10", response_model=List[GrantResponse])
def api_get_top10() -> List[GrantResponse]:
    with get_db() as db:
        grants = get_top_10_grants(db, limit=10)
        return [GrantResponse(**g.to_dict()) for g in grants]


@app.get("/api/grants", response_model=GrantListResponse)
def api_list_grants(
    status_filter: Optional[str] = Query(default=None, alias="status"),
) -> GrantListResponse:
    with get_db() as db:
        query = db.query(Grant)
        if status_filter and status_filter in {"draft", "published", "expired"}:
            query = query.filter(Grant.status == GrantStatus(status_filter))
        grants = query.order_by(Grant.updated_at.desc()).all()
        counts = _get_counts(db)
        return GrantListResponse(
            grants=[GrantResponse(**g.to_dict()) for g in grants],
            counts=counts,
        )


@app.get("/api/grants/{grant_id}", response_model=GrantResponse)
def api_get_grant(grant_id: int) -> GrantResponse:
    with get_db() as db:
        grant = get_grant_by_id(db, grant_id)
        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="المنحة غير موجودة في قاعدة البيانات.",
            )
        return GrantResponse(**grant.to_dict())


@app.post("/api/grants/{grant_id}/publish", response_model=GrantResponse)
def api_publish_grant(grant_id: int) -> GrantResponse:
    with get_db() as db:
        grant = publish(db, grant_id)
        if not grant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="تعذر العثور على المنحة المطلوب نشرها.",
            )
        return GrantResponse(**grant.to_dict())


# ---------------------------------------------------------------------------
# Task 2: Ingestion Sub-Agent (Agentic Scraper)
# ---------------------------------------------------------------------------

@app.post("/api/agents/ingest", response_model=GrantResponse)
async def api_ingest_url(payload: IngestRequest) -> GrantResponse:
    if payload.llm_api_key and payload.llm_api_key.strip():
        cleaned_key = payload.llm_api_key.strip()
        settings.LLM_API_KEY = cleaned_key
        _persist_env_key("LLM_API_KEY", cleaned_key)
    if payload.llm_model and payload.llm_model.strip():
        cleaned_model = payload.llm_model.strip()
        settings.LLM_MODEL = cleaned_model
        _persist_env_key("LLM_MODEL", cleaned_model)

    if not settings.LLM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="يرجى إدخال مفتاح الذكاء الاصطناعي (LLM API Key) لتفعيل الاستخلاص.",
        )
    try:
        grant = await asyncio.to_thread(ingest, payload.official_url.strip())
        return GrantResponse(**grant.to_dict())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            msg = "مفتاح الذكاء الاصطناعي غير صالح أو مرفوض من المزوّد؛ تأكد من إدخال مفتاح Gemini أو Groq أو OpenRouter أو OpenAI صحيح."
        elif code == 429:
            msg = "تم تجاوز الحد المجاني لمزوّد الذكاء الاصطناعي مؤقتاً؛ حاول بعد دقيقة أو جرّب نموذجاً آخر."
        elif code == 404:
            msg = "النموذج المحدّد غير متاح لهذا المفتاح؛ اختر نموذجاً متوافقاً مثل gemini-2.0-flash."
        else:
            msg = f"رفض المزوّد الطلب (رمز {code})؛ تحقق من الرابط أو المفتاح."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", payload.official_url)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="تعذّر فتح أو استخلاص الصفحة الرسمية؛ تأكد من أن الرابط يعمل ومفتاح الذكاء الاصطناعي صحيح.",
        )


# ---------------------------------------------------------------------------
# Task 4: External Deep Search Sub-Agent (ReAct)
# ---------------------------------------------------------------------------

@app.post("/api/agents/search")
async def api_external_search(payload: ExternalSearchRequest) -> Dict[str, Any]:
    if payload.tavily_api_key and payload.tavily_api_key.strip():
        cleaned_tavily = payload.tavily_api_key.strip()
        settings.TAVILY_API_KEY = cleaned_tavily
        _persist_env_key("TAVILY_API_KEY", cleaned_tavily)
    if payload.serper_api_key and payload.serper_api_key.strip():
        cleaned_serper = payload.serper_api_key.strip()
        settings.SERPER_API_KEY = cleaned_serper
        _persist_env_key("SERPER_API_KEY", cleaned_serper)

    if not (settings.TAVILY_API_KEY or settings.SERPER_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="يرجى إدخال مفتاح محرك البحث (Tavily أو Serper) لتفعيل البحث العميق.",
        )
    try:
        result = await asyncio.to_thread(
            search_external_scholarships,
            major=payload.major.strip(),
            degree_level=payload.degree_level.strip(),
            target_country=payload.target_country.strip() if payload.target_country else None,
            max_steps=payload.max_steps,
        )
        return result.model_dump()
    except Exception as exc:
        logger.exception("External search failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"فشل وكيل البحث الخارجي: {exc}",
        )


# ---------------------------------------------------------------------------
# Task 9: Monitor Service & Audit Logs
# ---------------------------------------------------------------------------

@app.post("/api/agents/monitor/run")
async def api_run_monitor() -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(run_monitor_cycle)
    except Exception as exc:
        logger.exception("Monitor run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"تعذر إتمام دورة المراقبة: {exc}",
        )


@app.get("/api/agents/monitor/logs", response_model=List[UpdateLogResponse])
def api_monitor_logs(limit: int = Query(default=50, ge=1, le=200)) -> List[UpdateLogResponse]:
    with get_db() as db:
        rows = (
            db.query(UpdateLog)
            .order_by(UpdateLog.detected_at.desc())
            .limit(limit)
            .all()
        )
        results: List[UpdateLogResponse] = []
        for row in rows:
            results.append(
                UpdateLogResponse(
                    id=row.id,
                    grant_id=row.grant_id,
                    grant_title=row.grant.title if row.grant else None,
                    field_name=row.field_name,
                    old_value=row.old_value,
                    new_value=row.new_value,
                    detected_at=row.detected_at.isoformat() if row.detected_at else None,
                    notified=row.notified,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Phase 8: Telegram Watcher (Creative Task)
# ---------------------------------------------------------------------------

@app.post("/api/agents/telegram/process")
async def api_telegram_process(payload: TelegramProcessRequest) -> Dict[str, Any]:
    if payload.dry_run or not settings.LLM_API_KEY:
        # Dry-run analysis: extract, filter official domains, and check DB duplicates without LLM calls
        def _dry_run() -> Dict[str, Any]:
            all_urls: List[str] = []
            seen = set()
            for post in payload.posts:
                for url in extract_urls_from_text(post):
                    if url not in seen:
                        seen.add(url)
                        all_urls.append(url)
            trusted, rejected = filter_official_urls(all_urls)
            skipped_existing: List[str] = []
            ready_for_ingest: List[str] = []
            with get_db() as db:
                for u in trusted:
                    if get_grant_by_url(db, u) is not None:
                        skipped_existing.append(u)
                    else:
                        ready_for_ingest.append(u)
            return {
                "mode": "dry_run",
                "posts_processed": len(payload.posts),
                "urls_extracted": len(all_urls),
                "trusted_urls_found": len(trusted),
                "ready_for_ingest": ready_for_ingest,
                "drafts_created": [],
                "skipped_existing": skipped_existing,
                "skipped_untrusted": [{"url": u, "reason": r} for u, r in rejected],
                "errors": [],
                "note": (
                    "وضع الفحص الآمن: تم استخراج الروابط الرسمية وفلترة المجمّعات والمكررات. "
                    + ("" if settings.LLM_API_KEY else "أدخل مفتاح الذكاء الاصطناعي في حقل الاستخلاص لتحويل الروابط الجديدة تلقائياً إلى مسودات.")
                ),
            }

        return await asyncio.to_thread(_dry_run)

    return await asyncio.to_thread(process_telegram_posts, payload.posts)


# ---------------------------------------------------------------------------
# Optional Static Frontend Serving (when frontend/dist is built)
# ---------------------------------------------------------------------------

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
