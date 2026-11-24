# Scholarship Orchestrator platform

A stateful, multi-agent scholarship-advising service for Yemeni and MENA
students. It holds a conversation in Arabic or English, maintains a student
profile, matches it against the verified grants in this repository, builds
application plans, drafts CVs / motivation letters / recommendation briefs
through a reflection loop, and advises on competitive gaps.

This document describes the `src/agent_platform` service that now lives in this
repository. It runs next to the original discovery/admin API and shares the
same `grants.db`.

---

## Repository layout

```
autonomous-discovery-web-research-subagent/
├── api/                    Discovery/admin FastAPI app (port 8001)
├── core/                   Role 2 database, models, repository, config
├── utils/                  Role 2 URL validation, extraction, evidence checks
├── seed/                   Verified top-10 seed data
├── integrations/           Telegram watcher
├── frontend/               ASAAS UI
├── ai/scholarship_agent/   Role 3 matcher, planner, schemas
├── src/agent_platform/     Orchestrator platform (port 8010)
│   ├── api.py              FastAPI boundary, routes, probes
│   ├── main.py             CLI entry point
│   ├── graph/              LangGraph supervisor + tools
│   ├── orchestrator.py     Deterministic fallback engine
│   ├── matching.py         Weighted scorer over published grants
│   ├── planning.py         Application-plan builder
│   ├── document_agent.py   Draft → critique → revise reflection loop
│   ├── advisor.py          Profile-strengthening report
│   ├── knowledge.py        Read model over grants.db
│   ├── memory.py           Sessions, transcripts, artifacts
│   ├── concurrency.py      Bounded-wait helper for external search
│   ├── guardrails.py       Input/output safety checks
│   └── data/               Runtime session DB (created automatically)
├── tests/                  Role 2 tests + admin-auth tests
├── tests/agent_platform/   Platform test suite
├── scripts/llm_smoke.py    Manual real-LLM smoke test
├── docs/PLATFORM.md        This document
└── docker-compose.yml      discovery (8001) + platform (8010)
```

The platform imports Role 2 (`core`, `utils`, `seed`, the root agents) and
Role 3 (`ai.scholarship_agent`) directly from the repository root. There is no
vendored copy of either project.

---

## Shared data

Both services read the same SQLite file:

| Service | Setting | Default |
|---|---|---|
| Platform | `GRANTS_DB_PATH` | `<repo>/grants.db` |
| Role 2 API | `DATABASE_URL` | `sqlite:///grants.db` (run from the repo root) |

The platform's session store is separate (`SESSIONS_DB_PATH`, default
`src/agent_platform/data/sessions.db`). Grants are seeded from
`seed/seed_db.py`; the platform only exposes `PUBLISHED` grants.

---

## Windows quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-deploy.txt pytest pytest-asyncio

# Seed/publish the verified top 10 (DRAFT by default)
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python seed\seed_db.py --publish

# Optional: create .env and fill values
Copy-Item .env.example .env

# Platform on 8010
$env:PYTHONPATH=".;src"
.\.venv\Scripts\python -m agent_platform.main --port 8010

# Discovery/admin API on 8001 (separate terminal)
.\.venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Interactive docs: <http://127.0.0.1:8010/docs> and
<http://127.0.0.1:8001/docs>.

Offline smoke test of the platform:

```powershell
$env:AGENT_OFFLINE="1"; $env:PYTHONPATH=".;src"
.\.venv\Scripts\python -m pytest -q
```

Manual real-LLM check (never prints the key):

```powershell
$env:LLM_API_KEY="..."; $env:LLM_MODEL="gemini-2.5-flash"
.\.venv\Scripts\python scripts\llm_smoke.py
```

---

## Configuration

All platform settings are optional; `.env` is loaded from the repository root
and real environment variables win. `LLM_MODEL` must be set explicitly for the
key in use — the old `gemini-1.5-flash` default may be retired by the provider.

### Model / agent

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | auto | `gemini`, `openai`, `openrouter`, `groq`, `custom`; auto-detected from the key prefix |
| `LLM_API_KEY` | empty | Shared with Role 2. Empty ⇒ deterministic engine |
| `LLM_MODEL` | provider default | Must support tool calling for the graph engine |
| `LLM_BASE_URL` | provider default | Gateway/self-hosted override |
| `LLM_TIMEOUT` | `25` | Seconds per provider call |
| `LLM_MAX_RETRIES` | `3` | Retries on HTTP 429/500/503/timeout, with 1s/2s/4s backoff + jitter |
| `LLM_FALLBACK_MODEL` | empty | Optional second model tried once before the deterministic engine |
| `AGENT_ENGINE` | `graph` | `graph` = LLM supervisor; `legacy` = deterministic only |
| `AGENT_OFFLINE` | `0` | `1` forces deterministic even with a key |
| `GRAPH_MAX_TOOL_ROUNDS` | `6` | Tool-calling budget |

### Data

| Variable | Default | Meaning |
|---|---|---|
| `GRANTS_DB_PATH` | `<repo>/grants.db` | Shared verified-grant database |
| `SESSIONS_DB_PATH` | `src/agent_platform/data/sessions.db` | Session store |
| `SESSION_TTL_DAYS` | `30` | Purge age for transcripts |
| `MAINTENANCE_INTERVAL_SECONDS` | `3600` | Purge interval |

### Search and matching

| Variable | Default | Meaning |
|---|---|---|
| `TAVILY_API_KEY` / `SERPER_API_KEY` | empty | Enable external discovery |
| `EXTERNAL_SEARCH_THRESHOLD` | `40` | Trigger discovery when the top match % is below this |
| `EXTERNAL_SEARCH_MIN_FIT` | `60` | Trigger discovery when the best semantic fit is below this |
| `EXTERNAL_SEARCH_TIMEOUT_SECONDS` | `75` | Hard wait for the discovery call; timeout degrades politely |

### Server / production controls

| Variable | Default | Meaning |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8010` | Platform bind address |
| `RELOAD` | `0` | Dev auto-reload |
| `API_KEY` | empty | Shared service key for `/api/*` (`X-API-Key` / Bearer) |
| `RATE_LIMIT_PER_MINUTE` | `0` | Per-IP request budget; `0` disables |
| `STRICT_SESSION_IDS` | `0` | `1` rejects client-chosen ids outside `^[A-Za-z0-9_-]{22,64}$` (422) |
| `ADMIN_API_KEY` | empty | Enables `X-Admin-Key` on Role 2 admin/write endpoints |
| `CORS_ORIGINS` | localhost origins | Comma-separated allowed origins |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `0` | Logging |

`ADMIN_API_KEY` protects all `/api/system/*` and `/api/agents/*` routes,
`POST /api/grants/{id}/publish`, and `GET /api/grants` in the discovery app.
`GET /api/grants/top10` and the SPA stay public. When it is set, the admin UI
needs a field for the key (follow-up on the Role 4 UI).

---

## API overview (platform, 8010)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Liveness, no I/O |
| `GET` | `/health/ready` | Readiness; 503 when grants/repos are unusable |
| `POST` | `/api/session` | Issue an unguessable session id |
| `GET` / `PUT` | `/api/profile/{session_id}` | Read / merge the stored profile |
| `POST` | `/api/chat` | One conversational turn |
| `POST` | `/api/chat/stream` | Same turn as Server-Sent Events |
| `POST` | `/api/documents` | Document reflection loop |
| `POST` | `/api/match` | Ranked grant matching |
| `POST` | `/api/plan` | Application plan |
| `POST` | `/api/advisor` | Profile-strengthening report |
| `POST` | `/api/session/{session_id}/reset` | Delete a session |
| `GET` | `/api/grants/top10` | Published grants |

When `STRICT_SESSION_IDS=1`, every route that carries a `session_id` rejects
non-matching ids with HTTP 422 `{"message": ...}`.

### Discovery/admin API (8001)

`uvicorn api.main:app` serves the original grants, ingestion, search, monitor,
and Telegram endpoints, plus the built SPA. Admin routes are listed in the
configuration section above.

---

## Testing

```powershell
$env:AGENT_OFFLINE="1"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pytest tests\agent_platform -q
```

- 113 platform tests, 39 Role 2 tests, 6 admin-auth tests (158 total).
- Fully offline: no key, no network.
- `tests/conftest.py` gives `tests/test_seed.py` a temporary SQLite database, so
  the suite never mutates the real `grants.db`.

---

## Deployment

`docker-compose.yml` builds one image and runs both services from it:

```powershell
Copy-Item .env.example .env   # fill the values you need
docker compose up --build
```

- `discovery` → `uvicorn api.main:app --host 0.0.0.0 --port 8001`
- `platform` → `python -m agent_platform.main --host 0.0.0.0 --port 8010`
- Both mount the named volume `platform-data` at `/app/data`, with
  `DATABASE_URL=sqlite:////app/data/grants.db`,
  `GRANTS_DB_PATH=/app/data/grants.db`, `SESSIONS_DB_PATH=/app/data/sessions.db`.
- `init: true` forwards SIGTERM for a clean SQLite shutdown.

`render.yaml` and `railway.json` still deploy the discovery app only.

---

## Known limits

- **Sparse grant requirements.** The seeded grants lack structured GPA and
  major requirements, so GPA and major barely affect the ranking; match quality
  rests mostly on the semantic-alignment score (plus degree, nationality and
  language checks).
- **`API_KEY` is a shared service key**, not per-user authentication. There is
  no user identity or session ownership check.
- **SQLite is a single-writer store.** Run one process per database file;
  horizontal scaling requires PostgreSQL.
- **Production settings:** set `STRICT_SESSION_IDS=1` and `ADMIN_API_KEY`, and
  give the platform a long random `API_KEY`.
- Documents and `.env` are plaintext; use a secret manager and encrypt at rest.
- `/docs` and `/openapi.json` stay public even when `API_KEY` is set.
- No CI pipeline or dependency lockfile is included.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: agent_platform` | Set `$env:PYTHONPATH=".;src"` or run through pytest's `pythonpath` |
| `/health/ready` 503 | Seed/publish grants: `python seed\seed_db.py --publish` |
| `/api/grants/top10` returns fewer than 10 | Grants are DRAFT; run the seed with `--publish` |
| `agent_engine: legacy` with a key set | Model lacks tool calling, or `AGENT_OFFLINE=1`, or `AGENT_ENGINE=legacy` |
| `database is locked` | More than one writer; run a single process |
| Admin UI gets 401 after setting `ADMIN_API_KEY` | The UI does not send `X-Admin-Key` yet (Role 4 follow-up) |
