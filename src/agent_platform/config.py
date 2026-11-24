"""Environment-driven configuration for the orchestrator platform.

No secret is ever hard-coded here.  All provider keys are read from the
environment or from a local ``.env`` file that is *not* committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

try:  # optional convenience; the platform never requires a .env file
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    pass

REPO_MATCHER = Path(
    os.getenv("REPO_MATCHER_PATH", str(PROJECT_ROOT))
).expanduser().resolve()
REPO_DISCOVERY = Path(
    os.getenv("REPO_DISCOVERY_PATH", str(PROJECT_ROOT))
).expanduser().resolve()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = "Scholarship Orchestrator"
    app_version: str = "1.0.0"

    # --- LLM provider (optional). When empty/offline, deterministic mode is used.
    llm_provider: str = os.getenv("LLM_PROVIDER", "").strip().lower()
    llm_api_key: str = os.getenv("LLM_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").strip()
    llm_timeout: float = _env_float("LLM_TIMEOUT", 25.0)
    llm_connect_timeout: float = _env_float("LLM_CONNECT_TIMEOUT", 5.0)
    llm_turn_budget_seconds: float = _env_float("LLM_TURN_BUDGET_SECONDS", 30.0)
    llm_fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 3)
    offline: bool = _env_bool("AGENT_OFFLINE", False)

    # --- Agent engine. "graph" = LangGraph supervisor + tools; "legacy" = the
    # deterministic rule-based orchestrator.  The graph automatically degrades to
    # the deterministic engine when no provider is configured.
    agent_engine: str = os.getenv("AGENT_ENGINE", "graph").strip().lower()
    graph_max_tool_rounds: int = _env_int("GRAPH_MAX_TOOL_ROUNDS", 6)

    # --- Data locations (absolute, independent from the current directory)
    grants_db_path: Path = Path(
        os.getenv("GRANTS_DB_PATH", str(PROJECT_ROOT / "grants.db"))
    ).expanduser().resolve()
    sessions_db_path: Path = Path(
        os.getenv("SESSIONS_DB_PATH", str(PACKAGE_DIR / "data" / "sessions.db"))
    ).expanduser().resolve()

    # --- Optional external search
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()
    serper_api_key: str = os.getenv("SERPER_API_KEY", "").strip()

    # --- Production controls (all disabled by default so local/offline use and
    # the test-suite are unchanged).  Turn them on before exposing the service.
    api_key: str = os.getenv("API_KEY", "").strip()
    rate_limit_per_minute: int = _env_int("RATE_LIMIT_PER_MINUTE", 0)
    strict_session_ids: bool = _env_bool("STRICT_SESSION_IDS", False)

    # --- Session retention: turn transcripts are purged after this many days.
    session_ttl_days: int = _env_int("SESSION_TTL_DAYS", 30)
    maintenance_interval_seconds: int = _env_int("MAINTENANCE_INTERVAL_SECONDS", 3600)

    # --- Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    log_json: bool = _env_bool("LOG_JSON", False)

    # --- Behaviour
    top_grants_limit: int = _env_int("TOP_GRANTS_LIMIT", 10)
    max_message_chars: int = _env_int("MAX_MESSAGE_CHARS", 4000)
    history_turns: int = _env_int("HISTORY_TURNS", 12)
    external_search_threshold: float = _env_float("EXTERNAL_SEARCH_THRESHOLD", 40.0)
    external_search_min_fit: float = _env_float("EXTERNAL_SEARCH_MIN_FIT", 60.0)
    external_search_timeout_seconds: float = _env_float(
        "EXTERNAL_SEARCH_TIMEOUT_SECONDS", 75.0
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:5174,http://localhost:3000,"
                "http://127.0.0.1:5173,http://127.0.0.1:8000,http://127.0.0.1:8001",
            ).split(",")
            if origin.strip()
        )
    )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key) and not self.offline

    @property
    def external_search_configured(self) -> bool:
        return bool(self.tavily_api_key or self.serper_api_key)


settings = Settings()

# Repositories are imported relatively; refuse to run silently from a wrong tree.
MISSING_REPOS: list[str] = [
    str(path.name)
    for path in (REPO_MATCHER, REPO_DISCOVERY)
    if not path.is_dir()
]
