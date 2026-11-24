"""Shared fixtures for the orchestrator test-suite.

The suite is fully offline: no LLM key and no search key is required, and
every test gets its own temporary SQLite databases so tests never touch the
real grants or session data.
"""

from __future__ import annotations

import os

# Must be set before ``agent_platform.config`` is imported anywhere.
os.environ.setdefault("AGENT_OFFLINE", "1")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("TAVILY_API_KEY", "")
os.environ.setdefault("SERPER_API_KEY", "")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest
from fastapi.testclient import TestClient

from agent_platform.api import Runtime, create_app
from agent_platform.knowledge import GrantRepository
from agent_platform.schemas import DegreeLevel, StudentProfile
from agent_platform import llm, matching


@pytest.fixture(autouse=True)
def _reset_llm_runtime_state() -> object:
    """Keep the process-wide circuit breaker and caches isolated per test."""
    llm.reset_runtime_state()
    matching.clear_semantic_cache()
    yield
    llm.reset_runtime_state()
    matching.clear_semantic_cache()


@pytest.fixture()
def runtime(tmp_path: Path) -> Runtime:
    service = Runtime(
        memory_path=tmp_path / "sessions.db",
        grants_path=tmp_path / "grants.db",
    )
    try:
        yield service
    finally:
        service.close()


@pytest.fixture()
def client(runtime: Runtime) -> TestClient:
    return TestClient(create_app(runtime))


@pytest.fixture()
def repository(tmp_path: Path) -> GrantRepository:
    return GrantRepository(tmp_path / "grants.db")


@pytest.fixture()
def complete_profile() -> StudentProfile:
    return StudentProfile(
        student_id="test-student",
        name="أحمد اليمني",
        email="ahmed@example.com",
        gpa=3.6,
        major="علوم الحاسوب",
        degree_level=DegreeLevel.BACHELOR,
        target_degree=DegreeLevel.MASTER,
        nationality="Yemeni",
        country_of_residence="Yemen",
        research_interests=["Artificial Intelligence"],
        target_fields=["Computer Science"],
        publications=0,
    )


@pytest.fixture()
def complete_profile_payload() -> dict:
    return {
        "name": "أحمد اليمني",
        "gpa": 3.6,
        "major": "علوم الحاسوب",
        "degree_level": "bachelor",
        "target_degree": "master",
        "nationality": "Yemeni",
    }
