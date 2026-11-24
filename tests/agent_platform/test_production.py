"""Production plumbing: probes, auth, rate limiting, retention and shutdown.

These tests exercise behaviour that is invisible in the happy-path API tests:
the health-probe split, request correlation, the optional API key, the
in-process rate limiter, session retention and the lifespan ownership rules.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agent_platform.api as api_module
from agent_platform.api import Runtime, create_app
from agent_platform.memory import MemoryStore
from agent_platform.observability import JsonLogFormatter, RequestIdFilter
from agent_platform.schemas import StudentProfile


# ---------------------------------------------------------------------------
# Probes and headers
# ---------------------------------------------------------------------------


def test_liveness_is_cheap_and_io_free(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_dependencies(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["grants_available"] == 10
    assert body["repos_present"] is True


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    response = client.get("/health/live")
    request_id = response.headers.get("X-Request-ID")
    assert request_id and len(request_id) >= 8


def test_request_id_from_the_client_is_preserved(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_security_headers_are_always_present(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_json_log_formatter_emits_valid_json() -> None:
    record = logging.LogRecord(
        "demo", logging.INFO, __file__, 1, "hello %s", ("world",), None
    )
    RequestIdFilter().filter(record)
    payload = json.loads(JsonLogFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "-"


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def test_api_key_is_optional_by_default(client: TestClient) -> None:
    assert client.get("/api/grants/top10").status_code == 200


def test_api_key_guards_api_but_not_probes(runtime: Runtime) -> None:
    client = TestClient(create_app(runtime, api_key="s3cret"))

    assert client.get("/api/grants/top10").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/health/live").status_code == 200
    assert client.get("/docs").status_code == 200

    wrong = client.get("/api/grants/top10", headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    assert wrong.headers["WWW-Authenticate"] == "Bearer"

    via_header = client.get("/api/grants/top10", headers={"X-API-Key": "s3cret"})
    assert via_header.status_code == 200

    via_bearer = client.get(
        "/api/grants/top10", headers={"Authorization": "Bearer s3cret"}
    )
    assert via_bearer.status_code == 200

    # The gate also protects the streaming endpoint and writes.
    assert client.post("/api/chat/stream", json={"message": "hi"}).status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429(runtime: Runtime) -> None:
    client = TestClient(create_app(runtime, rate_limit_per_minute=2))
    assert client.get("/api/grants/top10").status_code == 200
    assert client.get("/api/grants/top10").status_code == 200
    limited = client.get("/api/grants/top10")
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"


def test_rate_limit_never_applies_to_probes(runtime: Runtime) -> None:
    client = TestClient(create_app(runtime, rate_limit_per_minute=1))
    for _ in range(5):
        assert client.get("/health/live").status_code == 200


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def _backdate(memory: MemoryStore, session_id: str, stamp: str) -> None:
    conn = sqlite3.connect(memory.path)
    try:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (stamp, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_purge_removes_only_stale_sessions(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "sessions.db")
    try:
        memory.save_profile("old", StudentProfile(student_id="old", name="Old"))
        memory.append_message("old", "user", "hi")
        memory.save_profile("new", StudentProfile(student_id="new", name="New"))
        memory.append_message("new", "user", "hi")
        _backdate(memory, "old", "2020-01-01T00:00:00+00:00")

        removed = memory.purge_older_than(30)

        assert removed == {"sessions": 1, "messages": 1, "artifacts": 0}
        assert memory.load_profile("old").name is None
        assert memory.load_profile("new").name == "New"
        stats = memory.stats()
        assert stats["sessions"] == 1
        assert stats["messages"] == 1
    finally:
        memory.close()


def test_purge_is_a_noop_for_non_positive_days(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "sessions.db")
    try:
        memory.save_profile("keep", StudentProfile(student_id="keep", name="Keep"))
        _backdate(memory, "keep", "2020-01-01T00:00:00+00:00")

        assert memory.purge_older_than(0) == {
            "sessions": 0,
            "messages": 0,
            "artifacts": 0,
        }
        assert memory.load_profile("keep").name == "Keep"
    finally:
        memory.close()


def test_stats_reports_file_size(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "sessions.db")
    try:
        memory.save_profile("s", StudentProfile(student_id="s"))
        stats = memory.stats()
        assert set(stats) == {"sessions", "messages", "artifacts", "bytes"}
        assert stats["sessions"] == 1
        assert stats["bytes"] > 0
    finally:
        memory.close()


# ---------------------------------------------------------------------------
# Lifespan ownership
# ---------------------------------------------------------------------------


def test_injected_runtime_is_not_closed_by_the_app(runtime: Runtime) -> None:
    app = create_app(runtime)
    with TestClient(app):
        pass
    # The caller (fixture) owns it, so it must still be usable afterwards.
    assert runtime.memory.stats()["sessions"] == 0


def test_owned_runtime_is_closed_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owned = Runtime(
        memory_path=tmp_path / "owned-sessions.db",
        grants_path=tmp_path / "owned-grants.db",
    )
    monkeypatch.setattr(api_module, "get_runtime", lambda: owned)

    app = create_app(None)
    with TestClient(app):
        assert app.state.runtime_holder["owned"] is True

    with pytest.raises(sqlite3.ProgrammingError):
        owned.memory.stats()
