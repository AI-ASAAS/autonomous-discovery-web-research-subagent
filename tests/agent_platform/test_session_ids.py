"""Session ids: issuance plus optional strict validation."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from agent_platform.api import Runtime, create_app

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{22,64}$")


def test_new_session_endpoint_issues_unguessable_id(client: TestClient) -> None:
    response = client.post("/api/session")

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert SESSION_ID_RE.fullmatch(session_id)
    assert len(session_id) >= 32


def test_default_mode_accepts_legacy_ids(client: TestClient) -> None:
    response = client.post(
        "/api/chat", json={"session_id": "short", "message": "hello"}
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "short"


def test_strict_mode_rejects_invalid_ids_everywhere(runtime: Runtime) -> None:
    strict = TestClient(create_app(runtime, strict_session_ids=True))
    cases = [
        ("GET", "/api/profile/short", None),
        ("PUT", "/api/profile/short", {}),
        ("POST", "/api/chat", {"session_id": "short", "message": "hello"}),
        ("POST", "/api/chat/stream", {"session_id": "short", "message": "hello"}),
        ("POST", "/api/documents", {"session_id": "short"}),
        ("POST", "/api/match", {"session_id": "short"}),
        ("POST", "/api/plan", {"session_id": "short"}),
        ("POST", "/api/advisor", {"session_id": "short"}),
        ("POST", "/api/session/short/reset", None),
    ]

    for method, path, body in cases:
        kwargs = {"json": body} if body is not None else {}
        response = strict.request(method, path, **kwargs)
        assert response.status_code == 422, (method, path)
        assert "message" in response.json(), (method, path)


def test_strict_mode_accepts_issued_id(runtime: Runtime) -> None:
    strict = TestClient(create_app(runtime, strict_session_ids=True))

    session_id = strict.post("/api/session").json()["session_id"]
    profile = strict.get(f"/api/profile/{session_id}")
    chat = strict.post(
        "/api/chat", json={"session_id": session_id, "message": "hello"}
    )

    assert profile.status_code == 200
    assert chat.status_code == 200
    assert chat.json()["session_id"] == session_id
