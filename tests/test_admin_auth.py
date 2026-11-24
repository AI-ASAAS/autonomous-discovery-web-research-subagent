"""Admin-key middleware for the discovery API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

ADMIN_KEY = "test-admin-key-123456"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    with TestClient(app) as test_client:
        yield test_client


def test_admin_routes_require_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    cases = [
        ("GET", "/api/system/status"),
        ("GET", "/api/agents/monitor/logs"),
        ("GET", "/api/grants"),
        ("POST", "/api/grants/1/publish"),
    ]
    for method, path in cases:
        response = client.request(method, path)
        assert response.status_code == 401, (method, path)
        assert "message" in response.json()


def test_admin_routes_reject_wrong_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    response = client.get("/api/grants", headers={"X-Admin-Key": "wrong"})

    assert response.status_code == 401


def test_admin_routes_allow_correct_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    listing = client.get("/api/grants", headers={"X-Admin-Key": ADMIN_KEY})
    status = client.get("/api/system/status", headers={"X-Admin-Key": ADMIN_KEY})

    assert listing.status_code == 200
    assert "grants" in listing.json()
    assert status.status_code == 200


def test_top10_stays_public(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    response = client.get("/api/grants/top10")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unset_key_allows_access(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    assert client.get("/api/grants").status_code == 200
    assert client.get("/api/system/status").status_code == 200


def test_preflight_options_pass_with_key_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    response = client.options(
        "/api/grants",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Admin-Key",
        },
    )

    assert response.status_code == 200
