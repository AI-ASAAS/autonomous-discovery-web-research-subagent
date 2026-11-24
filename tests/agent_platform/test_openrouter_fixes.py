"""Tests for OpenRouter / LLM error classification and llm_used reporting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from agent_platform import llm
from agent_platform.api import Runtime, create_app
from agent_platform.config import Settings
from agent_platform.document_agent import DocumentAgent
from agent_platform.graph import model as model_module
from agent_platform.graph.service import AgentService
from agent_platform.knowledge import GrantRepository
from agent_platform.memory import MemoryStore
from agent_platform.orchestrator import Orchestrator
from agent_platform.schemas import GrantCard, StudentProfile


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "", body: Any = None) -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


class _ScriptedModel:
    def __init__(self, *, failures: int = 0, content: str = "ok", exc: Exception | None = None) -> None:
        self.failures = failures
        self.content = content
        self.exc = exc or _StatusError(503)
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return AIMessage(content=self.content)

    async def ainvoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        return self.invoke(messages, config=config, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


def _retrying(primary: Any, fallback: Any = None, sleeps: list[float] | None = None, max_retries: int = 2) -> Any:
    recorder = sleeps if sleeps is not None else []
    return model_module.RetryingChatModel(
        primary=primary,
        fallback=fallback,
        max_retries=max_retries,
        sleep=recorder.append,
    )


def _service(tmp_path: Path, model: Any) -> AgentService:
    memory = MemoryStore(tmp_path / "sessions.db")
    repository = GrantRepository(tmp_path / "grants.db")
    documents = DocumentAgent(use_llm=False)
    return AgentService(
        memory=memory,
        repository=repository,
        documents=documents,
        orchestrator=Orchestrator(
            memory=memory, repository=repository, documents=documents
        ),
        engine="graph",
        model_factory=lambda: model,
    )


# ---------------------------------------------------------------------------
# 1. llm_used reporting
# ---------------------------------------------------------------------------


def test_llm_used_is_false_when_circuit_breaker_is_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When circuit breaker is open, zero calls are made and llm_used is False."""
    llm.reset_runtime_state()
    monkeypatch.setattr(llm, "is_configured", lambda config=None: True)
    model = _ScriptedModel(failures=0, content="unreachable")
    service = _service(tmp_path, model)

    llm.circuit.open(retry_after=600)
    assert llm.circuit.is_open() is True

    response = service.chat("s1", "hello")

    assert response.llm_used is False
    assert response.model_calls == 0
    assert model.calls == 0
    llm.reset_runtime_state()


def test_llm_used_is_false_when_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When model raises and service falls back to deterministic, llm_used is False."""
    llm.reset_runtime_state()
    monkeypatch.setattr(llm, "is_configured", lambda config=None: True)
    model = _retrying(_ScriptedModel(failures=99, exc=_StatusError(500)), sleeps=[], max_retries=0)
    service = _service(tmp_path, model)

    response = service.chat("s1", "hello")

    assert response.llm_used is False
    assert response.model_calls == 1
    llm.reset_runtime_state()


def test_llm_used_is_true_when_model_call_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a model call completes successfully in this turn, llm_used is True."""
    llm.reset_runtime_state()
    monkeypatch.setattr(llm, "is_configured", lambda config=None: True)
    model = _retrying(_ScriptedModel(failures=0, content="Hello there!"), sleeps=[], max_retries=0)
    service = _service(tmp_path, model)

    response = service.chat("s1", "hello")

    assert response.llm_used is True
    assert response.model_calls == 1
    llm.reset_runtime_state()


# ---------------------------------------------------------------------------
# 2. Error classification in breaker
# ---------------------------------------------------------------------------


def test_permanent_error_opens_breaker_for_process_and_sets_health(tmp_path: Path) -> None:
    """401/402/403 or 400 model support errors open the breaker permanently and set llm_config_error."""
    llm.reset_runtime_state()

    # 401 Unauthorized
    exc_401 = _StatusError(401, "Invalid API key")
    err_class, delay, msg = llm.classify_error(exc_401)
    assert err_class == llm.ErrorClass.PERMANENT

    llm.note_provider_failure(exc_401)
    assert llm.circuit.is_open() is True
    assert llm.circuit.config_error() is not None
    assert "401" in llm.circuit.config_error()

    # Verify /health reflects the permanent config error
    runtime = Runtime(memory_path=tmp_path / "sessions.db")
    with TestClient(create_app(runtime)) as client:
        health = client.get("/health").json()
        assert health["llm_available"] is False
        assert health["llm_config_error"] is not None
        assert "401" in health["llm_config_error"]

    # 400 with model/tool support error
    llm.reset_runtime_state()
    exc_400 = _StatusError(400, "Model does not support tools", body={"error": "tool use is unsupported for this model"})
    err_class, _, msg = llm.classify_error(exc_400)
    assert err_class == llm.ErrorClass.PERMANENT

    llm.note_provider_failure(exc_400)
    assert llm.circuit.is_open() is True
    assert llm.circuit.config_error() is not None
    assert "400" in llm.circuit.config_error()
    llm.reset_runtime_state()


def test_account_quota_error_opens_breaker_for_reported_delay(tmp_path: Path) -> None:
    """429 with 'PerDay' or delay > 60s opens breaker for reported delay."""
    llm.reset_runtime_state()

    exc_429 = _StatusError(
        429,
        "ResourceExhausted",
        body={"error": {"details": [{"quotaId": "RequestsPerDayPerProject", "retryDelay": "120s"}]}},
    )
    err_class, delay, msg = llm.classify_error(exc_429)
    assert err_class == llm.ErrorClass.ACCOUNT_QUOTA
    assert delay == 120.0

    llm.note_provider_failure(exc_429)
    assert llm.circuit.is_open() is True
    assert llm.circuit.config_error() is None  # not a permanent config error
    assert 60.0 < llm.circuit.retry_after_seconds() <= 120.0

    runtime = Runtime(memory_path=tmp_path / "sessions.db")
    with TestClient(create_app(runtime)) as client:
        health = client.get("/health").json()
        assert health["llm_available"] is False
        assert health["llm_config_error"] is None
        assert health["llm_retry_after_seconds"] > 60.0
    llm.reset_runtime_state()


def test_transient_upstream_error_does_not_open_breaker_and_retries() -> None:
    """429/503 with upstream cues retries with 2s and 5s backoff and does NOT open breaker."""
    llm.reset_runtime_state()

    exc_upstream = _StatusError(
        429,
        "rate limit",
        body={"error": {"message": "Provider temporarily rate-limited", "limit_source": "upstream"}},
    )
    err_class, delay, msg = llm.classify_error(exc_upstream)
    assert err_class == llm.ErrorClass.TRANSIENT_UPSTREAM
    assert llm.is_retryable_error(exc_upstream) is True

    # Note provider failure should NOT open the circuit breaker
    llm.note_provider_failure(exc_upstream)
    assert llm.circuit.is_open() is False

    # Check backoff delays for upstream transient error: 2s, then 5s
    assert llm.retry_delay(0, exc_upstream) == 2.0
    assert llm.retry_delay(1, exc_upstream) == 5.0

    # RetryingChatModel retries 2 times with 2s and 5s backoff
    sleeps: list[float] = []
    primary = _ScriptedModel(failures=2, content="succeeded after 2 retries", exc=exc_upstream)
    model = _retrying(primary, sleeps=sleeps, max_retries=2)

    res = model.invoke([HumanMessage(content="test")])
    assert res.content == "succeeded after 2 retries"
    assert primary.calls == 3
    assert sleeps == [2.0, 5.0]
    assert llm.circuit.is_open() is False
    llm.reset_runtime_state()


def test_transient_upstream_uses_fallback_model() -> None:
    """When primary fails repeatedly on transient upstream, fallback model is tried once."""
    llm.reset_runtime_state()

    exc_upstream = _StatusError(503, "temporarily rate-limited upstream")
    primary = _ScriptedModel(failures=99, exc=exc_upstream)
    fallback = _ScriptedModel(failures=0, content="fallback response")
    sleeps: list[float] = []

    model = _retrying(primary, fallback=fallback, sleeps=sleeps, max_retries=2)
    res = model.invoke([HumanMessage(content="test")])

    assert res.content == "fallback response"
    assert primary.calls == 3
    assert fallback.calls == 1
    assert sleeps == [2.0, 5.0]
    assert llm.circuit.is_open() is False
    llm.reset_runtime_state()

