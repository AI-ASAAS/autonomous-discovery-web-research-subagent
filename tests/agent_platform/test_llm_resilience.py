"""LLM resilience: retries, fallback model, and deterministic-fallback flags."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_platform import llm
from agent_platform.config import Settings
from agent_platform.document_agent import DocumentAgent
from agent_platform.graph import model as model_module
from agent_platform.graph.service import AgentService
from agent_platform.knowledge import GrantRepository
from agent_platform.memory import MemoryStore
from agent_platform.orchestrator import Orchestrator


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ScriptedModel:
    """Minimal chat model that fails a fixed number of times, then answers."""

    def __init__(
        self, *, failures: int = 0, content: str = "ok", status: int = 503
    ) -> None:
        self.failures = failures
        self.content = content
        self.status = status
        self.calls = 0

    def invoke(
        self, messages: Any, config: Any = None, **kwargs: Any
    ) -> AIMessage:
        self.calls += 1
        if self.calls <= self.failures:
            raise _StatusError(self.status)
        return AIMessage(content=self.content)

    async def ainvoke(
        self, messages: Any, config: Any = None, **kwargs: Any
    ) -> AIMessage:
        return self.invoke(messages, config=config, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


def _settings(**overrides: Any) -> Settings:
    base = Settings(
        llm_api_key="test-key",
        llm_provider="openai",
        llm_model="primary-model",
        llm_base_url="https://example.invalid/v1",
        llm_max_retries=3,
        llm_fallback_model="",
        offline=False,
    )
    return replace(base, **overrides)


def _retrying(
    primary: Any,
    fallback: Any = None,
    sleeps: list[float] | None = None,
    max_retries: int = 3,
) -> Any:
    model_class = getattr(model_module, "RetryingChatModel")
    recorder = sleeps if sleeps is not None else []
    return model_class(
        primary=primary,
        fallback=fallback,
        max_retries=max_retries,
        sleep=recorder.append,
    )


# ---------------------------------------------------------------------------
# graph/model.py
# ---------------------------------------------------------------------------


def test_chat_model_retries_503_then_succeeds() -> None:
    primary = _ScriptedModel(failures=1, content="after retry")
    sleeps: list[float] = []
    model = _retrying(primary, sleeps=sleeps)

    result = model.invoke([HumanMessage(content="hi")])

    assert result.content == "after retry"
    assert primary.calls == 2
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] < 1.5


def test_chat_model_uses_fallback_after_repeated_503() -> None:
    primary = _ScriptedModel(failures=99, content="primary")
    fallback = _ScriptedModel(failures=0, content="fallback")
    sleeps: list[float] = []
    model = _retrying(primary, fallback=fallback, sleeps=sleeps)

    result = model.invoke([HumanMessage(content="hi")])

    assert result.content == "fallback"
    assert primary.calls == 3
    assert fallback.calls == 1
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] < 1.5
    assert 2.0 <= sleeps[1] < 2.5


def test_chat_model_raises_when_every_model_fails() -> None:
    primary = _ScriptedModel(failures=99)
    fallback = _ScriptedModel(failures=99)
    model = _retrying(primary, fallback=fallback, sleeps=[], max_retries=1)

    with pytest.raises(_StatusError):
        model.invoke([HumanMessage(content="hi")])

    assert primary.calls == 2
    assert fallback.calls == 1


def _graph_service(tmp_path: Any, model: Any) -> AgentService:
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


def test_service_marks_llm_used_false_after_graph_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even when a provider is configured (and the deterministic engine would
    # otherwise claim it), a failed graph must report the answer as non-LLM.
    monkeypatch.setattr(llm, "is_configured", lambda config=None: True)
    model = _retrying(_ScriptedModel(failures=99), sleeps=[], max_retries=0)
    service = _graph_service(tmp_path, model)

    response = service.chat("s1", "hello")

    assert service.engine == "graph"
    assert response.llm_used is False
    assert response.reply


def test_service_uses_deterministic_only_after_primary_and_fallback_fail(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "is_configured", lambda config=None: True)
    primary = _ScriptedModel(failures=99)
    fallback = _ScriptedModel(failures=99)
    model = _retrying(primary, fallback=fallback, sleeps=[], max_retries=0)
    service = _graph_service(tmp_path, model)

    response = service.chat("s1", "hello")

    assert primary.calls == 1
    assert fallback.calls == 1
    assert response.llm_used is False
    assert response.reply


def test_stream_marks_llm_used_false_after_graph_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "is_configured", lambda config=None: True)
    model = _retrying(_ScriptedModel(failures=99), sleeps=[], max_retries=0)
    service = _graph_service(tmp_path, model)

    events = list(service.stream("s1", "hello"))

    finals = [payload for kind, payload in events if kind == "final"]
    assert finals and finals[-1].llm_used is False
    assert finals[-1].reply


# ---------------------------------------------------------------------------
# graph/model.py temperature handling
# ---------------------------------------------------------------------------


def test_build_chat_model_omits_temperature_for_gemini_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(model_module, "ChatOpenAI", FakeChat)
    config = _settings(
        llm_provider="gemini", llm_model="gemini-3-pro-preview", llm_base_url=""
    )

    model_module.build_chat_model(config)

    assert "temperature" not in captured


def test_build_chat_model_keeps_temperature_for_other_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(model_module, "ChatOpenAI", FakeChat)
    config = _settings(
        llm_provider="gemini", llm_model="gemini-2.0-flash", llm_base_url=""
    )

    model_module.build_chat_model(config)

    assert captured["temperature"] == 0


# ---------------------------------------------------------------------------
# llm.py
# ---------------------------------------------------------------------------


def test_complete_retries_503_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_call(
        prompt: str,
        api_key: str,
        model: str,
        base_url: str,
        json_mode: bool,
        temperature: float,
        timeout: float,
    ) -> str:
        calls.append(model)
        if len(calls) == 1:
            raise _StatusError(503)
        return "ok"

    monkeypatch.setattr(llm, "_call_openai_compatible", fake_call)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    result = llm.complete("hi", config=_settings())

    assert result == "ok"
    assert calls == ["primary-model", "primary-model"]
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] < 1.5


def test_complete_uses_fallback_model_after_repeated_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_call(
        prompt: str,
        api_key: str,
        model: str,
        base_url: str,
        json_mode: bool,
        temperature: float,
        timeout: float,
    ) -> str:
        calls.append(model)
        if model == "primary-model":
            raise _StatusError(503)
        return "fallback-ok"

    monkeypatch.setattr(llm, "_call_openai_compatible", fake_call)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    result = llm.complete(
        "hi", config=_settings(llm_fallback_model="fallback-model")
    )

    assert result == "fallback-ok"
    assert calls == ["primary-model"] * 3 + ["fallback-model"]


def test_complete_raises_unavailable_when_everything_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_call(
        prompt: str,
        api_key: str,
        model: str,
        base_url: str,
        json_mode: bool,
        temperature: float,
        timeout: float,
    ) -> str:
        calls.append(model)
        raise _StatusError(503)

    monkeypatch.setattr(llm, "_call_openai_compatible", fake_call)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    with pytest.raises(llm.LLMUnavailable):
        llm.complete(
            "hi", config=_settings(llm_fallback_model="fallback-model")
        )

    assert calls == ["primary-model"] * 3 + ["fallback-model"]


def test_gemini_3_payload_omits_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            }

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            captured.update(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr(llm.httpx, "Client", FakeClient)

    llm._call_gemini("hi", "key", "gemini-3-pro-preview", False, 0.2, 10)

    assert "temperature" not in captured.get("generationConfig", {})
