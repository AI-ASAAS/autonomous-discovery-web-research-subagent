"""Quota handling, circuit breaker, per-turn budget and call accounting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient

from agent_platform import llm, matching
from agent_platform.config import settings
from agent_platform.document_agent import DocumentAgent
from agent_platform.graph import model as model_module
from agent_platform.graph.model import RetryingChatModel
from agent_platform.graph.service import AgentService
from agent_platform.knowledge import GrantRepository
from agent_platform.memory import MemoryStore
from agent_platform.orchestrator import Orchestrator
from agent_platform.schemas import DegreeLevel, GrantCard, StudentProfile

PROFILE = StudentProfile(
    student_id="s1",
    name="Ahmed",
    email="ahmed@example.com",
    gpa=3.6,
    major="Computer Science",
    degree_level=DegreeLevel.BACHELOR,
    target_degree=DegreeLevel.MASTER,
    nationality="Yemeni",
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _StatusError(Exception):
    def __init__(self, status: int, body: dict[str, Any] | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.body = body


class _ScriptedModel:
    def __init__(
        self,
        *,
        failures: int = 0,
        status: int = 503,
        body: dict[str, Any] | None = None,
        content: str = "ok",
        responses: list[AIMessage] | None = None,
    ) -> None:
        self.failures = failures
        self.status = status
        self.body = body
        self.content = content
        self.responses = list(responses or [])
        self.calls = 0
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.seen.append(list(messages))
        self.calls += 1
        if self.calls <= self.failures:
            raise _StatusError(self.status, self.body)
        if self.responses:
            return self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return AIMessage(content=self.content)


def _service(tmp_path: Path, factory: Any, *, profile: StudentProfile | None = None) -> AgentService:
    memory = MemoryStore(tmp_path / "sessions.db")
    repository = GrantRepository(tmp_path / "grants.db")
    documents = DocumentAgent(use_llm=False)
    if profile is not None:
        memory.save_profile(profile.student_id, profile)
    return AgentService(
        memory=memory,
        repository=repository,
        documents=documents,
        orchestrator=Orchestrator(
            memory=memory, repository=repository, documents=documents
        ),
        engine="graph",
        model_factory=factory,
    )


def _settings(**overrides: Any) -> Any:
    base = replace(
        settings,
        llm_api_key="test-key",
        llm_provider="openai",
        llm_model="primary-model",
        llm_base_url="https://example.invalid/v1",
        llm_fallback_model="",
        llm_max_retries=3,
        offline=False,
    )
    return replace(base, **overrides)


# ---------------------------------------------------------------------------
# Quota errors
# ---------------------------------------------------------------------------


def test_quota_429_is_not_retried_and_opens_the_circuit() -> None:
    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                },
                {"retryDelay": "35s"},
            ],
        }
    }
    primary = _ScriptedModel(failures=99, status=429, body=body)
    sleeps: list[float] = []
    model = RetryingChatModel(primary=primary, max_retries=5, sleep=sleeps.append)

    with pytest.raises(_StatusError):
        model.invoke([AIMessage(content="hi")])

    assert primary.calls == 1
    assert sleeps == []
    assert llm.circuit.is_open()
    assert llm.circuit.retry_after_seconds() == pytest.approx(35, abs=1)


def test_429_without_details_still_opens_the_circuit_with_the_default_delay() -> None:
    primary = _ScriptedModel(failures=99, status=429)
    model = RetryingChatModel(primary=primary, max_retries=5, sleep=lambda _: None)

    with pytest.raises(_StatusError):
        model.invoke([AIMessage(content="hi")])

    assert primary.calls == 1
    assert llm.circuit.is_open()
    assert llm.circuit.retry_after_seconds() == pytest.approx(600, abs=1)


def test_retry_cap_is_two_retries() -> None:
    primary = _ScriptedModel(failures=99, status=503)
    model = RetryingChatModel(primary=primary, max_retries=5, sleep=lambda _: None)

    with pytest.raises(_StatusError):
        model.invoke([AIMessage(content="hi")])

    assert primary.calls == 3


@pytest.mark.parametrize("status", [500])
def test_non_retryable_5xx_is_not_retried(status: int) -> None:
    primary = _ScriptedModel(failures=99, status=status)
    model = RetryingChatModel(primary=primary, max_retries=5, sleep=lambda _: None)

    with pytest.raises(_StatusError):
        model.invoke([AIMessage(content="hi")])

    assert primary.calls == 1


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_open_circuit_skips_the_model_until_the_delay_elapses(tmp_path: Path) -> None:
    clock = FakeClock()
    llm.circuit.set_clock(clock)
    llm.circuit.reset()
    factory_calls: list[int] = []
    primary = _ScriptedModel(failures=1, status=429, content="after the breaker")
    # Max retries 0: the first turn makes exactly one model call, which trips
    # the breaker.
    model = RetryingChatModel(primary=primary, max_retries=0)

    def factory() -> Any:
        factory_calls.append(1)
        return model

    service = _service(tmp_path, factory, profile=PROFILE)

    first = service.chat("s1", "hello")
    assert first.llm_used is False
    assert first.model_calls == 1
    assert len(factory_calls) == 1

    second = service.chat("s1", "hello")
    assert second.model_calls == 0
    assert len(factory_calls) == 1

    clock.advance(601)
    third = service.chat("s1", "hello")
    assert third.llm_used is True
    assert len(factory_calls) == 2
    assert primary.calls == 2


def test_health_exposes_circuit_state(client: TestClient) -> None:
    clock = FakeClock()
    llm.circuit.set_clock(clock)
    llm.circuit.reset()
    llm.circuit.open(120)

    body = client.get("/health").json()

    assert body["llm_available"] is False
    assert body["llm_retry_after_seconds"] == pytest.approx(120, abs=1)


# ---------------------------------------------------------------------------
# Per-turn budget
# ---------------------------------------------------------------------------


class _TimeoutModel:
    def __init__(self, clock: FakeClock, cost: float = 0.0) -> None:
        self.clock = clock
        self.cost = cost
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls += 1
        self.clock.advance(self.cost)
        raise TimeoutError("provider timeout")


def test_timeouts_stop_at_the_turn_budget() -> None:
    clock = FakeClock()
    budget = llm.TurnBudget(2.5, clock=clock)
    primary = _TimeoutModel(clock)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    model = RetryingChatModel(
        primary=primary, max_retries=5, sleep=sleep, budget=budget
    )

    with pytest.raises((TimeoutError, llm.LLMTurnBudgetExceeded)):
        model.invoke([AIMessage(content="hi")])

    assert clock.now <= 2.5
    assert primary.calls <= 3
    assert sum(sleeps) <= 2.5


def test_three_timeouts_never_exceed_the_budget() -> None:
    clock = FakeClock()
    budget = llm.TurnBudget(30.0, clock=clock)
    primary = _TimeoutModel(clock, cost=8.0)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    model = RetryingChatModel(
        primary=primary, max_retries=5, sleep=sleep, budget=budget
    )

    with pytest.raises((TimeoutError, llm.LLMTurnBudgetExceeded)):
        model.invoke([AIMessage(content="hi")])

    assert clock.now <= 30.0
    assert primary.calls <= 3


def test_build_chat_model_uses_a_quick_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "ChatOpenAI", FakeChat)

    model_module.build_chat_model(_settings(llm_connect_timeout=5))

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5


# ---------------------------------------------------------------------------
# Semantic cache and call accounting
# ---------------------------------------------------------------------------


def test_semantic_scores_are_cached_per_profile_and_grant_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matching.clear_semantic_cache()
    calls: list[str] = []
    card = GrantCard(
        grant_id="g1",
        title="Computer Science Grant",
        official_url="https://example.org",
        description="AI and data science",
        fields_of_study=["Computer Science"],
    )

    def fake_complete_json(prompt: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(prompt)
        return {"g1": 88}

    monkeypatch.setattr(matching.llm, "complete_json", fake_complete_json)
    config = _settings()

    first = matching._llm_semantic_scores(PROFILE, [card], config)
    second = matching._llm_semantic_scores(PROFILE, [card], config)

    assert first == {"g1": 88.0}
    assert second == first
    assert len(calls) == 1


def test_match_turn_uses_at_most_three_model_calls(tmp_path: Path) -> None:
    model = _ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "match_profile", "args": {"limit": 3}, "id": "c1"}
                ],
            ),
            AIMessage(content="Here are your matches."),
        ]
    )
    service = _service(tmp_path, lambda: model, profile=PROFILE)

    response = service.chat("s1", "Match my profile")

    assert response.model_calls <= 3
    assert model.calls == 2


def test_deadline_turn_uses_at_most_three_model_calls(tmp_path: Path) -> None:
    repository = GrantRepository(tmp_path / "grants.db")
    card = next(card for card in repository.list_published() if card.deadline)
    model = _ScriptedModel(responses=[AIMessage(content="Answered.")])
    service = _service(tmp_path, lambda: model, profile=PROFILE)

    response = service.chat(
        "s1", f"Is the deadline of the {card.title} 1999-01-01?"
    )

    assert response.model_calls <= 3
    assert model.calls == 1
