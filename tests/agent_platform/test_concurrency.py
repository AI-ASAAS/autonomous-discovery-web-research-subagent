"""External search is bounded by a timeout and degrades politely."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import external_search_sub_agent
from agent_platform.concurrency import OperationTimeoutError, run_with_timeout
from agent_platform.document_agent import DocumentAgent
from agent_platform.graph import tools as tools_module
from agent_platform.graph.tools import ToolContext, build_tools
from agent_platform.knowledge import GrantRepository
from agent_platform import orchestrator as orchestrator_module
from agent_platform.schemas import DegreeLevel, StudentProfile


class _SearchResult:
    def model_dump(self) -> dict:
        return {
            "results": [
                {"title": "Fake Grant", "official_url": "https://example.edu/fake"}
            ]
        }


def _slow_search(**_: object) -> _SearchResult:
    time.sleep(1.5)
    return _SearchResult()


def _fast_search(**_: object) -> _SearchResult:
    return _SearchResult()


PROFILE = StudentProfile(
    student_id="timeout",
    name="Test Student",
    gpa=3.6,
    major="Computer Science",
    degree_level=DegreeLevel.BACHELOR,
    target_degree=DegreeLevel.MASTER,
    nationality="Yemeni",
)


def _tool_context(repository: GrantRepository) -> ToolContext:
    return ToolContext(
        profile=PROFILE,
        repository=repository,
        documents=DocumentAgent(use_llm=False),
    )


def test_run_with_timeout_returns_fast_result() -> None:
    assert run_with_timeout(lambda: 42, 1.0) == 42


def test_external_tool_times_out_politely(
    repository: GrantRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {item.name: item for item in build_tools(_tool_context(repository))}
    monkeypatch.setattr(
        tools_module,
        "settings",
        SimpleNamespace(
            external_search_configured=True,
            external_search_timeout_seconds=1.0,
        ),
    )
    monkeypatch.setattr(
        external_search_sub_agent, "search_external_scholarships", _slow_search
    )

    payload = json.loads(tools["search_external_grants"].invoke({}))

    assert payload["ok"] is False
    assert "timed out" in payload["summary"].lower()


def test_external_tool_returns_fast_results(
    repository: GrantRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {item.name: item for item in build_tools(_tool_context(repository))}
    monkeypatch.setattr(
        tools_module,
        "settings",
        SimpleNamespace(
            external_search_configured=True,
            external_search_timeout_seconds=1.0,
        ),
    )
    monkeypatch.setattr(
        external_search_sub_agent, "search_external_scholarships", _fast_search
    )

    payload = json.loads(tools["search_external_grants"].invoke({}))

    assert payload["ok"] is True
    assert payload["data"]["results"][0]["title"] == "Fake Grant"


def test_legacy_engine_times_out_politely(
    repository: GrantRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = orchestrator_module.Orchestrator(
        repository=repository, documents=DocumentAgent(use_llm=False)
    )
    monkeypatch.setattr(
        orchestrator_module,
        "settings",
        SimpleNamespace(
            external_search_configured=True,
            external_search_timeout_seconds=1.0,
        ),
    )
    monkeypatch.setattr(
        external_search_sub_agent, "search_external_scholarships", _slow_search
    )

    reply, cards = engine._handle_external(PROFILE, [], "en", lambda *a, **k: None)

    assert "too long" in reply.lower()
    assert cards == []


def test_operation_timeout_is_a_timeout_error() -> None:
    assert issubclass(OperationTimeoutError, TimeoutError)
