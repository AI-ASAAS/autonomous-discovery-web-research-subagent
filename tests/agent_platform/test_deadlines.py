"""Deterministic deadline verdicts (offline, no model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from agent_platform import llm
from agent_platform.deadlines import DateClaim, apply_deadline_fact, extract_dates
from agent_platform.document_agent import DocumentAgent
from agent_platform.graph.prompts import build_system_prompt
from agent_platform.graph.service import AgentService
from agent_platform.knowledge import GrantRepository
from agent_platform.memory import MemoryStore
from agent_platform.orchestrator import Orchestrator
from agent_platform.schemas import DegreeLevel, GrantCard, StudentProfile

CARD = GrantCard(
    grant_id="g-chevening",
    title="Chevening Scholarship",
    official_url="https://chevening.org",
    provider="UK Government",
    country="United Kingdom",
    deadline="2026-06-03",
    accepts_yemeni="yes",
)

NO_DEADLINE_CARD = GrantCard(
    grant_id="g-no-deadline",
    title="Open Deadline Grant",
    official_url="https://example.org/grant",
    provider="Example",
    country="Germany",
    deadline=None,
    accepts_yemeni="yes",
)

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


class FakeRepository:
    """Small read-only repository that keeps the tests offline and fast."""

    def __init__(self, cards: list[GrantCard]) -> None:
        self._cards = cards

    def list_published(self, limit: int = 50) -> list[GrantCard]:
        return self._cards[:limit]

    def get(self, grant_id: str) -> GrantCard | None:
        return next((card for card in self._cards if card.grant_id == grant_id), None)

    def find_by_title(self, title: str) -> GrantCard | None:
        return next((card for card in self._cards if card.title == title), None)

    def search(self, query: str, limit: int = 1) -> list[GrantCard]:
        return []

    def count(self) -> int:
        return len(self._cards)


def _orchestrator(tmp_path: Path, cards: list[GrantCard]) -> Orchestrator:
    return Orchestrator(
        memory=MemoryStore(tmp_path / "sessions.db"),
        repository=FakeRepository(cards),  # type: ignore[arg-type]
        documents=DocumentAgent(use_llm=False),
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_iso_and_english_date_forms() -> None:
    assert DateClaim(6, 3, 2026) in extract_dates("deadline is 2026-06-03")
    assert DateClaim(6, 3, None) in extract_dates("is it June 3?")
    assert DateClaim(6, 3, 2026) in extract_dates("is it 3 June 2026?")
    assert DateClaim(6, 3, None) in extract_dates("is it Jun 3?")


def test_parse_arabic_date_forms() -> None:
    assert DateClaim(6, 3, None) in extract_dates("هل الموعد ٣ يونيو؟")
    assert DateClaim(6, 3, None) in extract_dates("هل الموعد 3 يونيو؟")
    assert DateClaim(6, 3, 2026) in extract_dates("هل الموعد يونيو 3 2026؟")
    assert DateClaim(1, 15, None) in extract_dates("الموعد 15 يناير")


# ---------------------------------------------------------------------------
# Legacy engine
# ---------------------------------------------------------------------------


def test_legacy_wrong_deadline_is_corrected(tmp_path: Path) -> None:
    reply = _orchestrator(tmp_path, [CARD]).chat(
        "s1", "Is the deadline of the Chevening Scholarship June 5?"
    ).reply

    assert reply.startswith("No: the verified deadline is 2026-06-03")


def test_legacy_right_deadline_is_confirmed(tmp_path: Path) -> None:
    reply = _orchestrator(tmp_path, [CARD]).chat(
        "s1", "Is the deadline of the Chevening Scholarship June 3?"
    ).reply

    assert reply.startswith("Yes: 2026-06-03")


def test_legacy_unknown_deadline_is_reported_as_unverified(tmp_path: Path) -> None:
    reply = _orchestrator(tmp_path, [NO_DEADLINE_CARD]).chat(
        "s1", "Is the deadline of the Open Deadline Grant June 3?"
    ).reply

    assert reply.startswith("I have no verified deadline for this grant")


def test_legacy_without_grant_focus_asks_which_scholarship(tmp_path: Path) -> None:
    reply = _orchestrator(tmp_path, [CARD]).chat(
        "s1", "Is the deadline June 3?"
    ).reply

    assert "Which scholarship" in reply


def test_legacy_uses_the_grant_from_the_previous_turn(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, [CARD])
    orchestrator.memory.save_profile("s1", PROFILE)
    first = orchestrator.chat("s1", "Match my profile").reply
    assert first

    reply = orchestrator.chat("s1", "Is the deadline June 3?").reply

    assert reply.startswith("Yes: 2026-06-03")


def test_legacy_arabic_verdict(tmp_path: Path) -> None:
    reply = _orchestrator(tmp_path, [CARD]).chat(
        "s1", "هل الموعد النهائي لمنحة Chevening هو ٣ يونيو؟"
    ).reply

    assert reply.startswith("نعم: 2026-06-03")


# ---------------------------------------------------------------------------
# Graph engine
# ---------------------------------------------------------------------------


class _ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.seen.append(list(messages))
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def _service(tmp_path: Path, cards: list[GrantCard], model: Any) -> AgentService:
    memory = MemoryStore(tmp_path / "sessions.db")
    repository = FakeRepository(cards)
    documents = DocumentAgent(use_llm=False)
    return AgentService(
        memory=memory,
        repository=repository,  # type: ignore[arg-type]
        documents=documents,
        orchestrator=Orchestrator(
            memory=memory,
            repository=repository,  # type: ignore[arg-type]
            documents=documents,
        ),
        engine="graph",
        model_factory=lambda: model,
    )


def test_graph_wrong_deadline_cannot_contradict_the_record(tmp_path: Path) -> None:
    model = _ScriptedModel([AIMessage(content="Yes, the deadline is June 5, 2026.")])
    reply = _service(tmp_path, [CARD], model).chat(
        "s1", "Is the deadline of the Chevening Scholarship June 5?"
    ).reply

    assert reply.startswith("No: the verified deadline is 2026-06-03")


def test_graph_unknown_deadline_is_reported_as_unverified(tmp_path: Path) -> None:
    model = _ScriptedModel([AIMessage(content="The deadline is June 3.")])
    reply = _service(tmp_path, [NO_DEADLINE_CARD], model).chat(
        "s1", "Is the deadline of the Open Deadline Grant June 3?"
    ).reply

    assert reply.startswith("I have no verified deadline for this grant")


def test_graph_without_grant_focus_asks_which_scholarship(tmp_path: Path) -> None:
    model = _ScriptedModel([AIMessage(content="The deadline is June 3.")])
    reply = _service(tmp_path, [CARD], model).chat(
        "s1", "Is the deadline June 3?"
    ).reply

    assert "Which scholarship" in reply


def test_graph_system_prompt_and_tool_envelope_carry_the_fact(tmp_path: Path) -> None:
    from agent_platform.graph.builder import build_graph
    from agent_platform.graph.tools import ToolContext, build_tools

    context = ToolContext(
        profile=PROFILE,
        repository=FakeRepository([CARD]),  # type: ignore[arg-type]
        documents=DocumentAgent(use_llm=False),
        deadline_fact={
            "status": "verdict",
            "verdict": "No: the verified deadline is 2026-06-03",
            "grant_id": CARD.grant_id,
            "grant_title": CARD.title,
        },
    )
    tools = build_tools(context)
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_grant_details",
                        "args": {"grant": CARD.grant_id},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    graph = build_graph(model=model, tools=tools, context=context)
    state = {
        "session_id": "s1",
        "message": "Is the deadline of the Chevening Scholarship June 5?",
        "language": "en",
        "profile": PROFILE.model_dump(mode="json"),
        "missing_fields": [],
        "guardrails": [],
        "steps": [],
        "tool_rounds": 0,
        "matches": [],
        "plan": [],
        "documents": [],
        "advisor": None,
        "sources": [],
        "tools_used": [],
        "blocked": False,
        "llm_used": True,
        "deadline_fact": context.deadline_fact,
    }

    final = graph.invoke(state)

    assert final["reply"].startswith("No: the verified deadline is 2026-06-03")
    assert any(
        "deadline_fact" in str(message)
        for messages in model.seen
        for message in messages
    )
    prompt = build_system_prompt(state)
    assert "VERIFIED DEADLINE FACT" in prompt
    assert "No: the verified deadline is 2026-06-03" in prompt


def test_apply_deadline_fact_prepends_and_replaces_clarification() -> None:
    from agent_platform.deadlines import DeadlineFact

    verdict = DeadlineFact(
        status="verdict",
        verdict="No: the verified deadline is 2026-06-03",
    )
    clarification = DeadlineFact(
        status="clarification",
        verdict="",
        clarification="Which scholarship do you mean?",
    )

    assert apply_deadline_fact("Yes, June 5.", verdict).startswith(
        "No: the verified deadline is 2026-06-03"
    )
    assert apply_deadline_fact("Anything", clarification) == (
        "Which scholarship do you mean?"
    )
