"""Agent service: the single entry point used by the HTTP layer.

The service owns the engine choice:

* **graph** (default) — a LangGraph supervisor with tools, active whenever an
  LLM provider is configured; it degrades gracefully to
* **legacy** — the deterministic orchestrator, which requires no provider and
  keeps the platform usable offline.

Both engines return the same :class:`ChatResponse`, so the FastAPI layer and
the front-ends are engine-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from langchain_core.language_models import BaseChatModel

from .. import llm
from ..config import settings
from ..deadlines import deadline_verdict, find_focused_grant, parse_date_from_text
from ..document_agent import DocumentAgent
from ..extraction import detect_language, merge_profile
from ..knowledge import GrantRepository
from ..memory import MemoryStore
from ..orchestrator import Orchestrator
from ..schemas import (
    AdvisorReport,
    ChatResponse,
    DocumentResult,
    GuardrailEvent,
    Intent,
    MatchResult,
    StudentProfile,
    ThoughtStep,
)
from .builder import build_graph
from .model import RetryingChatModel, build_chat_model
from .prompts import fallback_reply
from .tools import ToolContext, build_tools

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], BaseChatModel]


def summarize_artifacts(records: list[dict[str, Any]]) -> list[str]:
    """Turn stored session artifacts into compact lines for the supervisor prompt.

    This reuses the work the session already produced (matches, plans,
    documents, advisory reports) instead of recomputing everything every turn.
    """
    lines: list[str] = []
    for record in records:
        kind = record.get("kind")
        payload = record.get("payload") or {}
        if kind == "match":
            results = payload.get("results") or []
            if results:
                top = results[0]
                lines.append(
                    f"- match: {top.get('title', '?')} — "
                    f"{top.get('match_percentage', 0)}%"
                )
            else:
                lines.append("- match: no results")
        elif kind == "plan":
            plans = payload.get("plans") or payload.get("items") or []
            titles = [
                str(item.get("grant_title", "")) for item in plans[:3] if item
            ]
            titles = [title for title in titles if title]
            suffix = f" ({', '.join(titles)})" if titles else ""
            lines.append(f"- plan: {len(plans)} plans{suffix}")
        elif kind == "document":
            doc_type = payload.get("document_type", "document")
            grant = payload.get("grant_title", "")
            lines.append(f"- document: {doc_type} for {grant}".strip())
        elif kind == "advisor":
            actions = payload.get("actions") or []
            lines.append(f"- advisor: {len(actions)} improvement actions")
    return lines


class AgentService:
    """Engine selector + response assembler for the conversational agent."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        repository: GrantRepository,
        documents: DocumentAgent,
        orchestrator: Orchestrator | None = None,
        engine: str | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.memory = memory
        self.repository = repository
        self.documents = documents
        self.orchestrator = orchestrator or Orchestrator(
            memory=memory, repository=repository, documents=documents
        )
        self.requested_engine = (engine or settings.agent_engine or "graph").strip().lower()
        self._model_factory = model_factory

    # ------------------------------------------------------------------
    # Engine selection
    # ------------------------------------------------------------------
    @property
    def graph_enabled(self) -> bool:
        if self.requested_engine == "legacy":
            return False
        if llm.circuit.is_open():
            # A quota error already proved the key unusable; do not spend
            # another turn (or another model call) discovering it again.
            return False
        if self._model_factory is not None:
            return True
        return llm.is_configured()

    @property
    def engine(self) -> str:
        return "graph" if self.graph_enabled else "legacy"

    # ------------------------------------------------------------------
    # Public API (drop-in for the orchestrator)
    # ------------------------------------------------------------------
    def chat(
        self,
        session_id: str,
        message: str,
        *,
        profile_override: StudentProfile | None = None,
        on_step: Callable[[ThoughtStep], None] | None = None,
    ) -> ChatResponse:
        tokens = llm.begin_turn()
        try:
            if not self.graph_enabled:
                response = self.orchestrator.chat(
                    session_id,
                    message,
                    profile_override=profile_override,
                    on_step=on_step,
                )
                return response.model_copy(
                    update={
                        "llm_used": llm.llm_used_in_turn(),
                        "model_calls": llm.model_calls(),
                    }
                )
            try:
                return self._chat_graph(
                    session_id,
                    message,
                    profile_override=profile_override,
                    on_step=on_step,
                )
            except Exception:  # pragma: no cover - provider/transport failure
                logger.exception(
                    "Agent graph failed; falling back to the deterministic engine"
                )
                response = self.orchestrator.chat(
                    session_id,
                    message,
                    profile_override=profile_override,
                    on_step=on_step,
                )
                # No model produced this answer, so it must not be reported as
                # LLM-backed.
                return response.model_copy(
                    update={
                        "llm_used": llm.llm_used_in_turn(),
                        "model_calls": llm.model_calls(),
                    }
                )
        finally:
            llm.end_turn(tokens)

    def stream(self, session_id: str, message: str) -> Iterator[tuple[str, Any]]:
        tokens = llm.begin_turn()
        try:
            if not self.graph_enabled:
                for kind, payload in self.orchestrator.stream(session_id, message):
                    if kind == "final":
                        payload = payload.model_copy(
                            update={
                                "llm_used": llm.llm_used_in_turn(),
                                "model_calls": llm.model_calls(),
                            }
                        )
                    yield kind, payload
                return

            try:
                yield from self._stream_graph(session_id, message)
            except Exception:  # pragma: no cover - provider/transport failure
                logger.exception(
                    "Agent graph stream failed; falling back to the deterministic engine"
                )
                # The student must still receive a verified answer, even if the
                # provider failed mid-run; a repeated step is preferable to
                # silence.
                for kind, payload in self.orchestrator.stream(session_id, message):
                    if kind == "final":
                        payload = payload.model_copy(
                            update={
                                "llm_used": llm.llm_used_in_turn(),
                                "model_calls": llm.model_calls(),
                            }
                        )
                    yield kind, payload
        finally:
            llm.end_turn(tokens)

    # ------------------------------------------------------------------
    # Graph execution
    # ------------------------------------------------------------------
    def _prepare(
        self, session_id: str, message: str, profile_override: StudentProfile | None
    ) -> tuple[ToolContext, Any, dict[str, Any]]:
        language = detect_language(message)
        profile = self.memory.load_profile(session_id)
        if profile_override is not None:
            profile = _merge_override(profile, profile_override)
        profile = profile.model_copy(update={"student_id": session_id})

        history = [
            {"role": item.role, "content": item.content}
            for item in self.memory.history(session_id, limit=settings.history_turns)
        ]
        artifacts = summarize_artifacts(
            self.memory.artifacts(session_id, limit=6)
        )

        user_date = parse_date_from_text(message)
        cards = self.repository.list_published(limit=50)
        focused_grant = None
        deadline_verdict_str = None
        deadline_fact = None
        if user_date is not None:
            focused_grant = find_focused_grant(message, cards, history, self.repository)
            deadline_verdict_str = deadline_verdict(user_date, focused_grant, language)
            if focused_grant is not None:
                deadline_fact = (
                    f"PRE-COMPUTED VERIFIED FACT: The user asks about the deadline of {focused_grant.title}. "
                    f"The verified deadline is {focused_grant.deadline or 'NOT ANNOUNCED/NULL'}. "
                    f"Your answer MUST open with the exact verdict: '{deadline_verdict_str}'."
                )
            else:
                deadline_fact = (
                    f"PRE-COMPUTED VERIFIED FACT: The user mentions a date, but no specific scholarship "
                    f"is in focus. You MUST answer by asking the student which scholarship they are asking about: "
                    f"'{deadline_verdict_str}'."
                )

        context = ToolContext(
            profile=profile,
            repository=self.repository,
            documents=self.documents,
            user_date=user_date,
            deadline_verdict=deadline_verdict_str,
        )
        tools = build_tools(context)
        if self._model_factory:
            raw_model = self._model_factory()
            if not isinstance(raw_model, RetryingChatModel):
                model = RetryingChatModel(primary=raw_model, max_retries=0)
            else:
                model = raw_model
        else:
            model = build_chat_model()
        graph = build_graph(model=model, tools=tools, context=context)

        initial: dict[str, Any] = {
            "session_id": session_id,
            "message": message,
            "language": language,
            "history": history,
            "artifacts": artifacts,
            "profile": profile.model_dump(mode="json"),
            "missing_fields": profile.missing_required(),
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
            "llm_used": False,
            "deadline_fact": deadline_fact,
            "deadline_verdict": deadline_verdict_str,
            "user_date": user_date,
            "focused_grant": focused_grant.model_dump(mode="json") if focused_grant else None,
        }
        return context, graph, initial

    def _chat_graph(
        self,
        session_id: str,
        message: str,
        *,
        profile_override: StudentProfile | None,
        on_step: Callable[[ThoughtStep], None] | None,
    ) -> ChatResponse:
        context, graph, initial = self._prepare(session_id, message, profile_override)
        final_state = graph.invoke(initial)
        return self._respond(
            session_id, message, final_state, context, on_step=on_step
        )

    def _stream_graph(
        self, session_id: str, message: str
    ) -> Iterator[tuple[str, Any]]:
        context, graph, initial = self._prepare(session_id, message, profile_override=None)
        seen = 0
        final_state: dict[str, Any] = initial
        for snapshot in graph.stream(initial, stream_mode="values"):
            final_state = snapshot
            steps = snapshot.get("steps") or []
            for raw in steps[seen:]:
                yield ("step", ThoughtStep.model_validate(raw))
            seen = len(steps)
        yield ("final", self._respond(session_id, message, final_state, context))

    # ------------------------------------------------------------------
    # Response assembly / persistence
    # ------------------------------------------------------------------
    def _respond(
        self,
        session_id: str,
        message: str,
        state: dict[str, Any],
        context: ToolContext,
        *,
        on_step: Callable[[ThoughtStep], None] | None = None,
    ) -> ChatResponse:
        profile_data = state.get("profile") or context.profile.model_dump(mode="json")
        profile = StudentProfile.model_validate(profile_data).model_copy(
            update={"student_id": session_id}
        )
        language = state.get("language") or "ar"
        language = "en" if language == "en" else "ar"
        reply = (state.get("reply") or "").strip() or fallback_reply(language)
        verdict = state.get("deadline_verdict")
        if verdict:
            if state.get("focused_grant") is None and state.get("user_date") is not None:
                reply = verdict
            elif not reply.startswith(verdict):
                reply = f"{verdict}\n\n{reply}"

        steps = [ThoughtStep.model_validate(raw) for raw in state.get("steps") or []]
        if on_step:
            for item in steps:
                try:
                    on_step(item)
                except Exception:  # pragma: no cover - stream consumer failure
                    logger.debug("step callback failed", exc_info=True)

        matches = [MatchResult.model_validate(raw) for raw in state.get("matches") or []]
        documents = [
            DocumentResult.model_validate(raw) for raw in state.get("documents") or []
        ]
        advisor = (
            AdvisorReport.model_validate(state["advisor"])
            if state.get("advisor")
            else None
        )
        plan = state.get("plan") or []

        # Persist the turn only after the graph succeeded (the fallback engine
        # owns persistence when this path raises).
        self.memory.save_profile(session_id, profile)
        self.memory.append_message(session_id, "user", message)
        self.memory.append_message(session_id, "assistant", reply)
        if matches:
            self.memory.save_artifact(
                session_id, "match", {"results": [item.model_dump(mode="json") for item in matches]}
            )
        if plan:
            self.memory.save_artifact(session_id, "plan", {"items": plan})
        if advisor is not None:
            self.memory.save_artifact(
                session_id, "advisor", advisor.model_dump(mode="json")
            )
        for document in documents:
            self.memory.save_artifact(
                session_id, "document", document.model_dump(mode="json")
            )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            intent=Intent(state.get("intent") or Intent.SCHOLARSHIP_QUESTION.value),
            language=language,  # type: ignore[arg-type]
            steps=steps,
            guardrails=[
                GuardrailEvent.model_validate(raw) for raw in state.get("guardrails") or []
            ],
            profile=profile,
            missing_fields=profile.missing_required(),
            matches=matches,
            plan=plan,
            documents=documents,
            advisor=advisor,
            sources=sorted({url for url in state.get("sources") or [] if url}),
            llm_used=llm.llm_used_in_turn(),
            model_calls=llm.model_calls(),
        )


def _merge_override(
    profile: StudentProfile, override: StudentProfile
) -> StudentProfile:
    patch = {
        key: value
        for key, value in override.model_dump().items()
        if key != "student_id" and value not in (None, "", [])
    }
    return merge_profile(profile, patch) if patch else profile
