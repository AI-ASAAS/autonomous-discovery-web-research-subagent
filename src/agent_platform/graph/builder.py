"""Compile the supervisor/tools agent graph.

Topology::

    START -> intake -> (blocked) -----------------------------> finalize -> END
                    \\-> supervisor -> (tool calls) -> tools -/
                                    \\-> (answer) ----> finalize

* ``intake`` applies input guardrails, detects the language and merges the
  facts stated in the message into the student's profile.
* ``supervisor`` is the LLM with the tools bound; it either requests tools or
  writes the final answer.
* ``tools`` executes the requested tools and folds their structured output
  (matches, plan, documents, advisor) into the graph state.
* ``finalize`` applies output guardrails and derives the response intent.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

with warnings.catch_warnings():
    # Third-party PendingDeprecationWarning from langgraph's serializer import;
    # the default will change in a future release (not actionable here).
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change",
        category=PendingDeprecationWarning,
    )
    from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..extraction import detect_language, extract_profile_patch, merge_profile
from ..guardrails import check_input, check_output
from ..orchestrator import refusal_for
from ..schemas import GuardrailAction, Intent, StudentProfile, ThoughtStep
from .prompts import build_system_prompt, fallback_reply
from .state import AgentState
from .tools import ToolContext

logger = logging.getLogger(__name__)

# Tool -> response intent, in priority order (first match wins).
_INTENT_PRIORITY: tuple[tuple[str, str], ...] = (
    ("write_document", Intent.DOCUMENTS.value),
    ("build_application_plan", Intent.PLAN.value),
    ("advise_profile", Intent.ADVISOR.value),
    ("search_external_grants", Intent.EXTERNAL_SEARCH.value),
    ("match_profile", Intent.MATCH.value),
    ("list_verified_grants", Intent.TOP_GRANTS.value),
    ("get_grant_details", Intent.SCHOLARSHIP_QUESTION.value),
    ("update_profile", Intent.PROFILE_UPDATE.value),
)


def _step(
    name: str, label: str, status: str = "completed", detail: str = ""
) -> dict[str, Any]:
    return ThoughtStep(
        step=name, label=label, status=status, detail=detail  # type: ignore[arg-type]
    ).model_dump(mode="json")


def _message_text(message: Any) -> str:
    """Extract plain text from an ``AIMessage`` content payload."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _collect_payload(accumulator: dict[str, Any], raw: str) -> None:
    """Fold a tool observation envelope into structured response fields."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return
    if not isinstance(payload, dict) or not payload.get("ok", True):
        return
    kind = payload.get("kind")
    data = payload.get("data") or {}
    if kind == "match":
        accumulator["matches"].extend(data.get("results") or [])
    elif kind == "plan":
        accumulator["plan"] = data.get("plans") or accumulator["plan"]
    elif kind == "document":
        accumulator["documents"].append(data)
    elif kind == "advisor":
        accumulator["advisor"] = data
    elif kind == "profile" and isinstance(data, dict):
        accumulator["profile"] = data
    for url in payload.get("sources") or []:
        if url:
            accumulator["sources"].append(url)


def derive_intent(state: AgentState) -> str:
    if state.get("blocked"):
        return Intent.BLOCKED.value
    used = set(state.get("tools_used") or [])
    for tool_name, intent in _INTENT_PRIORITY:
        if tool_name in used:
            return intent
    if state.get("missing_fields"):
        return Intent.PROFILE_UPDATE.value
    return Intent.SCHOLARSHIP_QUESTION.value


def build_graph(
    *,
    model: BaseChatModel,
    tools: list[BaseTool],
    context: ToolContext | None = None,
    max_tool_rounds: int | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the supervisor graph for one conversational turn.

    ``context`` is the shared per-turn tool context: the intake node refreshes
    its profile from extracted facts, and tool-driven corrections are mirrored
    back into the graph state.
    """
    rounds_limit = max(1, int(max_tool_rounds or settings.graph_max_tool_rounds))
    bound_model = model.bind_tools(tools)
    registry: dict[str, BaseTool] = {tool_item.name: tool_item for tool_item in tools}

    # -- nodes ---------------------------------------------------------
    def intake_node(state: AgentState) -> dict[str, Any]:
        message = state.get("message") or ""
        language = state.get("language") or detect_language(message)
        decision = check_input(message, max_chars=settings.max_message_chars)
        guard_events = [decision.event.model_dump(mode="json")] if decision.event else []
        base: dict[str, Any] = {
            "language": language,
            "guardrails": guard_events,
            "messages": [HumanMessage(content=message)],
        }

        if decision.action == GuardrailAction.REFUSE:
            reply = refusal_for(decision.category, language)
            base.update(
                {
                    "blocked": True,
                    "reply": reply,
                    "steps": [
                        _step(
                            "intake",
                            "فحص الرسالة وحواجز الأمان",
                            "failed",
                            decision.category.value,
                        )
                    ],
                }
            )
            return base

        current = StudentProfile.model_validate(state.get("profile") or {})
        patch = extract_profile_patch(decision.text)
        if patch:
            current = merge_profile(current, patch)
        if context is not None:
            # Tools must observe the facts stated in this very message.
            context.profile = current
        missing = current.missing_required()
        base.update(
            {
                "profile": current.model_dump(mode="json"),
                "missing_fields": missing,
                "steps": [
                    _step("intake", "فحص الرسالة وحواجز الأمان", detail=decision.category.value),
                    _step(
                        "profile",
                        "تحديث ملف الطالب",
                        detail=f"completeness={current.completeness():.0f}%",
                    ),
                ],
            }
        )
        return base

    def supervisor_node(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=build_system_prompt(state))]
        messages.extend(state.get("messages") or [])
        rounds = int(state.get("tool_rounds") or 0)

        if rounds >= rounds_limit:
            # Budget exhausted: force an answer with tools disabled.
            answer = model.invoke(messages)
            return {
                "messages": [answer],
                "reply": _message_text(answer),
                "tool_rounds": rounds + 1,
                "steps": [
                    _step(
                        "supervisor",
                        "حد النمط التجريبي: إيقاف الأدوات وتلخيص النتيجة",
                        "skipped",
                        f"rounds={rounds}",
                    )
                ],
            }

        response = bound_model.invoke(messages)
        update: dict[str, Any] = {
            "messages": [response],
            "tool_rounds": rounds + 1,
        }
        calls = getattr(response, "tool_calls", None) or []
        if calls:
            update["steps"] = [
                _step(
                    "supervisor",
                    "اختيار الأدوات",
                    detail=", ".join(call.get("name", "?") for call in calls),
                )
            ]
        else:
            update["reply"] = _message_text(response)
            update["steps"] = [
                _step("supervisor", "صياغة الإجابة النهائية", detail="no tools requested")
            ]
        return update

    def tools_node(state: AgentState) -> dict[str, Any]:
        last = (state.get("messages") or [None])[-1]
        calls = getattr(last, "tool_calls", None) or []
        tool_messages: list[ToolMessage] = []
        accumulator: dict[str, Any] = {
            "matches": [],
            "plan": [],
            "documents": [],
            "advisor": None,
            "sources": [],
            "profile": None,
        }
        used: list[str] = []
        steps: list[dict[str, Any]] = []

        for call in calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            tool_item = registry.get(name)
            if tool_item is None:
                observation = json.dumps(
                    {"kind": "error", "ok": False, "summary": f"Unknown tool {name!r}."}
                )
                status = "failed"
            else:
                try:
                    observation = tool_item.invoke(args)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Tool %s failed", name)
                    observation = json.dumps(
                        {"kind": "error", "ok": False, "summary": str(exc)}
                    )
                    status = "failed"
                else:
                    status = "completed"
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=call.get("id") or name,
                    name=name,
                )
            )
            used.append(name)
            _collect_payload(accumulator, observation)
            steps.append(
                _step(
                    f"tool:{name}",
                    f"تنفيذ الأداة {name}",
                    status,
                    _tool_detail(observation),
                )
            )

        update: dict[str, Any] = {
            "messages": tool_messages,
            "tools_used": used,
            "steps": steps,
            "sources": accumulator["sources"],
            "matches": list(state.get("matches") or []) + accumulator["matches"],
        }
        if accumulator["plan"]:
            update["plan"] = accumulator["plan"]
        if accumulator["documents"]:
            update["documents"] = list(state.get("documents") or []) + accumulator["documents"]
        if accumulator["advisor"]:
            update["advisor"] = accumulator["advisor"]

        # A tool may have corrected the profile; mirror the shared object back.
        if accumulator["profile"]:
            merged = StudentProfile.model_validate(accumulator["profile"])
            if context is not None:
                context.profile = merged
                context.profile_dirty = True
            update["profile"] = merged.model_dump(mode="json")
            update["missing_fields"] = merged.missing_required()
        return update

    def finalize_node(state: AgentState) -> dict[str, Any]:
        language = state.get("language") or "ar"
        reply = (state.get("reply") or "").strip()
        if not reply:
            reply = fallback_reply(language)

        fact = state.get("deadline_fact") or (context.deadline_fact if context else None)
        if fact:
            from ..deadlines import apply_deadline_fact
            reply = apply_deadline_fact(reply, fact)

        report = check_output(reply)
        guard_events = list(state.get("guardrails") or [])
        guard_events.extend(event.model_dump(mode="json") for event in report.events)
        if not report.passed:
            reply = fallback_reply(language)

        intents = derive_intent(state)
        return {
            "reply": report.text if report.passed else reply,
            "guardrails": guard_events,
            "intent": intents,
            "steps": [
                _step("output_guard", "تدقيق المخرجات", detail=f"{len(report.issues)} issues"),
                _step("done", "تجهيز الرد", detail=f"intent={intents}"),
            ],
        }

    # -- edges ---------------------------------------------------------
    def after_intake(state: AgentState) -> str:
        return "blocked" if state.get("blocked") else "continue"

    def after_supervisor(state: AgentState) -> str:
        last = (state.get("messages") or [None])[-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake", after_intake, {"blocked": "finalize", "continue": "supervisor"}
    )
    graph.add_conditional_edges(
        "supervisor", after_supervisor, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def _tool_detail(observation: str) -> str:
    try:
        payload = json.loads(observation)
    except (TypeError, ValueError):
        return observation[:100]
    summary = payload.get("summary") or ""
    return str(summary)[:120]
