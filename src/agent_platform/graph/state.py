"""Shared state for the supervisor/tools graph.

The state is deliberately JSON-friendly: every node returns plain dicts so a
checkpointer could persist a run without custom serializers.  The only
non-trivial reducer is ``messages`` (conversation, append + de-duplicate by id)
and ``steps`` / ``sources`` / ``tools_used`` (append).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # -- conversation for the current turn ------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    history: list[dict[str, str]]
    # Compact summaries of earlier artifacts (matches, plans, documents,
    # advisor reports) so the agent can reuse the session's own work.
    artifacts: list[str]

    # -- request ---------------------------------------------------------
    session_id: str
    message: str
    language: str

    # -- profile ---------------------------------------------------------
    profile: dict[str, Any]
    missing_fields: list[str]

    # -- guardrails / observability -------------------------------------
    guardrails: list[dict[str, Any]]
    steps: Annotated[list[dict[str, Any]], operator.add]
    blocked: bool
    intent: str
    reply: str
    tool_rounds: int
    tools_used: Annotated[list[str], operator.add]

    # -- structured results produced by tools ---------------------------
    matches: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    advisor: dict[str, Any] | None
    sources: Annotated[list[str], operator.add]

    # -- meta ------------------------------------------------------------
    llm_used: bool
    deadline_fact: str | None
    deadline_verdict: str | None
    user_date: tuple[int, int, int | None] | None
    focused_grant: dict[str, Any] | None
