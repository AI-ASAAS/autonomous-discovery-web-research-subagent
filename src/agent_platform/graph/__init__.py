"""LangGraph-based agent engine.

The package turns the platform's domain capabilities (verified grants,
programmatic matching, planning, reflection documents, advisory and external
discovery) into **tools** that a supervisor model can call in a loop until it
can answer the student.

Public surface:

``AgentService``
    Drop-in replacement for ``Orchestrator.chat`` / ``Orchestrator.stream`` that
    runs the graph when a provider is configured and delegates to the
    deterministic orchestrator otherwise.

``build_graph``
    Compiles the supervisor/tools state graph (used by tests with a scripted
    model).
"""

from .service import AgentService
from .state import AgentState

__all__ = ["AgentService", "AgentState", "build_graph"]

from .builder import build_graph  # noqa: E402  (re-export after AgentState)
