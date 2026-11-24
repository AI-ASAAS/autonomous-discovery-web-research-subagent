"""Stateful scholarship orchestrator platform.

This package implements the missing pieces of the four-person plan:

* Person 1: central orchestrator with memory, proactive routing and guardrails.
* Person 4: reflection document agent, profile strengthening advisor and the
  FastAPI boundary that exposes both to the existing frontends.

It intentionally reuses the working sub-agents that already exist in the two
repositories (matcher, planner, grants repository, external search) instead of
rewriting them.
"""

__version__ = "1.0.0"
