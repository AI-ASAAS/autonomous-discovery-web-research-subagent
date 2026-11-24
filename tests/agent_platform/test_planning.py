"""Application planning builds concrete, deadline-ordered plans."""

from __future__ import annotations

from agent_platform.knowledge import GrantRepository
from agent_platform.matching import match_grants
from agent_platform.planning import build_plan, days_until, plan_summary
from agent_platform.schemas import StudentProfile


def test_plan_has_ranked_steps_and_documents(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    cards = repository.list_published(limit=50)
    report = match_grants(complete_profile, cards, use_llm=False)

    items = build_plan(complete_profile, report, cards, top_n=3)

    assert len(items) == 3
    assert [item["priority_rank"] for item in items] == [1, 2, 3]
    for item in items:
        assert item["grant_title"]
        assert item["official_url"].startswith("http")
        assert item["steps"], "plan must contain concrete steps"
        assert item["documents"], "plan must list required documents"
        assert item["tips"]
        assert 10 <= item["estimated_preparation_days"] <= 60


def test_plan_summary_uses_arabic_plural(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    cards = repository.list_published(limit=50)
    report = match_grants(complete_profile, cards, use_llm=False)

    summary = plan_summary(build_plan(complete_profile, report, cards, top_n=3))
    assert "خطتك تشمل 3 منح" in summary
    assert "الأولوية الأولى" in summary

    single = plan_summary(build_plan(complete_profile, report, cards, top_n=1))
    assert "منحة واحدة" in single


def test_empty_plan_summary() -> None:
    assert "لا توجد منح" in plan_summary([])


def test_days_until_handles_missing_deadline() -> None:
    assert days_until(None) == 0
    assert days_until("not-a-date") == 0
    assert days_until("2999-01-01") > 0
