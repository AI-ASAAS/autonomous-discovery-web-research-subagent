"""Matching delegates to the repository's pure scoring functions."""

from __future__ import annotations

import pytest

from agent_platform.knowledge import GrantRepository
from agent_platform.matching import IncompleteProfile, match_grants
from agent_platform.schemas import (
    DegreeLevel,
    LanguageLevel,
    LanguageScore,
    StudentProfile,
)

IELTS_65 = LanguageScore(
    language="English", level=LanguageLevel.B2, test_name="IELTS", score=6.5
)


def _computer_science_profile(gpa: float) -> StudentProfile:
    return StudentProfile(
        student_id="profile-a",
        name="Profile A",
        gpa=gpa,
        major="Computer Science",
        degree_level=DegreeLevel.BACHELOR,
        target_degree=DegreeLevel.MASTER,
        nationality="Yemeni",
        country_of_residence="Yemen",
        languages=[IELTS_65],
        research_interests=["Artificial Intelligence"],
        target_fields=["Computer Science"],
    )


def _marine_biology_profile() -> StudentProfile:
    return StudentProfile(
        student_id="profile-c",
        name="Profile C",
        gpa=3.0,
        major="Marine Biology",
        degree_level=DegreeLevel.BACHELOR,
        target_degree=DegreeLevel.MASTER,
        nationality="Yemeni",
        country_of_residence="Yemen",
        research_interests=[],
        target_fields=["Marine Biology"],
    )


def test_match_returns_sorted_scores(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    cards = repository.list_published(limit=50)
    assert len(cards) == 10

    report = match_grants(complete_profile, cards, use_llm=False)

    assert report.semantic_source == "deterministic"
    assert len(report.results) == 10
    assert 0.0 <= report.top_percentage <= 100.0
    percentages = [item.match_percentage for item in report.results]
    assert percentages == sorted(percentages, reverse=True)
    assert all(0.0 <= value <= 100.0 for value in percentages)
    assert report.results[0].title
    assert report.results[0].official_url.startswith("http")


def test_match_limit_is_respected(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    cards = repository.list_published(limit=50)
    report = match_grants(complete_profile, cards, limit=3, use_llm=False)
    assert len(report.results) == 3


def test_incomplete_profile_is_rejected(
    repository: GrantRepository,
) -> None:
    with pytest.raises(IncompleteProfile):
        match_grants(StudentProfile(), repository.list_published(limit=5))


def test_no_grants_triggers_external_search() -> None:
    profile = StudentProfile(
        name="أحمد",
        gpa=3.5,
        major="علوم الحاسوب",
        degree_level="bachelor",
        target_degree="master",
        nationality="Yemeni",
    )
    report = match_grants(profile, [], use_llm=False)
    assert report.results == []
    assert report.trigger_external_search is True
    assert "لا توجد منح" in (report.external_search_reason or "")


def _best_semantic(report) -> float:
    return max(
        (
            item.breakdown.semantic_alignment_score
            for item in report.results
            if not item.disqualifiers
        ),
        default=0.0,
    )


def test_computer_science_profile_does_not_trigger_external_search(
    repository: GrantRepository,
) -> None:
    """A strong, well-aligned match must stay internal."""
    report = match_grants(
        _computer_science_profile(3.6), repository.list_published(limit=50), use_llm=False
    )
    assert report.top_percentage >= 40.0
    assert _best_semantic(report) >= 60.0
    assert report.trigger_external_search is False


def test_marine_biology_profile_triggers_external_search(
    repository: GrantRepository,
) -> None:
    """A high numeric score with weak semantic fit still triggers search."""
    report = match_grants(
        _marine_biology_profile(), repository.list_published(limit=50), use_llm=False
    )
    assert report.top_percentage >= 40.0
    assert _best_semantic(report) < 60.0
    assert report.trigger_external_search is True
    assert "دلالي" in (report.external_search_reason or "")


def test_all_disqualified_grants_trigger_external_search(
    repository: GrantRepository,
) -> None:
    """When every grant is disqualified the top is below 40 and search triggers."""
    ineligible = repository.list_published(limit=1)[0].model_copy(
        update={"accepts_yemeni": "no"}
    )
    report = match_grants(
        _computer_science_profile(3.6), [ineligible], use_llm=False
    )
    assert report.results and report.results[0].disqualifiers
    assert report.top_percentage < 40.0
    assert report.trigger_external_search is True
    assert "أقل من الحد" in (report.external_search_reason or "")


def test_disqualified_grants_score_zero(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    cards = repository.list_published(limit=50)
    report = match_grants(complete_profile, cards, use_llm=False)
    for result in report.results:
        if result.disqualifiers:
            assert result.match_percentage == 0.0
