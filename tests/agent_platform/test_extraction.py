"""Fact extraction must never invent data and must merge safely."""

from __future__ import annotations

from agent_platform.extraction import (
    detect_language,
    extract_profile_patch,
    merge_profile,
)
from agent_platform.schemas import DegreeLevel, LanguageLevel, LanguageScore, StudentProfile

SENTENCE = (
    "اسمي أحمد اليمني، طالب يمني، خريج بكالوريوس في علوم الحاسوب "
    "بمعدل 3.6 وأريد الماجستير"
)


def test_extracts_complete_profile_from_one_sentence() -> None:
    patch = extract_profile_patch(SENTENCE)
    assert patch["name"] == "أحمد اليمني"
    assert patch["gpa"] == 3.6
    assert patch["major"] == "علوم الحاسوب"
    assert patch["degree_level"] == DegreeLevel.BACHELOR
    assert patch["target_degree"] == DegreeLevel.MASTER
    assert patch["nationality"] == "Yemeni"


def test_no_invention_on_small_talk() -> None:
    assert extract_profile_patch("مرحبا كيف حالك اليوم") == {}


def test_partial_facts_only() -> None:
    patch = extract_profile_patch("معدلي 3.2")
    assert patch == {"gpa": 3.2}


def test_language_detection() -> None:
    assert detect_language("أريد منحة ماجستير") == "ar"
    assert detect_language("I want a master scholarship") == "en"


def test_merge_keeps_known_values_and_upgrades_language() -> None:
    base = StudentProfile(
        name="أحمد",
        gpa=3.1,
        major="علوم الحاسوب",
        languages=[LanguageScore(language="English", level=LanguageLevel.B2)],
    )
    merged = merge_profile(
        base,
        {
            "gpa": 3.6,
            "languages": [{"language": "English", "level": "C1"}],
        },
    )
    assert merged.name == "أحمد"
    assert merged.major == "علوم الحاسوب"
    assert merged.gpa == 3.6
    assert merged.languages[0].level == LanguageLevel.C1


def test_merge_does_not_downgrade_language() -> None:
    base = StudentProfile(
        languages=[LanguageScore(language="English", level=LanguageLevel.C1)]
    )
    merged = merge_profile(
        base, {"languages": [{"language": "English", "level": "B1"}]}
    )
    assert merged.languages[0].level == LanguageLevel.C1


def test_completeness_and_missing_fields() -> None:
    empty = StudentProfile()
    assert empty.completeness() == 0.0
    assert set(empty.missing_required()) == {
        "name",
        "gpa",
        "major",
        "degree_level",
        "target_degree",
        "nationality",
    }
    full = StudentProfile(
        name="أحمد",
        gpa=3.6,
        major="علوم الحاسوب",
        degree_level=DegreeLevel.BACHELOR,
        target_degree=DegreeLevel.MASTER,
        nationality="Yemeni",
    )
    assert full.completeness() == 100.0
    assert full.missing_required() == []
