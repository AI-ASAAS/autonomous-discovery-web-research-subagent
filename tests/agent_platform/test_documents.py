"""The document agent drafts, critiques, revises and grounds every document."""

from __future__ import annotations

import pytest

from agent_platform.document_agent import DocumentAgent
from agent_platform.knowledge import GrantRepository
from agent_platform.schemas import DocumentType, StudentProfile


def _grant(repository: GrantRepository):
    return repository.list_published(limit=1)[0]


@pytest.mark.parametrize("document_type", list(DocumentType))
@pytest.mark.parametrize("language", ["ar", "en"])
def test_reflection_pipeline_is_complete(
    repository: GrantRepository,
    complete_profile: StudentProfile,
    document_type: DocumentType,
    language: str,
) -> None:
    result = DocumentAgent().generate(
        complete_profile, _grant(repository), document_type, language=language
    )

    assert [revision.stage for revision in result.revisions] == [
        "draft",
        "critique",
        "revised",
        "final",
    ]
    assert result.content.strip()
    assert result.content == result.revisions[-1].content
    assert result.provider == "deterministic"
    assert result.grounded is True
    assert not [issue for issue in result.issues if issue.severity == "critical"]
    # Documents must never contain live links; the official URL is presented
    # separately by the UI.
    assert "http://" not in result.content
    assert "https://" not in result.content


def test_brief_is_not_a_signed_recommendation(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    result = DocumentAgent().generate(
        complete_profile,
        _grant(repository),
        DocumentType.RECOMMENDATION_BRIEF,
        language="ar",
    )
    assert "ليست رسالة توصية" in result.content
    assert "[[" in result.content
    assert any(issue.code == "placeholder" for issue in result.issues)
    # Placeholders are a warning, never a critical failure.
    assert result.passed is True


def test_cv_contains_profile_facts(
    repository: GrantRepository, complete_profile: StudentProfile
) -> None:
    result = DocumentAgent().generate(
        complete_profile, _grant(repository), DocumentType.CV, language="en"
    )
    assert complete_profile.name in result.content
    assert complete_profile.major in result.content


def test_incomplete_profile_cannot_generate(
    repository: GrantRepository,
) -> None:
    with pytest.raises(ValueError):
        DocumentAgent().generate(
            StudentProfile(),
            _grant(repository),
            DocumentType.MOTIVATION_LETTER,
            language="ar",
        )
