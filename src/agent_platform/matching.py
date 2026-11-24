"""Hybrid matching adapter.

Scoring is delegated to the already-tested pure functions of the matcher
repository (hard filters for GPA, degree, language, nationality, major,
experience and publications).  The semantic-alignment component is computed
deterministically from the profile and the grant text, and is only replaced by
an LLM call when a provider is explicitly configured.

No network call happens in deterministic mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import Any, Iterable

from . import llm
from .config import settings
from .repos import setup_repo_paths
from .schemas import (
    DegreeLevel,
    GrantCard,
    LanguageLevel,
    LanguageScore,
    MatchReport,
    MatchResult,
    MatchTier,
    ScoreBreakdown,
    StudentProfile,
)

logger = logging.getLogger(__name__)

setup_repo_paths()

from ai.scholarship_agent.agents.matcher_evaluator_agent import (  # noqa: E402
    WEIGHTS,
    _tier,
    compute_programmatic_scores,
)
from ai.scholarship_agent.schemas.scholarship_schemas import (  # noqa: E402
    DegreeLevel as RepoDegreeLevel,
)
from ai.scholarship_agent.schemas.scholarship_schemas import (  # noqa: E402
    LanguageLevel as RepoLanguageLevel,
)
from ai.scholarship_agent.schemas.scholarship_schemas import (  # noqa: E402
    LanguageScore as RepoLanguageScore,
)
from ai.scholarship_agent.schemas.scholarship_schemas import (  # noqa: E402
    Scholarship as RepoScholarship,
)
from ai.scholarship_agent.schemas.scholarship_schemas import (  # noqa: E402
    ScholarshipRequirements as RepoScholarshipRequirements,
)
from ai.scholarship_agent.schemas.scholarship_schemas import (  # noqa: E402
    StudentProfile as RepoStudentProfile,
)

_DEGREE_MAP = {
    DegreeLevel.HIGH_SCHOOL: RepoDegreeLevel.HIGH_SCHOOL,
    DegreeLevel.BACHELOR: RepoDegreeLevel.BACHELOR,
    DegreeLevel.MASTER: RepoDegreeLevel.MASTER,
    DegreeLevel.PHD: RepoDegreeLevel.PHD,
}
_LANGUAGE_MAP = {
    LanguageLevel.A1: RepoLanguageLevel.A1,
    LanguageLevel.A2: RepoLanguageLevel.A2,
    LanguageLevel.B1: RepoLanguageLevel.B1,
    LanguageLevel.B2: RepoLanguageLevel.B2,
    LanguageLevel.C1: RepoLanguageLevel.C1,
    LanguageLevel.C2: RepoLanguageLevel.C2,
    LanguageLevel.NATIVE: RepoLanguageLevel.NATIVE,
}

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "will", "have", "has", "been", "their", "your", "our", "students", "student",
    "scholarship", "scholarships", "program", "programme", "study", "studies",
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "منحة", "منح", "دراسية", "الدراسة", "برنامج", "الطلاب", "الطالب",
}


class IncompleteProfile(ValueError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"Incomplete profile, missing: {', '.join(missing)}")


def _tokens(text: str) -> set[str]:
    words = re.split(r"[^\w\u0600-\u06FF]+", (text or "").lower())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def _to_repo_profile(profile: StudentProfile) -> RepoStudentProfile:
    missing = profile.missing_required()
    if missing:
        raise IncompleteProfile(missing)
    return RepoStudentProfile(
        student_id=profile.student_id,
        name=profile.name or "",
        degree_level=_DEGREE_MAP[profile.degree_level],  # type: ignore[index]
        gpa=float(profile.gpa),  # type: ignore[arg-type]
        major=profile.major or "",
        research_interests=profile.research_interests,
        languages=[
            RepoLanguageScore(
                language=item.language,
                level=_LANGUAGE_MAP[item.level],
                test_name=item.test_name,
                score=item.score,
            )
            for item in profile.languages
        ],
        nationality=profile.nationality or "",
        country_of_residence=profile.country_of_residence or "Yemen",
        experiences=profile.experiences,
        publications=profile.publications,
        extracurriculars=profile.extracurriculars,
        target_degree=_DEGREE_MAP[profile.target_degree],  # type: ignore[index]
        target_countries=profile.target_countries,
        target_fields=profile.target_fields,
    )


def _infer_degree_levels(text: str) -> list[RepoDegreeLevel]:
    lowered = text.lower()
    levels: list[RepoDegreeLevel] = []
    if re.search(r"\b(bachelor|undergraduate|بكالوريوس)\b", lowered):
        levels.append(RepoDegreeLevel.BACHELOR)
    if re.search(r"\b(master|postgraduate|ماجستير)\b", lowered):
        levels.append(RepoDegreeLevel.MASTER)
    if re.search(r"\b(phd|doctoral|doctorate|دكتوراه)\b", lowered):
        levels.append(RepoDegreeLevel.PHD)
    return levels or [RepoDegreeLevel.BACHELOR, RepoDegreeLevel.MASTER, RepoDegreeLevel.PHD]


def _parse_min_gpa(text: str, target_degree: DegreeLevel) -> float | None:
    """Convert a percentage requirement (e.g. 75% for master's) to a 4.0 GPA."""
    keywords = {
        DegreeLevel.BACHELOR: ("bachelor", "undergraduate", "بكالوريوس"),
        DegreeLevel.MASTER: ("master", "postgraduate", "ماجستير"),
        DegreeLevel.PHD: ("phd", "doctoral", "doctorate", "دكتوراه"),
    }[target_degree]
    candidates: list[float] = []
    generic: list[float] = []
    funding_words = (
        "tuition",
        "fund",
        "cover",
        "stipend",
        "allowance",
        "منحة",
        "تمويل",
        "تغطية",
    )
    academic_words = ("minimum", "at least", "achievement", "grade", "gpa", "معدل", "بمعدل")
    for match in re.finditer(r"(\d{2,3})\s*%", text):
        percentage = int(match.group(1))
        if not 40 <= percentage <= 100:
            continue
        context = text[max(0, match.start() - 80) : match.end() + 80].lower()
        if any(word in context for word in funding_words):
            continue
        if any(keyword in context for keyword in keywords):
            candidates.append(percentage * 4.0 / 100.0)
        elif any(word in context for word in academic_words):
            generic.append(percentage * 4.0 / 100.0)
    if candidates:
        return round(min(candidates), 2)
    if len(generic) == 1:
        return round(generic[0], 2)
    return None


def _parse_language_requirement(text: str) -> list[RepoLanguageScore]:
    lowered = text.lower()
    requirements: list[RepoLanguageScore] = []
    ielts = re.search(r"ielts[^0-9]{0,20}(\d(?:\.\d)?)", lowered)
    toefl = re.search(r"toefl[^0-9]{0,20}(\d{2,3})", lowered)
    if ielts:
        score = float(ielts.group(1))
        level = (
            RepoLanguageLevel.C1
            if score >= 7.0
            else RepoLanguageLevel.B2
            if score >= 5.5
            else RepoLanguageLevel.B1
        )
        requirements.append(
            RepoLanguageScore(
                language="English", level=level, test_name="IELTS", score=score
            )
        )
    elif toefl:
        score = float(toefl.group(1))
        level = RepoLanguageLevel.C1 if score >= 90 else RepoLanguageLevel.B2
        requirements.append(
            RepoLanguageScore(
                language="English", level=level, test_name="TOEFL", score=score
            )
        )
    return requirements


def _to_repo_scholarship(card: GrantCard, target_degree: DegreeLevel) -> RepoScholarship:
    text_blob = " ".join(
        [
            card.title,
            card.description,
            " ".join(card.eligibility),
            " ".join(card.criteria),
            " ".join(card.fields_of_study),
        ]
    )
    if card.accepts_yemeni == "yes":
        nationalities = ["Yemeni"]
    elif card.accepts_yemeni == "no":
        nationalities = ["__ineligible__"]
    else:
        nationalities = []

    funding_text = card.funding.lower()
    funding_type = (
        "full"
        if any(word in funding_text for word in ("full", "free", "tuition", "stipend", "كامل", "ممولة"))
        else "partial"
        if funding_text
        else "unknown"
    )

    return RepoScholarship(
        scholarship_id=card.grant_id,
        name=card.title,
        provider=card.provider or "Unknown provider",
        country=card.country or "Unknown",
        description=card.description or card.title,
        mission_statement=(card.description or card.title)[:600],
        requirements=RepoScholarshipRequirements(
            min_gpa=_parse_min_gpa(text_blob, target_degree),
            degree_levels=_infer_degree_levels(text_blob),
            eligible_majors=[],
            eligible_nationalities=nationalities,
            language_requirements=_parse_language_requirement(text_blob),
            required_documents=card.required_documents,
            min_publications=0,
        ),
        deadline=card.deadline or "2099-12-31",
        funding_type=funding_type,
        fields_of_study=card.fields_of_study,
        application_url=card.official_url,
        extra_info={"deadline_known": card.deadline is not None},
    )


def _deterministic_semantic(profile: StudentProfile, card: GrantCard) -> float:
    interest_text = " ".join(
        [profile.major or "", *profile.research_interests, *profile.target_fields]
    )
    interest_tokens = _tokens(interest_text)
    if not interest_tokens:
        return 50.0
    grant_tokens = _tokens(
        " ".join(
            [
                card.title,
                card.description,
                card.funding,
                *card.fields_of_study,
                *card.criteria,
            ]
        )
    )
    if not grant_tokens:
        return 45.0
    matched = len(interest_tokens & grant_tokens)
    ratio = matched / max(3, len(interest_tokens))
    score = 40.0 + 50.0 * min(1.0, ratio)
    major_tokens = _tokens(profile.major or "")
    if major_tokens & grant_tokens:
        score += 8.0
    return round(max(20.0, min(95.0, score)), 1)


_SEMANTIC_CACHE: dict[str, dict[str, float]] = {}
_SEMANTIC_CACHE_LOCK = threading.Lock()


def clear_semantic_cache() -> None:
    """Drop cached semantic scores (test isolation and explicit resets)."""
    with _SEMANTIC_CACHE_LOCK:
        _SEMANTIC_CACHE.clear()


def _semantic_cache_key(
    profile: StudentProfile, cards: list[GrantCard], config: Any
) -> str:
    payload = {
        "profile": profile.model_dump(mode="json"),
        "grants": [
            (
                card.grant_id,
                card.title,
                card.description,
                card.funding,
                tuple(card.fields_of_study),
            )
            for card in cards
        ],
        "provider": config.llm_provider,
        "model": config.llm_model,
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _llm_semantic_scores(
    profile: StudentProfile, cards: Iterable[GrantCard], config=settings
) -> dict[str, float]:
    cards = list(cards)
    if not cards or not config.llm_configured:
        return {}
    cache_key = _semantic_cache_key(profile, cards, config)
    with _SEMANTIC_CACHE_LOCK:
        cached = _SEMANTIC_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    payload = [
        {
            "grant_id": card.grant_id,
            "title": card.title,
            "mission": (card.description or "")[:350],
            "fields": card.fields_of_study,
        }
        for card in cards
    ]
    prompt = f"""You are a scholarship evaluator. Score the semantic alignment between
the student's academic interests and each grant mission on a 0-100 scale.

Student major: {profile.major}
Research interests: {", ".join(profile.research_interests)}
Target fields: {", ".join(profile.target_fields)}

Grants:
{json.dumps(payload, ensure_ascii=False)}

Return strict JSON: {{"<grant_id>": <integer 0-100>, ...}} with every grant id.
Do not add explanations or markdown."""
    try:
        data = llm.complete_json(prompt, temperature=0.0)
    except Exception as exc:
        logger.warning("LLM semantic scoring unavailable: %s", exc)
        return {}
    scores: dict[str, float] = {}
    for card in cards:
        raw = data.get(card.grant_id)
        if isinstance(raw, (int, float)) and 0 <= float(raw) <= 100:
            scores[card.grant_id] = float(raw)
    if scores:
        with _SEMANTIC_CACHE_LOCK:
            _SEMANTIC_CACHE[cache_key] = dict(scores)
    return scores


def _tier_from_percentage(percentage: float) -> MatchTier:
    return MatchTier(_tier(percentage).value)


def _reasons(
    profile: StudentProfile,
    card: GrantCard,
    breakdown: ScoreBreakdown,
    disqualifiers: list[str],
) -> list[str]:
    if disqualifiers:
        return disqualifiers[:3]
    reasons: list[str] = []
    if breakdown.gpa_score >= 80:
        reasons.append(f"المعدل التراكمي {profile.gpa:.2f}/4.00 قوي مقارنة بمتطلبات المنحة.")
    if breakdown.degree_score >= 100:
        reasons.append("المرحلة الدراسية المستهدفة مشمولة في شروط المنحة.")
    if breakdown.language_score >= 80:
        reasons.append("مستوى اللغة المسجل يستوفي متطلبات المنحة.")
    elif breakdown.language_score == 0:
        reasons.append("لا يوجد إثبات لغة مطابق لمتطلبات المنحة.")
    if breakdown.major_score >= 80:
        reasons.append(f"تخصصك ({profile.major}) متوافق مع مجالات المنحة.")
    if breakdown.semantic_alignment_score >= 60:
        reasons.append("اهتماماتك البحثية متوافقة مع رسالة المنحة.")
    if card.accepts_yemeni == "unclear":
        reasons.append("أهلية الطلاب اليمنيين غير مؤكدة صراحةً، تحقق من الموقع الرسمي.")
    if card.deadline:
        reasons.append(f"الموعد النهائي: {card.deadline}.")
    return reasons[:4]


def match_grants(
    profile: StudentProfile,
    cards: Iterable[GrantCard],
    *,
    limit: int | None = None,
    use_llm: bool = True,
    config=settings,
) -> MatchReport:
    """Rank grants for a profile with hard filters + semantic alignment."""
    card_list = list(cards)
    limit = limit or len(card_list)
    if not card_list:
        return MatchReport(
            results=[],
            top_percentage=0.0,
            trigger_external_search=True,
            external_search_reason="لا توجد منح منشورة في قاعدة البيانات حالياً.",
        )

    repo_profile = _to_repo_profile(profile)
    repo_scholarships = [
        _to_repo_scholarship(card, profile.target_degree)  # type: ignore[arg-type]
        for card in card_list
    ]
    programmatic = compute_programmatic_scores(repo_profile, repo_scholarships)

    llm_scores: dict[str, float] = {}
    if use_llm and config.llm_configured:
        llm_scores = _llm_semantic_scores(profile, card_list, config)

    by_id = {card.grant_id: card for card in card_list}
    results: list[MatchResult] = []
    for scholarship_id, scores in programmatic.items():
        card = by_id[scholarship_id]
        semantic = llm_scores.get(
            scholarship_id, _deterministic_semantic(profile, card)
        )
        disqualifiers = list(scores["disqualifiers"])
        if disqualifiers:
            total = 0.0
        else:
            total = (
                scores["gpa_score"] * WEIGHTS["gpa"]
                + scores["degree_score"] * WEIGHTS["degree"]
                + scores["language_score"] * WEIGHTS["language"]
                + scores["nationality_score"] * WEIGHTS["nationality"]
                + scores["major_score"] * WEIGHTS["major"]
                + semantic * WEIGHTS["semantic"]
                + scores["experience_score"] * WEIGHTS["experience"]
                + scores["publications_score"] * WEIGHTS["publications"]
            )
        total = round(max(0.0, min(100.0, total)), 1)
        breakdown = ScoreBreakdown(
            gpa_score=round(scores["gpa_score"], 1),
            degree_score=round(scores["degree_score"], 1),
            language_score=round(scores["language_score"], 1),
            nationality_score=round(scores["nationality_score"], 1),
            major_score=round(scores["major_score"], 1),
            semantic_alignment_score=round(semantic, 1),
            experience_score=round(scores["experience_score"], 1),
            publications_score=round(scores["publications_score"], 1),
        )
        results.append(
            MatchResult(
                grant_id=scholarship_id,
                title=card.title,
                match_percentage=total,
                tier=_tier_from_percentage(total),
                breakdown=breakdown,
                reasons=_reasons(profile, card, breakdown, disqualifiers),
                disqualifiers=disqualifiers,
                missing_documents=list(card.required_documents),
                deadline=card.deadline,
                official_url=card.official_url,
            )
        )

    results.sort(key=lambda item: (-item.match_percentage, item.deadline or "9999"))
    results = results[:limit]
    top = results[0].match_percentage if results else 0.0
    threshold = config.external_search_threshold
    min_fit = config.external_search_min_fit
    best_semantic = max(
        (
            item.breakdown.semantic_alignment_score
            for item in results
            if not item.disqualifiers
        ),
        default=0.0,
    )
    below_threshold = top < threshold
    weak_fit = best_semantic < min_fit
    reasons: list[str] = []
    if not results:
        reasons.append("لا توجد منح منشورة داخل قاعدة البيانات للمقارنة.")
    else:
        if below_threshold:
            reasons.append(
                f"أعلى نسبة تطابق داخلية {top:.1f}% أقل من الحد {threshold:.0f}%."
            )
        if weak_fit:
            reasons.append(
                f"أفضل توافق دلالي {best_semantic:.1f}% أقل من الحد الأدنى {min_fit:.0f}%."
            )
    trigger = not results or below_threshold or weak_fit
    return MatchReport(
        results=results,
        top_percentage=top,
        trigger_external_search=trigger,
        external_search_reason=" ".join(reasons) if trigger else None,
        semantic_source="llm" if llm_scores else "deterministic",
    )
