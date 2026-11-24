"""Reflection document agent (task 7).

Pipeline per document:

1. draft   - LLM when configured, otherwise a deterministic grounded writer.
2. critique - deterministic fact/grounding/a11y checks (no second model).
3. revise  - sanitize, or fall back to the grounded draft on critical issues.
4. final   - re-validate; expose the full revision log to the client.

The agent can only use facts that exist in the profile or the grant record.
It never writes a signed recommendation letter; it produces a briefing sheet
for the recommender instead.
"""

from __future__ import annotations

import json
import logging

from . import llm
from .config import settings
from .guardrails import check_grounding, check_output
from .schemas import (
    DocumentIssue,
    DocumentResult,
    DocumentRevision,
    DocumentType,
    GrantCard,
    StudentProfile,
)

logger = logging.getLogger(__name__)

_DEGREE_AR = {
    "bachelor": "بكالوريوس",
    "master": "ماجستير",
    "phd": "دكتوراه",
    "high_school": "ثانوية عامة",
}
_DEGREE_EN = {
    "bachelor": "Bachelor's degree",
    "master": "Master's degree",
    "phd": "Doctorate",
    "high_school": "High school diploma",
}


class DocumentAgent:
    def __init__(self, *, use_llm: bool = True) -> None:
        self.use_llm = use_llm and settings.llm_configured

    # -- public API ------------------------------------------------------
    def generate(
        self,
        profile: StudentProfile,
        grant: GrantCard,
        document_type: DocumentType,
        *,
        language: str = "ar",
    ) -> DocumentResult:
        missing = profile.missing_required()
        if missing:
            raise ValueError(
                "ملف الطالب غير مكتمل لإنشاء الوثيقة: " + ", ".join(missing)
            )
        language = "en" if language == "en" else "ar"

        draft, provider = self._draft(profile, grant, document_type, language)
        issues = self._critique(draft, profile, grant, document_type, language)

        revised = self._revise(draft, issues, profile, grant, document_type, language)
        final_report = check_output(revised, known_grants=[grant], allow_urls=False)
        final_text = final_report.text

        remaining = check_grounding(final_text, profile, grant) + final_report.issues
        critical = [issue for issue in remaining if issue.severity == "critical"]
        if critical:
            # Last line of defence: the deterministic draft is grounded by construction.
            final_text, _ = self._deterministic(
                profile, grant, document_type, language
            )
            final_report = check_output(
                final_text, known_grants=[grant], allow_urls=False
            )
            remaining = check_grounding(final_text, profile, grant) + final_report.issues
            critical = [issue for issue in remaining if issue.severity == "critical"]

        revisions = [
            DocumentRevision(
                version=1, stage="draft", content=draft, issues=[]
            ),
            DocumentRevision(
                version=2,
                stage="critique",
                content=self._critique_summary(issues),
                issues=issues,
            ),
            DocumentRevision(
                version=3, stage="revised", content=revised, issues=issues
            ),
            DocumentRevision(
                version=4, stage="final", content=final_text, issues=remaining
            ),
        ]

        return DocumentResult(
            document_type=document_type,
            grant_id=grant.grant_id,
            grant_title=grant.title,
            language=language,  # type: ignore[arg-type]
            content=final_text,
            revisions=revisions,
            passed=not critical,
            grounded=not any(i.code == "ungrounded_claim" for i in remaining),
            issues=remaining,
            provider=provider,  # type: ignore[arg-type]
        )

    # -- draft -----------------------------------------------------------
    def _draft(
        self,
        profile: StudentProfile,
        grant: GrantCard,
        document_type: DocumentType,
        language: str,
    ) -> tuple[str, str]:
        deterministic, _ = self._deterministic(profile, grant, document_type, language)
        if not self.use_llm:
            return deterministic, "deterministic"
        try:
            prompt = self._llm_prompt(profile, grant, document_type, language)
            draft = llm.complete(prompt, temperature=0.4, timeout=40)
            if len(draft.strip()) < 200:
                raise ValueError("LLM draft too short")
            return draft.strip(), "llm"
        except Exception as exc:  # pragma: no cover - provider dependent
            logger.warning("LLM document draft failed, using deterministic: %s", exc)
            return deterministic, "deterministic"

    def _llm_prompt(
        self,
        profile: StudentProfile,
        grant: GrantCard,
        document_type: DocumentType,
        language: str,
    ) -> str:
        facts = {
            "name": profile.name,
            "email": profile.email,
            "gpa": profile.gpa,
            "major": profile.major,
            "current_degree": profile.degree_level.value if profile.degree_level else None,
            "target_degree": profile.target_degree.value if profile.target_degree else None,
            "nationality": profile.nationality,
            "languages": [item.model_dump() for item in profile.languages],
            "research_interests": profile.research_interests,
            "target_fields": profile.target_fields,
            "target_countries": profile.target_countries,
            "experiences": profile.experiences,
            "publications": profile.publications,
            "extracurriculars": profile.extracurriculars,
        }
        grant_facts = {
            "title": grant.title,
            "provider": grant.provider,
            "country": grant.country,
            "fields_of_study": grant.fields_of_study,
            "required_documents": grant.required_documents,
            "funding": grant.funding,
        }
        kind = {
            DocumentType.CV: "academic CV",
            DocumentType.MOTIVATION_LETTER: "motivation letter",
            DocumentType.RECOMMENDATION_BRIEF: "briefing sheet for a recommender "
            "(not a signed letter)",
        }[document_type]
        output_language = "Arabic" if language == "ar" else "English"
        return f"""You are drafting a {kind} for a scholarship application.

STRICT RULES:
- Use ONLY the facts in the FACTS object. Never invent GPA, test scores,
  publications, employers, awards, dates, names or contact details.
- If a fact is missing, do not mention it. Do not use placeholders except for
  recommender details in a briefing sheet.
- Do not include any URL.
- Write in {output_language}, Markdown, 350-550 words.
- For a briefing sheet, explicitly state that the recommender must write and
  sign the final letter, and list verified achievements only.

STUDENT FACTS:
{json.dumps(facts, ensure_ascii=False, indent=2)}

GRANT FACTS:
{json.dumps(grant_facts, ensure_ascii=False, indent=2)}

Return only the document text."""

    def _deterministic(
        self,
        profile: StudentProfile,
        grant: GrantCard,
        document_type: DocumentType,
        language: str,
    ) -> tuple[str, list[DocumentIssue]]:
        if language == "en":
            text = {
                DocumentType.CV: _cv_en,
                DocumentType.MOTIVATION_LETTER: _motivation_en,
                DocumentType.RECOMMENDATION_BRIEF: _brief_en,
            }[document_type](profile, grant)
        else:
            text = {
                DocumentType.CV: _cv_ar,
                DocumentType.MOTIVATION_LETTER: _motivation_ar,
                DocumentType.RECOMMENDATION_BRIEF: _brief_ar,
            }[document_type](profile, grant)
        return text.strip(), []

    # -- critique / revise ----------------------------------------------
    def _critique(
        self,
        text: str,
        profile: StudentProfile,
        grant: GrantCard,
        document_type: DocumentType,
        language: str,
    ) -> list[DocumentIssue]:
        issues = check_grounding(text, profile, grant)
        lowered = text.lower()

        if profile.name and profile.name.lower() not in lowered:
            issues.append(
                DocumentIssue(
                    code="missing_fact",
                    severity="critical",
                    message="اسم الطالب غير موجود في الوثيقة.",
                )
            )
        if grant.title.lower() not in lowered and grant.title.split("(")[0].strip().lower() not in lowered:
            issues.append(
                DocumentIssue(
                    code="missing_fact",
                    severity="critical",
                    message="اسم المنحة المستهدفة غير موجود في الوثيقة.",
                )
            )
        if profile.major and profile.major.lower() not in lowered:
            issues.append(
                DocumentIssue(
                    code="missing_fact",
                    severity="warning",
                    message="التخصص غير مذكور في الوثيقة.",
                )
            )
        if document_type == DocumentType.MOTIVATION_LETTER and len(text) < 700:
            issues.append(
                DocumentIssue(
                    code="too_short",
                    severity="warning",
                    message="خطاب الدافع قصير جداً (أقل من 700 حرف).",
                )
            )
        if len(text) > 9000:
            issues.append(
                DocumentIssue(
                    code="too_long",
                    severity="warning",
                    message="الوثيقة طويلة جداً؛ اختصرها.",
                )
            )
        return issues

    def _revise(
        self,
        text: str,
        issues: list[DocumentIssue],
        profile: StudentProfile,
        grant: GrantCard,
        document_type: DocumentType,
        language: str,
    ) -> str:
        critical = [issue for issue in issues if issue.severity == "critical"]
        if critical:
            deterministic, _ = self._deterministic(
                profile, grant, document_type, language
            )
            return deterministic

        # Remove links (documents never need them) and normalize whitespace.
        report = check_output(text, known_grants=[grant], allow_urls=False)
        revised = report.text
        if any(issue.code == "too_short" for issue in issues):
            deterministic, _ = self._deterministic(
                profile, grant, document_type, language
            )
            if len(deterministic) > len(revised):
                revised = deterministic
        return revised

    @staticmethod
    def _critique_summary(issues: list[DocumentIssue]) -> str:
        if not issues:
            return "اجتازت المسودة جميع فحوص التدقيق الذاتي دون ملاحظات."
        lines = ["نتائج التدقيق الذاتي:"]
        for issue in issues:
            lines.append(f"- [{issue.severity}] {issue.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic Arabic writers
# ---------------------------------------------------------------------------


def _degree_ar(value: str | None) -> str:
    if value is None:
        return "غير محدد"
    return _DEGREE_AR.get(value, value)


def _languages_ar(profile: StudentProfile) -> str:
    if not profile.languages:
        return "لم تُذكر شهادات لغة بعد."
    parts = []
    for item in profile.languages:
        detail = item.level.value
        if item.test_name:
            detail += f" ({item.test_name}"
            if item.score is not None:
                detail += f" {item.score:g}"
            detail += ")"
        parts.append(f"{item.language} — {detail}")
    return "، ".join(parts)


def _languages_en(profile: StudentProfile) -> str:
    if not profile.languages:
        return "No language certificate recorded yet."
    parts = []
    for item in profile.languages:
        detail = item.level.value
        if item.test_name:
            detail += f" ({item.test_name}"
            if item.score is not None:
                detail += f" {item.score:g}"
            detail += ")"
        parts.append(f"{item.language} — {detail}")
    return ", ".join(parts)


def _cv_ar(profile: StudentProfile, grant: GrantCard) -> str:
    lines = [f"# {profile.name}", ""]
    contact = [value for value in (profile.email, profile.nationality) if value]
    if contact:
        lines.append(" | ".join(contact))
        lines.append("")
    lines += [
        "## الملف الأكاديمي",
        f"- الدرجة الحالية: {_degree_ar(profile.degree_level.value if profile.degree_level else None)}",
        f"- التخصص: {profile.major}",
        f"- المعدل التراكمي: {profile.gpa:.2f} / 4.00" if profile.gpa is not None else "- المعدل التراكمي: غير محدد",
        f"- المرحلة المستهدفة: {_degree_ar(profile.target_degree.value if profile.target_degree else None)}",
        "",
        "## اللغات",
        f"- {_languages_ar(profile)}",
        "",
    ]
    if profile.research_interests or profile.target_fields:
        lines += ["## الاهتمامات البحثية", ""]
        for item in (profile.research_interests or profile.target_fields):
            lines.append(f"- {item}")
        lines.append("")
    if profile.experiences:
        lines += ["## الخبرات", ""]
        for item in profile.experiences:
            lines.append(f"- {item}")
        lines.append("")
    if profile.publications:
        lines += ["## المنشورات", "", f"- عدد الأوراق المنشورة: {profile.publications}", ""]
    if profile.extracurriculars:
        lines += ["## الأنشطة", ""]
        for item in profile.extracurriculars:
            lines.append(f"- {item}")
        lines.append("")
    if profile.target_countries:
        lines += ["## الدول المستهدفة", f"- {'، '.join(profile.target_countries)}", ""]
    lines += [
        "## الهدف",
        f"- التقديم على «{grant.title}» في مجال {', '.join(grant.fields_of_study[:3]) or profile.major}.",
        "",
    ]
    return "\n".join(lines)


def _motivation_ar(profile: StudentProfile, grant: GrantCard) -> str:
    fields = "، ".join(grant.fields_of_study[:4]) or profile.major or ""
    interests = "، ".join(profile.research_interests[:3] or profile.target_fields[:3])
    experience_block = (
        "وقد صقلت ذلك عبر "
        + "؛ و".join(profile.experiences[:2])
        + ". "
        if profile.experiences
        else "وأعمل على تعزيز هذا المسار عبر مشاريع تطبيقية موجّهة. "
    )
    language_block = (
        f"أمتلك مستوى {_languages_ar(profile)}، وأخطط لاستكمال أي متطلب لغوي متبقٍ قبل الموعد النهائي. "
    )
    goal_block = (
        f"أطمح بعد التخرج إلى المساهمة في {'، '.join(profile.target_fields[:2]) or profile.major} "
        "خدمةً لسوق العمل والتنمية في بلدي."
    )
    return f"""# {profile.name}
{profile.email or ""}

إلى لجنة الاختيار في «{grant.title}» — {grant.provider}

الموضوع: خطاب دافع للتقديم على {grant.title}

تحية طيبة،

أكتب إليكم للتقديم على منحة «{grant.title}»، وأنا {profile.name}، حاصل/ة على درجة {_degree_ar(profile.degree_level.value if profile.degree_level else None)} في تخصص {profile.major} بمعدل تراكمي {profile.gpa:.2f} من 4.00، وأتقدم حالياً لمرحلة {_degree_ar(profile.target_degree.value if profile.target_degree else None)}.

ينبع اهتمامي بهذه المنحة من توافق أهدافها مع مساري الأكاديمي في {fields}. {(('وتركّز اهتماماتي البحثية على ' + interests + '. ') if interests else '')}{experience_block}{language_block}

أرى في «{grant.title}» بيئة مناسبة لتحويل معرفتي الأكاديمية إلى أثر ملموس، خصوصاً في المجالات التي ترعاها المنحة مثل {fields}. {goal_block}

أشكركم على وقت لجنة الاختيار، وأؤكد أن جميع المعلومات الواردة في هذا الخطاب مطابقة لملفي الأكاديمي الموثق.

تفضلوا بقبول فائق الاحترام،
{profile.name}
"""


def _brief_ar(profile: StudentProfile, grant: GrantCard) -> str:
    achievements = []
    if profile.gpa is not None:
        achievements.append(f"معدل تراكمي {profile.gpa:.2f}/4.00 في تخصص {profile.major}.")
    for item in profile.experiences[:4]:
        achievements.append(f"خبرة موثقة: {item}.")
    for item in (profile.research_interests or profile.target_fields)[:4]:
        achievements.append(f"اهتمام/عمل بحثي في {item}.")
    if profile.publications:
        achievements.append(f"عدد الأوراق المنشورة: {profile.publications}.")
    if not achievements:
        achievements.append("لا توجد إنجازات مذكورة في الملف بعد؛ يجب على الموصي الاعتماد على معرفته المباشرة فقط.")

    return f"""# موجز لكاتب رسالة التوصية

**تنبيه مهم:** هذه الوثيقة ليست رسالة توصية ولا تُوقّع نيابة عن الموصي. على الموصي كتابة الرسالة بنفسه وتوقيعها، والاستشهاد فقط بما يعرفه فعلاً عن الطالب.

## بيانات الطالب (من ملف المنصة)
- الاسم: {profile.name}
- التخصص: {profile.major}
- الدرجة الحالية: {_degree_ar(profile.degree_level.value if profile.degree_level else None)}
- المرحلة المستهدفة: {_degree_ar(profile.target_degree.value if profile.target_degree else None)}
- المعدل: {profile.gpa:.2f}/4.00
- اللغات: {_languages_ar(profile)}

## المنحة المستهدفة
- «{grant.title}» — {grant.provider}
- المجالات: {'، '.join(grant.fields_of_study) or 'غير محددة'}
- الوثائق المطلوبة: {'، '.join(grant.required_documents) or 'غير منشورة'}

## إنجازات موثقة يمكن الاستشهاد بها
""" + "\n".join(f"- {item}" for item in achievements) + f"""

## نقاط يُنصح بتغطيتها
- كيف عرف الموصي الطالب ومدة المعرفة.
- مثال محدد على أداء أكاديمي أو بحثي.
- مقارنة الطالب بأقرانه إن أمكن.
- توافق الطالب مع مجال المنحة «{grant.title}».

## حقول تحتاج تعبئة من الموصي
- [[اسم الموصي]]
- [[المسمى الوظيفي والجهة]]
- [[البريد الإلكتروني الرسمي]]
- [[التاريخ والتوقيع]]
"""


# ---------------------------------------------------------------------------
# Deterministic English writers
# ---------------------------------------------------------------------------


def _cv_en(profile: StudentProfile, grant: GrantCard) -> str:
    lines = [f"# {profile.name}", ""]
    contact = [value for value in (profile.email, profile.nationality) if value]
    if contact:
        lines += [" | ".join(contact), ""]
    lines += [
        "## Academic profile",
        f"- Current degree: {_DEGREE_EN.get(profile.degree_level.value, 'Not specified') if profile.degree_level else 'Not specified'}",
        f"- Major: {profile.major}",
        f"- GPA: {profile.gpa:.2f} / 4.00" if profile.gpa is not None else "- GPA: Not specified",
        f"- Target degree: {_DEGREE_EN.get(profile.target_degree.value, 'Not specified') if profile.target_degree else 'Not specified'}",
        "",
        "## Languages",
        f"- {_languages_en(profile)}",
        "",
    ]
    if profile.research_interests or profile.target_fields:
        lines += ["## Research interests", ""]
        lines += [f"- {item}" for item in (profile.research_interests or profile.target_fields)]
        lines.append("")
    if profile.experiences:
        lines += ["## Experience", ""]
        lines += [f"- {item}" for item in profile.experiences]
        lines.append("")
    if profile.publications:
        lines += ["## Publications", "", f"- Published papers: {profile.publications}", ""]
    if profile.extracurriculars:
        lines += ["## Activities", ""]
        lines += [f"- {item}" for item in profile.extracurriculars]
        lines.append("")
    lines += [
        "## Objective",
        f"- Applying to {grant.title} in {', '.join(grant.fields_of_study[:3]) or profile.major}.",
        "",
    ]
    return "\n".join(lines)


def _motivation_en(profile: StudentProfile, grant: GrantCard) -> str:
    fields = ", ".join(grant.fields_of_study[:4]) or profile.major or ""
    interests = ", ".join(profile.research_interests[:3] or profile.target_fields[:3])
    experience_block = (
        "I have developed this interest through "
        + "; ".join(profile.experiences[:2])
        + ". "
        if profile.experiences
        else "I am strengthening this path through applied projects. "
    )
    return f"""# {profile.name}
{profile.email or ""}

To the selection committee of {grant.title} — {grant.provider}

Subject: Motivation letter for {grant.title}

Dear Committee,

I am writing to apply for {grant.title}. I am {profile.name}, holder of a {_DEGREE_EN.get(profile.degree_level.value, '') if profile.degree_level else ''} in {profile.major} with a GPA of {profile.gpa:.2f}/4.00, currently applying for a {_DEGREE_EN.get(profile.target_degree.value, '') if profile.target_degree else ''}.

My interest in this grant stems from the close alignment between its objectives and my academic path in {fields}. {('My research interests focus on ' + interests + '. ') if interests else ''}{experience_block}My current language profile is: {_languages_en(profile)}. I am committed to completing any remaining language requirement before the deadline.

I see {grant.title} as the right environment to translate academic knowledge into measurable impact in the fields it supports, such as {fields}. After graduation I intend to contribute to {', '.join(profile.target_fields[:2]) or profile.major} in my home country.

Thank you for your time and consideration. All statements in this letter reflect my verified academic record.

Sincerely,
{profile.name}
"""


def _brief_en(profile: StudentProfile, grant: GrantCard) -> str:
    achievements = []
    if profile.gpa is not None:
        achievements.append(f"GPA {profile.gpa:.2f}/4.00 in {profile.major}.")
    for item in profile.experiences[:4]:
        achievements.append(f"Documented experience: {item}.")
    for item in (profile.research_interests or profile.target_fields)[:4]:
        achievements.append(f"Research interest in {item}.")
    if profile.publications:
        achievements.append(f"Published papers: {profile.publications}.")
    if not achievements:
        achievements.append(
            "No achievements are recorded yet; the recommender must rely only on first-hand knowledge."
        )
    return f"""# Briefing sheet for the recommender

**Important:** this is not a recommendation letter and must not be signed on the
student's behalf. The recommender must write and sign the final letter and cite
only first-hand knowledge.

## Student facts (from the platform profile)
- Name: {profile.name}
- Major: {profile.major}
- Current degree: {_DEGREE_EN.get(profile.degree_level.value, 'Not specified') if profile.degree_level else 'Not specified'}
- Target degree: {_DEGREE_EN.get(profile.target_degree.value, 'Not specified') if profile.target_degree else 'Not specified'}
- GPA: {profile.gpa:.2f}/4.00
- Languages: {_languages_en(profile)}

## Target grant
- {grant.title} — {grant.provider}
- Fields: {', '.join(grant.fields_of_study) or 'Not specified'}
- Required documents: {', '.join(grant.required_documents) or 'Not published'}

## Verified achievements to cite
""" + "\n".join(f"- {item}" for item in achievements) + f"""

## Suggested coverage
- How long and in what capacity the recommender knows the student.
- One concrete academic or research example.
- Comparison with peers, if appropriate.
- Fit with {grant.title}.

## Fields to complete by the recommender
- [[Recommender name]]
- [[Title and institution]]
- [[Official email]]
- [[Date and signature]]
"""
