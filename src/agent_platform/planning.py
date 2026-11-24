"""Strategic application planning.

Reuses the deterministic step templates from the planner repository but skips
the LLM enrichment call, so planning works offline and cannot invent deadlines.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .repos import setup_repo_paths
from .schemas import GrantCard, MatchReport, StudentProfile

setup_repo_paths()

from ai.scholarship_agent.tools.application_planner_tool import (  # noqa: E402
    _build_base_steps,
    _days_until,
)

_BASE_TIPS = (
    "طابق كل وثيقة مع نص شرط المنحة قبل الرفع، ولا تعتمد على نماذج عامة.",
    "اطلب رسائل التوصية مبكراً وزوّد الموصي بملخص إنجازاتك والمنحة المستهدفة.",
    "ارفع طلبك قبل الموعد النهائي بخمسة أيام على الأقل لتفادي ضغط البوابات.",
)
_OPEN_DEADLINE_TIP = "الموعد النهائي لم يُعلن على الصفحة الرسمية بعد؛ راقبها أسبوعياً."


def _document_rows(card: GrantCard) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in card.required_documents:
        rows.append(
            {
                "name": name,
                "status": "required",
                "description": "مطلوبة صراحةً في الصفحة الرسمية للمنحة.",
            }
        )
    if not rows:
        rows = [
            {
                "name": "سجل الدرجات الأكاديمي (Transcript)",
                "status": "required",
                "description": "لم تُنشر قائمة وثائق مفصلة؛ تحقق من الموقع الرسمي.",
            },
            {
                "name": "خطاب الدافع (Motivation Letter)",
                "status": "required",
                "description": "وثيقة أساسية في معظم المنح الحكومية.",
            },
            {
                "name": "السيرة الذاتية الأكاديمية (Academic CV)",
                "status": "required",
                "description": "وثيقة أساسية في معظم المنح الحكومية.",
            },
        ]
    return rows


def build_plan(
    profile: StudentProfile,
    report: MatchReport,
    cards: Iterable[GrantCard],
    *,
    top_n: int = 3,
) -> list[dict]:
    by_id = {card.grant_id: card for card in cards}
    viable = [
        match
        for match in report.results
        if match.match_percentage > 0 and not match.disqualifiers
    ][:top_n]
    if not viable:
        viable = report.results[:top_n]

    items: list[dict] = []
    for match in viable:
        card = by_id.get(match.grant_id)
        if not card:
            continue
        if card.deadline:
            days_left = _days_until(card.deadline)
            deadline_label = card.deadline
        else:
            days_left = 180
            deadline_label = "غير محدد (راقب الصفحة الرسمية)"
        steps = [
            step.model_dump() for step in _build_base_steps(days_left, card.title)
        ]
        tips = list(_BASE_TIPS)
        if not card.deadline:
            tips.append(_OPEN_DEADLINE_TIP)
        items.append(
            {
                "grant_id": card.grant_id,
                "grant_title": card.title,
                "official_url": card.official_url,
                "deadline": deadline_label,
                "match_percentage": match.match_percentage,
                "documents": _document_rows(card),
                "steps": steps,
                "estimated_preparation_days": min(60, max(10, days_left // 2)),
                "tips": tips,
            }
        )

    items.sort(key=lambda item: (item["deadline"] == "غير محدد (راقب الصفحة الرسمية)", item["deadline"]))
    for rank, item in enumerate(items, 1):
        item["priority_rank"] = rank
    return items


def _grants_phrase(count: int) -> str:
    """Correct Arabic plural for the number of grants in a sentence."""
    if count == 1:
        return "منحة واحدة"
    if count == 2:
        return "منحتين"
    if 3 <= count <= 10:
        return f"{count} منح"
    return f"{count} منحة"


def plan_summary(items: list[dict]) -> str:
    if not items:
        return "لا توجد منح مؤهلة حالياً لبناء خطة تقديم. فعّل البحث الخارجي أو أكمل بياناتك."
    first = items[0]
    return (
        f"خطتك تشمل {_grants_phrase(len(items))}. الأولوية الأولى: «{first['grant_title']}» "
        f"بنسبة تطابق {first['match_percentage']:.1f}% والموعد النهائي {first['deadline']}. "
        "ابدأ بجمع الوثائق الرسمية ومراسلة الموصين قبل الموعد بأسبوعين على الأقل."
    )


def days_until(deadline: str | None) -> int:
    if not deadline:
        return 0
    try:
        return max(0, (date.fromisoformat(deadline) - date.today()).days)
    except ValueError:
        return 0
