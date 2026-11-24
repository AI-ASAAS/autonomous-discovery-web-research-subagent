"""Date extraction and deterministic deadline verdicts for scholarship queries.

Ensures questions about grant deadlines receive an explicit verdict without
requiring an LLM, and provides verified facts to the supervisor graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .knowledge import GrantRepository
from .resolution import resolve_grant
from .schemas import GrantCard

_MONTHS_EN = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTHS_AR = {
    "يناير": 1,
    "فبراير": 2,
    "مارس": 3,
    "ابريل": 4,
    "أبريل": 4,
    "مايو": 5,
    "يونيو": 6,
    "يوليو": 7,
    "اغسطس": 8,
    "أغسطس": 8,
    "سبتمبر": 9,
    "اكتوبر": 10,
    "أكتوبر": 10,
    "نوفمبر": 11,
    "ديسمبر": 12,
}

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass(frozen=True)
class DateClaim:
    month: int
    day: int
    year: int | None = None


@dataclass
class DeadlineFact:
    status: str
    verdict: str
    clarification: str = ""
    grant_id: str | None = None
    grant_title: str | None = None


def _clean_text(text: str) -> str:
    return (text or "").translate(_ARABIC_DIGITS)


def extract_dates(text: str) -> list[DateClaim]:
    """Extract all DateClaim(month, day, year) instances from text."""
    cleaned = _clean_text(text)
    claims: list[DateClaim] = []

    # 1. ISO format: YYYY-MM-DD
    for match in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", cleaned):
        claims.append(DateClaim(int(match.group(2)), int(match.group(3)), int(match.group(1))))

    # 2. English: "June 3", "June 3rd", "June 3, 2026"
    month_names_en = "|".join(_MONTHS_EN.keys())
    for match in re.finditer(
        rf"\b({month_names_en})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:[,\s]+(\d{{4}}))?\b",
        cleaned,
        re.IGNORECASE,
    ):
        month = _MONTHS_EN[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        claims.append(DateClaim(month, day, year))

    # 3. English: "3 June", "3rd June", "3rd of June", "3 June 2026"
    for match in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_names_en})(?:[,\s]+(\d{{4}}))?\b",
        cleaned,
        re.IGNORECASE,
    ):
        day = int(match.group(1))
        month = _MONTHS_EN[match.group(2).lower()]
        year = int(match.group(3)) if match.group(3) else None
        claims.append(DateClaim(month, day, year))

    # 4. Arabic: "3 يونيو", "3 من يونيو", "3 يونيو 2026"
    month_names_ar = "|".join(_MONTHS_AR.keys())
    for match in re.finditer(
        rf"\b(\d{{1,2}})\s*(?:من\s+)?({month_names_ar})(?:\s+(\d{{4}}))?\b",
        cleaned,
    ):
        day = int(match.group(1))
        month = _MONTHS_AR[match.group(2)]
        year = int(match.group(3)) if match.group(3) else None
        claims.append(DateClaim(month, day, year))

    # 5. Arabic: "يونيو 3", "يونيو 3 2026"
    for match in re.finditer(
        rf"\b({month_names_ar})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?\b",
        cleaned,
    ):
        month = _MONTHS_AR[match.group(1)]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        claims.append(DateClaim(month, day, year))

    return claims


def parse_date_from_text(text: str) -> tuple[int, int, int | None] | None:
    """Extract (month, day, year) from English, Arabic, or ISO formatted text."""
    claims = extract_dates(text)
    if claims:
        first = claims[0]
        return (first.month, first.day, first.year)
    return None


def dates_match(user_date: tuple[int, int, int | None] | DateClaim, deadline_str: str | None) -> bool:
    """Return True if user_date matches deadline_str (YYYY-MM-DD)."""
    if not deadline_str:
        return False
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", deadline_str.strip())
    if not match:
        return False
    target_year = int(match.group(1))
    target_month = int(match.group(2))
    target_day = int(match.group(3))

    if isinstance(user_date, DateClaim):
        u_month, u_day, u_year = user_date.month, user_date.day, user_date.year
    else:
        u_month, u_day, u_year = user_date

    if u_year is not None and u_year != target_year:
        return False
    return u_month == target_month and u_day == target_day


def find_focused_grant(
    message: str,
    cards: list[GrantCard],
    history: list[Any] | None = None,
    repository: GrantRepository | None = None,
) -> GrantCard | None:
    """Find which grant is in focus from message (ordinals, names) or history."""
    if not cards:
        return None

    cleaned = _clean_text(message).lower()

    # 1. Check ordinal references
    if any(
        phrase in cleaned
        for phrase in (
            "first scholarship",
            "1st scholarship",
            "first grant",
            "first one",
            "المنحة الأولى",
            "المنحه الاولى",
            "أول منحة",
            "اول منحه",
            "اول منحة",
            "المنحة 1",
            "منحة 1",
        )
    ):
        return cards[0]
    if any(
        phrase in cleaned
        for phrase in (
            "second scholarship",
            "2nd scholarship",
            "second grant",
            "المنحة الثانية",
            "المنحه الثانيه",
            "ثاني منحة",
            "ثاني منحه",
            "المنحة 2",
        )
    ) and len(cards) > 1:
        return cards[1]
    if any(
        phrase in cleaned
        for phrase in (
            "third scholarship",
            "3rd scholarship",
            "third grant",
            "المنحة الثالثة",
            "المنحه الثالثه",
            "ثالث منحة",
            "ثالث منحه",
            "المنحة 3",
        )
    ) and len(cards) > 2:
        return cards[2]

    # 2. Check direct mention in message
    card = resolve_grant(message, cards, repository)
    if card is not None:
        return card

    # 3. Check previous conversation turns in history
    if history:
        for item in reversed(history[-6:]):
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "")
            else:
                content = getattr(item, "content", "")
            if content:
                hist_card = resolve_grant(content, cards, repository)
                if hist_card is not None:
                    return hist_card
        # If user ran match or viewed grants in previous turn, default to first card
        if any("match" in str(item).lower() or "chevening" in str(item).lower() for item in history[-4:]):
            return cards[0]

    return None


def deadline_verdict(
    user_date: tuple[int, int, int | None] | DateClaim | None,
    card: GrantCard | None,
    language: str = "en",
) -> str:
    """Produce the exact verdict opening required for a deadline question."""
    if card is None:
        if language == "ar":
            return "عن أي منحة تسأل؟ يرجى تحديد اسم المنحة."
        return "Which scholarship are you asking about? Please specify the scholarship name."

    if not card.deadline:
        if language == "ar":
            return "I have no verified deadline for this grant" if language == "en" else "I have no verified deadline for this grant"
        return "I have no verified deadline for this grant"

    if user_date is not None and dates_match(user_date, card.deadline):
        if language == "ar":
            return f"نعم: {card.deadline}"
        return f"Yes: {card.deadline}"

    if language == "ar":
        return f"لا: الموعد النهائي الموثق هو {card.deadline}"
    return f"No: the verified deadline is {card.deadline}"


def apply_deadline_fact(reply: str, fact: DeadlineFact | dict[str, Any] | None) -> str:
    """Apply pre-computed deadline fact to reply."""
    if not fact:
        return reply
    if isinstance(fact, dict):
        status = fact.get("status")
        verdict = fact.get("verdict", "")
        clarification = fact.get("clarification", "")
    else:
        status = fact.status
        verdict = fact.verdict
        clarification = fact.clarification

    if status == "clarification":
        return clarification or verdict
    if status == "verdict" and verdict:
        if not reply.startswith(verdict):
            return f"{verdict}\n\n{reply}" if reply else verdict
    return reply
