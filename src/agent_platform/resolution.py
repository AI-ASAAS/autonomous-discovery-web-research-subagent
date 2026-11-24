"""Natural-language resolution shared by the legacy and graph engines.

Both engines must agree on *which* grant the student means and *which* document
they asked for.  Keeping that logic in one place avoids a subtle regression:
switching the agent engine on must not lose the Arabic aliases ("شيفينغ",
"فولبرايت", ...) or the country hints that the deterministic engine supports.
"""

from __future__ import annotations

from typing import Iterable

from .guardrails import normalize
from .knowledge import GrantRepository
from .schemas import DocumentType, GrantCard

GRANT_ALIASES: dict[tuple[str, ...], str] = {
    ("chevening", "شيفينغ", "تشيفنينغ", "تشيفنغ"): "chevening",
    ("fulbright", "فولبرايت", "فلبرايت"): "fulbright",
    ("daad", "داد", "دائد"): "daad",
    ("turkiye", "türkiye", "turkiyeburslari", "burslari"): "türkiye",
    ("stipendium", "هنجاريا", "stipendium hungaricum"): "stipendium",
    ("isdb", "islamic development", "البنك الإسلامي", "البنك الاسلامي"): "islamic development",
    ("eiffel", "ايفل", "إيفل"): "eiffel",
    ("campuschina", "csc", "chinese government"): "chinese government",
    ("russian government", "education in russia"): "russian",
    ("romanian government", "mfa romania"): "romanian",
}

COUNTRY_ALIASES: dict[tuple[str, ...], str] = {
    ("تركيا", "توركيا", "turkey", "turkiye"): "Turkey",
    ("المجر", "هنجاريا", "hungary"): "Hungary",
    ("فرنسا", "الفرنسية", "france"): "France",
    ("الصين", "الصينية", "china"): "China",
    ("روسيا", "الروسية", "russia"): "Russia",
    ("رومانيا", "الرومانية", "romania"): "Romania",
    ("بريطانيا", "المملكة المتحدة", "uk", "united kingdom"): "United Kingdom",
    ("أمريكا", "امريكا", "الولايات المتحدة", "usa", "united states"): "United States",
    ("ألمانيا", "المانيا", "germany"): "Germany",
}


def resolve_grant(
    message: str,
    cards: Iterable[GrantCard],
    repository: GrantRepository | None = None,
) -> GrantCard | None:
    """Resolve a free-text grant reference against the verified catalogue.

    Order: known aliases -> exact/normalised title -> distinctive title prefix
    -> country hint -> repository token search.
    """
    card_list = list(cards)
    text = normalize(message)

    for aliases, canonical in GRANT_ALIASES.items():
        if any(alias in text for alias in aliases):
            for card in card_list:
                if canonical in card.title.lower():
                    return card

    for card in card_list:
        title = normalize(card.title)
        if title in text:
            return card
        short = title.split("(")[0].strip()
        if len(short) > 5 and short in text:
            return card

    for aliases, country in COUNTRY_ALIASES.items():
        if any(alias in text for alias in aliases):
            for card in card_list:
                if card.country == country:
                    return card

    if repository is not None:
        hits = repository.search(message, limit=1)
        return hits[0] if hits else None
    return None


def detect_document_type(message: str) -> DocumentType:
    """Detect the requested document type from Arabic or English wording."""
    text = normalize(message)
    if "توصية" in text or "توصيه" in text or "recommendation" in text:
        return DocumentType.RECOMMENDATION_BRIEF
    if "سيرة" in text or "سيره" in text or "cv" in text or "resume" in text:
        return DocumentType.CV
    return DocumentType.MOTIVATION_LETTER
