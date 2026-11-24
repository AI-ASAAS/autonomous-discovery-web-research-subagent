"""Tools the supervisor agent can call.

Each tool is a thin, *typed* wrapper over an existing domain capability.  The
tools never invent data: they read the verified grant database, run the
programmatic matcher, compose the application plan, drive the reflection
document agent, build the advisory report, or query the external discovery
sub-agent.

Return contract (every tool returns a JSON string):

``{"kind": <str>, "ok": <bool>, "summary": <str>, "data": {...},
   "sources": [<official urls>], "missing_fields": [...]}``

The graph's tool node uses ``kind``/``data`` to populate structured response
fields (matches, plan, documents, advisor) while the raw JSON is what the
model observes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, tool

from ..advisor import build_advisor_report
from ..concurrency import OperationTimeoutError, run_with_timeout
from ..config import settings
from ..document_agent import DocumentAgent
from ..extraction import merge_profile
from ..knowledge import GrantRepository
from ..matching import match_grants
from ..planning import build_plan
from ..resolution import detect_document_type, resolve_grant as resolve_grant_reference
from ..schemas import DegreeLevel, DocumentType, GrantCard, StudentProfile

logger = logging.getLogger(__name__)

_DOCUMENT_ALIASES: dict[str, DocumentType] = {
    "cv": DocumentType.CV,
    "resume": DocumentType.CV,
    "سيرة": DocumentType.CV,
    "سيره": DocumentType.CV,
    "سيرة ذاتية": DocumentType.CV,
    "motivation": DocumentType.MOTIVATION_LETTER,
    "motivation_letter": DocumentType.MOTIVATION_LETTER,
    "motivation letter": DocumentType.MOTIVATION_LETTER,
    "sop": DocumentType.MOTIVATION_LETTER,
    "statement of purpose": DocumentType.MOTIVATION_LETTER,
    "دافع": DocumentType.MOTIVATION_LETTER,
    "خطاب دافع": DocumentType.MOTIVATION_LETTER,
    "recommendation": DocumentType.RECOMMENDATION_BRIEF,
    "recommendation_brief": DocumentType.RECOMMENDATION_BRIEF,
    "توصية": DocumentType.RECOMMENDATION_BRIEF,
    "توصيه": DocumentType.RECOMMENDATION_BRIEF,
}

_DEGREE_ALIASES: dict[str, str] = {
    "bachelor": "bachelor",
    "bachelors": "bachelor",
    "undergraduate": "bachelor",
    "بكالوريوس": "bachelor",
    "master": "master",
    "masters": "master",
    "postgraduate": "master",
    "ماجستير": "master",
    "phd": "phd",
    "doctorate": "phd",
    "doctoral": "phd",
    "دكتوراه": "phd",
    "دكتوراة": "phd",
    "high_school": "high_school",
    "high school": "high_school",
    "ثانوية": "high_school",
    "ثانويه": "high_school",
}

_WRITABLE_FIELDS = {
    "name",
    "email",
    "gpa",
    "major",
    "degree_level",
    "target_degree",
    "nationality",
    "country_of_residence",
    "research_interests",
    "target_fields",
    "target_countries",
    "experiences",
    "extracurriculars",
    "publications",
}


def _envelope(
    kind: str,
    *,
    ok: bool = True,
    summary: str = "",
    data: Any = None,
    sources: list[str] | None = None,
    missing_fields: list[str] | None = None,
    **extra: Any,
) -> str:
    payload: dict[str, Any] = {"kind": kind, "ok": ok, "summary": summary}
    if data is not None:
        payload["data"] = data
    if sources:
        payload["sources"] = sources
    if missing_fields:
        payload["missing_fields"] = missing_fields
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, default=str)


def _card_summary(card: GrantCard) -> dict[str, Any]:
    return {
        "grant_id": card.grant_id,
        "title": card.title,
        "provider": card.provider,
        "country": card.country,
        "deadline": card.deadline,
        "funding": card.funding[:160],
        "official_url": card.official_url,
        "accepts_yemeni": card.accepts_yemeni,
    }


@dataclass
class ToolContext:
    """Per-turn dependencies and mutable profile shared by every tool."""

    profile: StudentProfile
    repository: GrantRepository
    documents: DocumentAgent
    profile_dirty: bool = False
    user_date: tuple[int, int, int | None] | None = None
    deadline_verdict: str | None = None
    deadline_fact: Any = None
    _cards: dict[str, GrantCard] = field(default_factory=dict)

    def cards(self, limit: int = 50) -> list[GrantCard]:
        for card in self.repository.list_published(limit=limit):
            self._cards[card.grant_id] = card
        return list(self._cards.values())

    def remember(self, card: GrantCard) -> None:
        self._cards[card.grant_id] = card

    def resolve_grant(self, query: str) -> GrantCard | None:
        query = (query or "").strip()
        if not query:
            return None
        card = self.repository.get(query)
        if card is not None:
            self.remember(card)
            return card
        # Reuse the aliases/country hints of the deterministic engine so graph
        # mode understands the same Arabic grant names ("شيفينغ", "فولبرايت").
        card = resolve_grant_reference(query, self.cards(), self.repository)
        if card is not None:
            self.remember(card)
        return card

    def best_grant(self) -> GrantCard | None:
        cards = self.cards()
        if not cards:
            return None
        if self.profile.missing_required():
            return cards[0]
        report = match_grants(self.profile, cards, limit=1, use_llm=False)
        if report.results:
            return self._cards.get(report.results[0].grant_id) or cards[0]
        return cards[0]


def _document_type(raw: str) -> DocumentType:
    key = (raw or "").strip().lower()
    mapped = _DOCUMENT_ALIASES.get(key)
    if mapped is not None:
        return mapped
    # Fall back to the shared detector for free-text Arabic/English wording.
    return detect_document_type(raw)


def _coerce_field(name: str, raw: Any) -> Any:
    if name in {"degree_level", "target_degree"}:
        key = str(raw).strip().lower()
        value = _DEGREE_ALIASES.get(key)
        if value is None:
            raise ValueError(f"Unsupported degree level: {raw!r}")
        return DegreeLevel(value)
    if name == "gpa":
        value = float(str(raw).replace(",", ".").strip())
        if not 0.0 <= value <= 4.0:
            raise ValueError("GPA must be between 0 and 4.")
        return value
    if name == "publications":
        return max(0, int(raw))
    if name in {
        "research_interests",
        "target_fields",
        "target_countries",
        "experiences",
        "extracurriculars",
    }:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        parts = [part.strip() for part in str(raw).split(",")]
        return [part for part in parts if part]
    return str(raw).strip()


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    """Create the tool set bound to a single conversational turn."""

    @tool
    def list_verified_grants(limit: int = 10) -> str:
        """List verified, published scholarships (title, country, deadline, link).

        Use this when the student asks which scholarships exist, wants a list or
        ranking of available grants, or has not named a specific grant yet.
        """
        cards = ctx.cards(limit=max(1, min(int(limit or 10), 50)))
        return _envelope(
            "grants",
            summary=f"{len(cards)} verified grants are available.",
            data={"grants": [_card_summary(card) for card in cards]},
            sources=[card.official_url for card in cards],
        )

    @tool
    def get_grant_details(grant: str) -> str:
        """Return the verified details of one scholarship.

        ``grant`` may be a grant id from ``list_verified_grants``, an exact
        title, or a distinctive fragment of the title.
        """
        card = ctx.resolve_grant(grant)
        if card is None:
            return _envelope(
                "grant",
                ok=False,
                summary=f"No verified grant matches {grant!r}.",
            )
        data = card.model_dump(mode="json")
        if ctx.deadline_verdict:
            data["precomputed_deadline_verdict"] = ctx.deadline_verdict
        if ctx.deadline_fact:
            data["deadline_fact"] = ctx.deadline_fact
        return _envelope(
            "grant",
            summary=f"Details for {card.title}.",
            data=data,
            sources=[card.official_url],
            deadline_verdict=ctx.deadline_verdict,
            deadline_fact=ctx.deadline_fact,
        )

    @tool
    def match_profile(limit: int = 10) -> str:
        """Compute the student's match percentage against every verified grant.

        Requires a complete profile (name, gpa, major, degree levels,
        nationality).  If fields are missing the tool returns them so you can ask
        the student; never invent values.
        """
        missing = ctx.profile.missing_required()
        if missing:
            return _envelope(
                "match",
                ok=False,
                summary="The profile is incomplete; ask for the missing fields.",
                missing_fields=missing,
            )
        cards = ctx.cards()
        report = match_grants(ctx.profile, cards, limit=max(1, min(int(limit or 10), 20)))
        summary = f"Top match is {report.top_percentage:.1f}%."
        if report.trigger_external_search:
            # Same threshold rule the deterministic engine uses.
            summary += " All internal matches are weak; consider search_external_grants."
        return _envelope(
            "match",
            summary=summary,
            data=report.model_dump(mode="json"),
            sources=[result.official_url for result in report.results],
            top_percentage=report.top_percentage,
        )

    @tool
    def build_application_plan(top_n: int = 3) -> str:
        """Build a ranked application plan (steps, documents, deadlines) for the
        student's best-matching grants. Requires a complete profile."""
        missing = ctx.profile.missing_required()
        if missing:
            return _envelope(
                "plan",
                ok=False,
                summary="The profile is incomplete; ask for the missing fields.",
                missing_fields=missing,
            )
        cards = ctx.cards()
        report = match_grants(ctx.profile, cards, limit=10)
        items = build_plan(
            ctx.profile, report, cards, top_n=max(1, min(int(top_n or 3), 10))
        )
        by_id = {card.grant_id: card for card in cards}
        sources = [
            by_id[item["grant_id"]].official_url
            for item in items
            if item.get("grant_id") in by_id
        ]
        return _envelope(
            "plan",
            summary=f"Built {len(items)} application plans.",
            data={"plans": items},
            sources=sources,
        )

    @tool
    def write_document(
        document_type: str = "motivation_letter",
        grant: str = "",
        language: str = "",
    ) -> str:
        """Draft a CV, motivation letter or recommendation brief for a grant.

        ``document_type`` is one of ``cv``, ``motivation_letter``,
        ``recommendation_brief``.  ``grant`` is an id/title; leave empty for the
        best-matching grant.  ``language`` is ``ar`` or ``en``.  The document is
        produced through draft -> critique -> revision and grounded in the
        student's real profile.
        """
        missing = ctx.profile.missing_required()
        if missing:
            return _envelope(
                "document",
                ok=False,
                summary="The profile is incomplete; ask for the missing fields.",
                missing_fields=missing,
            )
        card = ctx.resolve_grant(grant) or ctx.best_grant()
        if card is None:
            return _envelope(
                "document", ok=False, summary="No verified grant is available."
            )
        doc_type = _document_type(document_type)
        lang = "en" if str(language).strip().lower() == "en" else "ar"
        result = ctx.documents.generate(ctx.profile, card, doc_type, language=lang)
        return _envelope(
            "document",
            summary=(
                f"{doc_type.value} for {card.title} "
                f"({'grounded' if result.grounded else 'needs review'})."
            ),
            data=result.model_dump(mode="json"),
            sources=[card.official_url],
        )

    @tool
    def advise_profile() -> str:
        """Assess strengths, weaknesses, competitive gaps and concrete next actions
        to improve the student's scholarship competitiveness."""
        report = None
        if not ctx.profile.missing_required():
            report = match_grants(ctx.profile, ctx.cards(), limit=10, use_llm=False)
        advisor = build_advisor_report(ctx.profile, report)
        return _envelope(
            "advisor",
            summary=f"{len(advisor.actions)} improvement actions suggested.",
            data=advisor.model_dump(mode="json"),
        )

    @tool
    def search_external_grants() -> str:
        """Search the open web for scholarships outside the verified database.

        Only use when the student explicitly asks for external/other scholarships
        or when internal matches are exhausted.  Requires a search API key.
        """
        if not settings.external_search_configured:
            return _envelope(
                "external",
                ok=False,
                summary=(
                    "External search is not configured (missing Tavily/Serper key); "
                    "fall back to verified internal grants."
                ),
            )
        missing = [
            field
            for field in ("major", "target_degree")
            if field in ctx.profile.missing_required()
        ]
        if missing:
            return _envelope(
                "external",
                ok=False,
                summary="Major and target degree are required for external search.",
                missing_fields=missing,
            )
        try:
            from external_search_sub_agent import (  # type: ignore[import-not-found]
                search_external_scholarships,
            )

            output = run_with_timeout(
                search_external_scholarships,
                settings.external_search_timeout_seconds,
                major=ctx.profile.major or "",
                degree_level=(
                    ctx.profile.target_degree.value
                    if ctx.profile.target_degree
                    else "master"
                ),
                target_country=(
                    ctx.profile.target_countries[0]
                    if ctx.profile.target_countries
                    else None
                ),
            )
            data = (
                output.model_dump()
                if hasattr(output, "model_dump")
                else dict(getattr(output, "__dict__", {}))
            )
            results = data.get("results") or data.get("verified_results") or []
            return _envelope(
                "external",
                summary=f"{len(results)} external results found.",
                data={"results": results[:5]},
                sources=[
                    item.get("official_url") or item.get("url") or ""
                    for item in results[:5]
                ],
            )
        except OperationTimeoutError:
            logger.warning(
                "External search timed out after %ss",
                settings.external_search_timeout_seconds,
            )
            return _envelope(
                "external",
                ok=False,
                summary=(
                    "External search timed out; fall back to the verified "
                    "internal grants."
                ),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.exception("External search tool failed")
            return _envelope(
                "external", ok=False, summary=f"External search failed: {exc}"
            )

    @tool
    def update_profile(field: str, value: str) -> str:
        """Correct or add a single student profile field from explicit user input.

        ``field`` is one of name, email, gpa, major, degree_level,
        target_degree, nationality, country_of_residence, research_interests,
        target_fields, target_countries, experiences, extracurriculars,
        publications.  Never use this to invent or guess a value.
        """
        name = (field or "").strip().lower()
        if name not in _WRITABLE_FIELDS:
            return _envelope(
                "profile", ok=False, summary=f"Field {field!r} is not writable."
            )
        try:
            coerced = _coerce_field(name, value)
        except (TypeError, ValueError) as exc:
            return _envelope("profile", ok=False, summary=str(exc))
        ctx.profile = merge_profile(ctx.profile, {name: coerced})
        ctx.profile_dirty = True
        return _envelope(
            "profile",
            summary=f"Updated {name}.",
            data=ctx.profile.model_dump(mode="json"),
            missing_fields=ctx.profile.missing_required(),
        )

    return [
        list_verified_grants,
        get_grant_details,
        match_profile,
        build_application_plan,
        write_document,
        advise_profile,
        search_external_grants,
        update_profile,
    ]
