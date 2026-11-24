"""Read access to the verified grants database of the discovery repository.

The discovery sub-agent already owns the grant lifecycle (DRAFT -> PUBLISHED
-> EXPIRED) and stores anti-hallucination evidence.  This module does not
re-implement any of that; it opens the same SQLite file with an absolute path
and exposes a small, typed read model for the orchestrator.

If the database is empty (fresh checkout) it is seeded with the verified
top-10 baseline that ships with the repository - no network access.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .repos import setup_repo_paths
from .schemas import GrantCard

logger = logging.getLogger(__name__)

setup_repo_paths()

from core.models import Grant, GrantStatus, Base  # noqa: E402
from core.models import YemeniEligibilityStatus  # noqa: E402

_COUNTRY_HINTS: dict[str, str] = {
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "turkish": "Turkey",
    "turkey": "Turkey",
    "chevening": "United Kingdom",
    "uk ": "United Kingdom",
    "united kingdom": "United Kingdom",
    "daad": "Germany",
    "german": "Germany",
    "germany": "Germany",
    "hungar": "Hungary",
    "romania": "Romania",
    "romanian": "Romania",
    "fulbright": "United States",
    "u.s.": "United States",
    "united states": "United States",
    "china": "China",
    "chinese": "China",
    "russia": "Russia",
    "russian": "Russia",
    "eiffel": "France",
    "france": "France",
    "french": "France",
    "islamic development bank": "Saudi Arabia",
}

_EVIDENCE_FIELDS = (
    "title",
    "description",
    "eligibility",
    "funding",
    "deadline",
    "how_to_apply",
    "accepts_yemeni",
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _infer_country(text: str) -> str:
    lowered = _normalize(text)
    for hint, country in _COUNTRY_HINTS.items():
        if hint in lowered:
            return country
    return ""


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


class GrantRepository:
    """Typed, read-only facade over the discovery SQLite database."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        seed_if_empty: bool = True,
    ) -> None:
        self.db_path = Path(db_path or settings.grants_db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        self._session_factory = sessionmaker(
            bind=self._engine, autoflush=False, expire_on_commit=False
        )
        self._seed_lock = threading.Lock()
        Base.metadata.create_all(bind=self._engine)
        if seed_if_empty:
            self._seed_if_empty()

    # -- seeding ---------------------------------------------------------
    def _count(self) -> int:
        with self._session_factory() as session:
            return session.query(Grant).count()

    def _seed_if_empty(self) -> None:
        with self._seed_lock:
            if self._count() > 0:
                return
            try:
                from seed.seed_db import VERIFIED_TOP10_BASELINE
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Could not load verified seed baseline: %s", exc)
                return

            with self._session_factory() as session:
                for entry in VERIFIED_TOP10_BASELINE:
                    if (
                        session.query(Grant)
                        .filter(Grant.official_url == entry["official_url"])
                        .first()
                    ):
                        continue
                    reference = (
                        f"{entry.get('title', '')} {entry.get('description', '')} "
                        f"{entry.get('funding', '')}"
                    )
                    session.add(
                        Grant(
                            title=entry["title"],
                            official_url=entry["official_url"],
                            description=entry.get("description"),
                            eligibility=entry.get("eligibility"),
                            criteria=entry.get("criteria"),
                            fields_of_study=entry.get("fields_of_study"),
                            funding=entry.get("funding"),
                            required_documents=entry.get("required_documents"),
                            deadline=entry.get("deadline"),
                            how_to_apply=entry.get("how_to_apply"),
                            accepts_yemeni=entry.get(
                                "accepts_yemeni", YemeniEligibilityStatus.UNCLEAR
                            ),
                            evidence=entry.get("evidence", {}),
                            field_verification=entry.get("field_verification", {}),
                            status=GrantStatus.PUBLISHED,
                            content_hash=hashlib.sha256(
                                reference.encode("utf-8")
                            ).hexdigest(),
                        )
                    )
                session.commit()
            logger.info("Seeded %s verified grants into %s", self._count(), self.db_path)

    # -- reads -----------------------------------------------------------
    def list_published(
        self,
        limit: int | None = None,
        *,
        reference_date: date | None = None,
    ) -> list[GrantCard]:
        limit = limit or settings.top_grants_limit
        ref = reference_date or date.today()
        with self._session_factory() as session:
            rows = (
                session.query(Grant)
                .filter(Grant.status == GrantStatus.PUBLISHED)
                .filter(or_(Grant.deadline.is_(None), Grant.deadline >= ref))
                .order_by(Grant.deadline.is_(None), Grant.deadline.asc())
                .limit(limit)
                .all()
            )
            return [self._to_card(row) for row in rows]

    def get(self, grant_id: str) -> GrantCard | None:
        try:
            numeric_id = int(str(grant_id).lstrip("Gg"))
        except ValueError:
            return None
        with self._session_factory() as session:
            row = session.get(Grant, numeric_id)
            return self._to_card(row) if row else None

    def count(self) -> int:
        with self._session_factory() as session:
            return (
                session.query(Grant)
                .filter(Grant.status == GrantStatus.PUBLISHED)
                .count()
            )

    def search(self, text: str, *, limit: int = 3) -> list[GrantCard]:
        """Rank published grants by token overlap with free text."""
        tokens = {token for token in re.split(r"[^\w\u0600-\u06FF]+", text.lower()) if len(token) > 2}
        if not tokens:
            return []
        scored: list[tuple[float, GrantCard]] = []
        for card in self.list_published(limit=100):
            haystack = _normalize(
                " ".join(
                    [
                        card.title,
                        card.description,
                        card.funding,
                        " ".join(card.fields_of_study),
                        card.provider,
                        card.country,
                    ]
                )
            )
            overlap = sum(1 for token in tokens if token in haystack)
            if overlap:
                scored.append((overlap / len(tokens), card))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [card for score, card in scored[:limit] if score >= 0.2]

    def find_by_title(self, text: str) -> GrantCard | None:
        candidates = self.search(text, limit=1)
        return candidates[0] if candidates else None

    def urls(self) -> set[str]:
        return {card.official_url for card in self.list_published(limit=100)}

    # -- mapping ---------------------------------------------------------
    @staticmethod
    def _to_card(row: Grant) -> GrantCard:
        host = urlparse(row.official_url).hostname or ""
        accepts = row.accepts_yemeni
        if isinstance(accepts, YemeniEligibilityStatus):
            accepts_value = accepts.value
        else:
            accepts_value = str(accepts or "unclear").lower().split(".")[-1]
        if accepts_value not in {"yes", "no", "unclear"}:
            accepts_value = "unclear"

        evidence = row.evidence if isinstance(row.evidence, dict) else {}
        clean_evidence = {
            key: str(evidence[key])
            for key in _EVIDENCE_FIELDS
            if key in evidence and evidence[key]
        }

        text_blob = " ".join(
            filter(
                None,
                [
                    row.title,
                    row.description,
                    _as_text(row.eligibility),
                    _as_text(row.criteria),
                ],
            )
        )
        return GrantCard(
            grant_id=f"G{row.id}",
            title=row.title,
            official_url=row.official_url,
            provider=host.removeprefix("www."),
            country=_infer_country(text_blob),
            description=row.description or "",
            funding=row.funding or "",
            deadline=row.deadline.isoformat() if row.deadline else None,
            fields_of_study=list(row.fields_of_study or []),
            required_documents=list(row.required_documents or []),
            eligibility=list(row.eligibility or []),
            criteria=list(row.criteria or []),
            how_to_apply=row.how_to_apply,
            accepts_yemeni=accepts_value,  # type: ignore[arg-type]
            evidence=clean_evidence,
        )


def cards_to_sources(cards: Iterable[GrantCard]) -> list[str]:
    seen: list[str] = []
    for card in cards:
        if card.official_url and card.official_url not in seen:
            seen.append(card.official_url)
    return seen
