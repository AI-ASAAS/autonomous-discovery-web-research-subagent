"""Unified data contracts for the orchestrator.

These models are the single boundary between the discovery repository
(grants), the matcher repository (scores) and the conversational layer.
They deliberately mirror the existing sub-agent schemas without importing
them, so the orchestrator stays independently testable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DegreeLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"


class LanguageLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"
    NATIVE = "native"


class MatchTier(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    WEAK = "weak"


class Intent(str, Enum):
    GREET = "greet"
    PROFILE_UPDATE = "profile_update"
    TOP_GRANTS = "top_grants"
    MATCH = "match"
    PLAN = "plan"
    DOCUMENTS = "documents"
    ADVISOR = "advisor"
    EXTERNAL_SEARCH = "external_search"
    SCHOLARSHIP_QUESTION = "scholarship_question"
    OFF_TOPIC = "off_topic"
    BLOCKED = "blocked"


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    REDIRECT = "redirect"
    REFUSE = "refuse"


class GuardrailCategory(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    TOO_LONG = "too_long"
    PROMPT_INJECTION = "prompt_injection"
    SECRET_REQUEST = "secret_request"
    FABRICATION_REQUEST = "fabrication_request"
    PRIVACY_REQUEST = "privacy_request"
    UNSAFE = "unsafe"
    OFF_TOPIC = "off_topic"
    SECRET_LEAK = "secret_leak"
    UNVERIFIED_URL = "unverified_url"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    PROMPT_LEAK = "prompt_leak"
    URL_IN_DOCUMENT = "url_in_document"
    PLACEHOLDER = "placeholder"


class DocumentType(str, Enum):
    CV = "cv"
    MOTIVATION_LETTER = "motivation_letter"
    RECOMMENDATION_BRIEF = "recommendation_brief"


# ---------------------------------------------------------------------------
# Student profile
# ---------------------------------------------------------------------------

REQUIRED_PROFILE_FIELDS: tuple[str, ...] = (
    "name",
    "gpa",
    "major",
    "degree_level",
    "target_degree",
    "nationality",
)


class LanguageScore(BaseModel):
    language: str
    level: LanguageLevel
    test_name: str | None = None
    score: float | None = None


class StudentProfile(BaseModel):
    """Student profile that may still be incomplete during a conversation."""

    student_id: str = "default"
    name: str | None = None
    email: str | None = None
    gpa: float | None = Field(default=None, ge=0.0, le=4.0)
    major: str | None = None
    degree_level: DegreeLevel | None = None
    target_degree: DegreeLevel | None = None
    nationality: str | None = None
    country_of_residence: str | None = None
    languages: list[LanguageScore] = Field(default_factory=list)
    research_interests: list[str] = Field(default_factory=list)
    target_fields: list[str] = Field(default_factory=list)
    target_countries: list[str] = Field(default_factory=list)
    experiences: list[str] = Field(default_factory=list)
    extracurriculars: list[str] = Field(default_factory=list)
    publications: int = Field(default=0, ge=0)

    @field_validator("name", "major", "nationality", "country_of_residence")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None

    def missing_required(self) -> list[str]:
        missing: list[str] = []
        for field_name in REQUIRED_PROFILE_FIELDS:
            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)
        return missing

    def completeness(self) -> float:
        """Percentage of required fields filled (extra signals excluded)."""
        filled = len(REQUIRED_PROFILE_FIELDS) - len(self.missing_required())
        return round((filled / len(REQUIRED_PROFILE_FIELDS)) * 100.0, 1)


# ---------------------------------------------------------------------------
# Grants (unified view over the discovery repository)
# ---------------------------------------------------------------------------


class GrantCard(BaseModel):
    grant_id: str
    title: str
    official_url: str
    provider: str = ""
    country: str = ""
    description: str = ""
    funding: str = ""
    deadline: str | None = None
    fields_of_study: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    eligibility: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)
    how_to_apply: str | None = None
    accepts_yemeni: Literal["yes", "no", "unclear"] = "unclear"
    evidence: dict[str, str] = Field(default_factory=dict)

    @property
    def host(self) -> str:
        from urllib.parse import urlparse

        return (urlparse(self.official_url).hostname or "").lower()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    gpa_score: float = 0.0
    degree_score: float = 0.0
    language_score: float = 0.0
    nationality_score: float = 0.0
    major_score: float = 0.0
    semantic_alignment_score: float = 0.0
    experience_score: float = 0.0
    publications_score: float = 0.0


class MatchResult(BaseModel):
    grant_id: str
    title: str
    match_percentage: float = Field(ge=0.0, le=100.0)
    tier: MatchTier
    breakdown: ScoreBreakdown
    reasons: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    missing_documents: list[str] = Field(default_factory=list)
    deadline: str | None = None
    official_url: str = ""


class MatchReport(BaseModel):
    results: list[MatchResult] = Field(default_factory=list)
    top_percentage: float = 0.0
    trigger_external_search: bool = False
    external_search_reason: str | None = None
    semantic_source: Literal["deterministic", "llm"] = "deterministic"


# ---------------------------------------------------------------------------
# Documents and reflection loop
# ---------------------------------------------------------------------------


class DocumentIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str


class DocumentRevision(BaseModel):
    version: int
    stage: Literal["draft", "critique", "revised", "final"]
    content: str
    issues: list[DocumentIssue] = Field(default_factory=list)


class DocumentResult(BaseModel):
    document_type: DocumentType
    grant_id: str
    grant_title: str
    language: Literal["ar", "en"] = "ar"
    content: str
    revisions: list[DocumentRevision] = Field(default_factory=list)
    passed: bool = True
    grounded: bool = True
    issues: list[DocumentIssue] = Field(default_factory=list)
    provider: Literal["deterministic", "llm"] = "deterministic"


# ---------------------------------------------------------------------------
# Advisor (task 6)
# ---------------------------------------------------------------------------


class AdvisorAction(BaseModel):
    title: str
    why: str
    how: str
    timeframe: str
    priority: Literal["high", "medium", "low"] = "medium"


class AdvisorReport(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    competitive_gaps: list[str] = Field(default_factory=list)
    actions: list[AdvisorAction] = Field(default_factory=list)
    overall_competitiveness: str = ""


# ---------------------------------------------------------------------------
# Conversation boundary
# ---------------------------------------------------------------------------


class ThoughtStep(BaseModel):
    step: str
    label: str
    status: Literal["started", "completed", "skipped", "failed"] = "completed"
    detail: str = ""
    at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class GuardrailEvent(BaseModel):
    stage: Literal["input", "output"]
    category: GuardrailCategory
    action: GuardrailAction
    reason: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str = ""


class ChatRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=4000)
    profile: StudentProfile | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Intent
    language: Literal["ar", "en"] = "ar"
    steps: list[ThoughtStep] = Field(default_factory=list)
    guardrails: list[GuardrailEvent] = Field(default_factory=list)
    profile: StudentProfile = Field(default_factory=StudentProfile)
    missing_fields: list[str] = Field(default_factory=list)
    matches: list[MatchResult] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[DocumentResult] = Field(default_factory=list)
    advisor: AdvisorReport | None = None
    sources: list[str] = Field(default_factory=list)
    llm_used: bool = False
    model_calls: int = 0


class DocumentRequest(BaseModel):
    session_id: str = "default"
    document_type: DocumentType = DocumentType.MOTIVATION_LETTER
    grant_id: str | None = None
    grant_title: str | None = None
    language: Literal["ar", "en"] = "ar"
    profile: StudentProfile | None = None


class PlanRequest(BaseModel):
    session_id: str = "default"
    profile: StudentProfile | None = None
    top_n: int = Field(default=3, ge=1, le=10)


class MatchRequest(BaseModel):
    session_id: str = "default"
    profile: StudentProfile | None = None
    limit: int = Field(default=10, ge=1, le=20)


class AdvisorRequest(BaseModel):
    session_id: str = "default"
    profile: StudentProfile | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: Literal["deterministic", "llm"]
    llm_provider: str = ""
    agent_engine: str = "legacy"
    agent_ready: bool = False
    external_search_configured: bool = False
    grants_available: int = 0
    repos_present: bool = True
    llm_available: bool = True
    llm_retry_after_seconds: float = 0.0
    llm_config_error: str | None = None
