"""Central stateful orchestrator (Person 1).

Responsibilities
----------------
* keep a live student profile and transcript per session (MemorySaver-style),
* merge facts from each message without inventing anything,
* detect missing required fields and ask for them proactively,
* route to the existing sub-agents (matching, planning, external search),
* run the reflection document agent when asked,
* guard every input and every output,
* stream the reasoning steps for the UI.

The routing is deterministic and keyword based, so it is testable without an
LLM provider.  When a provider is configured it is only used for optional
free-form enrichment; the state transitions never depend on it.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
from typing import Any, Callable, Iterator

from . import guardrails, llm
from .advisor import build_advisor_report
from .concurrency import OperationTimeoutError, run_with_timeout
from .config import settings
from .deadlines import deadline_verdict, find_focused_grant, parse_date_from_text
from .document_agent import DocumentAgent
from .extraction import detect_language, extract_profile_patch, merge_profile
from .guardrails import check_input, check_output, is_domain_text, normalize
from .knowledge import GrantRepository
from .matching import IncompleteProfile, match_grants
from .memory import MemoryStore
from .planning import build_plan, plan_summary
from .resolution import (
    COUNTRY_ALIASES as _COUNTRY_ALIASES,
    GRANT_ALIASES as _GRANT_ALIASES,
    detect_document_type,
    resolve_grant as _resolve_grant_shared,
)
from .schemas import (
    AdvisorReport,
    ChatMessage,
    ChatResponse,
    DocumentResult,
    DocumentType,
    GrantCard,
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvent,
    Intent,
    MatchReport,
    MatchResult,
    StudentProfile,
    ThoughtStep,
)

logger = logging.getLogger(__name__)

_MISSING_QUESTIONS = {
    "name": ("ما اسمك الكامل؟", "What is your full name?"),
    "gpa": ("ما معدلك التراكمي على مقياس 4.00؟ (مثال: 3.40)", "What is your GPA on a 4.00 scale?"),
    "major": ("ما تخصصك الحالي؟", "What is your current major?"),
    "degree_level": (
        "ما درجتك الحالية؟ (بكالوريوس / ماجستير / دكتوراه)",
        "What is your current degree? (bachelor / master / PhD)",
    ),
    "target_degree": (
        "ما الدرجة التي تستهدفها؟ (بكالوريوس / ماجستير / دكتوراه)",
        "Which degree are you targeting? (bachelor / master / PhD)",
    ),
    "nationality": ("ما جنسيتك؟", "What is your nationality?"),
}

_TIER_AR = {
    "excellent": "ممتاز",
    "good": "جيد",
    "very_good": "جيد جداً",
    "fair": "مقبول",
    "weak": "ضعيف",
}

_GREETING_REPLY_AR = (
    "أهلاً بك في مستشار المنح الدراسية. سأبني ملفك خطوة بخطوة ثم أحسب لك نسب "
    "التطابق مع المنح الموثقة، وأجهّز خطة تقديم ووثائق عند الحاجة."
)
_GREETING_REPLY_EN = (
    "Welcome to the scholarship advisor. I will build your profile step by step, "
    "rank the verified grants for you, and prepare an application plan and documents."
)

_SAFE_FALLBACK_AR = (
    "تعذر توليد رد موثوق في هذه اللحظة. أعد صياغة سؤالك عن المنح أو ملفك الأكاديمي "
    "وسأجيبك من البيانات الموثقة فقط."
)
_SAFE_FALLBACK_EN = (
    "I could not produce a verified answer just now. Please rephrase your scholarship "
    "or profile question and I will answer from verified data only."
)

_REFUSAL_AR = {
    GuardrailCategory.PROMPT_INJECTION: "لا أستطيع تنفيذ تعليمات تتجاوز نطاق المنصة أو تكشف إعداداتها الداخلية.",
    GuardrailCategory.SECRET_REQUEST: "لا يمكنني عرض مفاتيح API أو أسرار النظام. يضبطها المشرف من لوحة الإدارة.",
    GuardrailCategory.FABRICATION_REQUEST: "لا يمكنني اختلاق شهادات أو خبرات أو تعديل بياناتك. أساعدك في إبراز إنجازاتك الحقيقية فقط.",
    GuardrailCategory.PRIVACY_REQUEST: "لا أستطيع الوصول إلى بيانات طالب آخر أو مشاركتها. أساعدك بملفك أنت فقط.",
    GuardrailCategory.UNSAFE: "هذا الطلب خارج نطاق خدمة المنح الدراسية. إن كنت تمر بضغط نفسي فتحدث إلى شخص تثق به أو مختص.",
    GuardrailCategory.TOO_LONG: "الرسالة أطول من الحد المسموح. اختصرها لأستطيع مساعدتك.",
    GuardrailCategory.EMPTY: "أرسل سؤالاً عن ملفك الأكاديمي أو المنح الدراسية.",
}
_REFUSAL_EN = {
    GuardrailCategory.PROMPT_INJECTION: "I cannot follow instructions that go outside the platform scope or reveal internal configuration.",
    GuardrailCategory.SECRET_REQUEST: "I cannot reveal API keys or system secrets. An administrator configures them in the admin panel.",
    GuardrailCategory.FABRICATION_REQUEST: "I cannot fabricate certificates, experience, or alter your data. I can only help present real achievements.",
    GuardrailCategory.PRIVACY_REQUEST: "I cannot access or share another student's data. I can only help with your own profile.",
    GuardrailCategory.UNSAFE: "This request is outside the scholarship domain. If you are under psychological pressure, please talk to someone you trust or a professional.",
    GuardrailCategory.TOO_LONG: "Your message is longer than the allowed limit. Please shorten it.",
    GuardrailCategory.EMPTY: "Please ask about your academic profile or scholarships.",
}

def missing_field_question(field: str, language: str) -> str:
    """Localized question for a missing profile field (shared with the agent)."""
    pair = _MISSING_QUESTIONS.get(field)
    if not pair:
        return field
    return pair[0 if language == "ar" else 1]


def refusal_for(category: GuardrailCategory, language: str) -> str:
    """Localized refusal text for a blocked request (shared with the agent engine)."""
    table = _REFUSAL_AR if language == "ar" else _REFUSAL_EN
    return table.get(category) or (
        "لا أستطيع تنفيذ هذا الطلب." if language == "ar" else "I cannot fulfil this request."
    )


_DOCUMENT_KEYWORDS = (
    "سيرة ذاتية", "سيره ذاتيه", "cv", "resume", "خطاب دافع", "خطاب الدافع",
    "رسالة دافع", "motivation letter", "statement of purpose", "sop",
    "رسالة توصية", "رساله توصيه", "recommendation", "خطاب توصية",
)
_PLAN_KEYWORDS = (
    "خطة", "خطه", "خطة تقديم", "خطه تقديم", "خارطة طريق", "خارطه طريق",
    "جدول", "roadmap", "plan", "خطوات التقديم", "كيف أقدم", "كيف اقدم",
    "ماذا أفعل", "ماذا افعل",
)
_ADVISOR_KEYWORDS = (
    "نقاط الضعف", "نقاط القوة", "نقاط قوتي", "نقاط ضعفي", "قوتي", "ضعفي",
    "طوير ملفي", "أطور ملفي", "اطور ملفي", "تحسين", "نصائح", "نصيحة",
    "advisor", "strengths", "weaknesses", "improve", "gap", "فجوات",
)
_MATCH_KEYWORDS = (
    "نسبة", "نسبه", "تطابق", "مطابقة", "مطابقه", "طابق", "طابقني", "قارن",
    "أي منحة", "اي منحه", "أي منح", "اي منح", "مناسبة لي", "مناسبه لي",
    "فرص", "فرصي", "chance", "match", "percentage", "rank",
)
_TOP_KEYWORDS = (
    "أفضل 10", "افضل 10", "أفضل عشر", "افضل عشر", "المنح المتاحة", "المنح المتاحه",
    "اعرض المنح", "أعرض المنح", "قائمة المنح", "قائمه المنح", "top 10",
    "list grants", "available scholarships", "المنح المنشورة",
)
_EXTERNAL_KEYWORDS = (
    "بحث خارجي", "منح خارجية", "منح خارجيه", "خارج المنصة", "خارج المنصه",
    "ابحث في الويب", "ابحث في الانترنت", "google", "قوقل", "external search",
    "search the web", "منح أخرى", "منح اخري", "ابحث لي",
)
_QUESTION_WORDS = (
    "هل", "ما", "ماذا", "كيف", "متى", "أين", "اين", "شروط", "موعد", "رابط",
    "تفاصيل", "what", "how", "when", "where", "deadline", "eligibility", "link",
    "requirements", "details",
)


class Orchestrator:
    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        repository: GrantRepository | None = None,
        documents: DocumentAgent | None = None,
    ) -> None:
        self.memory = memory or MemoryStore()
        self.repository = repository or GrantRepository()
        self.documents = documents or DocumentAgent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        session_id: str,
        message: str,
        *,
        profile_override: StudentProfile | None = None,
        on_step: Callable[[ThoughtStep], None] | None = None,
    ) -> ChatResponse:
        steps: list[ThoughtStep] = []
        guard_events: list[GuardrailEvent] = []

        def step(name: str, label: str, status: str = "completed", detail: str = "") -> None:
            item = ThoughtStep(step=name, label=label, status=status, detail=detail)  # type: ignore[arg-type]
            steps.append(item)
            if on_step:
                try:
                    on_step(item)
                except Exception:  # pragma: no cover - stream consumer failure
                    logger.debug("step callback failed", exc_info=True)

        profile = self.memory.load_profile(session_id)
        language = detect_language(message)
        self.memory.append_message(session_id, "user", message)

        # 1. Input guardrails
        decision = check_input(message, max_chars=settings.max_message_chars)
        if decision.event:
            guard_events.append(decision.event)
        step("input_guard", "فحص الرسالة وحواجز الأمان", detail=decision.category.value)

        if decision.action == GuardrailAction.REFUSE:
            reply = self._refusal(decision.category, language)
            self.memory.append_message(session_id, "assistant", reply)
            step("done", "تجهيز الرد", detail="blocked")
            return ChatResponse(
                session_id=session_id,
                reply=reply,
                intent=Intent.BLOCKED,
                language=language,  # type: ignore[arg-type]
                steps=steps,
                guardrails=guard_events,
                profile=profile,
                missing_fields=profile.missing_required(),
            )

        # 2. Merge explicit profile override, then extracted facts
        if profile_override is not None:
            profile = self._merge_override(profile, profile_override)
        patch = extract_profile_patch(decision.text)
        if patch:
            profile = merge_profile(profile, patch)
        self.memory.save_profile(session_id, profile)
        step(
            "profile",
            "تحديث ملف الطالب",
            detail=f"completeness={profile.completeness():.0f}%",
        )

        missing = profile.missing_required()
        intent = self._route(decision.text, profile, missing)
        step("route", "تحديد المسار", detail=intent.value)

        if decision.action == GuardrailAction.REDIRECT and intent != Intent.GREET:
            intent = Intent.OFF_TOPIC

        cards: list[GrantCard] = []
        matches: list[MatchResult] = []
        plan: list[dict] = []
        documents_out: list[DocumentResult] = []
        advisor_out: AdvisorReport | None = None
        sources: list[str] = []

        try:
            if intent in {Intent.GREET, Intent.PROFILE_UPDATE}:
                reply = self._profile_reply(profile, missing, language)
                if intent == Intent.GREET:
                    reply = (
                        _GREETING_REPLY_AR if language == "ar" else _GREETING_REPLY_EN
                    ) + "\n\n" + reply
                    cards = self.repository.list_published(limit=3)
                    step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")
                    reply += "\n\n" + self._top_grants_reply(cards, language, limit=3)
                    sources += [card.official_url for card in cards]

            elif intent == Intent.TOP_GRANTS:
                cards = self.repository.list_published()
                step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")
                reply = self._top_grants_reply(cards, language)
                sources += [card.official_url for card in cards]

            elif intent == Intent.MATCH:
                reply, matches, cards = self._handle_match(session_id, profile, missing, language, step)
                sources += [match.official_url for match in matches]

            elif intent == Intent.PLAN:
                reply, matches, cards, plan = self._handle_plan(
                    session_id, profile, missing, language, step
                )
                sources += [match.official_url for match in matches]
                if plan:
                    self.memory.save_artifact(
                        session_id, "plan", {"items": self._jsonable(plan)}
                    )

            elif intent == Intent.ADVISOR:
                reply, matches, cards, advisor_out = self._handle_advisor(
                    session_id, profile, missing, language, step
                )
                sources += [match.official_url for match in matches]
                if advisor_out:
                    self.memory.save_artifact(
                        session_id, "advisor", advisor_out.model_dump(mode="json")
                    )

            elif intent == Intent.DOCUMENTS:
                reply, documents_out, cards = self._handle_documents(
                    session_id, profile, missing, decision.text, language, step
                )
                sources += [card.official_url for card in cards]

            elif intent == Intent.EXTERNAL_SEARCH:
                reply, cards = self._handle_external(profile, missing, language, step)
                sources += [card.official_url for card in cards]

            elif intent == Intent.SCHOLARSHIP_QUESTION:
                reply, cards = self._handle_question(
                    decision.text, profile, missing, language, step, session_id=session_id
                )
                sources += [card.official_url for card in cards]

            elif intent == Intent.OFF_TOPIC:
                reply = self._off_topic_reply(language)
                cards = self.repository.list_published(limit=3)
                step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")

            else:  # pragma: no cover - exhaustive above
                reply = self._profile_reply(profile, missing, language)
        except IncompleteProfile as exc:
            step("route", "تحديد المسار", status="failed", detail="incomplete profile")
            reply = self._missing_reply(profile, exc.missing, language)

        # 3. Output guardrails
        output = check_output(reply, known_grants=cards)
        guard_events.extend(output.events)
        reply = output.text
        if not output.passed:
            reply = _SAFE_FALLBACK_AR if language == "ar" else _SAFE_FALLBACK_EN
        step("output_guard", "تدقيق المخرجات", detail=f"{len(output.issues)} issues")

        self.memory.append_message(session_id, "assistant", reply)
        step("done", "تجهيز الرد", detail=f"matches={len(matches)} docs={len(documents_out)}")

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            intent=intent,
            language=language,  # type: ignore[arg-type]
            steps=steps,
            guardrails=guard_events,
            profile=profile,
            missing_fields=missing,
            matches=matches,
            plan=self._jsonable(plan),
            documents=documents_out,
            advisor=advisor_out,
            sources=sorted(set(sources)),
            llm_used=llm.llm_used_in_turn(),
            model_calls=llm.model_calls(),
        )

    def stream(
        self, session_id: str, message: str
    ) -> Iterator[tuple[str, Any]]:
        """Yield ('step', ThoughtStep) live, then ('final', ChatResponse)."""
        events: queue.Queue[tuple[str, Any]] = queue.Queue()

        def worker() -> None:
            try:
                response = self.chat(
                    session_id,
                    message,
                    on_step=lambda item: events.put(("step", item)),
                )
                events.put(("final", response))
            except Exception as exc:  # pragma: no cover - surfaced to client
                logger.exception("Orchestrator stream failed")
                events.put(("error", str(exc)))
            finally:
                events.put(("end", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            kind, payload = events.get()
            if kind == "end":
                return
            yield kind, payload

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _handle_match(
        self,
        session_id: str,
        profile: StudentProfile,
        missing: list[str],
        language: str,
        step: Callable[..., None],
    ) -> tuple[str, list[MatchResult], list[GrantCard]]:
        if missing:
            return self._missing_reply(profile, missing, language), [], []
        cards = self.repository.list_published(limit=50)
        step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")
        report = match_grants(profile, cards)
        step("match", "حساب نسب التطابق", detail=f"top={report.top_percentage:.1f}%")
        self.memory.save_artifact(
            session_id, "match", report.model_dump(mode="json")
        )
        reply = self._match_reply(report, language)
        return reply, report.results[:10], cards

    def _handle_plan(
        self,
        session_id: str,
        profile: StudentProfile,
        missing: list[str],
        language: str,
        step: Callable[..., None],
    ) -> tuple[str, list[MatchResult], list[GrantCard], list[dict]]:
        if missing:
            return self._missing_reply(profile, missing, language), [], [], []
        cards = self.repository.list_published(limit=50)
        step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")
        report = match_grants(profile, cards)
        step("match", "حساب نسب التطابق", detail=f"top={report.top_percentage:.1f}%")
        items = build_plan(profile, report, cards, top_n=3)
        step("plan", "بناء خطة التقديم", detail=f"{len(items)} plans")
        reply = self._plan_reply(items, report, language)
        return reply, report.results[:10], cards, items

    def _handle_advisor(
        self,
        session_id: str,
        profile: StudentProfile,
        missing: list[str],
        language: str,
        step: Callable[..., None],
    ) -> tuple[str, list[MatchResult], list[GrantCard], AdvisorReport | None]:
        cards = self.repository.list_published(limit=50)
        step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")
        report: MatchReport | None = None
        if not missing:
            report = match_grants(profile, cards)
            step("match", "حساب نسب التطابق", detail=f"top={report.top_percentage:.1f}%")
        advisor = build_advisor_report(profile, report)
        step("advisor", "تحليل نقاط القوة والضعف", detail=f"{len(advisor.actions)} actions")
        reply = self._advisor_reply(advisor, missing, language)
        return reply, (report.results[:10] if report else []), cards, advisor

    def _handle_documents(
        self,
        session_id: str,
        profile: StudentProfile,
        missing: list[str],
        message: str,
        language: str,
        step: Callable[..., None],
    ) -> tuple[str, list[DocumentResult], list[GrantCard]]:
        if missing:
            return self._missing_reply(profile, missing, language), [], []

        cards = self.repository.list_published(limit=50)
        step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")
        grant = self._resolve_grant(message, cards)
        if grant is None:
            report = match_grants(profile, cards, limit=1)
            if not report.results:
                return (
                    "أكمل بياناتك أولاً أو حدد اسم المنحة لأتمكن من صياغة الوثيقة."
                    if language == "ar"
                    else "Complete your profile or name a grant before I can draft the document.",
                    [],
                    cards,
                )
            grant = next(
                (card for card in cards if card.grant_id == report.results[0].grant_id),
                None,
            )
        if grant is None:
            return (
                "لم أجد المنحة المطلوبة في قاعدة البيانات الموثقة. اذكر اسمها بدقة أو اطلب البحث الخارجي."
                if language == "ar"
                else "I could not find that grant in the verified database. Name it precisely or request external search.",
                [],
                cards,
            )

        doc_type = self._document_type(message)
        step("documents", "صياغة الوثائق ومراجعتها", detail=doc_type.value)
        result = self.documents.generate(profile, grant, doc_type, language=language)
        self.memory.save_artifact(
            session_id, "document", result.model_dump(mode="json")
        )
        reply = self._documents_reply(result, language)
        return reply, [result], [grant]

    def _handle_external(
        self,
        profile: StudentProfile,
        missing: list[str],
        language: str,
        step: Callable[..., None],
    ) -> tuple[str, list[GrantCard]]:
        required = [field for field in ("major", "target_degree") if field in missing]
        if required:
            reply = self._missing_reply(profile, required, language)
            return reply, []
        if not settings.external_search_configured:
            step("external", "البحث الخارجي", status="skipped", detail="no search key")
            cards = self.repository.list_published(limit=3)
            if language == "ar":
                reply = (
                    "البحث الخارجي غير مفعّل: يلزم مفتاح Tavily أو Serper في إعدادات الخادم. "
                    "يمكنني الآن ترشيح المنح الداخلية الموثقة لك:\n\n"
                    + self._top_grants_reply(cards, language, limit=3)
                )
            else:
                reply = (
                    "External search is not configured: a Tavily or Serper key is required. "
                    "Here are the verified internal grants in the meantime:\n\n"
                    + self._top_grants_reply(cards, language, limit=3)
                )
            return reply, cards

        step("external", "البحث الخارجي", detail="ReAct search")
        try:
            from external_search_sub_agent import search_external_scholarships  # noqa: WPS433

            output = run_with_timeout(
                search_external_scholarships,
                settings.external_search_timeout_seconds,
                major=profile.major or "",
                degree_level=(profile.target_degree.value if profile.target_degree else "master"),
                target_country=(profile.target_countries[0] if profile.target_countries else None),
            )
            data = (
                output.model_dump()
                if hasattr(output, "model_dump")
                else dict(getattr(output, "__dict__", {}))
            )
            results = data.get("results") or data.get("verified_results") or []
            if not results:
                step("external", "البحث الخارجي", status="completed", detail="0 results")
                return (
                    "لم أجد نتائج خارجية موثقة لهذا الملف حالياً. جرّب توسيع الدول أو التخصص."
                    if language == "ar"
                    else "I found no verified external results for this profile. Try broadening the country or field.",
                    [],
                )
            lines = [
                "نتائج البحث الخارجي الموثقة:" if language == "ar" else "Verified external results:"
            ]
            for index, item in enumerate(results[:5], 1):
                title = item.get("title") or item.get("name") or "Grant"
                url = item.get("official_url") or item.get("url") or ""
                lines.append(f"{index}. {title} — {url}")
                if item.get("reason"):
                    lines.append(f"   {item['reason']}")
            return "\n".join(lines), []
        except OperationTimeoutError:
            logger.warning(
                "External search timed out after %ss",
                settings.external_search_timeout_seconds,
            )
            step("external", "البحث الخارجي", status="failed", detail="timeout")
            return (
                "استغرق البحث الخارجي وقتاً أطول من الحد المسموح. سأعتمد على المنح الموثقة داخلياً."
                if language == "ar"
                else "External search took too long and was stopped. I will rely on the verified internal grants.",
                [],
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.exception("External search failed")
            step("external", "البحث الخارجي", status="failed", detail=str(exc)[:120])
            return (
                "تعذر إكمال البحث الخارجي الآن (خطأ في الشبكة أو المزود). سأعتمد على المنح الموثقة داخلياً."
                if language == "ar"
                else "External search failed (network or provider error). I will rely on verified internal grants.",
                [],
            )

    def _handle_question(
        self,
        message: str,
        profile: StudentProfile,
        missing: list[str],
        language: str,
        step: Callable[..., None],
        session_id: str | None = None,
    ) -> tuple[str, list[GrantCard]]:
        cards = self.repository.list_published(limit=50)
        step("knowledge", "تحميل المنح الموثقة", detail=f"{len(cards)} grants")

        history = (
            [
                {"role": item.role, "content": item.content}
                for item in self.memory.history(session_id, limit=6)
            ]
            if session_id
            else []
        )
        user_date = parse_date_from_text(message)
        if user_date is not None:
            focused = find_focused_grant(message, cards, history, self.repository)
            verdict = deadline_verdict(user_date, focused, language)
            if focused is None:
                return verdict, []
            reply = self._deadline_details_reply(verdict, focused, language)
            return reply, [focused]

        grant = self._resolve_grant(message, cards)
        if grant:
            reply = self._grant_details_reply(grant, language)
            if not missing:
                reply += (
                    "\n\nيمكنني الآن إعداد خطة تقديم مخصصة وخطاب دافع لهذه المنحة إن رغبت."
                    if language == "ar"
                    else "\n\nI can now prepare a tailored application plan and motivation letter for this grant."
                )
            return reply, [grant]
        if not missing:
            report = match_grants(profile, cards, limit=3)
            step("match", "حساب نسب التطابق", detail=f"top={report.top_percentage:.1f}%")
            reply = (
                "لم أجد اسم المنحة في قاعدة البيانات الموثقة. إليك أقرب الخيارات المحسوبة لملفك:\n\n"
                if language == "ar"
                else "I could not find that grant in the verified database. Here are the closest matches for your profile:\n\n"
            )
            reply += self._match_reply(report, language)
            return reply, cards
        reply = self._missing_reply(profile, missing, language)
        reply += (
            "\n\nأو اطلب مني عرض المنح الموثقة المتاحة حالياً."
            if language == "ar"
            else "\n\nOr ask me to list the currently verified grants."
        )
        return reply, []

    # ------------------------------------------------------------------
    # Routing / helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _contains(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _has_question_word(text: str) -> bool:
        """Match question words without false positives inside longer words.

        Arabic stemming makes plain substring search unsafe: the question word
        "ما" appears inside "ماجستير" (master).  Short words therefore require
        token boundaries while longer words may match as substrings.
        """
        for word in _QUESTION_WORDS:
            if len(word) <= 3:
                if re.search(
                    r"(?<![\w\u0600-\u06FF])" + re.escape(word) + r"(?![\w\u0600-\u06FF])",
                    text,
                ):
                    return True
            elif word in text:
                return True
        return False

    def _route(self, message: str, profile: StudentProfile, missing: list[str]) -> Intent:
        text = normalize(message)
        if self._contains(text, _EXTERNAL_KEYWORDS) and (
            "منح" in text or "scholarship" in text or "بحث" in text or "search" in text
        ):
            return Intent.EXTERNAL_SEARCH
        if self._contains(text, _DOCUMENT_KEYWORDS):
            return Intent.DOCUMENTS
        if self._contains(text, _PLAN_KEYWORDS):
            return Intent.PLAN
        if self._contains(text, _ADVISOR_KEYWORDS):
            return Intent.ADVISOR
        if self._contains(text, _MATCH_KEYWORDS):
            return Intent.MATCH
        if self._contains(text, _TOP_KEYWORDS):
            return Intent.TOP_GRANTS
        if re.match(
            r"^(hi|hello|hey|salam|salaam|thanks|thank you|ok|okay|سلام|مرحبا|مرحباً|اهلا|أهلا|هاي|صباح الخير|مساء الخير|شكرا|شكراً|تمام|حسنا)[!. ]*$",
            text,
        ):
            return Intent.GREET
        if any(
            word in text
            for word in ("ملفي", "معدلي", "تخصصي", "بياناتي", "my profile", "my gpa")
        ):
            return Intent.PROFILE_UPDATE
        if (
            self._looks_like_grant_question(text)
            or (
                ("منحه" in text or "منحة" in text or "scholarship" in text or "deadline" in text or "موعد" in text)
                and self._has_question_word(text)
            )
            or parse_date_from_text(text) is not None
        ):
            return Intent.SCHOLARSHIP_QUESTION
        if self._has_question_word(text) and not is_domain_text(text):
            return Intent.OFF_TOPIC
        if missing:
            return Intent.PROFILE_UPDATE
        return Intent.MATCH

    @staticmethod
    def _looks_like_grant_question(text: str) -> bool:
        if not Orchestrator._has_question_word(text):
            return False
        return any(
            alias in text for aliases in _GRANT_ALIASES for alias in aliases
        )

    def _resolve_grant(self, message: str, cards: list[GrantCard]) -> GrantCard | None:
        return _resolve_grant_shared(message, cards, self.repository)

    @staticmethod
    def _document_type(message: str) -> DocumentType:
        return detect_document_type(message)

    @staticmethod
    def _merge_override(profile: StudentProfile, override: StudentProfile) -> StudentProfile:
        patch: dict[str, Any] = {}
        for key, value in override.model_dump().items():
            if key == "student_id":
                continue
            if value in (None, "", []):
                continue
            patch[key] = value
        return merge_profile(profile, patch)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    # ------------------------------------------------------------------
    # Reply composers
    # ------------------------------------------------------------------
    def _refusal(self, category: GuardrailCategory, language: str) -> str:
        return refusal_for(category, language)

    def _missing_reply(
        self, profile: StudentProfile, missing: list[str], language: str
    ) -> str:
        questions = [_MISSING_QUESTIONS[field][0 if language == "ar" else 1] for field in missing[:3]]
        if language == "ar":
            head = (
                f"اكتمل ملفك بنسبة {profile.completeness():.0f}%. لأقدم لك مطابقة دقيقة أحتاج:"
            )
            body = "\n".join(f"- {question}" for question in questions)
            tail = "يمكنك كتابة كل الإجابات في رسالة واحدة."
        else:
            head = (
                f"Your profile is {profile.completeness():.0f}% complete. To rank grants accurately I need:"
            )
            body = "\n".join(f"- {question}" for question in questions)
            tail = "You can answer all of them in one message."
        return f"{head}\n{body}\n\n{tail}"

    def _profile_reply(
        self, profile: StudentProfile, missing: list[str], language: str
    ) -> str:
        if missing:
            return self._missing_reply(profile, missing, language)
        if language == "ar":
            return (
                f"ملفك مكتمل بنسبة 100%: {profile.name} — {profile.major} — "
                f"معدل {profile.gpa:.2f}/4.00 — المرحلة المستهدفة "
                f"{profile.target_degree.value if profile.target_degree else ''}. "
                "يمكنني الآن حساب نسب التطابق أو بناء خطة تقديم أو صياغة وثائق."
            )
        return (
            f"Your profile is complete: {profile.name} — {profile.major} — "
            f"GPA {profile.gpa:.2f}/4.00 — target "
            f"{profile.target_degree.value if profile.target_degree else ''}. "
            "I can now rank grants, build an application plan, or draft documents."
        )

    def _off_topic_reply(self, language: str) -> str:
        if language == "ar":
            return (
                "أنا مستشار منح دراسية فقط، لذلك سأبقى ضمن هذا النطاق. "
                "يمكنك سؤالي عن ملفك الأكاديمي، شروط المنح، نسب التطابق، خطة التقديم، أو الوثائق."
            )
        return (
            "I am a scholarship advisor and stay within that scope. "
            "Ask me about your academic profile, grant requirements, match scores, application plans, or documents."
        )

    def _top_grants_reply(
        self, cards: list[GrantCard], language: str, *, limit: int | None = None
    ) -> str:
        if not cards:
            return (
                "لا توجد منح منشورة في قاعدة البيانات حالياً."
                if language == "ar"
                else "There are no published grants in the database right now."
            )
        cards = cards[: (limit or settings.top_grants_limit)]
        header = (
            "أفضل المنح المتاحة حالياً (منشورة وموثقة):"
            if language == "ar"
            else "Top available grants (published and verified):"
        )
        lines = [header, ""]
        for index, card in enumerate(cards, 1):
            deadline = card.deadline or ("غير معلن" if language == "ar" else "not announced")
            funding = (card.funding or "")[:110]
            lines.append(f"{index}. {card.title}")
            lines.append(
                f"   {'الموعد' if language == 'ar' else 'Deadline'}: {deadline} | "
                f"{'الدولة' if language == 'ar' else 'Country'}: {card.country or '—'} | "
                f"{'التمويل' if language == 'ar' else 'Funding'}: {funding or '—'}"
            )
            lines.append(f"   {'الرابط الرسمي' if language == 'ar' else 'Official link'}: {card.official_url}")
        lines.append("")
        lines.append(
            "اذكر اسم أي منحة لتفاصيلها، أو اطلب \"طابق ملفي\" لحساب نسب التطابق."
            if language == "ar"
            else "Name any grant for details, or ask me to \"match my profile\"."
        )
        return "\n".join(lines)

    def _match_reply(self, report: MatchReport, language: str) -> str:
        if not report.results:
            return (
                "لا توجد منح موثقة لمطابقتها حالياً."
                if language == "ar"
                else "There are no verified grants to match against right now."
            )
        header = (
            "نسب التطابق مع المنح الموثقة (مرتبة تنازلياً):"
            if language == "ar"
            else "Match scores against verified grants (descending):"
        )
        lines = [header, ""]
        for index, match in enumerate(report.results[:5], 1):
            tier = _TIER_AR.get(match.tier.value, match.tier.value) if language == "ar" else match.tier.value
            reason = match.reasons[0] if match.reasons else ""
            lines.append(f"{index}. {match.title} — {match.match_percentage:.1f}% ({tier})")
            if reason:
                lines.append(f"   {reason}")
            if match.disqualifiers:
                lines.append(
                    f"   {'عوائق' if language == 'ar' else 'Blockers'}: {match.disqualifiers[0]}"
                )
        lines.append("")
        if report.trigger_external_search:
            lines.append(
                "أعلى نسبة داخلية أقل من 40%؛ أنصح بتفعيل البحث الخارجي (يتطلب مفتاح Tavily/Serper)."
                if language == "ar"
                else "The best internal match is below 40%; external search is recommended (requires a Tavily/Serper key)."
            )
        else:
            lines.append(
                "النسب محسوبة من شروط المنحة المنشورة وملفك الحالي، وليست وعداً بالقبول."
                if language == "ar"
                else "Scores are computed from published requirements and your current profile; they are not a guarantee of admission."
            )
        return "\n".join(lines)

    def _plan_reply(
        self, items: list[dict], report: MatchReport, language: str
    ) -> str:
        if language == "ar":
            head = plan_summary(items)
            lines = [head, ""]
            for item in items:
                lines.append(
                    f"{item['priority_rank']}. {item['grant_title']} — "
                    f"{item['match_percentage']:.1f}% — {item['deadline']}"
                )
                for step in item["steps"][:3]:
                    lines.append(f"   - {step['title']}: {step['description'][:110]}")
                lines.append(f"   - الوثائق المطلوبة: {len(item['documents'])} وثيقة")
                lines.append("")
            lines.append("اطلب \"اكتب خطاب الدافع\" لصياغة الوثيقة للمنحة الأولى.")
            return "\n".join(lines)

        lines = [
            f"Your plan covers {len(items)} grants, ordered by deadline and match score.",
            "",
        ]
        for item in items:
            lines.append(
                f"{item['priority_rank']}. {item['grant_title']} — "
                f"{item['match_percentage']:.1f}% — {item['deadline']}"
            )
            for step in item["steps"][:3]:
                lines.append(f"   - {step['title']}: {step['description'][:110]}")
            lines.append("")
        lines.append('Ask me to "write the motivation letter" for the first grant.')
        return "\n".join(lines)

    def _advisor_reply(
        self, report: AdvisorReport, missing: list[str], language: str
    ) -> str:
        if language == "ar":
            lines = ["تقرير تقوية الملف:", ""]
            if missing:
                lines.append(
                    "الملف غير مكتمل؛ النتائج التالية أولية. الحقول الناقصة: "
                    + "، ".join(missing)
                )
                lines.append("")
            lines.append("نقاط القوة:")
            lines += [f"- {item}" for item in report.strengths] or ["- لا توجد بعد."]
            lines.append("")
            lines.append("نقاط الضعف:")
            lines += [f"- {item}" for item in report.weaknesses] or ["- لا توجد."]
            if report.competitive_gaps:
                lines.append("")
                lines.append("الفجوات التنافسية:")
                lines += [f"- {item}" for item in report.competitive_gaps]
            lines.append("")
            lines.append("خطة التحسين:")
            for action in report.actions[:6]:
                lines.append(
                    f"- [{action.priority}] {action.title} — {action.how} ({action.timeframe})"
                )
            lines.append("")
            lines.append(report.overall_competitiveness)
            return "\n".join(lines)

        lines = ["Profile strengthening report:", ""]
        lines.append("Strengths:")
        lines += [f"- {item}" for item in report.strengths] or ["- None yet."]
        lines.append("")
        lines.append("Weaknesses:")
        lines += [f"- {item}" for item in report.weaknesses] or ["- None."]
        if report.competitive_gaps:
            lines.append("")
            lines.append("Competitive gaps:")
            lines += [f"- {item}" for item in report.competitive_gaps]
        lines.append("")
        lines.append("Action plan:")
        for action in report.actions[:6]:
            lines.append(f"- [{action.priority}] {action.title} — {action.how} ({action.timeframe})")
        lines.append("")
        lines.append(report.overall_competitiveness)
        return "\n".join(lines)

    def _documents_reply(self, result: DocumentResult, language: str) -> str:
        head = (
            "تم توليد الوثيقة عبر حلقة: مسودة ثم نقد ذاتي ثم مراجعة نهائية.\n"
            if language == "ar"
            else "The document was generated through draft, self-critique, and final revision.\n"
        )
        if result.issues:
            head += (
                "\nملاحظات التدقيق: "
                + "؛ ".join(issue.message for issue in result.issues[:3])
                + "\n"
                if language == "ar"
                else "\nReview notes: "
                + "; ".join(issue.message for issue in result.issues[:3])
                + "\n"
            )
        return f"{head}\n{result.content}"

    def _deadline_details_reply(self, verdict: str, card: GrantCard, language: str) -> str:
        if language == "ar":
            lines = [
                verdict,
                "",
                f"### {card.title}",
                f"- الجهة: {card.provider or 'غير محددة'}",
                f"- الدولة: {card.country or 'غير محددة'}",
                f"- الرابط الرسمي: {card.official_url}",
            ]
        else:
            lines = [
                verdict,
                "",
                f"### {card.title}",
                f"- Provider: {card.provider or 'Not specified'}",
                f"- Country: {card.country or 'Not specified'}",
                f"- Official link: {card.official_url}",
            ]
        return "\n".join(lines)

    def _grant_details_reply(self, card: GrantCard, language: str) -> str:
        accepts = {
            "yes": "نعم" if language == "ar" else "Yes",
            "no": "لا" if language == "ar" else "No",
            "unclear": "غير مؤكد" if language == "ar" else "Unclear",
        }[card.accepts_yemeni]
        if language == "ar":
            lines = [
                f"### {card.title}",
                f"- الجهة: {card.provider or 'غير محددة'}",
                f"- الدولة: {card.country or 'غير محددة'}",
                f"- الموعد النهائي: {card.deadline or 'غير معلن'}",
                f"- التمويل: {card.funding or 'غير منشور'}",
                f"- أهلية اليمنيين: {accepts}",
                f"- المجالات: {'، '.join(card.fields_of_study) or 'غير محددة'}",
            ]
            if card.eligibility:
                lines.append("- الشروط:")
                lines += [f"  - {item}" for item in card.eligibility[:6]]
            if card.required_documents:
                lines.append("- الوثائق المطلوبة:")
                lines += [f"  - {item}" for item in card.required_documents[:8]]
            if card.how_to_apply:
                lines.append(f"- طريقة التقديم: {card.how_to_apply}")
            lines.append(f"- الرابط الرسمي: {card.official_url}")
            if card.evidence:
                lines.append(
                    "- المصدر: اقتباسات حرفية موثقة من الصفحة الرسمية (حقل evidence)."
                )
            return "\n".join(lines)

        lines = [
            f"### {card.title}",
            f"- Provider: {card.provider or 'Not specified'}",
            f"- Country: {card.country or 'Not specified'}",
            f"- Deadline: {card.deadline or 'Not announced'}",
            f"- Funding: {card.funding or 'Not published'}",
            f"- Yemeni eligibility: {accepts}",
            f"- Fields: {', '.join(card.fields_of_study) or 'Not specified'}",
        ]
        if card.eligibility:
            lines.append("- Eligibility:")
            lines += [f"  - {item}" for item in card.eligibility[:6]]
        if card.required_documents:
            lines.append("- Required documents:")
            lines += [f"  - {item}" for item in card.required_documents[:8]]
        if card.how_to_apply:
            lines.append(f"- How to apply: {card.how_to_apply}")
        lines.append(f"- Official link: {card.official_url}")
        return "\n".join(lines)
