"""Input/output guardrails and anti-hallucination checks.

The rules are deterministic so they can be unit-tested and so the agent never
depends on a second model to police the first one.  They cover:

* prompt injection and instruction-leak attempts,
* requests to reveal secrets or fabricate credentials,
* requests for another student's private data,
* unsafe content,
* off-topic redirection (the model stays inside the scholarship domain),
* output redaction of secrets, unverified URLs and prompt leakage,
* grounding of numeric claims in generated documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .schemas import (
    DocumentIssue,
    GrantCard,
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvent,
    StudentProfile,
)

_TRUSTED_HOSTS = {
    "turkiyeburslari.gov.tr",
    "chevening.org",
    "daad.de",
    "stipendiumhungaricum.hu",
    "campusfrance.org",
    "isdb.org",
    "education-in-russia.com",
    "campuschina.org",
    "foreign.fulbrightonline.org",
    "mae.ro",
}
_TRUSTED_SUFFIXES = (
    ".gov",
    ".gov.tr",
    ".gov.uk",
    ".edu",
    ".edu.tr",
    ".ac.uk",
    ".ac.jp",
)

_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-or-[0-9A-Za-z_\-]{10,}"),
    re.compile(r"sk-[0-9A-Za-z_\-]{20,}"),
    re.compile(r"tvly-[0-9A-Za-z_\-]{10,}"),
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),
    re.compile(r"Bearer\s+[0-9A-Za-z._\-]{12,}"),
    re.compile(r"(?i)(x-api-key|api[_-]?key)\s*[:=]\s*[^\s,;]{12,}"),
)

_PROMPT_LEAK_MARKERS = (
    "system prompt",
    "you are the orchestrator",
    "required_profile_fields",
    "تعليمات النظام",
    "برومبت النظام",
    "_secret_patterns",
)

_DOMAIN_WORDS = (
    "منح", "منحة", "دراس", "جامع", "تخصص", "معدل", "لغه", "لغات", "انجليز",
    "ايلتس", "توفل", "تقديم", "ابحث", "بحث", "خطه", "سيره", "دافع", "توصي",
    "ماجستير", "بكالوريوس", "دكتوراه", "دوله", "دول", "خارج", "سفر", "ترشيح",
    "قبول", "ملف", "فرصه", "زماله", "مقابله", "وثائق", "شهاده", "ترجمه",
    "scholarship", "grant", "study", "university", "master", "phd", "bachelor",
    "gpa", "ielts", "toefl", "apply", "application", "admission", "cv",
    "resume", "motivation", "recommendation", "abroad", "country", "profile",
    "major", "degree", "funding", "tuition", "deadline", "eligibility",
)
_GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|salam|salaam|thanks|thank you|ok|okay|"
    r"سلام|مرحبا|مرحباً|اهلا|أهلا|هاي|صباح الخير|مساء الخير|شكرا|شكراً|تمام|حسنا)[!. ]*$"
)

_INJECTION_PATTERNS = (
    re.compile(r"(ignore|disregard|forget).{0,30}(previous|prior|above|system|all).{0,25}(instruction|prompt|rule)"),
    re.compile(r"(reveal|show|print|repeat|leak).{0,30}(system|hidden|initial|developer).{0,20}(prompt|instruction|message)"),
    re.compile(r"(system prompt|developer mode|jailbreak|dan mode|do anything now)"),
    re.compile(r"(تجاهل|تخطي|الغ|الغي|تجاوز).{0,25}(التعليمات|الاوامر|القواعد|القيود)"),
    re.compile(r"(اكشف|اعرض|اطبع|كرر|سرب).{0,25}(تعليمات|برومبت|اعدادات|رساله).{0,25}(النظام|الداخليه|المخفيه)"),
    re.compile(r"(انت الان|تصرف ك|تظاهر بانك|تخيل انك).{0,40}(بدون قيود|مخترق|هاكر|غير مقيد)"),
)

_SECRET_REQUEST_PATTERNS = (
    re.compile(r"(reveal|show|give|tell|print|send|what is).{0,30}(api|secret|private|access).{0,15}(key|token|password)"),
    re.compile(r"(اعطني|اكشف|اعرض|اخبرني|هات|ارسل|اطبع).{0,25}(مفتاح|كلمه المرور|الرمز السري|توكن|باسورد)"),
    re.compile(r"(مفتاح|كلمه المرور|الرمز السري|توكن|باسورد).{0,20}(الapi|السري|الخاص|الوصول|الخادم)"),
)
_SECRET_INFO_ALLOW = re.compile(
    r"(how to get|where to get|where can i|from where|كيف احصل|من اين|اين اجد|وين احصل|كيف اضيف|طريقه اضافه)"
)

_FABRICATION_PATTERNS = (
    re.compile(r"(fake|forge|fabricate|invent|make up|doctor|exaggerate).{0,30}(certificate|transcript|degree|gpa|grade|experience|reference|recommendation|award)"),
    re.compile(r"(اختلق|اخترع|زور|اصنع|زيد|غير|بالغ).{0,30}(شهاده|سجل درجات|درجه|معدل|خبره|توصيه مزوره|وسام|جائزه)"),
    re.compile(r"(غير حقيقيه|مزور|وهميه).{0,20}(خبره|شهاده|درجه|معدل)"),
)

_PRIVACY_PATTERNS = (
    re.compile(r"(another|other|different).{0,20}(student|applicant|user).{0,20}(data|profile|email|phone|information)"),
    re.compile(r"(بيانات|ملف|معلومات).{0,20}(طالب|متقدم|مستخدم).{0,10}(اخر|آخر|اخرى|اخري|غيري)"),
    re.compile(r"(بيانات|معلومات).{0,15}(شخص|شخص اخر|احد اخر)"),
)

_UNSAFE_PATTERNS = (
    re.compile(r"(suicide|kill myself|self harm|end my life)"),
    re.compile(r"(انتحار|اقتل نفسي|ايذاء النفس|انهي حياتي)"),
    re.compile(r"(hack|hacking|exploit|malware|ddos|phishing)"),
    re.compile(r"(اختراق|هاكر|فيروس|تصيد|استغلال ثغره)"),
    re.compile(r"(سلاح|مخدرات|قتل|عنف|weapon|drugs|murder)"),
)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\]]+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")


@dataclass
class InputDecision:
    allowed: bool
    action: GuardrailAction
    category: GuardrailCategory
    reason: str
    text: str = ""
    event: GuardrailEvent | None = None


@dataclass
class OutputReport:
    text: str
    passed: bool
    issues: list[DocumentIssue] = field(default_factory=list)
    events: list[GuardrailEvent] = field(default_factory=list)


def normalize(text: str) -> str:
    lowered = text.lower()
    lowered = _DIACRITICS.sub("", lowered)
    for source, target in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ؤ", "و"), ("ئ", "ي")):
        lowered = lowered.replace(source, target)
    return re.sub(r"\s+", " ", lowered).strip()


def is_domain_text(text: str) -> bool:
    normalized = normalize(text)
    return any(word in normalized for word in _DOMAIN_WORDS)


def _clean_text(text: str) -> str:
    cleaned = _CONTROL_CHARS.sub(" ", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _event(
    stage: str, category: GuardrailCategory, action: GuardrailAction, reason: str
) -> GuardrailEvent:
    return GuardrailEvent(
        stage=stage,  # type: ignore[arg-type]
        category=category,
        action=action,
        reason=reason,
    )


def _decision(
    category: GuardrailCategory,
    action: GuardrailAction,
    reason: str,
    text: str,
) -> InputDecision:
    return InputDecision(
        allowed=action != GuardrailAction.REFUSE,
        action=action,
        category=category,
        reason=reason,
        text=text,
        event=_event("input", category, action, reason),
    )


def check_input(text: str, *, max_chars: int = 4000) -> InputDecision:
    cleaned = _clean_text(text)
    if not cleaned:
        return _decision(GuardrailCategory.EMPTY, GuardrailAction.REFUSE, "الرسالة فارغة.", cleaned)
    if len(cleaned) > max_chars:
        return _decision(
            GuardrailCategory.TOO_LONG,
            GuardrailAction.REFUSE,
            f"الرسالة أطول من الحد المسموح ({max_chars} حرفاً).",
            cleaned,
        )

    normalized = normalize(cleaned)

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            return _decision(
                GuardrailCategory.PROMPT_INJECTION,
                GuardrailAction.REFUSE,
                "لا أستطيع تنفيذ تعليمات تتجاوز نطاق المنصة أو تكشف إعداداتها الداخلية.",
                cleaned,
            )

    for pattern in _SECRET_REQUEST_PATTERNS:
        if pattern.search(normalized) and not _SECRET_INFO_ALLOW.search(normalized):
            return _decision(
                GuardrailCategory.SECRET_REQUEST,
                GuardrailAction.REFUSE,
                "لا يمكنني عرض مفاتيح API أو أسرار النظام. يمكن للمشرف ضبط المفاتيح من لوحة الإدارة.",
                cleaned,
            )

    for pattern in _FABRICATION_PATTERNS:
        if pattern.search(normalized):
            return _decision(
                GuardrailCategory.FABRICATION_REQUEST,
                GuardrailAction.REFUSE,
                "لا يمكنني اختلاق شهادات أو خبرات أو تعديل بياناتك. يمكنني مساعدتك في إبراز إنجازاتك الحقيقية فقط.",
                cleaned,
            )

    for pattern in _PRIVACY_PATTERNS:
        if pattern.search(normalized):
            return _decision(
                GuardrailCategory.PRIVACY_REQUEST,
                GuardrailAction.REFUSE,
                "لا أستطيع الوصول إلى بيانات طالب آخر أو مشاركتها. يمكنني مساعدتك بملفك أنت فقط.",
                cleaned,
            )

    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(normalized):
            return _decision(
                GuardrailCategory.UNSAFE,
                GuardrailAction.REFUSE,
                "هذا الطلب خارج نطاق خدمة المنح الدراسية ولا يمكنني المساعدة فيه. إن كنت تمر بضغط نفسي فتحدث إلى شخص تثق به أو مختص.",
                cleaned,
            )

    has_domain = any(word in normalized for word in _DOMAIN_WORDS)
    is_greeting = bool(_GREETING_PATTERN.match(normalized))
    if not has_domain and not is_greeting and len(normalized) > 20:
        return _decision(
            GuardrailCategory.OFF_TOPIC,
            GuardrailAction.REDIRECT,
            "أنا مستشار منح دراسية فقط؛ سأعيد توجيه النقاش إلى ملفك الأكاديمي والمنح.",
            cleaned,
        )

    return _decision(GuardrailCategory.OK, GuardrailAction.ALLOW, "", cleaned)


def _host_allowed(host: str, known_hosts: set[str]) -> bool:
    host = (host or "").lower()
    if not host:
        return False
    if host in known_hosts or any(host == h or host.endswith("." + h) for h in known_hosts):
        return True
    if host in _TRUSTED_HOSTS:
        return True
    return any(host.endswith(suffix) for suffix in _TRUSTED_SUFFIXES)


def _strip_trailing_punctuation(url: str) -> str:
    return url.rstrip(".,;:!؟?،")


def check_output(
    text: str,
    *,
    known_grants: list[GrantCard] | None = None,
    allow_urls: bool = True,
) -> OutputReport:
    issues: list[DocumentIssue] = []
    events: list[GuardrailEvent] = []
    sanitized = text or ""

    # 1. Never echo a credential.
    for pattern in _SECRET_PATTERNS:
        if pattern.search(sanitized):
            sanitized = pattern.sub("[محجوب]", sanitized)
            issues.append(
                DocumentIssue(
                    code="secret_leak",
                    severity="critical",
                    message="تم حجب نص يشبه مفتاحاً سرياً من المخرجات.",
                )
            )

    # 2. Never echo internal instructions.
    lowered = normalize(sanitized)
    for marker in _PROMPT_LEAK_MARKERS:
        if normalize(marker) in lowered:
            issues.append(
                DocumentIssue(
                    code="prompt_leak",
                    severity="critical",
                    message="تم حجب إشارة إلى تعليمات النظام الداخلية.",
                )
            )
            sanitized = re.sub(
                r"(?i).{0,80}(system prompt|required_profile_fields|you are the orchestrator).{0,80}",
                "[محجوب]",
                sanitized,
            )

    # 3. Only official or trusted URLs may survive; documents allow none.
    known_hosts = {card.host for card in (known_grants or []) if card.host}
    removed: list[str] = []
    for raw_url in list(_URL_PATTERN.findall(sanitized)):
        url = _strip_trailing_punctuation(raw_url)
        host = (urlparse(url).hostname or "").lower()
        if not allow_urls:
            sanitized = sanitized.replace(raw_url, "[رابط محذوف]")
            continue
        if not _host_allowed(host, known_hosts):
            removed.append(url)
            sanitized = sanitized.replace(raw_url, "[رابط غير موثق تم حذفه]")
    if removed:
        issues.append(
            DocumentIssue(
                code="unverified_url",
                severity="warning",
                message="تم حذف روابط غير رسمية: " + ", ".join(sorted(set(removed))[:3]),
            )
        )
    if not allow_urls and _URL_PATTERN.search(text or ""):
        issues.append(
            DocumentIssue(
                code="url_in_document",
                severity="warning",
                message="الوثائق الرسمية لا تتضمن روابط؛ تم حذفها.",
            )
        )

    for issue in issues:
        category = GuardrailCategory(issue.code)
        events.append(_event("output", category, GuardrailAction.REDIRECT, issue.message))

    passed = not any(issue.severity == "critical" for issue in issues)
    return OutputReport(text=sanitized.strip(), passed=passed, issues=issues, events=events)


_GPA_PATTERN = re.compile(
    r"(?:gpa|معدل(?:ي|ك|ه)?(?:\s+التراكمي)?)[^\d]{0,18}(\d(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)
_IELTS_PATTERN = re.compile(r"ielts[^\d]{0,20}(\d(?:[.,]\d)?)", re.IGNORECASE)
_TOEFL_PATTERN = re.compile(r"toefl[^\d]{0,20}(\d{2,3})", re.IGNORECASE)
_PUBLICATIONS_PATTERN = re.compile(
    r"(?:published|publications?|نشرت|عدد الأوراق|اوراق بحثيه)[^\d]{0,15}(\d{1,2})",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PLACEHOLDER_PATTERN = re.compile(r"\[\[[^\]]{1,80}\]\]")


def _as_float(value: str) -> float:
    return float(value.replace(",", "."))


def check_grounding(
    text: str,
    profile: StudentProfile,
    grant: GrantCard | None = None,
) -> list[DocumentIssue]:
    """Detect numeric/identity claims that are not present in the profile."""
    issues: list[DocumentIssue] = []

    for match in _GPA_PATTERN.finditer(text):
        value = _as_float(match.group(1))
        if profile.gpa is None or abs(value - float(profile.gpa)) > 0.011:
            issues.append(
                DocumentIssue(
                    code="ungrounded_claim",
                    severity="critical",
                    message=f"المعدل المذكور ({value}) لا يطابق ملف الطالب.",
                )
            )

    language_tests = {
        (item.test_name or "").upper(): item.score
        for item in profile.languages
        if item.test_name and item.score is not None
    }
    for pattern, label in ((_IELTS_PATTERN, "IELTS"), (_TOEFL_PATTERN, "TOEFL")):
        for match in pattern.finditer(text):
            value = _as_float(match.group(1))
            expected = language_tests.get(label)
            if expected is None or abs(value - float(expected)) > 0.011:
                issues.append(
                    DocumentIssue(
                        code="ungrounded_claim",
                        severity="critical",
                        message=f"درجة {label} المذكورة ({value:g}) غير موجودة في ملف الطالب.",
                    )
                )

    for match in _PUBLICATIONS_PATTERN.finditer(text):
        value = int(match.group(1))
        if value != profile.publications:
            issues.append(
                DocumentIssue(
                    code="ungrounded_claim",
                    severity="critical",
                    message=f"عدد الأوراق المذكور ({value}) لا يطابق ملف الطالب ({profile.publications}).",
                )
            )

    for email in _EMAIL_PATTERN.findall(text):
        if profile.email and email.lower() != profile.email.lower():
            issues.append(
                DocumentIssue(
                    code="ungrounded_claim",
                    severity="critical",
                    message="البريد المذكور لا يطابق بريد ملف الطالب.",
                )
            )

    if grant and _URL_PATTERN.search(text):
        issues.append(
            DocumentIssue(
                code="url_in_document",
                severity="warning",
                message="الوثيقة تحتوي رابطاً؛ يُفضّل إزالته من النص الرسمي.",
            )
        )

    if _PLACEHOLDER_PATTERN.search(text):
        issues.append(
            DocumentIssue(
                code="placeholder",
                severity="warning",
                message="الوثيقة تحتوي حقولاً ناقصة بين [[...]] تحتاج تعبئة يدوية.",
            )
        )

    # De-duplicate identical messages while keeping order.
    seen: set[str] = set()
    unique: list[DocumentIssue] = []
    for issue in issues:
        key = f"{issue.code}:{issue.message}"
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique
