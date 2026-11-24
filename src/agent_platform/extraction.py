"""Deterministic profile extraction from Arabic/English chat messages.

This is intentionally rule-based: it never invents a value, it only copies
what the student wrote.  The orchestrator uses it to fill the live profile and
to detect which required fields are still missing (proactive routing).
"""

from __future__ import annotations

import re

from .schemas import DegreeLevel, LanguageLevel, LanguageScore, StudentProfile

_AR_RANGE = re.compile(r"[\u0600-\u06FF]")

_DEGREE_WORDS = {
    "bachelor": DegreeLevel.BACHELOR,
    "undergraduate": DegreeLevel.BACHELOR,
    "بكالوريوس": DegreeLevel.BACHELOR,
    "master": DegreeLevel.MASTER,
    "masters": DegreeLevel.MASTER,
    "postgraduate": DegreeLevel.MASTER,
    "ماجستير": DegreeLevel.MASTER,
    "phd": DegreeLevel.PHD,
    "doctorate": DegreeLevel.PHD,
    "doctoral": DegreeLevel.PHD,
    "دكتوراه": DegreeLevel.PHD,
}

_TARGET_CUES = (
    "أريد", "اريد", "أرغب", "ارغب", "أخطط", "اخطط", "هدفي", "أقدم على",
    "اقدم على", "apply for", "applying for", "target", "want to study",
    "أسعى", "اسعى",
)
_CURRENT_CUES = (
    "خريج", "خريجة", "طالب", "طالبة", "حاصل على", "حاصلة على", "أنهيت",
    "انهيت", "بكالوريوس في", "ماجستير في", "i hold", "i graduated",
    "graduated", "my degree", "i studied", "i am studying", "أدرس", "ادرس",
)

_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "Artificial Intelligence": (
        "artificial intelligence", "ai", "الذكاء الاصطناعي", "ذكاء اصطناعي",
    ),
    "Machine Learning": ("machine learning", "تعلم الآلة", "تعلم الالة", "ml"),
    "NLP": ("nlp", "natural language processing", "معالجة اللغة", "اللغة الطبيعية"),
    "Data Science": ("data science", "علوم البيانات", "تحليل البيانات"),
    "Cybersecurity": ("cybersecurity", "الأمن السيبراني", "الامن السيبراني", "أمن سيبراني"),
    "Engineering": ("engineering", "هندسة", "الهندسة"),
    "Medicine": ("medicine", "medical", "طب", "الطب"),
    "Public Health": ("public health", "الصحة العامة", "صحة عامة"),
    "Economics": ("economics", "اقتصاد", "الاقتصاد"),
    "Public Policy": ("public policy", "السياسات العامة", "سياسات عامة", "حكم رشيد"),
    "International Relations": ("international relations", "العلاقات الدولية"),
    "Renewable Energy": ("renewable energy", "الطاقة المتجددة", "طاقة متجددة"),
    "Water Resources": ("water resources", "الموارد المائية", "المياه"),
    "Agriculture": ("agriculture", "agricultural", "زراعة", "الزراعة"),
    "Law": ("law", "قانون", "القانون"),
    "Computer Science": ("computer science", "علوم الحاسوب", "الحاسوب"),
}

_COUNTRY_MAP: dict[str, str] = {
    "ألمانيا": "Germany", "المانيا": "Germany", "germany": "Germany",
    "تركيا": "Turkey", "turkey": "Turkey", "turkiye": "Turkey",
    "بريطانيا": "United Kingdom", "المملكة المتحدة": "United Kingdom",
    "uk": "United Kingdom", "united kingdom": "United Kingdom", "england": "United Kingdom",
    "أمريكا": "United States", "امريكا": "United States", "الولايات المتحدة": "United States",
    "usa": "United States", "united states": "United States",
    "الصين": "China", "china": "China",
    "روسيا": "Russia", "russia": "Russia",
    "فرنسا": "France", "france": "France",
    "المجر": "Hungary", "hungary": "Hungary",
    "رومانيا": "Romania", "romania": "Romania",
    "كندا": "Canada", "canada": "Canada",
    "اليابان": "Japan", "japan": "Japan",
    "هولندا": "Netherlands", "netherlands": "Netherlands",
}

_NATIONALITY_MAP: dict[str, str] = {
    "يمني": "Yemeni", "يمنية": "Yemeni", "yemeni": "Yemeni",
    "سعودي": "Saudi", "سعودية": "Saudi", "saudi": "Saudi",
    "مصري": "Egyptian", "مصرية": "Egyptian", "egyptian": "Egyptian",
    "سوداني": "Sudanese", "sudanese": "Sudanese",
    "أردني": "Jordanian", "اردني": "Jordanian", "jordanian": "Jordanian",
    "سوري": "Syrian", "syrian": "Syrian",
}

_LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]


def detect_language(text: str) -> str:
    arabic = len(_AR_RANGE.findall(text or ""))
    letters = sum(1 for char in text or "" if char.isalpha())
    if letters == 0:
        return "ar"
    return "ar" if arabic / letters >= 0.25 else "en"


def _level_from_ielts(score: float) -> LanguageLevel:
    if score >= 8.0:
        return LanguageLevel.C2
    if score >= 7.0:
        return LanguageLevel.C1
    if score >= 5.5:
        return LanguageLevel.B2
    if score >= 4.5:
        return LanguageLevel.B1
    return LanguageLevel.A2


def _level_from_toefl(score: float) -> LanguageLevel:
    if score >= 110:
        return LanguageLevel.C1
    if score >= 80:
        return LanguageLevel.B2
    if score >= 60:
        return LanguageLevel.B1
    return LanguageLevel.A2


def _parse_level(raw: str) -> LanguageLevel | None:
    value = raw.upper().replace(" ", "")
    for level in LanguageLevel:
        if value == level.value.upper():
            return level
    return None


def _clean_capture(value: str, stops: tuple[str, ...] = ()) -> str:
    value = re.split(r"[،,.;\n!؟?]", value, maxsplit=1)[0]
    for stop in stops:
        index = value.find(stop)
        if index > 0:
            value = value[:index]
    return " ".join(value.split())[:120]


def _extract_gpa(text: str) -> float | None:
    patterns = (
        r"(?:gpa|معدلي|معدل(?:ي)?\s+التراكمي|المعدل\s+التراكمي|معدل)\D{0,15}(\d{1,3}(?:[.,]\d{1,2})?)",
        r"(\d(?:[.,]\d{1,2})?)\s*(?:/|من)\s*4",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1).replace(",", "."))
        if value > 4.0:
            if value > 100.0:
                continue
            value = round(value / 25.0, 2)
        if 0.0 <= value <= 4.0:
            return value
    return None


def _extract_degrees(text: str) -> tuple[DegreeLevel | None, DegreeLevel | None]:
    lowered = text.lower()
    target: DegreeLevel | None = None
    current: DegreeLevel | None = None
    for word, degree in _DEGREE_WORDS.items():
        for match in re.finditer(re.escape(word), lowered):
            # Only the immediate prefix decides whether this mention is the
            # current degree or the target degree; a wide window would let
            # "graduate with a bachelor's and want a master's" confuse both.
            prefix = lowered[max(0, match.start() - 22) : match.start()]
            if any(cue in prefix for cue in _TARGET_CUES):
                target = target or degree
            elif any(cue in prefix for cue in _CURRENT_CUES):
                current = current or degree
            else:
                target = target or degree
    return current, target


def _extract_languages(text: str) -> list[LanguageScore]:
    lowered = text.lower()
    found: dict[str, LanguageScore] = {}

    ielts = re.search(r"ielts\D{0,20}(\d(?:[.,]\d)?)", lowered)
    if ielts:
        score = float(ielts.group(1).replace(",", "."))
        found["english"] = LanguageScore(
            language="English", level=_level_from_ielts(score), test_name="IELTS", score=score
        )

    toefl = re.search(r"toefl\D{0,20}(\d{2,3})", lowered)
    if toefl:
        score = float(toefl.group(1))
        found["english"] = LanguageScore(
            language="English", level=_level_from_toefl(score), test_name="TOEFL", score=score
        )

    for match in re.finditer(
        r"(?:english|الإنجليزية|الانجليزية|انجليزي|إنجليزي)\s*[:\-]?\s*(A1|A2|B1|B2|C1|C2)",
        text,
        re.IGNORECASE,
    ):
        level = _parse_level(match.group(1))
        if level:
            found.setdefault("english", LanguageScore(language="English", level=level))

    if re.search(r"(?:لغتي الأم|اللغة الأم|native).{0,20}(العربية|arabic)", lowered):
        found["arabic"] = LanguageScore(language="Arabic", level=LanguageLevel.NATIVE)

    return list(found.values())


def _extract_fields(text: str) -> list[str]:
    lowered = text.lower()
    fields: list[str] = []
    for canonical, aliases in _FIELD_MAP.items():
        for alias in aliases:
            if len(alias) <= 3:
                pattern = (
                    r"(?<![\w\u0600-\u06FF])"
                    + re.escape(alias)
                    + r"(?![\w\u0600-\u06FF])"
                )
            else:
                pattern = re.escape(alias)
            if re.search(pattern, lowered):
                fields.append(canonical)
                break
    return fields


def _extract_countries(text: str) -> list[str]:
    lowered = text.lower()
    countries: list[str] = []
    for alias, canonical in _COUNTRY_MAP.items():
        if alias in lowered and canonical not in countries:
            countries.append(canonical)
    return countries


def _extract_nationality(text: str) -> str | None:
    lowered = text.lower()
    for alias, canonical in _NATIONALITY_MAP.items():
        if re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", lowered):
            return canonical
    return None


def extract_profile_patch(text: str) -> dict:
    patch: dict = {}
    lowered = text.lower()

    name = re.search(
        r"(?:اسمي|اسمى|my name is|i am called)\s+([A-Za-z\u0600-\u06FF][\w\u0600-\u06FF\- ]{1,40})",
        text,
        re.IGNORECASE,
    )
    if name:
        cleaned = _clean_capture(
            name.group(1),
            stops=("،", "وأنا", "وانا", "طالب", "طالبة", " and "),
        )
        if cleaned:
            patch["name"] = cleaned

    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email:
        patch["email"] = email.group(0)

    gpa = _extract_gpa(text)
    if gpa is not None:
        patch["gpa"] = gpa

    current, target = _extract_degrees(text)
    if current:
        patch["degree_level"] = current
    if target:
        patch["target_degree"] = target

    major = re.search(
        r"(?:تخصصي|تخصصى|أدرس|ادرس|خريجة|خريج(?!ة)|بكالوريوس في|ماجستير في|"
        r"major(?:ing)? in|studying|graduated in|my major is)\s*[:：]?\s*"
        r"([A-Za-z\u0600-\u06FF][\w\u0600-\u06FF /&,\-]{1,50})",
        text,
        re.IGNORECASE,
    )
    if major:
        captured = _clean_capture(
            major.group(1),
            stops=(
                "،",
                "وأ",
                "معدلي",
                "بمعدل",
                "وأريد",
                "أريد",
                "وارغب",
                "وأخطط",
                "في جامعة",
                "من جامعة",
            ),
        )
        if captured:
            # Drop a dangling Arabic suffix (e.g. the "ة" left by "خريجة")
            # and any leading degree word, leaving only the field of study.
            captured = re.sub(r"^[\u0621-\u064A]\s+(?=بكالوريوس|ماجستير|دكتوراه)", "", captured)
            captured = re.sub(
                r"^(?:بكالوريوس|ماجستير|دكتوراه|دبلوم|ثانوية عامة|"
                r"bachelor(?:'s)?|master(?:'s)?|ph\.?d|diploma|high school)"
                r"\s*(?:في|in)?\s*",
                "",
                captured,
                flags=re.IGNORECASE,
            ).strip()
        if captured:
            patch["major"] = captured

    nationality = _extract_nationality(text)
    if nationality:
        patch["nationality"] = nationality

    residence = re.search(
        r"(?:أقيم في|اقيم في|أعيش في|اعيش في|مقيم في|i live in|resident of|residing in)\s+"
        r"([A-Za-z\u0600-\u06FF][\w\u0600-\u06FF ]{1,40})",
        text,
        re.IGNORECASE,
    )
    if residence:
        captured = _clean_capture(residence.group(1), stops=("،", "وأنا", "وانا"))
        if captured:
            patch["country_of_residence"] = captured

    languages = _extract_languages(text)
    if languages:
        patch["languages"] = languages

    fields = _extract_fields(text)
    if fields:
        patch["research_interests"] = fields
        patch["target_fields"] = list(fields)

    countries = _extract_countries(text)
    if countries:
        patch["target_countries"] = countries

    publications = re.search(
        r"(?:نشرت|عدد الأوراق|publications?|published)\D{0,12}(\d{1,2})",
        lowered,
    )
    if publications:
        patch["publications"] = int(publications.group(1))

    experience = re.search(
        r"(?:لدي خبرة في|خبرتي في|عملت في|تدربت في|i worked (?:in|as)|experience in)\s*"
        r"([^.,;\n]{3,120})",
        text,
        re.IGNORECASE,
    )
    if experience:
        captured = _clean_capture(experience.group(1))
        if captured and captured not in ("", "لا"):
            patch.setdefault("experiences", []).append(captured)

    if re.search(r"(?:أنشطة|انشطة|extracurricular|volunteer|تطوع)", lowered):
        activity = re.search(
            r"(?:أنشطة|انشطة|تطوع|volunteer(?:ing)?)\s*(?:في|in)?\s*([^.,;\n]{3,100})",
            text,
            re.IGNORECASE,
        )
        if activity:
            captured = _clean_capture(activity.group(1))
            if captured:
                patch.setdefault("extracurriculars", []).append(captured)

    return patch


def _language_rank(level_value: str) -> int:
    return _LEVEL_ORDER.index(level_value) if level_value in _LEVEL_ORDER else 0


def merge_profile(profile: StudentProfile, patch: dict) -> StudentProfile:
    """Merge an extracted patch into a profile without erasing known data."""
    data = profile.model_dump()

    for key, value in patch.items():
        if value is None:
            continue
        if key == "languages":
            by_language = {
                item["language"].lower(): item for item in (data.get(key) or [])
            }
            for raw in value:
                candidate_model = (
                    raw if isinstance(raw, LanguageScore) else LanguageScore.model_validate(raw)
                )
                candidate = candidate_model.model_dump()
                current = by_language.get(candidate["language"].lower())
                if current is None or _language_rank(
                    candidate["level"]
                ) > _language_rank(current["level"]):
                    by_language[candidate["language"].lower()] = candidate
            data[key] = list(by_language.values())
        elif key in {
            "research_interests",
            "target_fields",
            "target_countries",
            "experiences",
            "extracurriculars",
        }:
            existing = list(data.get(key) or [])
            for item in value:
                if item not in existing:
                    existing.append(item)
            data[key] = existing
        else:
            data[key] = value

    return StudentProfile.model_validate(data)
