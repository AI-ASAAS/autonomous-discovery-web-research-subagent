"""Profile strengthening advisor (task 6).

Every observation is derived from a concrete field of the profile; the module
never assumes a certificate exists or invents a missing achievement.
"""

from __future__ import annotations

from .schemas import (
    AdvisorAction,
    AdvisorReport,
    DegreeLevel,
    LanguageLevel,
    MatchReport,
    StudentProfile,
)

_LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]


def _english_level(profile: StudentProfile) -> str | None:
    for item in profile.languages:
        if item.language.lower() == "english":
            return item.level.value
    return None


def _level_index(value: str | None) -> int:
    return _LEVEL_ORDER.index(value) if value in _LEVEL_ORDER else 0


def build_advisor_report(
    profile: StudentProfile,
    report: MatchReport | None = None,
) -> AdvisorReport:
    strengths: list[str] = []
    weaknesses: list[str] = []
    gaps: list[str] = []
    actions: list[AdvisorAction] = []

    if profile.gpa is not None:
        if profile.gpa >= 3.5:
            strengths.append(
                f"معدل تراكمي مرتفع ({profile.gpa:.2f}/4.00) يتجاوز متطلبات معظم المنح."
            )
        elif profile.gpa >= 3.0:
            strengths.append(
                f"معدل تراكمي جيد ({profile.gpa:.2f}/4.00) ويستوفي الحدود الشائعة (3.0)."
            )
        else:
            weaknesses.append(
                f"المعدل التراكمي ({profile.gpa:.2f}/4.00) أقل من الحد الشائع في المنح التنافسية (3.0)."
            )
            actions.append(
                AdvisorAction(
                    title="ارفع معدلك أو استهدف منحاً لا تشترط حداً مرتفعاً",
                    why="الفلترة البرمجية في معظم المنح تستبعد المعدلات الأقل من 3.0.",
                    how="أعد دراسة مواد يمكن رفع معدلها، وفي الوقت نفسه فلتر المنح التي تكتفي بخبرة أو مشروع بحثي.",
                    timeframe="الفصل الدراسي القادم",
                    priority="high",
                )
            )

    english = _english_level(profile)
    test_names = {
        (item.test_name or "").upper()
        for item in profile.languages
        if item.test_name
    }
    if english and _level_index(english) >= _level_index("B2"):
        strengths.append(f"مستوى الإنجليزية {english} يفتح معظم المنح التي تطلب B2 أو أعلى.")
    elif english:
        weaknesses.append(
            f"مستوى الإنجليزية الحالي ({english}) أقل من B2 المطلوب في معظم المنح."
        )
        actions.append(
            AdvisorAction(
                title="ارفع مستوى الإنجليزية إلى B2 على الأقل",
                why="معظم المنح الحكومية تشترط B2 أو ما يعادلها (IELTS 5.5-6.0).",
                how="دورة مكثفة 8-12 أسبوعاً مع اختبار تجريبي أسبوعي، ثم احجز IELTS.",
                timeframe="2-3 أشهر",
                priority="high",
            )
        )
    else:
        weaknesses.append("لا يوجد أي إثبات لمستوى اللغة الإنجليزية في ملفك.")
        actions.append(
            AdvisorAction(
                title="أضف إثبات لغة إنجليزية معتمداً",
                why="بدون درجة لغة ستُستبعد تلقائياً من معظم المنح.",
                how="قدّم IELTS Academic أو TOEFL iBT وسجّل الدرجة في ملفك.",
                timeframe="6-8 أسابيع",
                priority="high",
            )
        )

    if english and english != "native" and not ({"IELTS", "TOEFL"} & test_names):
        gaps.append(
            "لديك مستوى لغة موصوف لكن بدون شهادة اختبار معياري (IELTS/TOEFL)."
        )
        actions.append(
            AdvisorAction(
                title="حوّل مستواك اللغوي إلى شهادة رسمية",
                why="الوصف الذاتي لا يُقبل كإثبات في طلبات المنح.",
                how="احجز أقرب موعد IELTS أو TOEFL وأضف الدرجة إلى ملفك.",
                timeframe="خلال شهرين",
                priority="medium",
            )
        )

    if profile.research_interests or profile.target_fields:
        strengths.append(
            "اهتمامات أكاديمية محددة: "
            + "، ".join((profile.research_interests or profile.target_fields)[:4])
            + "."
        )
    else:
        weaknesses.append("لم تحدد اهتماماتك البحثية أو مجالاتك المستهدفة.")
        actions.append(
            AdvisorAction(
                title="حدد 2-3 اهتمامات بحثية دقيقة",
                why="المطابقة الدلالية مع رسالة المنحة تعتمد على وضوح الاهتمامات.",
                how="اختر موضوعين من مقرراتك المفضلة واربطهما بمشكلة مجتمعية أو بحثية.",
                timeframe="هذا الأسبوع",
                priority="high",
            )
        )

    if profile.publications > 0:
        strengths.append(f"لديك {profile.publications} ورقة بحثية منشورة.")
    elif profile.target_degree in {DegreeLevel.MASTER, DegreeLevel.PHD}:
        gaps.append("لا توجد أوراق بحثية منشورة، وهي عامل ترجيح في المنح البحثية.")
        actions.append(
            AdvisorAction(
                title="أضف مخرجاً بحثياً واحداً على الأقل",
                why="المنح البحثية تعتبر النشر دليلاً على الجاهزية.",
                how="شارك في مؤتمر طلابي أو أعد ورقة مراجعة مع مشرفك ثم اذكرها في السيرة.",
                timeframe="3-6 أشهر",
                priority="medium",
            )
        )

    if profile.experiences:
        strengths.append(f"خبرات موثقة في ملفك: {len(profile.experiences)} عنصراً.")
    else:
        weaknesses.append("لا توجد خبرات عملية أو تطوعية مذكورة.")
        actions.append(
            AdvisorAction(
                title="ابدأ تجربة عملية أو تطوعية قصيرة",
                why="الخبرة تعوّض جزئياً عن المعدل وتقوّي خطاب الدافع.",
                how="تدريب عن بُعد أو عمل تطوعي في منظمة محلية لمدة 4-8 أسابيع.",
                timeframe="الشهر القادم",
                priority="medium",
            )
        )

    if profile.target_countries:
        strengths.append("دول مستهدفة محددة: " + "، ".join(profile.target_countries) + ".")
        if len(profile.target_countries) == 1:
            gaps.append("التركيز على دولة واحدة يقلل فرص القبول؛ نوّع الخيارات.")
    else:
        weaknesses.append("لم تحدد الدول التي ترغب بالدراسة فيها.")

    if report and report.results:
        strong = [m for m in report.results if m.match_percentage >= 60]
        if strong:
            strengths.append(
                f"لديك {len(strong)} منحة داخلية بنسبة تطابق 60% أو أعلى؛ أعلاها "
                f"«{strong[0].title}» بنسبة {strong[0].match_percentage:.1f}%."
            )
        else:
            weaknesses.append(
                "لا توجد منحة داخلية تتجاوز 60% حالياً؛ البحث الخارجي ضروري."
            )

    actions.append(
        AdvisorAction(
            title="جهّز حزمة وثائق موحدة",
            why="كل المنح تطلب السيرة الذاتية وخطاب الدافع وسجل الدرجات ورسائل التوصية.",
            how="استخدم وكيل المستندات لتوليد السيرة وخطاب الدافع، ثم راجعها يدوياً قبل الرفع.",
            timeframe="أسبوعان",
            priority="medium",
        )
    )
    actions.append(
        AdvisorAction(
            title="قدّم على 3-5 منح متوازنة",
            why="الاعتماد على منحة واحدة يرفع مخاطر الرفض.",
            how="اختر منحتين بنسبة تطابق عالية ومنحتين متوسطة ومنحة طموحة، ورتبها حسب الموعد النهائي.",
            timeframe="وفق المواعيد",
            priority="medium",
        )
    )

    overall = (
        f"الملف مكتمل بنسبة {profile.completeness():.0f}% من الحقول الأساسية. "
        "نقاط القوة أعلاه قابلة للتوظيف فوراً في خطاب الدافع، والعناصر الناقصة "
        "قابلة للمعالجة خلال 2-6 أشهر."
    )

    return AdvisorReport(
        strengths=strengths,
        weaknesses=weaknesses,
        competitive_gaps=gaps,
        actions=actions,
        overall_competitiveness=overall,
    )
