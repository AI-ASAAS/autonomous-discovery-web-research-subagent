# منح أساس — وكيل الاستكشاف والبحث المستقل للمنح الدراسية
### Autonomous Discovery & Web Research Sub-Agent (Full-Stack AI Agents + ASAAS UI)

> **المطوّر:** [Anas Abdulaziz Moqbel Al-Aghbari (@Anas-ALAghbari)](https://github.com/Anas-ALAghbari) — منظمة **[AI-ASAAS](https://github.com/AI-ASAAS)**

منصة متكاملة من وكلاء الذكاء الاصطناعي المستقلين لاستكشاف المنح الدراسية العالمية المخصّصة للطلاب اليمنيين، والتحقق منها ومراقبتها وفق مبدأ صارم ضد الهلوسة: **«الكود يتحقق والموديل ينقل»**، ومبنية بالكامل وفق معايير **[AI-ASAAS/skills](https://github.com/AI-ASAAS/skills)** (AI + Backend FastAPI + Frontend `asaas-ui`).

للاطلاع على معمارية وتشغيل منصة الوكلاء والمنسق الحواري (`agent_platform` على المنفذ 8010)، راجع [دليل المنصة (docs/PLATFORM.md)](docs/PLATFORM.md).

---

## ✨ أبرز المزايا والقدرات

* **أفضل 10 منح حقيقية متاحة (`/`)**: قاعدة بيانات SQLite (وضع WAL، قابلة للنقل إلى PostgreSQL) بفهرس مركّب على `(status, deadline)` تعرض المنح المنشورة وغير المنتهية فقط مرتبة بأقرب موعد إغلاق، مع نافذة تفاصيل تعرض **الاقتباس الحرفي الموثّق بالكود (`evidence`)** لكل حقل.
* **وكيل الاستخلاص الذاتي بالرابط (`/admin` — `ingestion_sub_agent.py`)**: يأخذ الرابط الرسمي للمنحة، يتحقق من النطاق الموثوق و`HTTP 200`، يتتبع رابطاً فرعياً واحداً (نفس النطاق، عمق 1) للشروط أو التمويل، يستخرج البيانات بمخطط Pydantic صارم (`temperature = 0`)، ويحفظها حصراً كمسودة `DRAFT` لاعتماد الإدارة.
* **وكيل البحث الخارجي العميق ReAct (`/search` — `external_search_sub_agent.py`)**: يعمل بنمط `Thought ➔ Action ➔ Observation` بحد أقصى **5 خطوات** عبر Tavily أو Serper مع قائمة منسدلة لـ **22 دولة مستهدفة**، يزور الصفحات الرسمية فعلياً، يتحقق من قبول الطلاب اليمنيين بنص صريح (`yes` / `no` / `unclear`)، ويعرض سجل التفكير بالكامل بالعربية المختصرة.
* **مراقب التحديثات والمواعيد (`/monitor` — `monitor_service.py`)**: يفحص الروابط الرسمية للمنح المنشورة عبر بصمة `SHA-256`، يسجل التغييرات في `UpdateLog` دون تعديل المنحة المنشورة تلقائياً، يطلق تنبيهات الإغلاق القريب (قبل 7 أيام و 3 أيام)، ويسم المنتهية بـ `EXPIRED` مع جدولة يومية عبر `APScheduler`.
* **راصد قنوات التيليجرام (`integrations/telegram_watcher.py`)**: يقرأ منشورات القنوات المعتمدة عبر `Telethon`، يستخرج الروابط الرسمية فقط بالـ Regex، يرفض روابط `t.me` والمجمّعات الإعلانية، يزيل المكرر مقابل قاعدة البيانات، ويمرّر الروابط الجديدة مباشرة إلى `ingest()`.
* **واجهة ASAAS UI متجاوبة (`frontend/`)**: خطوط ثمانية (`Sans` / `Serif Display` / `Serif Text` / `<Swash>`)، تصميم Liquid Glass، دعم كامل للعربية `RTL` والإنجليزية `LTR`، وضع ليلي/نهاري، استجابة كاملة للجوال (`375px` بصفر تمرير أفقي)، وحقول إدخال يدوية آمنة لمفاتيح الـ API مع تفعيل فوري بدون إعادة تشغيل الخادم.

---

## 🏛️ الهيكلية المعمارية (4-Layer Architecture)

```text
Frontend (React 19 + Vite + TypeScript + asaas-ui)
  ├── /          أفضل 10 منح + نافذة الأدلة الحرفية
  ├── /search    وكيل البحث الخارجي العميق ReAct + عارض سجل التفكير
  ├── /admin     وكيل الاستخلاص بالرابط + اعتماد المسودات + راصد التيليجرام
  └── /monitor   مراقب التحديثات + جاهزية المفاتيح + سجل UpdateLog
        │
        ▼
FastAPI HTTP Boundary (api/main.py + api/schemas.py)
        │
        ▼
Agents Layer:
  ├── ingestion_sub_agent.py           (الاستخلاص الذاتي وتتبع الرابط الفرعي)
  ├── external_search_sub_agent.py     (وكيل البحث العميق ReAct — حد 5 خطوات)
  ├── monitor_service.py               (مراقب البصمات وتنبيهات الإغلاق والجدولة)
  └── integrations/telegram_watcher.py (راصد قنوات التيليجرام)
        │
        ▼
Verification & Anti-Hallucination Layer:
  ├── utils/url_validator.py           (HTTP 200 + نطاق رسمي + مطابقة الاسم)
  ├── utils/evidence_check.py          (تطبيع النصوص وتصفير أي حقل بلا اقتباس حرفي)
  ├── utils/llm_extractor.py           (Pydantic v2 + temperature=0 + توجيه تلقائي للمزوّد)
  └── prompts/*.md                     (ملفات برومبت مرقّمة ومفصولة عن الكود)
        │
        ▼
Data Layer (SQLite WAL / PostgreSQL Portable):
  ├── core/config.py                   (إعدادات البيئة)
  ├── core/models.py                   (Grant, GrantStatus, YemeniEligibilityStatus, UpdateLog)
  ├── core/db.py                       (WAL Mode + Foreign Keys + Busy Timeout)
  └── core/repository.py               (get_top_10_grants, save_draft, publish, log_update)
```

---

## 🔑 مزوّدو الذكاء الاصطناعي والبحث المدعومون (توجيه تلقائي)

يدعم النظام إدخال المفاتيح يدوياً من الواجهة مباشرة (في بطاقة الاستخلاص والبحث وصفحة المراقب) مع حفظ محلي آمن في `.env` وإخفاء الأسرار (`AIza••••9876`)، أو إضافتها مباشرة في ملف `.env`:

| الاستخدام | المزوّد المدعوم | بادئة المفتاح | رابط الحصول على مفتاح |
|---|---|---|---|
| **الاستخلاص بالذكاء الاصطناعي (`LLM_API_KEY`)** | Google Gemini *(موصى به — مجاني)* | `AIza...` | [Google AI Studio](https://aistudio.google.com/apikey) |
| **الاستخلاص بالذكاء الاصطناعي (`LLM_API_KEY`)** | Groq *(سريع ومجاني)* | `gsk_...` | [Groq Console](https://console.groq.com/keys) |
| **الاستخلاص بالذكاء الاصطناعي (`LLM_API_KEY`)** | OpenRouter | `sk-or-...` | [OpenRouter Keys](https://openrouter.ai/keys) |
| **الاستخلاص بالذكاء الاصطناعي (`LLM_API_KEY`)** | OpenAI | `sk-...` | [OpenAI Platform](https://platform.openai.com/api-keys) |
| **البحث الخارجي العميق (`TAVILY_API_KEY`)** | Tavily *(أساسي — 1000 بحث مجاني)* | `tvly-...` | [Tavily Dashboard](https://app.tavily.com) |
| **البحث الخارجي العميق (`SERPER_API_KEY`)** | Serper *(بديل اختياري)* | — | [Serper.dev](https://serper.dev) |
| **راصد التيليجرام (`TELEGRAM_API_ID` / `HASH`)** | Telegram Telethon *(اختياري)* | — | [my.telegram.org](https://my.telegram.org) |

> **تنبيه ذكي مدمج**: يكتشف النظام تلقائياً إذا تم إدخال مفتاح بحث Tavily (`tvly-...`) في حقل الذكاء الاصطناعي ويعرض توجيهاً فورياً بالعربية.

---

## 🚀 التثبيت والتشغيل المحلي

### 1. تجهيز بيئة Python والخادم الخلفي
```bash
# إنشاء وتفعيل البيئة الافتراضية
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# تثبيت الاعتماديات
pip install -r requirements.txt

# نسخ ملف البيئة
cp .env.example .env

# تعبئة قاعدة البيانات بأول 10 منح حقيقية ونشرها
python seed/seed_db.py --publish
```

### 2. بناء وتشغيل الواجهة الأمامية (ASAAS UI)
```bash
cd frontend
npm install
npm run build
cd ..

# تشغيل خادم FastAPI (يخدم الـ API والواجهة المبنية معاً)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
# افتح المتصفح على: http://127.0.0.1:8001
```

### 3. تشغيل الوكلاء مباشرة من سطر الأوامر (CLI)
```bash
# وكيل البحث الخارجي العميق ReAct:
python external_search_sub_agent.py --major "Computer Science" --degree "Master" --country "Germany"

# تشغيل دورة فحص فورية للمراقب:
python monitor_service.py --now

# تشغيل المراقب المجدول يومياً (APScheduler):
python monitor_service.py --schedule --interval 24

# قراءة آخر منشورات قنوات التيليجرام المعتمدة:
python integrations/telegram_watcher.py --limit 20
```

---

## 🔌 نقاط نهاية الـ API (FastAPI Endpoints)

| المسار | الطريقة | الوظيفة |
|---|---|---|
| `/api/grants/top10` | `GET` | استرجاع أفضل 10 منح منشورة وغير منتهية مرتبة بأقرب موعد |
| `/api/grants?status=` | `GET` | عرض المنح مع الفلترة (`all` / `draft` / `published` / `expired`) والإحصائيات |
| `/api/grants/{id}/publish` | `POST` | اعتماد ونشر منحة من الإدارة |
| `/api/agents/ingest` | `POST` | استخلاص منحة من رابط رسمي كمسودة `DRAFT` (يدعم تمرير المفتاح مباشرة) |
| `/api/agents/search` | `POST` | تشغيل وكيل البحث العميق ReAct وإرجاع النتائج الموثّقة + سجل التفكير |
| `/api/agents/monitor/run` | `POST` | تنفيذ دورة فحص فورية للمواقع الرسمية والمواعيد |
| `/api/agents/monitor/logs` | `GET` | استرجاع سجل التحديثات والتغييرات `UpdateLog` |
| `/api/agents/telegram/process` | `POST` | فحص منشورات تيليجرام واستخراج الروابط الرسمية وفلترتها |
| `/api/system/status` | `GET` | حالة جاهزية المفاتيح وإحصائيات القاعدة (مع قناع أمان بدون كشف أسرار) |
| `/api/system/keys` | `POST` | حفظ وتفعيل مفاتيح الـ API يدوياً وفورياً بدون إعادة تشغيل |

---

## ☁️ النشر السحابي (Docker / Railway / Render / Cloudflare)

المشروع مجهّز بالكامل للنشر السحابي مع تعبئة تلقائية لأول 10 منح عند أول إقلاع:
* **Docker**: يحتوي على [`Dockerfile`](Dockerfile) متعدد المراحل (يبني واجهة React/Vite ثم يشغّل FastAPI على `${PORT}`).
* **Railway**: جاهز تلقائياً عبر [`railway.json`](railway.json).
* **Render**: جاهز عبر [`render.yaml`](render.yaml).
* **Cloudflare Quick Tunnel** (رابط HTTPS عام فوري من جهازك):
  ```bash
  cloudflared tunnel --url http://127.0.0.1:8001
  ```

---

## 🧪 الاختبارات والتحقق من الجودة

```bash
# تشغيل جميع اختبارات Python (39 اختباراً للوحدة والتكامل E2E والـ API)
pytest -v

# فحص TypeScript وبناء الإنتاج للواجهة الأمامية
cd frontend && npm run build
```

| ملف الاختبار | نطاق التغطية |
|---|---|
| `tests/test_repository.py` | استرجاع أفضل 10 منح، استبعاد المسودات والمنتهية، والترتيب |
| `tests/test_url_validator.py` | رفض المجمّعات الإعلانية والروابط الميتة وغير المطابقة |
| `tests/test_evidence_check.py` | تطبيع النصوص وتصفير الحقول ذات الاقتباسات المختلقة |
| `tests/test_llm_extractor.py` | البرومبتات المرقّمة ومخطط Pydantic وتنظيف JSON |
| `tests/test_ingestion.py` | الاستخلاص بصفحة HTML محلية وتتبع الرابط الفرعي والصفحات الناقصة |
| `tests/test_seed.py` | تعبئة المنح الـ 10 الحقيقية ونشرها |
| `tests/test_monitor.py` | تغير البصمة والمواعيد وتنبيهات الـ 7 أيام والـ 3 أيام والانتهاء |
| `tests/test_external_search.py` | حلقة ReAct، حد الـ 5 خطوات، وفحص قبول اليمنيين |
| `tests/test_telegram_watcher.py` | استخراج روابط تيليجرام، رفض `t.me` والمجمّعات، ومنع التكرار |
| `tests/test_api.py` | نقاط FastAPI، قناع الأسرار، وصيغة أخطاء ASAAS الموحدة |
| `tests/test_e2e_pipeline.py` | السيناريو الكامل من البداية للنهاية + اختبار الهلوسة المتعمّد |

---

## ✅ قائمة التسليم النهائية
- [x] قاعدة بيانات مهيأة تحتوي على أول 10 منح حقيقية بروابط رسمية وأهلية موثّقة لليمنيين
- [x] منظومة منع هلوسة صارمة: كل حقل موثّق باقتباس حرفي محقق بالكود، والمجهول يرجع `null`
- [x] كل المنح الجديدة تدخل بحالة `draft` والنشر حصراً بقرار الإدارة
- [x] وكيل البحث العميق ReAct بحد 5 خطوات، ورفض كامل للروابط غير الرسمية
- [x] مراقب دوري بالبصمة الرقمية وتنبيهات 7 و 3 أيام وجدولة `APScheduler`
- [x] راصد قنوات التيليجرام لاستخراج الروابط الرسمية تلقائياً
- [x] واجهة أمامية كاملة بحزمة `asaas-ui` (متجاوبة للهاتف، ليل/نهار، عربي/إنجليزي، وإدخال يدوي للمفاتيح)
- [x] لا توجد أي مفاتيح سرية مكشوفة في المستودع (`.env` خارج Git)
- [x] جميع الاختبارات (**39/39**) وبناء الإنتاج يمرون بنسبة 100%
