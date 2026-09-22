import html

import streamlit as st

from components.cards import card_grid
from components.diagrams import mermaid
from config import APP_AUTHOR_AR, APP_AUTHOR_EN, APP_NAME_AR, APP_NAME_EN, APP_SUBTITLE_AR
from content.quizzes import QUIZZES
from core.curriculum import LEARNING_PATHS, MODULES, ROADMAP, get_module, tracked_modules
from core.page import footer
from core.state import is_complete, mark_visited, progress_pct

mark_visited("home")

st.html(
    f"""
<div class="ds-hero">
  <div class="ds-kicker">مقرر جامعي تفاعلي · Interactive university-level course</div>
  <h1>{html.escape(APP_NAME_AR)}</h1>
  <div class="ds-en">{html.escape(APP_NAME_EN)}</div>
  <p>{html.escape(APP_SUBTITLE_AR)}</p>
  <div class="ds-author">إعداد وتطوير: <b>{html.escape(APP_AUTHOR_AR)}</b> · <span class="ds-en">{html.escape(APP_AUTHOR_EN)}</span></div>
</div>
"""
)

# ---------------------------------------------------------------- actions
current = st.session_state.get("current_module")
if not current or current == "home":
    pending = [m for m in tracked_modules() if not is_complete(m.id)]
    current = pending[0].id if pending else "what_is_ds"
with st.container(horizontal=True):
    if st.button("ابدأ الوحدة الأولى", type="primary", icon=":material/rocket_launch:"):
        st.switch_page(get_module("what_is_ds").file)
    if st.button(f"تابع التعلّم: {get_module(current).title_ar}", icon=":material/play_arrow:"):
        st.switch_page(get_module(current).file)
    if st.button("استكشف مجموعة بيانات", icon=":material/data_exploration:"):
        st.switch_page(get_module("dataset_explorer").file)

# --------------------------------------------------------------- overview
st.markdown("## ماذا ستتعلّم؟")
st.markdown(
    "مقرر يسير من **البيانات الخام** إلى **التحليل والنمذجة والتواصل**، ثم إلى **الأتمتة وعلم البيانات بمساعدة الذكاء "
    "الاصطناعي**، مع الحفاظ على المبدأ: *Automation should assist judgment, not erase judgment*. "
    "كل مفهوم يُعرض بتسلسل: الفكرة ← الحدس ← الرياضيات ← المثال ← الكود ← التشغيل ← الرسم ← التفسير ← الأخطاء ← التمرين."
)
card_grid([
    ("فهم البيانات وجودتها", "الأنواع، المصادر، أبعاد الجودة، التحقق بالقواعد، ولوحة جودة بأوزان معلنة."),
    ("التشخيص والتنظيف", "القيم المفقودة بآلياتها، التكرارات، القيم الشاذة، الاتساق، التحويل والترميز."),
    ("الاستكشاف والإحصاء", "EDA منظم، اختيار الرسم، فترات الثقة، الاختبارات مع حجم الأثر."),
    ("النمذجة دون تسرّب", "هندسة الخصائص، التقسيم الصحيح، Leakage، التقييم والعتبات."),
    ("التنقيب", "قواعد الارتباط، التجميع، كشف الشذوذ، وتقليل الأبعاد مع رسوم متحركة."),
    ("علم البيانات الحديث", "Automated EDA، التنظيف الآلي، AutoML، المراقبة، LLMs ووكلاء البيانات."),
], columns=3)

# ----------------------------------------------------------------- roadmap
st.markdown("## خريطة الرحلة · Course roadmap")
done_ids = st.session_state.get("completed", set())
chips = []
for i, (en, ar, mid) in enumerate(ROADMAP):
    cls = "ds-step done" if mid in done_ids else "ds-step"
    chips.append(f'<span class="{cls}">{html.escape(en)} · {html.escape(ar)}</span>')
    if i < len(ROADMAP) - 1:
        chips.append('<span class="ds-arrow">→</span>')
st.html(f'<div class="ds-roadmap">{"".join(chips)}</div>')

step = st.pills("اختر محطة من الخريطة لمعرفة ما تغطيه", [r[0] for r in ROADMAP], key="home_roadmap",
                default="Data Quality")
if step:
    en, ar, mid = next(r for r in ROADMAP if r[0] == step)
    m = get_module(mid)
    with st.container(border=True):
        st.markdown(f"**{ar} · {en}** — {m.description}")
        if m.objectives:
            st.markdown("\n".join(f"- {o}" for o in m.objectives))
        st.page_link(m.file, label=f"اذهب إلى: {m.title_ar}", icon=m.icon)

with st.expander("المخطط الكامل لدورة حياة المقرر", icon=":material/account_tree:"):
    mermaid("""
flowchart LR
  A[Raw Data] --> B[Collection] --> C[Understanding] --> D[Validation] --> E[Quality]
  E --> F[Cleaning] --> G[EDA] --> H[Statistics] --> I[Feature Eng.]
  I --> J[Data Mining] --> K[Machine Learning] --> L[Evaluation] --> M[Interpretation]
  M --> N[Communication] --> O[Automation] --> P[AI-Assisted DS]
  L -. "leakage / poor fit" .-> F
  O -. "human review" .-> E
""")

# ---------------------------------------------------------------- progress
st.markdown("## تقدّمك · Progress overview")
quiz_scores = st.session_state.get("quiz_scores", {})
with st.container(horizontal=True):
    st.metric("نسبة الإنجاز", f"{progress_pct():.0f}%", border=True)
    st.metric("وحدات مكتملة", f"{len(done_ids)} / {len(tracked_modules())}", border=True)
    st.metric("اختبارات أُنجزت", f"{len(quiz_scores)} / {len(QUIZZES)}", border=True)
    st.metric("مختبرات زرتها", len(st.session_state.get("labs_visited", set())), border=True)
st.caption("التقدّم محفوظ في جلستك؛ يمكنك تصديره واستيراده من صفحة «تقدّمي».")

# ------------------------------------------------------------ learning paths
st.markdown("## مسارات التعلّم · Learning paths")
tabs = st.tabs([p["title"] for p in LEARNING_PATHS.values()])
for tab, path in zip(tabs, LEARNING_PATHS.values()):
    with tab:
        st.markdown(path["desc"])
        done = sum(1 for mid in path["modules"] if is_complete(mid))
        st.progress(done / len(path["modules"]), text=f"{done} من {len(path['modules'])} وحدات")
        for mid in path["modules"]:
            m = get_module(mid)
            st.page_link(m.file, label=f"{m.title_ar} · {m.title_en}",
                         icon=":material/check_circle:" if is_complete(mid) else m.icon)

# --------------------------------------------------------------- info card
st.markdown("## بطاقة المقرر · Course information")
n_lessons = len([m for m in MODULES if m.kind == "lesson"])
n_labs = len([m for m in MODULES if m.kind == "lab"]) + len([m for m in MODULES if m.lab and m.kind == "lesson"])
with st.container(border=True):
    c1, c2 = st.columns(2)
    c1.markdown(
        f"**المطوّر:** {APP_AUTHOR_AR} · {APP_AUTHOR_EN}  \n"
        "**الجمهور:** طلاب الجامعة، المتدربون المهنيون، والدارسون ذاتيًا.  \n"
        "**اللغة:** العربية مع المصطلحات العلمية بالإنجليزية.  \n"
        "**المتطلب:** أساسيات Python وpandas مفيدة وليست شرطًا للبدء."
    )
    c2.markdown(
        f"**الوحدات النظرية:** {n_lessons}  \n"
        f"**المختبرات والأنشطة التفاعلية:** {n_labs}  \n"
        f"**الاختبارات:** {len(QUIZZES)} اختبارًا بأسئلة مشروحة  \n"
        "**المشاريع:** 6 دراسات حالة + قصة بيانات + مشروع نهائي  \n"
        "**مستويات الشرح:** مبتدئ / متقدم / بحثي (من الشريط الجانبي)"
    )
footer()
