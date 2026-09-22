import importlib.metadata as md
import platform

import numpy as np
import pandas as pd
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least

page_header("reproducibility")

st.markdown("## ما قابلية إعادة الإنتاج؟")
st.markdown("أن يحصل شخص آخر (أو أنت بعد ستة أشهر) على **نفس النتائج** من **نفس البيانات والكود**. تتطلب تثبيت أربعة أشياء: "
            "**الكود، البيانات، البيئة، والعشوائية**.")
mermaid("""
flowchart LR
  C[Code<br/>Git] --> R[Same result]
  D[Data<br/>versioned snapshot] --> R
  E[Environment<br/>pinned versions] --> R
  S[Randomness<br/>fixed seeds] --> R
  R --> T[Trust & auditability]
""")

st.markdown("## العشوائية والبذور Random seeds")
c1, c2 = st.columns(2)
use_seed = c1.toggle("ثبّت البذرة", True, key="rep_seed")
if c2.button("أعد التشغيل", icon=":material/replay:", key="rep_run"):
    st.session_state["rep_runs"] = st.session_state.get("rep_runs", 0) + 1
rng = np.random.default_rng(42) if use_seed else np.random.default_rng()
sample = rng.normal(0, 1, 5).round(4)
st.code(f"rng = np.random.default_rng({'42' if use_seed else ''})\nrng.normal(0, 1, 5)\n# -> {sample.tolist()}", language="python")
st.caption("مع البذرة: نفس الأرقام في كل تشغيل. بدونها: تتغير كل مرة. ثبّت أيضًا random_state في scikit-learn "
           "(train_test_split، النماذج، KFold مع shuffle).")

st.markdown("## البيئة Environment")
tab1, tab2, tab3 = st.tabs(["venv + pip", "uv", "pyproject.toml"])
with tab1:
    st.code("python -m venv .venv\n# Windows: .venv\\Scripts\\activate   |   macOS/Linux: source .venv/bin/activate\n"
            "pip install -r requirements.txt\npip freeze > requirements-lock.txt   # exact versions used", language="bash")
with tab2:
    st.code("uv venv\nuv pip install -r requirements.txt\n# or, project mode:\nuv sync        # installs from pyproject.toml + uv.lock",
            language="bash")
    st.caption("uv مدير حزم سريع جدًا بملف قفل uv.lock يثبّت كل الإصدارات الانتقالية.")
with tab3:
    st.code('[project]\nname = "my-analysis"\nrequires-python = ">=3.10,<3.14"\n'
            'dependencies = ["pandas>=2.2", "scikit-learn>=1.5"]', language="toml")
st.markdown("**بيئة هذه المنصة الآن:**")
pkgs = ["streamlit", "pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "plotly"]
vers = []
for p in pkgs:
    try:
        vers.append({"package": p, "version": md.version(p)})
    except md.PackageNotFoundError:
        vers.append({"package": p, "version": "not installed"})
st.dataframe(pd.DataFrame(vers + [{"package": "python", "version": platform.python_version()}]), hide_index=True)

st.markdown("## Git وإدارة الإصدارات")
st.code('git init\ngit add src/ notebooks/ requirements.txt\ngit commit -m "Clean customer data: unit fixes, sentinel handling"\n'
        "git tag analysis-v1.0   # tag the exact code behind a report", language="bash")
st.markdown("- لا تضع البيانات الكبيرة أو الأسرار في Git (استخدم `.gitignore` و`st.secrets`).\n"
            "- رسالة الـcommit تشرح **لماذا** لا **ماذا** فقط.")

st.markdown("## Notebook مقابل Script")
comparison_table([
    {"": "الاستكشاف", "Notebook": "ممتاز (تفاعلي، رسوم)", "Script / module": "أقل مرونة"},
    {"": "إعادة الإنتاج", "Notebook": "خطر: ترتيب الخلايا والحالة المخفية", "Script / module": "تنفيذ من البداية للنهاية"},
    {"": "الاختبار", "Notebook": "صعب", "Script / module": "pytest"},
    {"": "مراجعة الكود", "Notebook": "diff صعب", "Script / module": "diff واضح"},
    {"": "الممارسة الجيدة", "Notebook": "Restart & Run All قبل المشاركة", "Script / module": "انقل المنطق المستقر إلى وحدات"},
])
why("انقل كل منطق يُعاد استخدامه من الـNotebook إلى دوال في ملفات .py مع اختبارات.",
    "هكذا بُنيت هذه المنصة: utils/ تحوي المنطق وtests/ تختبره، والصفحات تعرضه فقط.")

st.markdown("## Pipelines وتتبع التجارب وإصدارات البيانات")
comparison_table([
    {"المفهوم": "Pipeline", "الأداة": "scikit-learn Pipeline، Makefile، DVC stages", "الفائدة": "خطوات ثابتة الترتيب قابلة لإعادة التشغيل"},
    {"المفهوم": "Experiment tracking", "الأداة": "MLflow، Weights & Biases", "الفائدة": "تسجيل المعاملات والمقاييس والنماذج لكل تجربة"},
    {"المفهوم": "Data versioning", "الأداة": "DVC، lakeFS، لقطات Parquet مؤرخة", "الفائدة": "معرفة أي نسخة بيانات أنتجت أي نتيجة"},
    {"المفهوم": "Random seeds", "الأداة": "random_state، np.random.default_rng", "الفائدة": "نتائج حتمية"},
    {"المفهوم": "Environment pinning", "الأداة": "requirements lock، uv.lock، Docker", "الفائدة": "نفس الإصدارات في كل مكان"},
])
st.code('import mlflow\nwith mlflow.start_run():\n    mlflow.log_params({"model": "hgb", "learning_rate": 0.1, "seed": 42})\n'
        '    mlflow.log_metric("cv_auc", 0.781)\n    mlflow.log_artifact("reports/figure.png")', language="python")

if at_least("advanced"):
    st.markdown("## متقدم: هيكل مشروع مقترح")
    st.code("project/\n├── data/raw/            # read-only\n├── data/processed/\n├── src/                 # functions + pipeline\n"
            "├── notebooks/           # exploration only\n├── tests/\n├── reports/\n├── pyproject.toml / requirements.txt\n"
            "└── README.md            # how to reproduce every result", language="text")
if at_least("research"):
    researcher_note(["انشر الكود والبيانات (أو بيانات اصطناعية مكافئة) مع الورقة حيثما أمكن.",
                     "سجّل مسبقًا خطة التحليل وافصل النتائج الاستكشافية.",
                     "ميّز بين Reproducibility (نفس البيانات والكود) وReplicability (بيانات جديدة ونفس السؤال)."])
real_world(["هل يستطيع زميل تشغيل التحليل من الصفر بتعليمات README فقط؟", "هل البيانات الخام محفوظة دون تعديل؟",
            "هل كل رقم في التقرير يأتي من كود محفوظ؟", "هل الإصدارات مثبتة؟"])

page_footer("reproducibility",
            takeaways=["ثبّت الكود والبيانات والبيئة والعشوائية.", "Notebooks للاستكشاف؛ الوحدات والاختبارات للمنطق.",
                       "تتبع التجارب وإصدارات البيانات يربط كل نتيجة بمصدرها."],
            mistakes=["تعديل البيانات الخام في مكانها.", "requirements بلا إصدارات.", "مشاركة Notebook لم يُشغّل من البداية."])
