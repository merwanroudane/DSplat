import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit
from utils.plotting import plot

page_header("automl")

st.markdown("## مكونات AutoML")
mermaid("""
flowchart LR
  D[Data + target + metric] --> S[Search space<br/>models × hyperparameters × preprocessing]
  S --> ST[Search strategy<br/>random / Bayesian / successive halving]
  ST --> V[Validation<br/>cross-validation]
  V --> B{Budget left?}
  B -- yes --> ST
  B -- no --> L[Leaderboard] --> E[Ensemble / best model] --> T[Final test on untouched data]
""")
comparison_table([
    {"المكون": "Search space", "المعنى": "النماذج والمعاملات والمعالجات الممكنة", "القرار البشري": "ما المسموح؟ (قابلية التفسير، الزمن)"},
    {"المكون": "Search strategy", "المعنى": "عشوائي، Bayesian optimization، Hyperband", "القرار البشري": "عادة افتراضي الأداة"},
    {"المكون": "Budget", "المعنى": "زمن أو عدد تجارب", "القرار البشري": "حسب الموارد والأهمية"},
    {"المكون": "Validation", "المعنى": "CV بمخطط مناسب", "القرار البشري": "حاسم: Group/Time split عند الحاجة"},
    {"المكون": "Metric", "المعنى": "ما يُعظّم", "القرار البشري": "يجب أن يعكس تكلفة الأخطاء"},
])

st.markdown("## Mini-AutoML حقيقي")
st.caption("بحث عشوائي في 4 عائلات نماذج على بيانات القروض، بتقييم 3-fold CV (ROC-AUC)، ثم اختبار نهائي على بيانات محجوزة.")
FEATS = ["age", "income", "loan_amount", "term_months", "late_payments", "debt_to_income", "credit_history_years"]
c1, c2, c3 = st.columns(3)
budget = c1.slider("الميزانية (عدد التجارب)", 4, 40, 16, key="aml_budget")
seed = c2.number_input("البذرة", 0, 100, 0, key="aml_seed")
leak = c3.toggle("أضف العمود المسرّب sent_to_collections", False, key="aml_leak")


def sample_config(rng: np.random.Generator) -> tuple[str, dict]:
    family = rng.choice(["logreg", "rf", "hgb", "knn"])
    if family == "logreg":
        return family, {"C": float(10 ** rng.uniform(-3, 2))}
    if family == "rf":
        return family, {"n_estimators": int(rng.choice([100, 200])), "max_depth": int(rng.choice([3, 5, 8, 12])),
                        "min_samples_leaf": int(rng.choice([1, 5, 20]))}
    if family == "hgb":
        return family, {"learning_rate": float(10 ** rng.uniform(-2, -0.5)), "max_depth": int(rng.choice([2, 3, 5])),
                        "max_iter": int(rng.choice([100, 200]))}
    return family, {"n_neighbors": int(rng.choice([5, 15, 31, 61]))}


def build(family: str, params: dict):
    if family == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **params))
    if family == "rf":
        return RandomForestClassifier(random_state=0, n_jobs=-1, **params)
    if family == "hgb":
        return HistGradientBoostingClassifier(random_state=0, **params)
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))


@st.cache_data(show_spinner="AutoML يبحث…")
def run_automl(budget: int, seed: int, leak: bool) -> tuple[pd.DataFrame, dict]:
    d = credit()
    feats = FEATS + (["sent_to_collections"] if leak else [])
    Xtr, Xte, ytr, yte = train_test_split(d[feats], d["default"], test_size=0.25, random_state=42, stratify=d["default"])
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    rows, configs = [], []
    for trial in range(budget):
        fam, params = sample_config(rng)
        configs.append((fam, params))
        t0 = time.perf_counter()
        s = cross_val_score(build(fam, params), Xtr, ytr, cv=cv, scoring="roc_auc")
        rows.append({"trial": trial + 1, "model": fam, "params": str(params), "cv_auc": s.mean(), "cv_sd": s.std(),
                     "seconds": time.perf_counter() - t0})
    lb = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    best = lb.iloc[0]
    best_model = build(*configs[int(best["trial"]) - 1])
    best_model.fit(Xtr, ytr)
    test_auc = roc_auc_score(yte, best_model.predict_proba(Xte)[:, 1])
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[FEATS], ytr)
    base_auc = roc_auc_score(yte, base.predict_proba(Xte[FEATS])[:, 1])
    return lb, {"test_auc": test_auc, "baseline_auc": base_auc, "best_cv": best["cv_auc"]}


if st.button("شغّل AutoML", type="primary", icon=":material/play_arrow:", key="aml_run"):
    st.session_state["aml_ran"] = True
if st.session_state.get("aml_ran"):
    lb, summary = run_automl(budget, int(seed), leak)
    c1, c2, c3 = st.columns(3)
    c1.metric("أفضل CV AUC (على التدريب)", f"{summary['best_cv']:.3f}")
    c2.metric("AUC النهائي على الاختبار المحجوز", f"{summary['test_auc']:.3f}", f"{summary['test_auc'] - summary['best_cv']:+.3f}")
    c3.metric("خط الأساس: Logistic Regression", f"{summary['baseline_auc']:.3f}")
    st.dataframe(lb.round(4), hide_index=True, height=300)
    fig = go.Figure()
    for fam, color in zip(["logreg", "rf", "hgb", "knn"], [PALETTE["purple"], PALETTE["coral"], PALETTE["amber"], PALETTE["sky"]]):
        sub = lb[lb["model"] == fam]
        fig.add_trace(go.Scatter(x=sub["trial"], y=sub["cv_auc"], mode="markers", name=fam, marker=dict(size=11, color=color),
                                 error_y=dict(type="data", array=sub["cv_sd"], visible=True)))
    best_so_far = lb.sort_values("trial")["cv_auc"].cummax()
    fig.add_trace(go.Scatter(x=lb.sort_values("trial")["trial"], y=best_so_far, mode="lines", name="best so far",
                             line=dict(color="black", dash="dot")))
    fig.update_layout(xaxis_title="trial", yaxis_title="CV ROC-AUC", height=380, legend=dict(orientation="h", y=1.12))
    plot(fig)
    if leak:
        st.error("مع العمود المسرّب يصل AutoML إلى أداء شبه مثالي: **الأداة لا تعرف أن الخاصية غير متاحة وقت القرار.** "
                 "AutoML يعظّم المقياس بأي إشارة متاحة.", icon=":material/warning:")
    why("قارن دائمًا بخط أساس بسيط وقيّم على بيانات لم يرها البحث.",
        "كثرة التجارب على نفس الـCV ترفع أفضل نتيجة بالصدفة (Winner's curse)، والفرق عن نموذج بسيط قد لا يبرر التعقيد وفقدان التفسير.")
else:
    st.info("اضغط «شغّل AutoML» لبدء البحث (بضع ثوانٍ).", icon=":material/info:")

st.markdown("## أطر AutoML الحالية")
st.caption("الحالة كما تحققنا منها في سبتمبر 2026 (انظر docs/research_notes.md). لم تُضمَّن كاعتماديات لأنها ثقيلة على الاستضافة المجانية.")
comparison_table([
    {"الإطار": "FLAML (Microsoft)", "الحالة": "نشط", "القوة": "خفيف وموفّر للموارد، ضبط سريع", "مثال": "AutoML().fit(X, y, task='classification', time_budget=60)"},
    {"الإطار": "AutoGluon (AWS)", "الحالة": "نشط", "القوة": "Stacking قوي للجداول والنصوص والصور", "مثال": "TabularPredictor(label='y').fit(train)"},
    {"الإطار": "TPOT", "الحالة": "أُعيد بناؤه (نسخة جديدة)", "القوة": "بحث جيني في Pipelines كاملة", "مثال": "TPOTClassifier(...).fit(X, y)"},
    {"الإطار": "auto-sklearn", "الحالة": "آخر إصدار رسمي 2023 — يُعامل كإرث بحثي", "القوة": "Meta-learning + Bayesian optimization", "مثال": "Linux فقط تقريبًا"},
    {"الإطار": "H2O AutoML", "الحالة": "نشط", "القوة": "موزع، Leaderboard شامل", "مثال": "يتطلب Java"},
])

if at_least("advanced"):
    st.markdown("## متقدم: Automated feature engineering")
    st.markdown("أدوات مثل Featuretools تولّد خصائص تجميعية آليًا عبر الجداول المترابطة (Deep Feature Synthesis). "
                "الخطر الأكبر: خصائص تتضمن معلومات بعد لحظة القرار ما لم تُحدد «Cutoff time» لكل مثال.")
if at_least("research"):
    researcher_note(["أبلغ عن فضاء البحث والميزانية والبذرة ومخطط التحقق لأي نتيجة AutoML.",
                     "استخدم Nested CV أو مجموعة اختبار محجوزة لتقدير أداء «الإجراء» كاملًا.",
                     "قارن مع خطوط أساس قوية؛ كثير من المكاسب المعلنة تتلاشى أمام نموذج مضبوط جيدًا."])
real_world(["هل كل الخصائص متاحة وقت القرار؟", "هل مخطط CV يناسب البيانات (Group/Time)؟", "هل يحتاج المستخدم نموذجًا قابلًا للتفسير؟",
            "هل المكسب يستحق تكلفة الصيانة؟"])

page_footer("automl",
            takeaways=["AutoML = فضاء بحث + استراتيجية + ميزانية + تحقق.", "لا يمنع Leakage ولا يختار المقياس الصحيح.",
                       "قيّم على بيانات محجوزة وقارن بخط أساس."],
            mistakes=["الوثوق بأفضل CV دون اختبار نهائي.", "ترك الأداة تختار المقياس.", "Random CV لبيانات زمنية/مجمّعة."])
