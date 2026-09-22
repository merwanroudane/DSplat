from datetime import date

import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from components.dataset_viewer import dataset_card, dataset_picker
from config import APP_AUTHOR_EN, APP_NAME_EN
from core.page import footer, page_header
from core.state import mark_lab
from utils.profiling import detect_issues, overview, top_correlations
from utils.types import numeric_columns

page_header("capstone")
mark_lab("capstone")
st.markdown("المشروع النهائي: من **البيانات الخام** إلى **تقرير تحليلي**. لكل مرحلة: إرشاد، دليل آلي من البيانات، وملاحظاتك. "
            "في النهاية يُجمع كل شيء في تقرير Markdown قابل للتحميل.")

name, df = dataset_picker("cap_ds", default="credit")
dataset_card(name)
ov = overview(df)
issues = detect_issues(df)
nums = numeric_columns(df)
target = st.selectbox("متغير الهدف (إن وُجد)", ["(none)"] + list(df.columns),
                      index=(list(df.columns).index("default") + 1) if "default" in df.columns else 0, key="cap_target")

STAGES = [
    ("Problem definition", "حدد السؤال والقرار ومعيار النجاح ومن سيستخدم النتيجة.",
     lambda: "—"),
    ("Data import", "صف المصدر والصيغة والترميز وأي أسطر معطوبة.",
     lambda: f"{ov['rows']:,} rows × {ov['columns']} columns, {ov['memory_mb']:.2f} MB."),
    ("Understanding", "وحدة التحليل، المتغيرات وأدوارها (معرّف/خصائص/هدف)، التصميم (مقطعي/زمني/Panel).",
     lambda: f"{ov['numeric']} numeric and {ov['categorical']} categorical columns."),
    ("Validation", "اكتب قواعد التحقق الأساسية ونتائجها.",
     lambda: f"{sum(1 for i in issues if i.kind == 'detected')} detected rule-type issues."),
    ("Quality", "قيّم الأبعاد: الاكتمال، التفرد، الصلاحية، الاتساق.",
     lambda: f"Missing cells {ov['missing_pct']:.2f}%, duplicate rows {ov['duplicate_rows']}."),
    ("Cleaning", "كل خطوة: ماذا ولماذا وكم صفًا تأثر.",
     lambda: "; ".join(sorted({i.issue.split(' ')[0] + ' ' + i.column for i in issues[:6]})) or "No issues detected."),
    ("EDA", "أهم التوزيعات والعلاقات والمفاجآت.",
     lambda: ", ".join(f"{r.var_1}~{r.var_2} ρ={r.r:.2f}" for r in top_correlations(df, 3).itertuples()) or "—"),
    ("Statistical analysis", "الاختبارات المناسبة مع حجم الأثر وفترات الثقة والافتراضات.",
     lambda: "Use the Test Selector in the Statistical Tests module."),
    ("Feature engineering", "خصائص مبنية على المجال، متاحة وقت القرار.",
     lambda: "—"),
    ("Modeling", "خط أساس ثم نموذج أقوى؛ مخطط تقسيم يطابق الاستخدام.",
     lambda: baseline_text()),
    ("Evaluation", "مقاييس مناسبة للتكلفة، فحص التسرب، تحليل الأخطاء.",
     lambda: "Check the leakage checklist; compare to the baseline above."),
    ("Interpretation", "ماذا تعني النتائج؟ ما حدودها؟ ارتباطي أم سببي؟",
     lambda: "—"),
    ("Visualization", "رسوم صادقة تجيب كل منها عن سؤال واحد.",
     lambda: "—"),
    ("Final report", "ملخص تنفيذي، منهجية، نتائج، حدود، توصيات، وملحق قرارات.",
     lambda: "Assembled below."),
]


@st.cache_data(show_spinner="حساب خط الأساس…")
def baseline_scores(data: pd.DataFrame, tgt: str) -> str:
    X = data[[c for c in numeric_columns(data) if c != tgt]].copy()
    y = data[tgt]
    ok = y.notna()
    X, y = X[ok], y[ok]
    if X.shape[1] == 0:
        return "No numeric features."
    if y.nunique() <= 10:
        dummy = cross_val_score(DummyClassifier(strategy="prior"), X, y, cv=5, scoring="roc_auc" if y.nunique() == 2 else "accuracy").mean()
        model = cross_val_score(HistGradientBoostingClassifier(random_state=0), X, y, cv=5,
                                scoring="roc_auc" if y.nunique() == 2 else "accuracy").mean()
        metric = "ROC-AUC" if y.nunique() == 2 else "accuracy"
    else:
        dummy = -cross_val_score(DummyRegressor(), X, y, cv=5, scoring="neg_mean_absolute_error").mean()
        model = -cross_val_score(HistGradientBoostingRegressor(random_state=0), X, y, cv=5, scoring="neg_mean_absolute_error").mean()
        metric = "MAE"
    return f"5-fold CV {metric}: dummy baseline = {dummy:.3f}, gradient boosting (numeric features only) = {model:.3f}."


def baseline_text() -> str:
    if target == "(none)":
        return "No target selected."
    return baseline_scores(df, target)


st.session_state.setdefault("cap_done", {})
done_count = sum(bool(st.session_state.get(f"cap_chk_{i}")) for i in range(len(STAGES)))
st.progress(done_count / len(STAGES), text=f"{done_count} من {len(STAGES)} مراحل مكتملة")
notes = {}
for i, (title, guide, evidence) in enumerate(STAGES):
    with st.expander(f"{i + 1}. {title}", icon=":material/check_circle:" if st.session_state.get(f"cap_chk_{i}") else
                     ":material/radio_button_unchecked:"):
        st.markdown(f"**الإرشاد:** {guide}")
        ev = evidence()
        st.markdown(f"**دليل آلي من البيانات:** `{ev}`")
        notes[title] = st.text_area("ملاحظاتك وقراراتك", key=f"cap_note_{i}", height=100)
        st.checkbox("أنهيت هذه المرحلة", key=f"cap_chk_{i}")
        notes[title + "__evidence"] = ev

lines = [f"# Capstone report — {name}", f"_Prepared {date.today().isoformat()} with {APP_NAME_EN} ({APP_AUTHOR_EN})._", ""]
for i, (title, _g, _e) in enumerate(STAGES):
    lines.append(f"## {i + 1}. {title}")
    lines.append(f"*Automated evidence:* {notes[title + '__evidence']}")
    lines.append(notes[title].strip() or "_(not written yet)_")
    lines.append("")
log = st.session_state.get("decision_log", [])
if log:
    lines.append("## Appendix — decision log")
    lines += [f"- [{d['time']}] {d['source']} · {d['item']}: **{d['decision']}** — {d['reason']}" for d in log]
report = "\n".join(lines)
st.download_button("حمّل التقرير النهائي Markdown", report.encode("utf-8"), f"capstone_{name}.md", "text/markdown", type="primary",
                   icon=":material/download:")
with st.expander("معاينة التقرير", icon=":material/description:"):
    with st.container(key="ltr-capstone"):
        st.markdown(report)
st.markdown("### معايير التقييم (Rubric)")
st.dataframe(pd.DataFrame({
    "المعيار": ["وضوح السؤال", "جودة البيانات موثقة", "قرارات تنظيف مبررة", "EDA مرتبط بالسؤال", "اختبار/نموذج مناسب",
                "لا تسرّب", "تفسير حذر مع حدود", "رسوم صادقة", "قابلية إعادة الإنتاج"],
    "الوزن": [10, 10, 15, 10, 15, 10, 15, 5, 10]}), hide_index=True)
footer()
