import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import why
from components.dataset_viewer import dataset_card
from core.page import footer, page_header
from core.state import log_decision
from core.theme import PALETTE
from utils.cleaning import CUSTOMER_STEPS, apply_customer_steps, compare_to_truth
from utils.datasets import customers_clean, customers_raw
from utils.plotting import plot
from utils.profiling import detect_issues
from utils.quality import DEFAULT_WEIGHTS, quality_report, weighted_score
from utils.validation import CUSTOMER_RULES

page_header("cleaning_lab")
st.markdown("نظّف بيانات التحدي بنفسك: فعّل الخطوات التي تقتنع بها، وشاهد الأثر **قبل/بعد** على الجودة والتوزيعات، "
            "وقارن النتيجة بالحقيقة المرجعية (نملكها لأن البيانات اصطناعية).")
dataset_card("customers_raw")
raw = customers_raw()

st.markdown("### 1. اختر الخطوات")


def _select_all() -> None:
    for step_key, *_ in CUSTOMER_STEPS:
        st.session_state[f"cl_step_{step_key}"] = True


st.button("فعّل كل الخطوات (الحل المرجعي)", key="cl_all", icon=":material/done_all:", on_click=_select_all)
chosen = []
for key, label, why_txt, _fn, code in CUSTOMER_STEPS:
    c1, c2 = st.columns([3, 2])
    if c1.checkbox(label, key=f"cl_step_{key}", help=why_txt):
        chosen.append(key)
    c2.caption(why_txt)

clean, log = apply_customer_steps(raw, chosen)
if log:
    st.dataframe(pd.DataFrame(log), hide_index=True)

st.markdown("### 2. قبل مقابل بعد")


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


q_before = quality_report(raw, CUSTOMER_RULES, "customer_id")
q_after = quality_report(clean, CUSTOMER_RULES, "customer_id")
with st.container(horizontal=True):
    st.metric("الصفوف", len(clean), len(clean) - len(raw), border=True)
    st.metric("صفوف مكررة", int(clean.duplicated().sum()), int(clean.duplicated().sum() - raw.duplicated().sum()),
              delta_color="inverse", border=True)
    n_after, n_before = len(detect_issues(clean)), len(detect_issues(raw))
    st.metric("المشكلات المكتشفة", n_after, n_after - n_before, delta_color="inverse", border=True)
    s_b, s_a = weighted_score(q_before, DEFAULT_WEIGHTS), weighted_score(q_after, DEFAULT_WEIGHTS)
    st.metric("درجة الجودة", f"{100 * s_a:.1f}%", f"{100 * (s_a - s_b):+.1f}", border=True)
t1, t2, t3, t4 = st.tabs(["أبعاد الجودة", "الأنواع والفقد", "التوزيعات", "الملخصات"])
with t1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(q_before), y=[100 * v for v in q_before.values()], name="before", marker_color=PALETTE["softred"]))
    fig.add_trace(go.Bar(x=list(q_after), y=[100 * v for v in q_after.values()], name="after", marker_color=PALETTE["teal"]))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 105], title="%"), height=340)
    plot(fig)
with t2:
    comp = pd.DataFrame({"dtype before": raw.dtypes.astype(str), "missing before": raw.isna().sum()}).join(
        pd.DataFrame({"dtype after": clean.dtypes.astype(str), "missing after": clean.isna().sum()}), how="outer")
    st.dataframe(comp)
    st.caption("لاحظ: الفقد **يزداد** بعد التنظيف — لأن القيم الدلالية والمستحيلة صارت NaN صريحة. هذا تحسن لا تدهور: "
               "المفقود أصبح مرئيًا وقابلًا للمعالجة المبررة.")
with t3:
    col = st.selectbox("المتغير", ["annual_income", "age", "height_cm", "monthly_spend", "satisfaction"], key="cl_dist")
    c1, c2 = st.columns(2)
    for cc, data, title, color in ((c1, raw, "قبل", PALETTE["softred"]), (c2, clean, "بعد", PALETTE["teal"])):
        with cc:
            x = num(data[col]).dropna()
            use_log = col == "annual_income"
            fig = go.Figure(go.Histogram(x=np.log10(x[x > 0]) if use_log else x, nbinsx=50, marker_color=color))
            fig.update_layout(title=f"{title}: {'log10 ' if use_log else ''}{col}", height=300)
            plot(fig)
with t4:
    cols = ["age", "annual_income", "monthly_spend", "num_orders", "height_cm", "satisfaction"]
    before = pd.DataFrame({c: num(raw[c]) for c in cols}).describe().T[["count", "mean", "50%", "std", "min", "max"]]
    after = pd.DataFrame({c: num(clean[c]) for c in cols}).describe().T[["count", "mean", "50%", "std", "min", "max"]]
    st.dataframe(pd.concat({"before": before, "after": after}, axis=1).round(2))

st.markdown("### 3. المقارنة مع الحقيقة المرجعية")
acc = compare_to_truth(clean.drop_duplicates("customer_id"), customers_clean(), "customer_id",
                       ["age", "gender", "country", "annual_income", "membership", "num_orders", "height_cm", "satisfaction"])
fig = go.Figure(go.Bar(x=acc["matches_truth_%"], y=acc["column"], orientation="h", marker_color=PALETTE["purple"],
                       text=acc["matches_truth_%"].map(lambda v: f"{v:.1f}%"), textposition="auto"))
fig.update_layout(xaxis=dict(range=[0, 100], title="% of rows matching the truth"), height=340)
plot(fig)
why("لا تتوقع 100%: القيم التي كانت مفقودة أو مستحيلة في الخام لا يمكن استرجاعها بالتنظيف.",
    "التنظيف الجيد يجعل الأخطاء **مرئية** (NaN) بدل أن تبقى أرقامًا خاطئة. استرجاعها يحتاج المصدر أو تعويضًا مبررًا.")

st.markdown("### 4. صدّر")
code = "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv('customers_raw.csv')\n\n" + "\n\n".join(
    f"# {label}\n{snippet}" for key, label, _w, _f, snippet in CUSTOMER_STEPS if key in chosen)
c1, c2, c3 = st.columns(3)
c1.download_button("البيانات المنظفة CSV", clean.to_csv(index=False).encode("utf-8-sig"), "customers_cleaned.csv", "text/csv",
                   icon=":material/download:")
c2.download_button("الكود المكافئ .py", code.encode("utf-8"), "cleaning_steps.py", "text/x-python", icon=":material/code:")
if c3.button("سجّل خطواتي في سجل القرارات", key="cl_logbtn", icon=":material/bookmark:", disabled=not chosen):
    for row in log:
        log_decision("cleaning_lab", "customers_raw", row["الخطوة"], row["لماذا؟"])
    st.toast(f"سُجّلت {len(log)} خطوة.")
with st.expander("الكود المكافئ للخطوات المختارة", icon=":material/code:"):
    st.code(code, language="python")
footer()
