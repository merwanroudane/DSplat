import pandas as pd
import plotly.express as px
import streamlit as st

from components.callouts import why
from components.dataset_viewer import UPLOADED, dataset_card, dataset_picker, overview_metrics
from config import MAX_UPLOAD_MB
from core.page import footer, page_header, page_link
from core.theme import SEQUENCE
from utils.data_loader import initial_warnings, load_bytes
from utils.plotting import heatmap, plot, sample_for_plot
from utils.profiling import (categorical_summary, column_summary, detect_issues, iqr_outlier_counts, issues_frame,
                             numeric_summary, recommended_next_steps)
from utils.types import numeric_columns
from utils.validation import CUSTOMER_RULES, run_rules

page_header("dataset_explorer")

with st.expander("ارفع ملفًا (CSV / Excel / JSON)", icon=":material/upload_file:",
                 expanded=st.session_state.get("uploaded_df") is None):
    up = st.file_uploader(f"حتى {MAX_UPLOAD_MB} MB — يبقى في جلستك فقط", type=["csv", "txt", "xlsx", "xls", "json"], key="dx_up")
    if up is not None:
        res = load_bytes(up.getvalue(), up.name)
        for e in res.errors:
            st.error(e)
        for w in res.warnings:
            st.warning(w)
        if res.ok:
            st.session_state["uploaded_df"] = res.df
            st.session_state["uploaded_name"] = up.name
            st.success(f"حُمّل «{up.name}»: {res.df.shape[0]:,} × {res.df.shape[1]}. اختره من القائمة أدناه.")

name, df = dataset_picker("dx_ds", default=UPLOADED if st.session_state.get("uploaded_df") is not None else "customers_raw")
dataset_card(name)
overview_metrics(df)

tabs = st.tabs(["معاينة", "الأنواع", "الفقد", "التكرار", "رقمي", "فئوي", "التعدد", "القيم المتطرفة", "الارتباطات",
                "التحقق", "التحذيرات والتوصيات"])
with tabs[0]:
    st.dataframe(df.head(100))
with tabs[1]:
    st.dataframe(column_summary(df), hide_index=True)
with tabs[2]:
    miss = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        fig = px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x": "% missing", "y": ""},
                     color_discrete_sequence=SEQUENCE)
        fig.update_layout(height=max(250, 28 * len(miss)))
        plot(fig)
    else:
        st.success("لا قيم مفقودة صريحة (تحقق من القيم الدلالية في تبويب «الأنواع»).")
with tabs[3]:
    st.metric("صفوف مكررة تمامًا", int(df.duplicated().sum()))
    key_col = st.selectbox("افحص تكرار عمود كمفتاح", list(df.columns), key="dx_key")
    st.metric(f"قيم مكررة في {key_col}", int(df[key_col].dropna().duplicated().sum()))
with tabs[4]:
    ns = numeric_summary(df)
    if ns.empty:
        st.info("لا أعمدة رقمية.")
    else:
        st.dataframe(ns)
        col = st.selectbox("توزيع", list(ns.index), key="dx_num")
        fig = px.histogram(df, x=col, nbins=50, marginal="box", color_discrete_sequence=SEQUENCE)
        fig.update_layout(height=360)
        plot(fig)
with tabs[5]:
    cs = categorical_summary(df)
    if cs.empty:
        st.info("لا أعمدة فئوية.")
    else:
        st.dataframe(cs, hide_index=True)
with tabs[6]:
    card = pd.DataFrame({"unique": df.nunique(), "unique_ratio": (df.nunique() / max(1, len(df))).round(3)})
    card["hint"] = pd.cut(card["unique_ratio"], [-0.01, 0.0001, 0.05, 0.5, 0.95, 1.01],
                          labels=["ثابت", "فئوي منخفض", "متوسط", "عالي التعدد", "شبه فريد (معرّف؟)"])
    st.dataframe(card.sort_values("unique", ascending=False))
with tabs[7]:
    oc = iqr_outlier_counts(df)
    if oc.empty:
        st.info("لا أعمدة رقمية كافية.")
    else:
        st.dataframe(oc, hide_index=True)
        st.caption("مرشحة للفحص فقط — انظر أداة القرار في وحدة القيم الشاذة.")
with tabs[8]:
    nums = [c for c in numeric_columns(df) if df[c].nunique() > 2][:15]
    if len(nums) >= 2:
        plot(heatmap(df[nums].corr(method="spearman"), "Spearman correlation"), height=480)
        c1, c2 = st.columns(2)
        xa = c1.selectbox("X", nums, key="dx_x")
        ya = c2.selectbox("Y", nums, index=1, key="dx_y")
        d, _ = sample_for_plot(df[[xa, ya]].dropna())
        plot(px.scatter(d, x=xa, y=ya, opacity=0.5, color_discrete_sequence=SEQUENCE), height=360)
    else:
        st.info("تحتاج عمودين رقميين على الأقل.")
with tabs[9]:
    if name in ("customers_raw", "customers_clean"):
        st.dataframe(run_rules(df, CUSTOMER_RULES), hide_index=True)
    else:
        st.info("حزمة قواعد جاهزة متاحة لبيانات العملاء. لبياناتك استخدم «باني القواعد» في وحدة التحقق من صحة البيانات.")
        page_link("data_validation")
with tabs[10]:
    for w in initial_warnings(df):
        st.warning(w, icon=":material/warning:")
    f = issues_frame(detect_issues(df))
    st.dataframe(f[["column", "issue", "kind_ar", "severity_ar", "evidence", "action", "why"]], hide_index=True, height=360)
    st.markdown("### الخطوات التالية المقترحة")
    for i, s in enumerate(recommended_next_steps(df, detect_issues(df)), 1):
        st.markdown(f"{i}. {s}")
why("ابدأ من تبويب «الأنواع» ثم «التحذيرات» قبل أي رسم.",
    "الأنواع الخاطئة والقيم الدلالية تفسد كل الإحصاءات والرسوم اللاحقة بصمت.")
st.download_button("حمّل ملخص الأعمدة CSV", column_summary(df).to_csv(index=False).encode("utf-8-sig"), "column_summary.csv",
                   "text/csv", icon=":material/download:")
with st.container(horizontal=True):
    page_link("auto_report", label="ولّد تقريرًا كاملًا قابلًا للتحميل")
    page_link("eda_lab", label="استكشف بصريًا في مختبر EDA")
footer()
