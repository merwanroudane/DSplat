import pandas as pd
import streamlit as st

from components.callouts import why
from components.dataset_viewer import dataset_card, dataset_picker
from components.diagrams import mermaid
from core.page import footer, page_header
from core.state import log_decision
from utils.pipeline import OPTIONS, PipelineConfig, _cat, _num, generate_pandas_code, generate_sklearn_code, mermaid_diagram, run_pipeline

page_header("pipeline_builder")
st.markdown("ابنِ خط معالجة من ست مراحل، اعرض مخططه، شغّله، وحمّل كود Python المكافئ (نسخة pandas تعليمية + نسخة "
            "scikit-learn آمنة من التسرب).")

name, df = dataset_picker("pb_ds", default="customers_clean",
                          allowed=["customers_clean", "survey", "credit", "students", "panel", "wine", "iris"])
dataset_card(name)
if name == "survey":
    df["work_hours"] = df["work_hours"].replace(-99, float("nan"))
    st.caption("حُوّلت Sentinel -99 في work_hours إلى NaN قبل البدء.")

with st.form("pb_form"):
    st.markdown("### إعداد المراحل")
    c1, c2, c3 = st.columns(3)
    dup = c1.selectbox("1. التكرار", list(OPTIONS["duplicates"]), format_func=OPTIONS["duplicates"].get, index=1)
    key = c1.selectbox("المفتاح (عند اختيار حذف تكرار المفتاح)", ["(none)"] + list(df.columns))
    mnum = c2.selectbox("2. الفقد الرقمي", list(OPTIONS["missing_num"]), format_func=OPTIONS["missing_num"].get, index=4)
    mcat = c2.selectbox("2. الفقد الفئوي", list(OPTIONS["missing_cat"]), format_func=OPTIONS["missing_cat"].get, index=2)
    out = c3.selectbox("3. القيم المتطرفة", list(OPTIONS["outliers"]), format_func=OPTIONS["outliers"].get, index=1)
    c4, c5, c6 = st.columns(3)
    tr = c4.selectbox("4. التحويل", list(OPTIONS["transform"]), format_func=OPTIONS["transform"].get, index=1)
    sc = c5.selectbox("5. القياس", list(OPTIONS["scaling"]), format_func=OPTIONS["scaling"].get, index=1)
    enc = c6.selectbox("6. الترميز", list(OPTIONS["encoding"]), format_func=OPTIONS["encoding"].get, index=1)
    excl_default = [c for c in df.columns if c.endswith("_id") or c in ("churned", "default", "result")]
    exclude = st.multiselect("أعمدة تُستبعد من المعالجة (المعرّفات والهدف)", list(df.columns), default=excl_default)
    target = st.selectbox("الهدف (لنسخة scikit-learn)", ["(none)"] + list(df.columns),
                          index=(list(df.columns).index(excl_default[-1]) + 1) if excl_default else 0)
    submitted = st.form_submit_button("ابنِ Pipeline", type="primary", icon=":material/account_tree:")

cfg = PipelineConfig(dup, None if key == "(none)" else key, mnum, mcat, out, tr, sc, enc, tuple(exclude))
st.markdown("### المخطط")
mermaid(mermaid_diagram(cfg))

if submitted:
    st.session_state["pb_ran"] = True
if st.session_state.get("pb_ran"):
    result, log = run_pipeline(df, cfg)
    st.markdown("### التشغيل ومخرجات كل مرحلة")
    EXPLAIN = {
        "1. Duplicates": "يزيل السجلات المكررة قبل أي إحصاء حتى لا تُحسب مرتين.",
        "2. Missing values": "يعوّض أو يحذف المفقود؛ لاحظ أن الإحصاءات (الوسيط/المتوسط) تُحسب هنا على كل البيانات.",
        "3. Outliers": "يعلّم أو يقص أو يحذف القيم خارج 1.5×IQR؛ الحذف أقوى قرار وأخطره.",
        "4. Transformation": "يغيّر شكل التوزيع (log/Yeo-Johnson) لتقليل الالتواء.",
        "5. Scaling": "يوحّد المقاييس للخوارزميات المعتمدة على المسافة أو التنظيم.",
        "6. Encoding": "يحوّل الفئات إلى أرقام؛ One-hot يضيف عمودًا لكل فئة.",
    }
    log_df = pd.DataFrame(log)
    log_df["لماذا هذه المرحلة؟"] = log_df["stage"].map(EXPLAIN)
    st.dataframe(log_df, hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"**قبل** — {df.shape[0]} × {df.shape[1]}")
    c1.dataframe(df.head(8))
    c2.markdown(f"**بعد** — {result.shape[0]} × {result.shape[1]}")
    c2.dataframe(result.head(8))
    if cfg.outliers == "remove_iqr":
        st.warning("اخترت حذف الصفوف الشاذة: راجع كم صفًا حُذف ولماذا. في بيانات ملتوية قد تحذف أهم الحالات.", icon=":material/warning:")
    num, cat = _num(df, cfg), _cat(df, cfg)
    t1, t2 = st.tabs(["كود pandas (تسلسلي)", "كود scikit-learn (آمن من التسرب)"])
    code_pd = generate_pandas_code(cfg, name, num, cat)
    code_sk = generate_sklearn_code(cfg, num, cat, None if target == "(none)" else target)
    with t1:
        st.code(code_pd, language="python")
    with t2:
        st.code(code_sk, language="python")
    why("في النمذجة استخدم نسخة scikit-learn.",
        "النسخة التسلسلية تحسب الوسيط والمقياس على كامل البيانات؛ Pipeline + ColumnTransformer يتعلمان من التدريب فقط "
        "ويطبقان نفس المعالجة على الاختبار والإنتاج.")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("البيانات بعد المعالجة", result.to_csv(index=False).encode("utf-8-sig"), f"{name}_processed.csv",
                       "text/csv", icon=":material/download:")
    c2.download_button("كود pandas", code_pd.encode("utf-8"), "pipeline_pandas.py", "text/x-python", icon=":material/code:")
    c3.download_button("كود scikit-learn", code_sk.encode("utf-8"), "pipeline_sklearn.py", "text/x-python", icon=":material/code:")
    if c4.button("سجّل الإعداد", key="pb_log", icon=":material/bookmark:"):
        log_decision("pipeline_builder", name, "pipeline config", str(cfg))
        st.toast("سُجّل.")
else:
    st.info("اضبط المراحل ثم اضغط «ابنِ Pipeline».", icon=":material/info:")
footer()
