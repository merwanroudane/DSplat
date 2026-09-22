import importlib.util

import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.dataset_viewer import dataset_card, dataset_picker, overview_metrics
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import heatmap, plot
from utils.profiling import categorical_summary, column_summary, iqr_outlier_counts, numeric_summary, top_correlations
from utils.types import numeric_columns

page_header("automated_eda")

st.markdown("## ماذا تفعل أدوات الاستكشاف الآلي؟")
comparison_table([
    {"القدرة": "Schema inference", "المعنى": "استنتاج الأعمدة والأنواع", "الحدود": "dtype ≠ المعنى"},
    {"القدرة": "Semantic type detection", "المعنى": "بريد، معرّف، تاريخ كنص، أرقام كنص", "الحدود": "رموز رقمية تُفهم كأرقام"},
    {"القدرة": "Distribution detection", "المعنى": "ملخصات، التواء، رسوم تلقائية", "الحدود": "لا تعرف ما هو «طبيعي» للمجال"},
    {"القدرة": "Relationship discovery", "المعنى": "ارتباطات، تفاعلات مرشحة", "الحدود": "ارتباطات زائفة كثيرة"},
    {"القدرة": "Cardinality analysis", "المعنى": "فئات كثيرة/نادرة، أعمدة ثابتة", "الحدود": "لا تعرف أي الفئات خطأ إملائي"},
    {"القدرة": "Anomaly & pattern discovery", "المعنى": "قيم متطرفة، أنماط فقد", "الحدود": "لا تميّز الخطأ من الحقيقي النادر"},
    {"القدرة": "Feature suggestions", "المعنى": "تحويلات، ترميزات مقترحة", "الحدود": "قد تقترح خصائص مسربة"},
    {"القدرة": "Metadata discovery", "المعنى": "أحجام، ذاكرة، مفاتيح مرشحة", "الحدود": "لا تعرف المالك أو التعريف الرسمي"},
])

st.markdown("## المحلل المدمج · Built-in profiler")
st.caption("مكتوب داخل المنصة (utils/profiling.py) بلا اعتماديات ثقيلة، ويعمل على Streamlit Community Cloud.")
name, df = dataset_picker("aeda_ds", default="survey")
dataset_card(name)
overview_metrics(df)
t1, t2, t3, t4, t5 = st.tabs(["الأعمدة والأنواع الدلالية", "الرقمية", "الفئوية", "القيم المتطرفة", "العلاقات"])
with t1:
    cs = column_summary(df)
    st.dataframe(cs, hide_index=True, column_config={"missing_%": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                                  format="%.1f%%")})
    mismatch = cs[cs["semantic_type"].isin(["numeric-as-text", "date-as-text"])]
    if len(mismatch):
        st.warning("أعمدة نوعها المخزن لا يطابق معناها: " + ", ".join(mismatch["column"]), icon=":material/warning:")
with t2:
    ns = numeric_summary(df)
    if ns.empty:
        st.info("لا أعمدة رقمية.")
    else:
        st.dataframe(ns)
    nums = [c for c in numeric_columns(df) if df[c].nunique() > 2]
    if nums:
        sel = st.selectbox("توزيع", nums, key="aeda_num")
        fig = go.Figure(go.Histogram(x=df[sel], nbinsx=50, marker_color=PALETTE["coral"]))
        fig.update_layout(height=280)
        plot(fig)
with t3:
    cat = categorical_summary(df)
    if cat.empty:
        st.info("لا أعمدة فئوية.")
    else:
        st.dataframe(cat, hide_index=True)
with t4:
    oc = iqr_outlier_counts(df)
    if oc.empty:
        st.info("لا أعمدة رقمية كافية.")
    else:
        st.dataframe(oc, hide_index=True)
    st.caption("عدد القيم خارج 1.5×IQR لكل عمود — مؤشر للفحص، لا قائمة حذف.")
with t5:
    tc = top_correlations(df)
    st.dataframe(tc, hide_index=True)
    cols = [c for c in numeric_columns(df) if df[c].nunique() > 2][:12]
    if len(cols) >= 2:
        plot(heatmap(df[cols].corr(method="spearman"), "Spearman"), height=460)

why("استخدم الاستكشاف الآلي كنقطة بداية لقائمة أسئلة، لا كقائمة إجابات.",
    "التقرير الآلي يعرض كل شيء بنفس الوزن؛ مهمتك ترتيب الأولويات حسب السؤال ومعرفة المجال.")

st.markdown("## الأدوات الخارجية (اختيارية)")
st.caption("الحالة كما تحققنا منها في سبتمبر 2026 — تحقق دائمًا من صفحة PyPI قبل الاعتماد (انظر docs/research_notes.md).")
comparison_table([
    {"الأداة": "fg-data-profiling (سابقًا ydata-profiling / pandas-profiling)", "الحالة": "أُعيدت التسمية في أبريل 2026؛ الاسم القديم متوقف عن التحديث",
     "الاستيراد": "from data_profiling import ProfileReport", "ملاحظة": "تقرير HTML شامل؛ ثقيل نسبيًا"},
    {"الأداة": "Sweetviz", "الحالة": "آخر إصدار 2.3.x؛ نشاط محدود", "الاستيراد": "import sweetviz", "ملاحظة": "مقارنة مجموعتين (train/test) بصريًا"},
    {"الأداة": "skimpy", "الحالة": "خفيفة وتعمل في الطرفية", "الاستيراد": "from skimpy import skim", "ملاحظة": "ملخص سريع شبيه بـskimr في R"},
    {"الأداة": "missingno", "الحالة": "وضع صيانة (إصلاحات فقط)", "الاستيراد": "import missingno", "ملاحظة": "رسوم أنماط الفقد"},
    {"الأداة": "D-Tale", "الحالة": "واجهة ويب تفاعلية", "الاستيراد": "import dtale", "ملاحظة": "مفهوم: جدول تفاعلي مع رسوم؛ غير مناسب للنشر العام"},
])
avail = {pkg: importlib.util.find_spec(mod) is not None for pkg, mod in
         [("fg-data-profiling", "data_profiling"), ("ydata-profiling", "ydata_profiling"), ("sweetviz", "sweetviz"),
          ("skimpy", "skimpy"), ("missingno", "missingno")]}
st.markdown("**المثبت في هذه البيئة:** " + " · ".join(f"{k}: {'✅' if v else '—'}" for k, v in avail.items()))
st.code("# pip install fg-data-profiling\nfrom data_profiling import ProfileReport\n"
        'ProfileReport(df, title="Survey profile", minimal=True).to_file("profile.html")', language="python")
st.markdown("لم نجعل هذه الأدوات اعتماديات إلزامية: المنصة لا تعتمد على أداة واحدة، وبعضها ثقيل على الخوادم المجانية. "
            "المحلل المدمج يغطي الأساسيات، والتقرير الكامل القابل للتحميل في:")
page_link("auto_report")

if at_least("advanced"):
    st.markdown("## متقدم: Profiling على بيانات كبيرة")
    st.markdown("مع ملايين الصفوف: استخدم عينة ممثلة للرسوم، وحساب الملخصات على كامل البيانات بمحركات عمودية "
                "(Polars/DuckDB)، ووضع `minimal=True` في أدوات التقرير لتخطي الحسابات المكلفة (الارتباطات الكاملة، التفاعلات).")
if at_least("research"):
    researcher_note(["احفظ تقرير Profiling مع كل إصدار بيانات كجزء من قابلية إعادة الإنتاج.",
                     "الأداة التي تتوقف صيانتها تصبح دينًا تقنيًا؛ فضّل الأدوات النشطة أو كودًا بسيطًا تملكه."])
real_world(["هل الأداة تعمل محليًا دون إرسال البيانات؟", "هل ترخيصها مناسب؟", "هل تحتمل حجم بياناتك؟",
            "من سيقرأ التقرير ويحوله إلى قرارات؟"])

page_footer("automated_eda",
            takeaways=["الاستكشاف الآلي يسرّع الاكتشاف لا الفهم.", "النوع الدلالي أهم من dtype.",
                       "تحقق من حالة صيانة الأدوات قبل الاعتماد عليها.", "المحلل المدمج يكفي للأساسيات."],
            mistakes=["اعتبار التقرير الآلي تحليلًا نهائيًا.", "الاعتماد على مكتبة متوقفة الصيانة.", "رفع بيانات حساسة لأدوات سحابية."])
