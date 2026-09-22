import pandas as pd
import plotly.express as px
import streamlit as st

from components.callouts import domain, real_world, researcher_note
from components.cards import comparison_table
from components.dataset_viewer import dataset_card, dataset_picker
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import SEQUENCE
from utils.datasets import daily_sales, panel
from utils.plotting import plot
from utils.types import infer_semantic_type, is_numeric

page_header("variable_types")

st.markdown("## شجرة أنواع المتغيرات · Variable type tree")
mermaid("""
flowchart TB
  V[Variable] --> N[Numerical]
  V --> C[Categorical]
  V --> O[Other types]
  N --> NC[Continuous<br/>height, income]
  N --> ND[Discrete<br/>number of orders]
  C --> CN[Nominal<br/>country, color]
  C --> CO[Ordinal<br/>education, Likert]
  C --> CB[Binary<br/>churned yes/no]
  O --> OD[Date/Time]
  O --> OT[Text]
  O --> OG[Geospatial]
  O --> OM[Image / Audio / Video]
  O --> OS[Sequence]
""")

comparison_table([
    {"النوع": "Continuous", "الوصف": "أي قيمة ضمن مدى", "عمليات ذات معنى": "متوسط، انحراف، ارتباط", "مثال": "الطول 172.4 سم"},
    {"النوع": "Discrete", "الوصف": "قيم معدودة (غالبًا صحيحة)", "عمليات ذات معنى": "متوسط، عدّ، توزيع Poisson", "مثال": "عدد الطلبات"},
    {"النوع": "Nominal", "الوصف": "فئات بلا ترتيب", "عمليات ذات معنى": "تكرار، منوال", "مثال": "الدولة"},
    {"النوع": "Ordinal", "الوصف": "فئات مرتبة، مسافات غير مضمونة", "عمليات ذات معنى": "وسيط، مئينات، رتب", "مثال": "مستوى التعليم"},
    {"النوع": "Binary", "الوصف": "فئتان", "عمليات ذات معنى": "نسبة، Odds", "مثال": "غادر/لم يغادر"},
    {"النوع": "Boolean", "الوصف": "True/False منطقي", "عمليات ذات معنى": "نسبة True", "مثال": "email_valid"},
    {"النوع": "Date/Time", "الوصف": "نقاط أو فترات زمنية", "عمليات ذات معنى": "فروق، استخراج مكونات", "مثال": "تاريخ التسجيل"},
    {"النوع": "Text", "الوصف": "نص حر", "عمليات ذات معنى": "تمثيل رقمي (TF-IDF، Embeddings)", "مثال": "مراجعة منتج"},
    {"النوع": "Geospatial", "الوصف": "إحداثيات أو أشكال", "عمليات ذات معنى": "مسافات، تجميع مكاني", "مثال": "خط العرض والطول"},
])
domain("الرقم لا يجعل المتغير رقميًا: الرمز البريدي 20000 ورقم الهاتف ورمز المنتج **تسميات Nominal**. "
       "اسأل: هل للجمع والمتوسط معنى؟ إن لم يكن، فالمتغير ليس كميًا.")

st.markdown("## حسب البنية")
comparison_table([
    {"البنية": "Structured", "الوصف": "جداول بمخطط ثابت", "أمثلة": "CSV، SQL", "أدوات": "pandas, SQL"},
    {"البنية": "Semi-structured", "الوصف": "بنية مرنة بمفاتيح وتداخل", "أمثلة": "JSON، XML، سجلات Logs", "أدوات": "json_normalize"},
    {"البنية": "Unstructured", "الوصف": "بلا مخطط جدولي", "أمثلة": "نصوص، صور، صوت، فيديو", "أدوات": "NLP، رؤية حاسوبية"},
])
with st.expander("مثال: من JSON متداخل إلى جدول", icon=":material/data_object:"):
    records = [{"id": 1, "customer": {"name": "A", "city": "Rabat"}, "items": 3},
               {"id": 2, "customer": {"name": "B", "city": "Cairo"}, "items": 1}]
    st.json(records)
    st.code("pd.json_normalize(records)", language="python")
    st.dataframe(pd.json_normalize(records), hide_index=True)

st.markdown("## حسب التصميم الزمني/البحثي")
design = st.segmented_control("التصميم", ["Cross-sectional", "Time series", "Panel"], default="Panel", key="vt_design")
if design == "Cross-sectional":
    st.markdown("وحدات كثيرة في لحظة واحدة. الافتراض الشائع: استقلال المشاهدات.")
    st.dataframe(panel().query("year == 2018").head(8), hide_index=True)
elif design == "Time series":
    ts = daily_sales()
    st.markdown("وحدة واحدة عبر الزمن. المشاهدات المتتالية **مترابطة** (Autocorrelation).")
    fig = px.line(ts, x="date", y="sales", color_discrete_sequence=SEQUENCE)
    plot(fig, height=300)
else:
    p = panel()
    st.markdown("وحدات متعددة × فترات متعددة. المفتاح `(firm_id, year)`؛ الشركات تدخل وتخرج (Unbalanced).")
    firms = p["firm_id"].unique()[:8]
    fig = px.line(p[p["firm_id"].isin(firms)], x="year", y="revenue", color="firm_id",
                  color_discrete_sequence=SEQUENCE)
    plot(fig, height=320)
    counts = p.groupby("firm_id")["year"].count()
    st.caption(f"عدد السنوات لكل شركة يتراوح بين {counts.min()} و{counts.max()} ← Panel غير متوازن.")
comparison_table([
    {"التصميم": "Longitudinal", "الوصف": "نفس الأفراد عبر الزمن (مرادف قريب لـPanel في العلوم الاجتماعية)"},
    {"التصميم": "Repeated measures", "الوصف": "قياسات متكررة لنفس الوحدة تحت ظروف مختلفة (تجارب)"},
    {"التصميم": "Event data", "الوصف": "سجل أحداث بأوقات غير منتظمة (نقرات، أعطال)"},
    {"التصميم": "Transaction data", "الوصف": "كل صف معاملة؛ الكيان يتكرر مشروعًا"},
    {"التصميم": "Streaming data", "الوصف": "تدفق مستمر يُعالج لحظيًا أو بنوافذ زمنية"},
])

st.markdown("## مفتش المتغيرات · Interactive Variable Inspector")
st.caption("يستنتج النوع الدلالي Semantic type لكل عمود، مع السبب. جرّب بيانات التحدي الخام لترى الأرقام المخزنة كنص.")
name, data = dataset_picker("vt_ds", default="customers_raw")
dataset_card(name)
col = st.selectbox("العمود", list(data.columns), key="vt_col")
s = data[col]
sem, reason = infer_semantic_type(s, col)
with st.container(horizontal=True):
    st.metric("dtype المخزّن", str(s.dtype), border=True)
    st.metric("النوع الدلالي المستنتج", sem, border=True)
    st.metric("قيم مختلفة", int(s.nunique()), border=True)
    st.metric("مفقود", f"{s.isna().mean():.1%}", border=True)
st.info(f"**السبب:** {reason}", icon=":material/psychology:")
c1, c2 = st.columns([1, 1])
with c1:
    if is_numeric(s):
        plot(px.histogram(s.dropna(), nbins=40, color_discrete_sequence=SEQUENCE, title=f"توزيع {col}"), height=300)
    else:
        vc = s.astype(str).value_counts().head(15)
        plot(px.bar(x=vc.values, y=vc.index, orientation="h", color_discrete_sequence=SEQUENCE,
                    title=f"أكثر القيم تكرارًا: {col}"), height=320)
with c2:
    st.markdown("**عينة من القيم الفريدة:**")
    st.write(", ".join(map(str, s.dropna().unique()[:20])))
    user_type = st.radio("ما النوع الذي تراه صحيحًا من حيث المعنى؟",
                         ["Continuous", "Discrete", "Nominal", "Ordinal", "Binary", "Date/Time", "Text", "Identifier"],
                         horizontal=False, key="vt_user", index=None)
    if user_type:
        st.success(f"اخترت **{user_type}** مقابل استنتاج آلي **{sem}**. إن اختلفا، فأيهما أصح؟ "
                   "الآلة ترى الشكل، وأنت ترى المعنى — هذا جوهر Human-in-the-loop.")

if at_least("advanced"):
    st.markdown("## متقدم: مستويات القياس (Stevens, 1946)")
    comparison_table([
        {"المستوى": "Nominal", "العمليات المسموحة": "= ≠", "الإحصاءات": "المنوال، التكرار"},
        {"المستوى": "Ordinal", "العمليات المسموحة": "< >", "الإحصاءات": "الوسيط، المئينات، Spearman"},
        {"المستوى": "Interval", "العمليات المسموحة": "+ −", "الإحصاءات": "المتوسط، الانحراف (الصفر اعتباطي: °C)"},
        {"المستوى": "Ratio", "العمليات المسموحة": "× ÷", "الإحصاءات": "كل ما سبق + النسب (صفر حقيقي: الدخل)"},
    ])
    st.markdown("درجة الحرارة 20°C ليست «ضعف» 10°C (Interval)، بينما دخل 20,000 ضعف 10,000 (Ratio).")
if at_least("research"):
    researcher_note(["معاملة Likert كمتغير فترة Interval قرار منهجي يجب التصريح به وتبريره.",
                     "في Panel data استخدم أخطاء معيارية عنقودية Clustered SE أو نماذج Fixed/Random effects."])
real_world(["هل النوع المخزن يطابق المعنى؟", "هل الأعداد الصحيحة القليلة القيم رموز لفئات؟",
            "هل التصميم يسمح بافتراض الاستقلال؟", "هل هناك بُعد زمني مخفي في البيانات؟"])

page_footer("variable_types",
            takeaways=["النوع الدلالي أهم من dtype المخزن.", "الترتيب يميز Ordinal عن Nominal.",
                       "تصميم البيانات (مقطعي/زمني/Panel) يحدد الطرق الصالحة والتقسيم الصحيح."],
            mistakes=["حساب متوسط الرمز البريدي.", "اعتبار مشاهدات Panel مستقلة.", "ترك التواريخ نصوصًا."])
