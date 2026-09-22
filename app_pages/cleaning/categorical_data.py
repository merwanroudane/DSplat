import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from core.page import page_footer, page_header
from core.state import at_least, log_decision
from core.theme import PALETTE
from utils.cleaning import COUNTRY_SYNONYMS, canonical_labels
from utils.datasets import customers_raw, survey, transactions
from utils.plotting import plot

page_header("categorical_data")
raw = customers_raw()

st.markdown("## أنواع المتغيرات الفئوية")
comparison_table([
    {"النوع": "Nominal", "مثال": "الدولة، اللون", "العمليات": "تكرار، منوال", "في pandas": "category أو str"},
    {"النوع": "Ordinal", "مثال": "التعليم، Likert", "العمليات": "ترتيب، وسيط", "في pandas": "pd.Categorical(ordered=True)"},
    {"النوع": "Binary", "مثال": "غادر/لم يغادر", "العمليات": "نسبة", "في pandas": "bool أو 0/1"},
    {"النوع": "High cardinality", "مثال": "رمز المنتج (آلاف)", "العمليات": "تجميع، ترميز خاص", "في pandas": "category لتوفير الذاكرة"},
])

st.markdown("## USA · usa · U.S.A · United States")
col = st.selectbox("العمود", ["country", "gender", "membership"], key="cat_col")
before = raw[col].astype(str).value_counts(dropna=False)
step1 = raw[col].where(raw[col].isna(), raw[col].astype(str).str.strip())
step2 = step1.where(step1.isna(), step1.astype(str).str.lower())
syn = COUNTRY_SYNONYMS if col == "country" else ({"f": "Female", "female": "Female", "m": "Male", "male": "Male"}
                                                  if col == "gender" else None)
step3 = canonical_labels(raw[col], syn)
stages = {"الأصل": raw[col], "strip()": step1, "strip + lower()": step2, "قاموس + الصيغة الأكثر تكرارًا": step3}
counts = {k: int(v.nunique(dropna=True)) for k, v in stages.items()}
fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), marker_color=[PALETTE["softred"], PALETTE["amber"],
                                                                             PALETTE["purple"], PALETTE["teal"]],
                       text=list(counts.values()), textposition="outside"))
fig.update_layout(title=f"عدد الفئات المختلفة في {col} بعد كل خطوة", height=320)
plot(fig)
c1, c2 = st.columns(2)
c1.markdown("**قبل**")
c1.dataframe(before.rename("count"), height=300)
c2.markdown("**بعد**")
c2.dataframe(step3.value_counts(dropna=False).rename("count"), height=300)
comparison_table([
    {"المشكلة": "Whitespace", "مثال": "' Egypt' و'Egypt'", "الحل": "str.strip()"},
    {"المشكلة": "Case sensitivity", "مثال": "'MALE' و'male'", "الحل": "str.lower() / str.title()"},
    {"المشكلة": "Typos", "مثال": "'Moroco'", "الحل": "مقاييس التشابه + مراجعة"},
    {"المشكلة": "Synonyms / abbreviations", "مثال": "'KSA' و'Saudi Arabia'", "الحل": "قاموس مجال موثق"},
    {"المشكلة": "Punctuation", "مثال": "'U.S.A' و'USA'", "الحل": "حذف الرموز في مفتاح المقارنة"},
    {"المشكلة": "Unknown categories", "مثال": "'Other'، 'N/A'، '?'", "الحل": "توحيدها في Missing/Unknown صريحة"},
])

st.markdown("### ابنِ قاموس التوحيد بنفسك (Human-in-the-loop)")
variants = sorted(raw["country"].dropna().astype(str).unique())
mapping_df = pd.DataFrame({"variant": variants, "canonical": [COUNTRY_SYNONYMS.get(v.strip().lower().replace(".", ""),
                                                                                   v.strip()) for v in variants]})
edited = st.data_editor(mapping_df, hide_index=True, disabled=["variant"], key="cat_map", height=320)
mapped = raw["country"].map(dict(zip(edited["variant"], edited["canonical"])))
st.markdown(f"النتيجة: **{mapped.nunique()}** دولة مختلفة.")
if st.button("سجّل هذا القاموس في سجل القرارات", key="cat_log", icon=":material/bookmark:"):
    log_decision("categorical_data", "country mapping", f"{len(variants)} variants → {mapped.nunique()} countries",
                 "قاموس توحيد روجع يدويًا")
    st.toast("سُجّل القرار.")
st.code("mapping = {'USA': 'United States', 'usa': 'United States', 'U.S.A': 'United States', 'KSA': 'Saudi Arabia', ...}\n"
        "df['country'] = df['country'].str.strip().map(mapping).fillna(df['country'].str.strip())", language="python")

st.markdown("## الفئات النادرة وHigh cardinality")
tr = transactions()
vc = raw["country"].pipe(canonical_labels, COUNTRY_SYNONYMS).value_counts(normalize=True)
thr = st.slider("اعتبر الفئة نادرة إذا كانت نسبتها أقل من", 0.005, 0.1, 0.01, 0.005, format="%.3f", key="cat_thr")
rare = vc[vc < thr]
st.markdown(f"الفئات النادرة: {', '.join(f'{k} ({v:.2%})' for k, v in rare.items()) or 'لا شيء'}")
why("افحص الفئة النادرة قبل دمجها في «Other».",
    "«Iceland» هنا عميل حقيقي واحد (قيمة صحيحة نادرة)؛ دمجها مقبول للنمذجة لكن لا يُحذف السجل. "
    "أما الخطأ الإملائي (مثل «Moroco» لو ظهر) فيُصحح إلى فئته الصحيحة ولا يُدمج في Other.")
st.markdown(f"**High cardinality:** `customer_id` في المعاملات له {tr['customer_id'].nunique()} قيمة؛ One-hot سينتج مئات الأعمدة. "
            "الحلول: التجميع حسب الهدف، Frequency/Target encoding (انظر وحدة الترميز)، أو التجميع الهرمي (المنتج ← الفئة).")

st.markdown("## المتغيرات الترتيبية Ordinal")
sv = survey()


def _ordered(use_order: bool):
    s = sv["education"]
    if use_order:
        s = pd.Categorical(s, categories=["Primary", "Secondary", "Bachelor", "Master", "PhD"], ordered=True)
        out = pd.Series(s).value_counts().sort_index()
    else:
        out = s.value_counts().sort_index()
    return out.rename("count").to_frame()


code_lab("cat_ord", "الترتيب الأبجدي مقابل الترتيب المنطقي",
         lambda p: ('edu = pd.Categorical(df["education"],\n'
                    '    categories=["Primary", "Secondary", "Bachelor", "Master", "PhD"], ordered=True)\n'
                    "pd.Series(edu).value_counts().sort_index()") if p["use_order"]
         else 'df["education"].value_counts().sort_index()   # alphabetical!',
         _ordered, params=lambda: {"use_order": st.toggle("استخدم Categorical مرتب", True, key="cat_useord")},
         explanation="بدون ترتيب صريح يُفرز Bachelor قبل Primary أبجديًا — كل رسم وجدول ومقارنة سيكون مضللًا.")

if at_least("advanced"):
    st.markdown("## متقدم: ذاكرة category")
    s_obj = tr["product"].astype(object)
    s_cat = tr["product"].astype("category")
    st.markdown(f"عمود product: object = {s_obj.memory_usage(deep=True) / 1024:.0f} KB ← category = "
                f"{s_cat.memory_usage(deep=True) / 1024:.0f} KB. النوع category يخزن كل فئة مرة واحدة مع رموز صحيحة.")
if at_least("research"):
    researcher_note(["عند ترميز نص حر إلى فئات يدويًا، قِس الاتفاق بين المرمّزين (Cohen's kappa).",
                     "وثّق القاموس كملف مستقل مع الإصدار، لا داخل الكود فقط."])
real_world(["من يملك القائمة الرسمية للفئات؟", "هل تظهر فئات جديدة مع الزمن؟", "هل «Other» تخفي فئة مهمة كبرت؟",
            "هل التوحيد يغيّر معنى (مثل دمج مدينتين مختلفتين بنفس الاسم)؟"])

page_footer("categorical_data",
            takeaways=["التطبيع (strip/lower) أولًا، ثم قاموس مجال موثق.", "الفئات النادرة تُفحص قبل الدمج.",
                       "المتغير الترتيبي يحتاج ترتيبًا صريحًا."],
            mistakes=["دمج الأخطاء الإملائية في Other.", "الفرز الأبجدي لمتغير ترتيبي.", "One-hot لمتغير بآلاف الفئات."])
