import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab, explain_code
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import customers_raw
from utils.plotting import plot
from utils.types import to_number

page_header("numerical_cleaning")
raw = customers_raw()

st.markdown("## مشكلات البيانات الرقمية")
comparison_table([
    {"المشكلة": "أرقام مخزنة كنص", "مثال": "'$61,200'، '54,000'", "العلاج": "إزالة الرموز ثم pd.to_numeric(errors='coerce')"},
    {"المشكلة": "قيم سالبة مستحيلة", "مثال": "num_orders = −1", "العلاج": "NaN + التحقق من المصدر"},
    {"المشكلة": "أصفار مستحيلة", "مثال": "الطول = 0، السعر = 0", "العلاج": "NaN (أو قاعدة مجال: هدية مجانية؟)"},
    {"المشكلة": "Infinity", "مثال": "نسبة = 5/0", "العلاج": "np.isinf ← NaN، وإصلاح القسمة"},
    {"المشكلة": "أخطاء الفاصلة العشرية", "مثال": "1250 بدل 12.50", "العلاج": "نسب ثابتة ×10ⁿ + مراجعة"},
    {"المشكلة": "عدم اتساق الوحدات", "مثال": "متر/سم، آلاف/وحدات", "العلاج": "قاعدة تحويل موثقة"},
    {"المشكلة": "مخالفة المدى", "مثال": "رضا = 9 في مقياس 1–5", "العلاج": "قواعد مدى"},
    {"المشكلة": "مشكلات المقياس Scale", "مثال": "دخل بالملايين وعمر بالعشرات", "العلاج": "Scaling عند الحاجة"},
    {"المشكلة": "الدقة والتقريب", "مثال": "تكدس الأعمار عند 30، 40، 50", "العلاج": "تفسير حذر، مؤشرات Heaping"},
    {"المشكلة": "الالتواء والذيول الثقيلة", "مثال": "الإنفاق والدخل", "العلاج": "وسيط، تحويل log، نماذج متينة"},
])

st.markdown("## التحويل الآمن من نص إلى رقم")
col = st.selectbox("العمود", ["annual_income", "age", "satisfaction"], key="num_col")
s = raw[col]
naive = pd.to_numeric(s, errors="coerce")
smart = to_number(s)
c1, c2, c3 = st.columns(3)
c1.metric("قيم أصلية غير فارغة", int(s.notna().sum()))
c2.metric("فشل pd.to_numeric مباشرة", int((naive.isna() & s.notna()).sum()))
c3.metric("فشل بعد إزالة $ , %", int((smart.isna() & s.notna()).sum()))
failed = s[smart.isna() & s.notna()].value_counts()
st.markdown("**القيم التي ما زالت تفشل:** " + (", ".join(f"`{k}` ×{v}" for k, v in failed.items()) or "لا شيء"))
explain_code(
    'x = pd.to_numeric(s.str.replace(r"[$,%\\s]", "", regex=True), errors="coerce")',
    [("s.str.replace(..., regex=True)", "يحذف رموز العملة والفواصل والنسبة والمسافات"),
     ("pd.to_numeric", "يحوّل النص إلى رقم"), ('errors="coerce"', "ما لا يمكن تحويله يصبح NaN بدل رفع خطأ")],
    output="Series رقمية float64",
    interpretation="القيم التي تحولت إلى NaN ليست «مفقودة» أصلًا: عدّها وراجعها قبل المتابعة، فقد تكشف صيغة لم تتوقعها.")
why("لا تستخدم errors='coerce' دون عدّ ما تحول إلى NaN.", "الإكراه الصامت يحوّل أخطاء التنسيق إلى فقد غير مرئي.")

st.markdown("## اكتشاف أخطاء ترتيب الحجم (الوحدات والفواصل)")
x = smart.dropna()
x = x[x > 0]
fig = go.Figure(go.Histogram(x=np.log10(x), nbinsx=70, marker_color=PALETTE["purple"]))
fig.update_layout(title=f"log10({col})", height=300)
plot(fig)
if col == "annual_income":
    st.markdown("الكتلة اليسرى (≈1.5–2) قيم مسجلة **بالآلاف**. قاعدة التصحيح: القيم بين 1 و1000 ×1000. "
                "والقيم عند 7 (9,999,999) Sentinel.")

st.markdown("## القيم المستحيلة وInfinity")
demo = pd.DataFrame({"revenue": [100.0, 250.0, 0.0, 80.0], "visits": [10, 0, 0, 8]})
demo["revenue_per_visit"] = demo["revenue"] / demo["visits"]
st.dataframe(demo, hide_index=True)
st.markdown("250/0 = **inf** و0/0 = **NaN**. `isna()` لا يكتشف inf:")
st.code("np.isinf(df['revenue_per_visit']).sum()   # -> 1\n"
        "df['revenue_per_visit'] = df['revenue_per_visit'].replace([np.inf, -np.inf], np.nan)", language="python")

st.markdown("## الالتواء والذيول الثقيلة")
spend = raw["monthly_spend"].dropna()
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Histogram(x=spend, nbinsx=60, marker_color=PALETTE["coral"]))
    fig.add_vline(x=spend.mean(), line_color=PALETTE["purple"], annotation_text="mean")
    fig.add_vline(x=spend.median(), line_color=PALETTE["teal"], annotation_text="median", line_dash="dash")
    fig.update_layout(title="monthly_spend", height=300)
    plot(fig)
with c2:
    fig = go.Figure(go.Histogram(x=np.log1p(spend), nbinsx=60, marker_color=PALETTE["amber"]))
    fig.update_layout(title="log1p(monthly_spend)", height=300)
    plot(fig)
st.markdown(f"Skewness: {spend.skew():.2f} ← {np.log1p(spend).skew():.2f} بعد log1p. المتوسط ({spend.mean():.1f}) "
            f"أكبر من الوسيط ({spend.median():.1f}) بسبب الذيل. التفاصيل في وحدة «التحويلات والقياس».")


def _precision(decimals: int):
    d = pd.DataFrame({"price": [12.345, 7.891, 3.456, 10.004]})
    d["rounded"] = d["price"].round(decimals)
    d["qty"] = [1000, 2000, 1500, 3000]
    d["revenue_exact"] = d["price"] * d["qty"]
    d["revenue_rounded"] = d["rounded"] * d["qty"]
    return [d, f"الفرق الكلي في الإيراد بسبب التقريب: **{(d['revenue_exact'] - d['revenue_rounded']).sum():.2f}**"]


code_lab("num_round", "الدقة والتقريب Precision & rounding",
         lambda p: f"d['rounded'] = d['price'].round({p['decimals']})\nd['revenue_rounded'] = d['rounded'] * d['qty']",
         _precision, params=lambda: {"decimals": st.slider("عدد المنازل العشرية", 0, 3, 1, key="num_dec")},
         explanation="قرّب في العرض فقط، واحتفظ بالدقة الكاملة في الحساب؛ التقريب المبكر يتراكم عبر الكميات الكبيرة.")

if at_least("advanced"):
    st.markdown("## متقدم: أنواع pandas القابلة للقيم الفارغة")
    st.markdown("`Int64` (بحرف كبير) يسمح بأعداد صحيحة مع `pd.NA`، بينما `int64` يتحول إلى float عند وجود NaN. "
                "استخدم `df['orders'].astype('Int64')` للحفاظ على معنى العدّ.")
if at_least("research"):
    researcher_note(["في بيانات الأعمار الذاتية افحص Age heaping (تكدس عند مضاعفات 5).",
                     "أبلغ عن كل قاعدة تحويل وحدات وعدد الحالات المتأثرة."])
real_world(["ما الوحدة المتوقعة لكل عمود رقمي؟", "هل الصفر قيمة حقيقية أم رمز؟", "هل تغيّرت العملة أو الوحدة عبر الزمن؟",
            "كم قيمة فشلت في التحويل ولماذا؟"])

page_footer("numerical_cleaning",
            takeaways=["حوّل النص إلى رقم بعد إزالة الرموز، ثم عدّ ما فشل.", "المقياس اللوغاريتمي يكشف أخطاء الوحدات.",
                       "inf ليست NaN؛ اكشفها صراحة.", "الوسيط أصدق من المتوسط في البيانات الملتوية."],
            mistakes=["coerce صامت دون عدّ.", "حذف قيم أخطاء الوحدة بدل تصحيحها.", "التقريب قبل الحساب."])
