import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit
from utils.drift import ks_test, psi, psi_label, schema_diff
from utils.plotting import plot
from utils.validation import Rule, run_rules

page_header("observability")

st.markdown("## أعمدة مراقبة البيانات")
st.markdown("**Data observability** تطبيق فكرة مراقبة الأنظمة على البيانات: اكتشاف أن شيئًا تغيّر **قبل** أن يلاحظه المستخدم "
            "في لوحة خاطئة أو نموذج متدهور. بحجم يناسب علم البيانات (لا تخصص هندسة بيانات):")
comparison_table([
    {"العمود": "Freshness الحداثة", "السؤال": "هل وصلت البيانات في موعدها؟", "مؤشر": "زمن آخر تحديث مقارنة بالـSLA"},
    {"العمود": "Volume الحجم", "السؤال": "هل عدد الصفوف طبيعي؟", "مؤشر": "انخفاض مفاجئ 40% = ملف ناقص"},
    {"العمود": "Schema المخطط", "السؤال": "هل تغيرت الأعمدة أو الأنواع؟", "مؤشر": "Schema diff"},
    {"العمود": "Distribution التوزيع", "السؤال": "هل تغيّر توزيع القيم؟", "مؤشر": "PSI، KS، نسب الفقد"},
    {"العمود": "Lineage النسب", "السؤال": "من أين جاءت البيانات وما الذي يعتمد عليها؟", "مؤشر": "رسم الاعتماديات"},
])

st.markdown("## محاكي الانجراف · Data drift simulator")
st.caption("المرجع: دخل المتقدمين للقروض في بيانات التدريب. الحالي: بيانات «الإنتاج» بعد تغيير تصنعه أنت.")
ref = np.log(credit()["income"].to_numpy())
c1, c2, c3 = st.columns(3)
shift = c1.slider("إزاحة المتوسط (بوحدات SD)", -1.5, 1.5, 0.3, 0.1, key="obs_shift")
scale = c2.slider("مضاعف التشتت", 0.5, 2.0, 1.0, 0.1, key="obs_scale")
missing = c3.slider("نسبة فقد جديدة في الإنتاج", 0.0, 0.5, 0.0, 0.05, key="obs_miss")
rng = np.random.default_rng(1)
cur = rng.choice(ref, 1500)
cur = ref.mean() + (cur - ref.mean()) * scale + shift * ref.std() + rng.normal(0, 0.02, len(cur))
p_val, table = psi(ref, cur)
ks_stat, ks_p = ks_test(ref, cur)
with st.container(horizontal=True):
    st.metric("PSI", f"{p_val:.3f}", psi_label(p_val), delta_color="off", border=True)
    st.metric("KS statistic", f"{ks_stat:.3f}", f"p = {ks_p:.2g}", delta_color="off", border=True)
    st.metric("نسبة الفقد الحالية", f"{missing:.0%}", "كانت 0%", delta_color="inverse", border=True)
fig = go.Figure()
fig.add_trace(go.Histogram(x=ref, name="reference (train)", opacity=0.55, histnorm="probability density",
                           marker_color=PALETTE["purple"], nbinsx=50))
fig.add_trace(go.Histogram(x=cur, name="current (production)", opacity=0.55, histnorm="probability density",
                           marker_color=PALETTE["coral"], nbinsx=50))
fig.update_layout(barmode="overlay", xaxis_title="log(income)", height=340, legend=dict(orientation="h", y=1.12))
plot(fig)
formula(r"\text{PSI} = \sum_{b=1}^{B} (c_b - r_b)\,\ln\frac{c_b}{r_b}", title="Population Stability Index",
        symbols={"r_b": "نسبة المرجع في الفئة b (فئات مئينية من المرجع)", "c_b": "نسبة البيانات الحالية في نفس الفئة"},
        intuition="يقارن النسب فئةً فئة؛ قواعد إرشادية شائعة في الصناعة: < 0.1 مستقر، 0.1–0.25 متوسط، > 0.25 كبير.")
with st.expander("مساهمة كل فئة في PSI", icon=":material/table_chart:"):
    st.dataframe(table.round(3), hide_index=True)
why("مع n كبير لا تعتمد على p-value اختبار KS وحده للإنذار.",
    "مع آلاف الصفوف يرفض KS لأي فرق تافه. PSI وحجم الفرق (KS statistic) أنسب لتحديد عتبات الإنذار، مع ربطها بأثر الأعمال.")
comparison_table([
    {"النوع": "Data drift (covariate shift)", "ما الذي تغيّر؟": "توزيع المدخلات P(X)", "مثال": "متقدمون أصغر سنًا بعد حملة تسويق"},
    {"النوع": "Concept drift", "ما الذي تغيّر؟": "العلاقة P(y|X)", "مثال": "نفس الدخل صار أخطر بعد أزمة اقتصادية"},
    {"النوع": "Label drift", "ما الذي تغيّر؟": "نسبة الهدف P(y)", "مثال": "ارتفاع معدل التعثر العام"},
    {"النوع": "Schema drift", "ما الذي تغيّر؟": "البنية", "مثال": "income صار نصًا بفواصل"},
])

st.markdown("## انجراف المخطط Schema drift")
old = {"application_id": "string", "age": "int64", "income": "float64", "employment": "string", "loan_amount": "float64"}
new = {"application_id": "string", "age": "int64", "income": "string", "employment_type": "string", "loan_amount": "float64",
       "channel": "string"}
st.dataframe(schema_diff(old, new), hide_index=True)
st.caption("إعادة تسمية employment ← employment_type تكسر أي Pipeline يبحث عن الاسم القديم؛ وتحول income إلى نص يجعل "
           "النموذج يتلقى NaN بصمت. فحص المخطط عند كل دفعة يمنع ذلك.")

st.markdown("## نسب البيانات Lineage")
mermaid("""
flowchart LR
  CRM[(CRM system)] --> RAW[raw.applications]
  CB[(Credit bureau API)] --> RAW2[raw.bureau]
  RAW --> STG[staging.applications_clean]
  RAW2 --> STG
  STG --> FEAT[features.credit_v3]
  FEAT --> MODEL[default_model v3]
  FEAT --> DASH[Risk dashboard]
  MODEL --> DEC[Loan decisions]
""")
st.caption("عند مشكلة في raw.bureau نعرف فورًا أن النموذج واللوحة والقرارات متأثرة (تحليل الأثر Impact analysis).")

st.markdown("## عقود البيانات Data contracts")
st.markdown("اتفاق صريح بين **منتج** البيانات و**مستهلكها** على المخطط والجودة والتحديث، يُختبر آليًا عند كل تسليم.")
st.code("""dataset: loan_applications
owner: credit-operations@bank.example
freshness: daily by 06:00 UTC
schema:
  application_id: {type: string, unique: true, required: true}
  age:            {type: integer, min: 18, max: 100}
  income:         {type: number, min: 0}
  employment:     {type: string, allowed: [Salaried, Self-employed, Public sector, Unemployed]}
quality:
  max_missing_pct: {income: 5}
on_violation: quarantine batch and alert owner""", language="yaml")
batch = credit().head(400).copy()
batch.loc[batch.index[:12], "income"] = np.nan
batch.loc[batch.index[20:23], "age"] = 150
batch.loc[batch.index[30:32], "employment"] = "Freelance"
rules = [Rule("unique", "application_id", {}, "application_id unique"), Rule("range", "age", {"min": 18, "max": 100}, "age 18–100"),
         Rule("range", "income", {"min": 0}, "income ≥ 0"),
         Rule("category", "employment", {"allowed": ["Salaried", "Self-employed", "Public sector", "Unemployed"]}, "employment allowed"),
         Rule("not_null", "income", {}, "income not null")]
res = run_rules(batch, rules)
st.dataframe(res[["rule", "checked", "failed", "pass_rate_%", "status"]], hide_index=True)
miss_pct = batch["income"].isna().mean() * 100
st.markdown(f"نسبة فقد income في الدفعة = **{miss_pct:.1f}%** (الحد 5%) ← "
            + ("✅ ضمن العقد." if miss_pct <= 5 else "❌ مخالفة: تُعزل الدفعة ويُنبَّه المالك."))

if at_least("advanced"):
    st.markdown("## متقدم: الأدوات")
    comparison_table([
        {"الأداة": "Evidently", "الاستخدام": "تقارير Drift وجودة النماذج (مفتوحة المصدر)"},
        {"الأداة": "NannyML", "الاستخدام": "تقدير الأداء دون تسميات + Drift"},
        {"الأداة": "Great Expectations / Pandera", "الاستخدام": "التحقق عند كل دفعة"},
        {"الأداة": "OpenLineage / dbt docs", "الاستخدام": "النسب والاعتماديات"},
    ])
if at_least("research"):
    researcher_note(["عتبات PSI إرشادية من الصناعة لا قواعد إحصائية؛ اضبطها بتاريخ بياناتك.",
                     "Drift في المدخلات لا يعني بالضرورة تدهور الأداء؛ اربط المراقبة بمقاييس الأداء عند توفر التسميات."])
real_world(["ما الـSLA المتوقع للحداثة؟", "من يُنبَّه عند المخالفة؟", "هل يتوقف Pipeline أم يعزل الدفعة؟",
            "هل تُحفظ لقطات التوزيع المرجعي مع كل إصدار نموذج؟"])

page_footer("observability",
            takeaways=["خمسة أعمدة: الحداثة، الحجم، المخطط، التوزيع، النسب.", "PSI وKS لقياس انجراف التوزيع.",
                       "عقد البيانات يحول التوقعات إلى اختبارات آلية.", "Data drift ≠ Concept drift."],
            mistakes=["الاعتماد على p-value اختبار KS مع n ضخم.", "المراقبة دون مالك يُنبَّه.", "تجاهل تغيّر المخطط الصامت."])
