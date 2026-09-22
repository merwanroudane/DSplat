import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import domain, real_world, researcher_note, why
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.cleaning import parse_mixed_dates
from utils.datasets import customers_raw
from utils.plotting import plot
from utils.types import to_number
from utils.validation import Rule, check

page_header("consistency")
raw = customers_raw()

st.markdown("## أنواع القيم غير الصالحة")
comparison_table([
    {"النوع": "Impossible values", "مثال": "age = 250، num_orders = −1", "الكشف": "قواعد مدى من معرفة المجال"},
    {"النوع": "Out-of-domain categories", "مثال": "satisfaction = 9 في مقياس 1–5", "الكشف": "قائمة القيم المسموحة"},
    {"النوع": "Unit inconsistencies", "مثال": "الطول 1.75 (متر) بجانب 175 (سم)", "الكشف": "توزيع ثنائي القمة، نسب ثابتة ×100 أو ×1000"},
    {"النوع": "Format inconsistencies", "مثال": "2023-05-01 و01/05/2023", "الكشف": "Regex وصيغ صريحة"},
    {"النوع": "Cross-field contradictions", "مثال": "آخر شراء قبل التسجيل", "الكشف": "قواعد بين الحقول"},
    {"النوع": "Label inconsistencies", "مثال": "USA / usa / U.S.A", "الكشف": "تطبيع ومجموعات التشابه"},
    {"النوع": "Cross-source conflicts", "مثال": "دخلان مختلفان لنفس العميل", "الكشف": "مقارنة المصادر بالمفتاح"},
])

st.markdown("## 1. أخطاء الوحدات: مثال الطول")
h = raw["height_cm"]
fig = go.Figure(go.Histogram(x=h, nbinsx=80, marker_color=PALETTE["purple"]))
fig.update_layout(title="height_cm: كتلة صغيرة قرب 1.5–2 وكتلة كبيرة قرب 150–190", height=300, xaxis_title="height_cm")
plot(fig)
small = h[h < 3]
st.markdown(f"{len(small)} قيمة أقل من 3: لا يمكن أن تكون سنتيمترات لبالغ. ضربها في 100 يعطي "
            f"[{(small * 100).min():.0f}, {(small * 100).max():.0f}] — ضمن المدى الطبيعي تمامًا. **هذا تصحيح لا حذف.**")
st.code("metres = df['height_cm'] < 3\ndf.loc[metres, 'height_cm'] *= 100   # documented unit rule", language="python")

st.markdown("## 2. أخطاء الوحدات: مثال الدخل")
inc = to_number(raw["annual_income"])
fig = go.Figure(go.Histogram(x=np.log10(inc[inc > 0]), nbinsx=60, marker_color=PALETTE["coral"]))
fig.update_layout(title="log10(annual_income): كتلة حول 1.5–2 (عشرات) وكتلة حول 4.5 (عشرات الآلاف)", height=300,
                  xaxis_title="log10(income)")
plot(fig)
st.markdown(f"{int(inc.between(1, 1000).sum())} قيمة بين 1 و1000 ← على الأرجح مسجلة **بالآلاف**. "
            f"و{int((inc == 0).sum())} قيمة = 0، و{int((inc >= 9e6).sum())} قيمة = 9,999,999 (Sentinel متكرر).")
domain("المقياس اللوغاريتمي يكشف أخطاء «ترتيب الحجم» Order of magnitude: قيمتان تختلفان بعامل 1000 تظهران ككتلتين منفصلتين.")

st.markdown("## 3. التناقض بين الحقول والزمن")
res = check(raw, Rule("temporal", "last_purchase_date", {"other": "signup_date", "op": ">="}))
bad = raw.loc[res.failing_index, ["customer_id", "signup_date", "last_purchase_date"]]
st.markdown(f"**{res.failed}** عميلًا آخر شراء لهم قبل تاريخ تسجيلهم:")
st.dataframe(bad.head(10), hide_index=True)
st.caption("احذر: signup_date بصيغ مختلطة. المقارنة الصحيحة تتطلب تحويل التواريخ أولًا بصيغ صريحة.")

dates = raw["signup_date"].astype(str)
patterns = dates.str.replace(r"\d", "9", regex=True).value_counts()
st.markdown("**أنماط صيغ التاريخ في signup_date** (كل رقم ← 9):")
st.dataframe(patterns.rename("count").to_frame(), width="content")
parsed = parse_mixed_dates(raw["signup_date"])
st.markdown(f"بعد التحويل بصيغ صريحة: **{int(parsed.isna().sum())}** قيمة غير قابلة للتحويل "
            f"(منها الفارغة و«2023-13-45» المستحيلة).")
why("استخدم قائمة صيغ صريحة بدل format='mixed' أو الاستنتاج التلقائي.",
    "«01/05/2023» قد تُقرأ 5 يناير أو 1 مايو بصمت. الصيغة الصريحة إما تنجح بالمعنى المقصود أو تفشل بوضوح.")

st.markdown("## 4. التسميات غير المتسقة")
col = st.selectbox("العمود", ["country", "gender", "membership"], key="cons_col")
vals = raw[col].dropna().astype(str)
key = vals.str.strip().str.lower().str.replace(r"[^a-z]", "", regex=True).rename("normalized key")
groups = vals.groupby(key).agg(lambda v: sorted(set(v))).rename("variants").reset_index()
groups.columns = ["normalized key", "variants"]
groups["n_variants"] = groups["variants"].map(len)
groups["variants"] = groups["variants"].map(lambda v: " | ".join(repr(x) for x in v))
st.dataframe(groups.sort_values("n_variants", ascending=False), hide_index=True)
st.caption("التطبيع (strip + lower + حذف الرموز) يجمع أغلب التباينات. لكنه لا يعرف أن «KSA» = «Saudi Arabia» "
           "أو «Maroc» = «Morocco»: هنا يلزم قاموس مجال يراجعه إنسان.")

if at_least("advanced"):
    st.markdown("## متقدم: التحقق من الاتساق بين المصادر")
    st.markdown("عند دمج مصدرين، احسب لكل مفتاح مشترك نسبة تطابق كل حقل. انخفاض التطابق في حقل معين يكشف "
                "تعريفات مختلفة (دخل إجمالي مقابل صافي) لا أخطاء عشوائية.")
if at_least("research"):
    researcher_note(["وثّق كل قاعدة تصحيح (مثل ×1000) مع عدد الحالات والدليل عليها.",
                     "قاعدة التصحيح الخاطئة أسوأ من القيمة المفقودة: تحقق من عينة يدويًا قبل التعميم."])
real_world(["هل تغيّرت وحدة القياس أو تعريف الحقل عبر الزمن؟", "هل الأنظمة المصدر تستخدم إعدادات محلية مختلفة؟",
            "من يملك قاموس التوحيد؟", "هل تحقق عينة يدوية من صحة قاعدة التصحيح؟"])

page_footer("consistency",
            takeaways=["القيم المستحيلة تُكتشف بقواعد مجال، والوحدات بأنماط التوزيع.", "التصحيح المبرر أفضل من الحذف.",
                       "التواريخ تُحوّل بصيغ صريحة، والتسميات بقاموس موثق."],
            mistakes=["حذف قيم خطأ الوحدة بدل تصحيحها.", "الاعتماد على الاستنتاج التلقائي للتواريخ.", "توحيد التسميات آليًا دون مراجعة."])
