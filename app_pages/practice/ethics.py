import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit
from utils.plotting import plot

page_header("ethics")

st.markdown("## الخصوصية والبيانات الحساسة")
comparison_table([
    {"الفئة": "Direct identifiers", "أمثلة": "الاسم، رقم الهوية، الهاتف، البريد", "المعالجة": "حذف أو استبدال بمعرّف مستعار"},
    {"الفئة": "Quasi-identifiers", "أمثلة": "تاريخ الميلاد، الرمز البريدي، الجنس، المهنة", "المعالجة": "تعميم (سنة بدل تاريخ)، تجميع، k-anonymity"},
    {"الفئة": "Sensitive attributes", "أمثلة": "الصحة، الدين، الأصل، البيانات المالية، البيومترية", "المعالجة": "أساس قانوني، وصول مقيد، تقليل"},
])
st.markdown("**Data minimization:** اجمع وعالج فقط ما يلزم للغرض المحدد، واحذفه عند انتهاء الحاجة. "
            "**Consent:** الموافقة المستنيرة للغرض المعلن، وإعادة الاستخدام لغرض مختلف تحتاج مراجعة.")

st.markdown("### جرّب: Pseudonymization")
demo = pd.DataFrame({"name": ["Amina B.", "Youssef K.", "Sara M."], "birth_date": ["1990-03-14", "1985-11-02", "2001-07-21"],
                     "zip": ["20250", "10110", "30000"], "income": [52000, 71000, 23000]})
salt = st.text_input("Salt سري (في الواقع يُحفظ في st.secrets لا في الكود)", "course-demo-salt", key="eth_salt")
anon = demo.assign(person_id=demo["name"].map(lambda n: hashlib.sha256((salt + n).encode()).hexdigest()[:10]),
                   birth_year=demo["birth_date"].str[:4], zip_region=demo["zip"].str[:2] + "xxx").drop(columns=["name", "birth_date", "zip"])
c1, c2 = st.columns(2)
c1.markdown("**الأصل**")
c1.dataframe(demo, hide_index=True)
c2.markdown("**بعد الاستبدال والتعميم**")
c2.dataframe(anon, hide_index=True)
why("الاستعارة Pseudonymization ليست إخفاء هوية كاملًا Anonymization.",
    "من يملك الـSalt أو بيانات مساعدة قد يعيد الربط. سنة الميلاد + المنطقة + الجنس قد تميّز شخصًا في مجتمع صغير.")

st.markdown("## التحيّز والعدالة")
st.markdown("نموذج بسيط للموافقة على القروض (موافقة إذا كانت احتمالية التعثر أقل من عتبة) — كيف تختلف النتائج بين المجموعات؟")
d = credit()
risk = (0.9 * d["debt_to_income"] + 0.25 * d["late_payments"] + 0.3 * (d["employment"] == "Unemployed")
        + 0.15 * (d["home_ownership"] == "Rent") - 0.01 * d["credit_history_years"])
thr = st.slider("عتبة الموافقة (درجة الخطر القصوى)", float(risk.quantile(0.2)), float(risk.quantile(0.95)),
                float(risk.quantile(0.7)), key="eth_thr")
d["approved"] = risk <= thr
group = st.selectbox("قارن حسب", ["home_ownership", "employment", "region"], key="eth_group")
rates = d.groupby(group).agg(approval_rate=("approved", "mean"), default_rate=("default", "mean"), n=("approved", "size"))
fig = go.Figure()
fig.add_trace(go.Bar(x=rates.index, y=rates["approval_rate"], name="approval rate", marker_color=PALETTE["purple"]))
fig.add_trace(go.Bar(x=rates.index, y=rates["default_rate"], name="actual default rate", marker_color=PALETTE["coral"]))
fig.update_layout(barmode="group", yaxis_tickformat=".0%", height=340, legend=dict(orientation="h", y=1.12))
plot(fig)
di = rates["approval_rate"].min() / rates["approval_rate"].max()
st.metric("Disparate impact ratio (أدنى معدل موافقة ÷ أعلى معدل)", f"{di:.2f}",
          "أقل من 0.8 (قاعدة الأربعة أخماس الإرشادية)" if di < 0.8 else "≥ 0.8", delta_color="inverse" if di < 0.8 else "off")
st.caption("الفجوة قد تعكس فروقًا حقيقية في المخاطر أو تحيزًا تاريخيًا في البيانات — المقاييس تكشف ولا تحكم وحدها.")
comparison_table([
    {"المقياس": "Demographic parity", "يطلب": "معدلات قرار إيجابي متساوية", "الحذر": "يتجاهل الفروق الحقيقية في الأهلية"},
    {"المقياس": "Equal opportunity", "يطلب": "Recall متساوٍ للمؤهلين", "الحذر": "يحتاج تسميات موثوقة"},
    {"المقياس": "Equalized odds", "يطلب": "TPR وFPR متساويان", "الحذر": "قد يتعارض مع المعايرة"},
    {"المقياس": "Calibration by group", "يطلب": "نفس الاحتمال = نفس الخطر لكل مجموعة", "الحذر": "لا يضمن تساوي الفرص"},
])
st.markdown("**حذف المتغير المحمي لا يكفي:** متغيرات أخرى قد تكون وكيلًا Proxy له (المنطقة، نوع السكن).")

st.markdown("## قائمة الاستخدام المسؤول")
for i, item in enumerate(["الغرض محدد ومشروع وموثق", "البيانات الضرورية فقط (Minimization)", "أساس قانوني/موافقة لكل استخدام",
                          "المعرّفات المباشرة محذوفة أو مستعارة", "الوصول مقيد ومسجل", "قياس الفروق بين المجموعات قبل النشر",
                          "آلية اعتراض ومراجعة بشرية للقرارات المؤثرة", "لا ترسل بيانات شخصية لخدمات خارجية دون اتفاقية",
                          "خطة حذف عند انتهاء الغرض"]):
    st.checkbox(item, key=f"eth_chk_{i}")
st.markdown("هذه مقدمة تعليمية وليست استشارة قانونية؛ ارجع للأطر المحلية (مثل قوانين حماية البيانات الوطنية أو GDPR) ولجهة الأخلاقيات في مؤسستك.")

if at_least("advanced"):
    st.markdown("## متقدم: الخصوصية التفاضلية Differential privacy")
    st.markdown("إضافة ضوضاء محسوبة إلى النتائج المجمعة بحيث لا يتغير الناتج كثيرًا بوجود أو غياب فرد واحد (معامل ε). "
                "تُستخدم في الإحصاءات الرسمية (مثل تعداد الولايات المتحدة 2020).")
if at_least("research"):
    researcher_note(["احصل على موافقة لجنة الأخلاقيات IRB قبل جمع بيانات بشرية.",
                     "افصح عن مصادر التحيز المحتملة في بياناتك وحدود التعميم.",
                     "Barocas, Hardt & Narayanan: Fairness and Machine Learning — مرجع مفتوح."])
real_world(["من قد يتضرر من خطأ النموذج؟", "هل يستطيع الشخص معرفة سبب القرار والاعتراض؟", "هل البيانات جُمعت لهذا الغرض؟",
            "هل الأداء متقارب بين المجموعات؟"])

page_footer("ethics",
            takeaways=["قلّل البيانات، استعر المعرّفات، وقيّد الوصول.", "الاستعارة ليست إخفاء هوية كاملًا.",
                       "قِس الفروق بين المجموعات؛ مقاييس العدالة متعددة ومتعارضة أحيانًا.", "حذف المتغير المحمي لا يكفي."],
            mistakes=["جمع كل شيء «للاحتياط».", "اعتبار حذف الاسم إخفاءً للهوية.", "رفع بيانات شخصية لخدمات AI عامة."])
