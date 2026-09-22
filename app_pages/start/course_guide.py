import streamlit as st

from components.cards import card_grid, comparison_table
from components.diagrams import mermaid
from core.curriculum import GROUPS, MODULES, get_module
from core.page import footer, page_header

page_header("course_guide")

st.markdown("## كيف تستخدم المنصة؟")
card_grid([
    ("Learn · تعلّم", "شرح منظم لكل مفهوم: ما هو؟ لماذا؟ الحدس، الرياضيات، الكود، والتفسير."),
    ("Explore · استكشف", "غيّر المعاملات بالمنزلقات والقوائم وشاهد النتائج والرسوم تتغير فورًا."),
    ("Practice · تدرّب", "اختبارات بأسئلة مشروحة، وتمارين بثلاثة مستويات مع إجابات نموذجية."),
    ("Project · طبّق", "دراسات حالة كاملة وقصة بيانات ومشروع نهائي من البيانات الخام إلى التقرير."),
], columns=2)

st.markdown("## مستويات الشرح")
st.markdown("اختر المستوى من **الشريط الجانبي**. المحتوى الأساسي يظهر للجميع، ويُضاف المزيد حسب المستوى:")
comparison_table([
    {"المستوى": "مبتدئ Beginner", "ماذا يظهر؟": "شرح مبسط، أكواد قصيرة، أمثلة مباشرة، رياضيات تدريجية."},
    {"المستوى": "متقدم Advanced", "ماذا يظهر؟": "افتراضات، حالات حدية، تشخيصات، مفاضلات Trade-offs، طرق متقدمة."},
    {"المستوى": "بحثي Research", "ماذا يظهر؟": "ملاحظات منهجية، تحذيرات إحصائية، إعادة الإنتاج، وتوصيات كتابة التقارير."},
])

st.markdown("## بنية كل وحدة")
mermaid("""
flowchart LR
  H[Header: objectives, prerequisites, difficulty, time] --> C[Concept & intuition] --> M[Mathematics]
  M --> K[Code & run] --> V[Visualization & interpretation] --> X[Mistakes & real-world checks]
  X --> Q[Quiz] --> E[Exercises] --> N[Next module]
""")

st.markdown("## خريطة المتطلبات السابقة · Prerequisite map")
group = st.selectbox("اعرض المجموعة", list(GROUPS), format_func=GROUPS.get, index=2, key="guide_group")
mods = [m for m in MODULES if m.group == group]
lines = ["flowchart TB"]
ids_in_group = {m.id for m in mods}
for m in mods:
    lines.append(f'  {m.id}["{m.title_en}"]')
    for p in m.prereqs:
        if p not in ids_in_group:
            lines.append(f'  {p}["{get_module(p).title_en}"]:::ext')
        lines.append(f"  {p} --> {m.id}")
lines.append("  classDef ext fill:#F1F3F5,color:#5C677D,stroke:#CED4DA,stroke-dasharray:3 3;")
mermaid("\n".join(lines))
st.caption("المربعات الباهتة متطلبات من مجموعات أخرى. الترتيب يضمن مثلًا: أنواع البيانات قبل الترميز، "
           "والقيم المفقودة قبل التعويض، والتقسيم قبل المعالجة المعرّضة للتسرب، والجودة قبل التنظيف الآلي.")

st.markdown("## مسارات مقترحة حسب الهدف")
comparison_table([
    {"أنت": "طالب جديد", "ابدأ بـ": "مسار الأساسيات (الصفحة الرئيسية)", "ثم": "مختبر EDA ودراسة حالة العملاء"},
    {"أنت": "محلل بيانات", "ابدأ بـ": "مسار الجودة والتنظيف", "ثم": "باني خط المعالجة والتقرير الآلي"},
    {"أنت": "باحث", "ابدأ بـ": "القيم المفقودة والاختبارات الإحصائية (مستوى بحثي)", "ثم": "قابلية إعادة الإنتاج والمشروع النهائي"},
    {"أنت": "مهتم بالنمذجة", "ابدأ بـ": "مسار النمذجة", "ثم": "تسرب البيانات وAutoML"},
])

st.markdown("## ملاحظات عملية")
st.markdown(
    "- **الأمان:** لا يُنفذ أي كود تكتبه أنت؛ أزرار «تشغيل» تشغّل دوال معدة مسبقًا تطابق الكود المعروض.\n"
    "- **بياناتك:** الملفات المرفوعة تبقى في جلستك فقط ولا تُرسل لأي خدمة خارجية.\n"
    "- **التقدّم:** يُحفظ في الجلسة؛ صدّره من صفحة «تقدّمي» لتستعيده لاحقًا.\n"
    "- **الحركة:** فعّل «تقليل الحركة» من الشريط الجانبي إن كانت الرسوم المتحركة مزعجة."
)
footer()
