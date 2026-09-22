import streamlit as st

from components.cards import comparison_table
from config import APP_AUTHOR_AR, APP_AUTHOR_EN, APP_NAME_AR, APP_NAME_EN, APP_VERSION
from core.curriculum import MODULES
from core.page import footer, page_header

page_header("about")
st.markdown(f"## {APP_NAME_AR}")
st.markdown(f"##### {APP_NAME_EN} · v{APP_VERSION}")
with st.container(border=True):
    st.markdown(f"**إعداد وتطوير:** {APP_AUTHOR_AR}  \n**Developed by:** {APP_AUTHOR_EN}")
    st.markdown("منصة تعليمية عربية لمقرر شامل في علم البيانات، تجمع الشرح النظري والرياضي والتفاعلي والمختبرات والمشاريع، "
                "مع تركيز خاص على **جودة البيانات** و**الأتمتة المسؤولة** بمبدأ الإنسان في الحلقة.")

st.markdown("## المنهجية التعليمية")
st.markdown("كل مفهوم: **Concept → Intuition → Mathematics → Example → Code → Run → Visualization → Interpretation → "
            "Mistakes → Exercise → Quiz → Real-world context**، بثلاثة مستويات شرح (مبتدئ، متقدم، بحثي).")
comparison_table([
    {"العنصر": "الوحدات والصفحات", "العدد": len(MODULES)},
    {"العنصر": "مجموعات البيانات المدمجة", "العدد": "11 (9 اصطناعية + Iris + Wine)"},
    {"العنصر": "اللغة", "العدد": "العربية مع المصطلحات الإنجليزية"},
])

st.markdown("## التقنية")
st.markdown(
    "- **Streamlit** (تنقل متعدد الصفحات بـ`st.navigation`)، **pandas**، **NumPy**، **SciPy**، **statsmodels**، "
    "**scikit-learn**، **Plotly**.\n"
    "- محلل بيانات، ومحرك تحقق، ومولّد مشكلات، ومولّد تقارير — مكتوبة داخل المنصة دون اعتماديات ثقيلة.\n"
    "- **الأمان:** لا `eval`/`exec` لمدخلات المستخدم؛ أزرار «تشغيل» تنفذ دوال معدة مسبقًا؛ الملفات المرفوعة تبقى في الجلسة "
    "ولا تُرسل لأي خدمة خارجية.\n"
    "- **الخصوصية:** كل البيانات المدمجة اصطناعية أو عامة وغير حساسة."
)
st.markdown("## ملاحظة")
st.markdown("المحتوى تعليمي. النتائج الإحصائية تُعرض كأدلة تحت افتراضات محددة، لا كإثباتات. "
            "الملاحظات والتصحيحات مرحب بها لتحسين المقرر.")
footer()
