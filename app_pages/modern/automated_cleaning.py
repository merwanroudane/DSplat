import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.dataset_viewer import dataset_card, dataset_picker
from components.diagrams import mermaid
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot
from utils.profiling import KIND_AR, SEVERITY_AR, detect_issues, issues_frame

page_header("automated_cleaning")

st.markdown("## أساليب التنظيف")
comparison_table([
    {"الأسلوب": "Manual", "كيف": "المحلل يفحص ويقرر لكل حالة", "القوة": "فهم عميق", "الضعف": "بطيء، غير قابل للتوسع، غير متسق"},
    {"الأسلوب": "Rule-based", "كيف": "قواعد صريحة (مدى، Regex، قواميس)", "القوة": "شفاف، قابل للتدقيق", "الضعف": "لا يغطي غير المتوقع"},
    {"الأسلوب": "Statistical", "كيف": "IQR، MAD، اختبارات التوزيع", "القوة": "بلا معرفة مسبقة كثيرة", "الضعف": "يخلط النادر الحقيقي بالخطأ"},
    {"الأسلوب": "ML-based", "كيف": "نماذج تتنبأ بالقيمة المتوقعة وتعلّم الانحراف، أو Isolation Forest", "القوة": "يلتقط أنماطًا معقدة", "الضعف": "صندوق أسود، يحتاج بيانات نظيفة للتعلم"},
    {"الأسلوب": "AI-assisted", "كيف": "LLM يقترح قواعد أو قواميس توحيد أو كودًا", "القوة": "سريع في الصياغة والتوحيد النصي", "الضعف": "هلوسة، يحتاج تحققًا وتنفيذًا محليًا"},
    {"الأسلوب": "Automated", "كيف": "Pipeline يطبق القواعد آليًا عند كل تحديث", "القوة": "ثبات وسرعة", "الضعف": "خطأ واحد يتكرر بصمت"},
])
mermaid("""
flowchart LR
  A[Data] --> B[Detect issues<br/>rules + statistics] --> C[Classify:<br/>detected / potential / recommendation]
  C --> D[Evidence + severity + why] --> E{Reversible & low risk?}
  E -- yes --> F[Auto-apply on a copy + log]
  E -- no --> G[Human review queue]
  G --> H[Approved actions] --> F
  F --> I[Re-validate]
""")

st.markdown("## مولّد المشكلات · Issue detector")
st.caption("يكتشف ويقترح ويشرح — **ولا يعدّل البيانات أبدًا**. التطبيق يتم في المختبر الهجين بعد اعتمادك.")
name, df = dataset_picker("acl_ds", default="customers_raw")
dataset_card(name)
issues = detect_issues(df)
f = issues_frame(issues)
c1, c2 = st.columns([1, 2])
with c1:
    sev_counts = f["severity_ar"].value_counts().reindex(list(SEVERITY_AR.values())).fillna(0)
    fig = go.Figure(go.Bar(x=sev_counts.index, y=sev_counts.values,
                           marker_color=[PALETTE["softred"], PALETTE["coral"], PALETTE["amber"], PALETTE["sky"]]))
    fig.update_layout(title="حسب الخطورة", height=300)
    plot(fig)
with c2:
    kinds = st.pills("النوع", list(KIND_AR.values()), selection_mode="multi", default=list(KIND_AR.values()), key="acl_kinds")
    sevs = st.pills("الخطورة", list(SEVERITY_AR.values()), selection_mode="multi", default=list(SEVERITY_AR.values()), key="acl_sev")
view = f[f["kind_ar"].isin(kinds or []) & f["severity_ar"].isin(sevs or [])]
st.dataframe(view[["column", "issue", "kind_ar", "severity_ar", "evidence", "action", "why"]], hide_index=True,
             column_config={"kind_ar": "النوع", "severity_ar": "الخطورة", "evidence": "الدليل", "action": "الإجراء المقترح",
                            "why": "لماذا؟"}, height=420)
st.markdown("**التمييز بين الفئات:**\n- **مشكلة مكتشفة Detected:** مخالفة واضحة لقاعدة (نوع خاطئ، قيمة مستحيلة).\n"
            "- **مشكلة محتملة Potential:** نمط غير معتاد يحتاج تحقيقًا (قيم متطرفة، فئات نادرة).\n"
            "- **توصية Recommendation:** ممارسة مقترحة وليست خطأ (تحويل log، استبعاد المعرّف).")
why("لا يُنفّذ حذف أو تعديل غير قابل للتراجع تلقائيًا.",
    "المشكلة «المحتملة» قد تكون أهم حالة في البيانات (عميل ينفق 25 ضعف المتوسط)، والتصحيح الآلي الخاطئ ينتشر بصمت إلى كل تحليل لاحق.")
page_link("automation_lab", label="طبّق الإجراءات بعد المراجعة في المختبر الهجين")

if at_least("advanced"):
    st.markdown("## متقدم: التنظيف بمساعدة ML")
    st.markdown("فكرة: درّب نموذجًا يتنبأ بكل حقل من الحقول الأخرى على بيانات موثوقة؛ القيم البعيدة جدًا عن التنبؤ مرشحة لأخطاء "
                "إدخال. أدوات بحثية مثل HoloClean تستخدم الاستدلال الاحتمالي مع القيود. الخطر: تطبيع الحالات النادرة الحقيقية.")
if at_least("research"):
    researcher_note(["أبلغ عن قواعد التنظيف الآلي كجزء من المنهجية، مع عدد الحالات لكل قاعدة.",
                     "قارن نتائج التحليل قبل وبعد التنظيف الآلي (تحليل حساسية)."])
real_world(["أي القواعد يمكن تطبيقها آليًا بأمان؟", "من يراجع المحتمل؟", "هل يُحفظ الأصل دائمًا؟",
            "هل تُراقب نسب المشكلات عبر الزمن؟"])

page_footer("automated_cleaning",
            takeaways=["التنظيف الآلي = اكتشاف + تصنيف + دليل + سبب، لا حذف آلي.", "Detected ≠ Potential ≠ Recommendation.",
                       "القرارات غير القابلة للتراجع تمر بمراجعة بشرية."],
            mistakes=["تطبيق كل الاقتراحات دفعة واحدة.", "تعديل البيانات الأصلية في مكانها.", "الثقة بتصحيحات LLM دون تحقق."])
