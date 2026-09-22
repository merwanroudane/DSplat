import streamlit as st

from components.callouts import intuition, real_world, researcher_note, why
from components.cards import card_grid, comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header, page_link
from core.state import at_least

page_header("traditional_vs_modern")

st.markdown("## ثلاثة أنماط عمل")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### Manual · يدوي")
    mermaid("""
flowchart TB
  A[Inspect data] --> B[Write code] --> C[Check distributions] --> D[Clean] --> E[Build features] --> F[Model] --> G[Interpret]
""")
with c2:
    st.markdown("### Automated · آلي")
    mermaid("""
flowchart TB
  A[Automated profiling] --> B[Type inference] --> C[Anomaly flags] --> D[Chart suggestions] --> E[AutoML] --> F[AI summary]
""")
with c3:
    st.markdown("### Hybrid · هجين")
    mermaid("""
flowchart TB
  A[Automatic detection] --> B[Human review] --> C[Domain validation] --> D[Controlled transformation] --> E[Re-analysis] --> F[Documentation]
""")

st.markdown("## المقارنة الشاملة")
comparison_table([
    {"البعد": "السرعة", "Manual": "بطيء", "Automated": "سريع جدًا", "Hybrid": "سريع مع نقاط توقف مقصودة"},
    {"البعد": "الفهم العميق", "Manual": "عالٍ", "Automated": "منخفض", "Hybrid": "عالٍ حيث يهم"},
    {"البعد": "قابلية التكرار", "Manual": "ضعيفة إن لم يُوثق", "Automated": "عالية", "Hybrid": "عالية + سجل قرارات"},
    {"البعد": "معرفة المجال", "Manual": "مدمجة", "Automated": "غائبة", "Hybrid": "مدمجة في المراجعة"},
    {"البعد": "الأخطاء الشائعة", "Manual": "إغفال، إرهاق، عدم اتساق", "Automated": "قرارات عمياء، تحويلات خفية", "Hybrid": "Automation bias إن كانت المراجعة شكلية"},
    {"البعد": "التوسع", "Manual": "صعب", "Automated": "سهل", "Hybrid": "جيد"},
    {"البعد": "المساءلة", "Manual": "واضحة", "Automated": "غامضة", "Hybrid": "واضحة وموثقة"},
])
intuition("الأتمتة ممتازة في **الاكتشاف والحساب المتكرر**، والإنسان ضروري في **التفسير والقرارات غير القابلة للتراجع**. "
          "المبدأ المحوري للمقرر: **Automation should assist judgment, not erase judgment.**")

st.markdown("## ماذا نؤتمت وماذا لا نؤتمت؟")
comparison_table([
    {"المهمة": "حساب الملخصات ونسب الفقد", "القرار": "✅ أتمتة كاملة", "السبب": "حساب حتمي قابل للتحقق"},
    {"المهمة": "استنتاج الأنواع الدلالية", "القرار": "🟡 أتمتة + مراجعة", "السبب": "الآلة ترى الشكل لا المعنى"},
    {"المهمة": "تحويل Sentinels إلى NaN", "القرار": "🟡 اقتراح + اعتماد", "السبب": "يحتاج قاموس بيانات"},
    {"المهمة": "حذف القيم الشاذة", "القرار": "🔴 قرار بشري", "السبب": "غير قابل للتراجع وقد يحذف حالات حقيقية مهمة"},
    {"المهمة": "دمج التكرارات المتعارضة", "القرار": "🔴 قرار بشري", "السبب": "يحتاج معرفة مصدر الحقيقة"},
    {"المهمة": "اقتراح الرسوم", "القرار": "✅ أتمتة", "السبب": "منخفض المخاطر"},
    {"المهمة": "البحث عن أفضل نموذج", "القرار": "🟡 AutoML + فحص Leakage", "السبب": "يستخدم أي إشارة حتى المسربة"},
    {"المهمة": "تفسير النتائج السببية", "القرار": "🔴 بشري", "السبب": "يحتاج تصميمًا ومعرفة مجال"},
])
why("ابدأ مشاريعك الجديدة بنمط Hybrid.",
    "يجمع سرعة الأتمتة مع حكم الإنسان، ويترك سجلًا يمكن تدقيقه — وهذا ما تطلبه المؤسسات والمجلات العلمية.")

st.markdown("## جرّب الأنماط الثلاثة")
card_grid([("المختبر الهجين", "نفس البيانات بثلاثة أنماط: أنت تقرر، أو المنصة، أو المنصة تقترح وأنت تعتمد."),
           ("التنظيف الآلي", "مولد مشكلات بدرجة الخطورة والدليل وسبب التوصية."),
           ("الإنسان في الحلقة", "مخطط تفاعلي وسجل قرارات.")], columns=3)
with st.container(horizontal=True):
    page_link("automation_lab")
    page_link("automated_cleaning")
    page_link("human_in_loop")

if at_least("research"):
    researcher_note(["Automation bias موثق في الطيران والطب: الخبراء يقبلون اقتراحات الآلة الخاطئة أكثر مما ينبغي.",
                     "صمم واجهات المراجعة لتعرض الدليل وعدم اليقين، لا القرار فقط."])
real_world(["ما القرارات غير القابلة للتراجع في Pipeline؟", "من يراجع؟ وكم وقتًا لديه؟", "هل تُسجل القرارات؟",
            "كيف نعرف أن الأتمتة ما زالت صحيحة بعد تغيّر البيانات؟"])

page_footer("traditional_vs_modern",
            takeaways=["اليدوي عميق وبطيء، الآلي سريع وأعمى، الهجين يجمع الاثنين.",
                       "أتمت الحساب والاكتشاف؛ أبقِ القرارات غير القابلة للتراجع بشرية.", "وثّق كل قرار."],
            mistakes=["أتمتة الحذف.", "مراجعة بشرية شكلية.", "رفض الأتمتة كليًا في المهام المتكررة."])
