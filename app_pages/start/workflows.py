import streamlit as st

from components.animation import stepper
from components.callouts import intuition, real_world, researcher_note
from components.cards import comparison_table
from components.diagrams import flow, mermaid
from core.page import page_footer, page_header
from core.state import at_least

page_header("workflows")

st.markdown(
    "مشاريع البيانات الناجحة لا تبدأ بالنموذج. توجد أطر عمل Frameworks تنظّم الرحلة من السؤال إلى الأثر. "
    "نعرض أهمها ثم نبني **Workflow جامعًا** يعتمده هذا المقرر."
)

tab1, tab2, tab3, tab4 = st.tabs(["CRISP-DM", "OSEMN", "KDD", "Modern ML lifecycle"])
with tab1:
    st.markdown("**CRISP-DM** (Cross-Industry Standard Process for Data Mining، 1999–2000): ست مراحل دورية.")
    mermaid("""
flowchart LR
  BU[Business Understanding] <--> DU[Data Understanding]
  DU --> DP[Data Preparation]
  DP <--> MO[Modeling]
  MO --> EV[Evaluation]
  EV --> DE[Deployment]
  EV -. "back to the question" .-> BU
""")
    st.markdown("- **القوة:** يبدأ بالمشكلة، دوري، مستقل عن الأداة.\n- **الضعف:** لا يغطي المراقبة بعد النشر ولا تنظيم الفريق.")
with tab2:
    st.markdown("**OSEMN** (Mason & Wiggins, 2010): Obtain → Scrub → Explore → Model → iNterpret.")
    flow(["Obtain", "Scrub", "Explore", "Model", "iNterpret"])
    st.markdown("- **القوة:** بسيط وسهل التذكر للتعليم.\n- **الضعف:** يغفل فهم المشكلة والنشر والمراقبة.")
with tab3:
    st.markdown("**KDD** (Fayyad, Piatetsky-Shapiro & Smyth, 1996): عملية اكتشاف المعرفة، والتنقيب إحدى خطواتها.")
    flow(["Selection", "Preprocessing", "Transformation", "Data Mining", "Interpretation / Evaluation", "Knowledge"])
    st.markdown("- **القوة:** يوضح أن Data Mining خطوة لا العملية كلها.\n- **الضعف:** تقني، يفتقر لسياق الأعمال.")
with tab4:
    st.markdown("**دورة حياة ML الحديثة (MLOps):** تضيف ما بعد النشر: المراقبة، الانجراف، وإعادة التدريب.")
    mermaid("""
flowchart LR
  P[Problem framing] --> D[Data collection & validation] --> F[Features] --> T[Training] --> E[Evaluation]
  E --> R[Deployment] --> M[Monitoring]
  M -- drift / decay --> D
  M -- new requirements --> P
""")

st.markdown("## مقارنة الأطر")
comparison_table([
    {"الإطار": "CRISP-DM", "يبدأ بـ": "فهم الأعمال", "دوري؟": "نعم", "النشر والمراقبة": "نشر فقط", "الاستخدام الأنسب": "مشاريع مؤسسية"},
    {"الإطار": "OSEMN", "يبدأ بـ": "الحصول على البيانات", "دوري؟": "ضمنيًا", "النشر والمراقبة": "لا", "الاستخدام الأنسب": "التعليم والتحليل السريع"},
    {"الإطار": "KDD", "يبدأ بـ": "اختيار البيانات", "دوري؟": "نعم", "النشر والمراقبة": "لا", "الاستخدام الأنسب": "بحث التنقيب"},
    {"الإطار": "ML lifecycle", "يبدأ بـ": "صياغة المشكلة", "دوري؟": "نعم", "النشر والمراقبة": "نعم (أساسي)", "الاستخدام الأنسب": "منتجات ML"},
])

st.markdown("## الـWorkflow الجامع لهذا المقرر")
st.markdown("يجمع أفضل ما في الأطر: البداية بالسؤال (CRISP-DM)، والتحقق والجودة قبل النمذجة، والمراقبة (MLOps)، "
            "والتوثيق وقابلية إعادة الإنتاج (البحث العلمي)، والإنسان في الحلقة (الأتمتة الحديثة).")
STEPS = [
    ("Question", "صياغة السؤال والقرار ومعيار النجاح. ما الذي سيتغير إن عرفنا الإجابة؟"),
    ("Data collection", "من أين البيانات؟ ما وحدة التحليل؟ ما حدود التغطية والتحيز؟"),
    ("Validation", "قواعد صريحة: المخطط، الأنواع، المدى، المنطق بين الحقول."),
    ("Quality & cleaning", "تشخيص المفقود والتكرار والشذوذ والاتساق، وقرارات موثقة لا آلية."),
    ("EDA", "فهم التوزيعات والعلاقات وتوليد الفرضيات، مع فصل الاستكشاف عن التأكيد."),
    ("Modeling / inference", "اختيار الطريقة حسب الهدف: وصف، استدلال، تنبؤ، أو سببية."),
    ("Evaluation", "مقاييس مناسبة، تقسيم يحترم التصميم، فحص Leakage، وتحليل حساسية."),
    ("Communication", "رسوم صادقة، حجم أثر وعدم يقين، حدود واضحة، وتوصيات قابلة للتنفيذ."),
    ("Automation & monitoring", "Pipeline قابل لإعادة التشغيل، مراقبة الانجراف، ومراجعة بشرية في نقاط محددة."),
    ("Documentation", "سجل القرارات، الإصدارات، البذور العشوائية، وقاموس البيانات."),
]


def _render(i: int) -> None:
    flow([s[0] for s in STEPS[:5]], highlight=i if i < 5 else None)
    flow([s[0] for s in STEPS[5:]], highlight=i - 5 if i >= 5 else None)
    with st.container(border=True):
        st.markdown(f"**{i + 1}. {STEPS[i][0]}** — {STEPS[i][1]}")


stepper("workflow", len(STEPS), _render, labels=[s[0] for s in STEPS])

intuition("الأطر خرائط لا قضبان قطار: المشروع الحقيقي يعود كثيرًا إلى الخلف. اكتشاف Leakage في التقييم يعيدك "
          "إلى التحضير، واكتشاف أن النموذج يجيب سؤالًا خاطئًا يعيدك إلى البداية.")

if at_least("research"):
    researcher_note([
        "في البحث العلمي أضف: **التسجيل المسبق Pre-registration** للفرضيات والتحليل قبل رؤية البيانات.",
        "افصل بوضوح بين النتائج الاستكشافية والتأكيدية في التقرير.",
        "احفظ لقطة Snapshot من البيانات والبيئة لكل نتيجة منشورة.",
    ])
real_world(["من صاحب القرار النهائي ومتى يحتاج النتيجة؟", "ما تكلفة الخطأ من كل نوع؟",
            "هل البيانات ستتغير بعد النشر؟ ومن سيراقبها؟", "ما الحد الأدنى من الأداء الذي يجعل المشروع مفيدًا؟"])

page_footer("workflows",
            takeaways=["كل الأطر تبدأ أو يجب أن تبدأ بسؤال واضح.", "العملية دورية؛ الرجوع للخلف علامة نضج لا فشل.",
                       "الجودة والتحقق والتوثيق مراحل أساسية لا ملحقات."],
            mistakes=["القفز مباشرة إلى النمذجة.", "اعتبار النشر نهاية المشروع دون مراقبة.",
                      "الالتزام الحرفي بإطار واحد مهما كان السياق."])
