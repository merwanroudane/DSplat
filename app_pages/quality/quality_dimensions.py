import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.dataset_viewer import dataset_card
from components.diagrams import mermaid
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import customers_clean, customers_raw
from utils.plotting import plot
from utils.quality import DEFAULT_WEIGHTS, DIMENSIONS, quality_report, weighted_score
from utils.validation import CUSTOMER_RULES

page_header("quality_dimensions")

st.markdown("## الجودة = الملاءمة للاستخدام · Fitness for use")
st.markdown(
    "لا توجد بيانات «جيدة» بالمطلق. بيانات دقيقة لكنها متأخرة شهرًا ممتازة لتقرير سنوي وسيئة لقرار يومي. "
    "لذلك نقيس الجودة عبر **أبعاد Dimensions** مستقلة، ونزنها حسب الاستخدام."
)
comparison_table([
    {"البعد": "Completeness الاكتمال", "السؤال": "هل القيم المطلوبة موجودة؟", "مثال على الخلل": "دخل مفقود لـ20% من العملاء"},
    {"البعد": "Accuracy الدقة", "السؤال": "هل القيمة تطابق الواقع؟", "مثال على الخلل": "عمر مسجل 34 والحقيقي 43"},
    {"البعد": "Consistency الاتساق", "السؤال": "هل التمثيل موحد وغير متناقض؟", "مثال على الخلل": "USA / usa / U.S.A"},
    {"البعد": "Validity الصلاحية", "السؤال": "هل القيمة ضمن النوع والمدى والصيغة؟", "مثال على الخلل": "عمر = 250"},
    {"البعد": "Uniqueness التفرّد", "السؤال": "هل يظهر الكيان مرة واحدة؟", "مثال على الخلل": "عميل مسجل مرتين"},
    {"البعد": "Timeliness الحداثة", "السؤال": "هل البيانات حديثة بما يكفي؟", "مثال على الخلل": "مخزون يُحدَّث شهريًا لقرار يومي"},
    {"البعد": "Integrity السلامة", "السؤال": "هل العلاقات بين الحقول والجداول صحيحة؟", "مثال على الخلل": "طلب لعميل غير موجود"},
    {"البعد": "Reliability الموثوقية", "السؤال": "هل تنتج عملية الجمع نفس النتيجة باستمرار؟", "مثال على الخلل": "جهاز قياس ينجرف"},
    {"البعد": "Conformity المطابقة", "السؤال": "هل تتبع الصيغة المعيارية؟", "مثال على الخلل": "تواريخ بصيغ مختلطة"},
    {"البعد": "Relevance الصلة", "السؤال": "هل تجيب البيانات عن السؤال؟", "مثال على الخلل": "بيانات مبيعات لسؤال عن الرضا"},
])
st.caption("ملاحظة: Accuracy لا يمكن قياسها آليًا دون مرجع حقيقة Ground truth؛ هنا نملك الحقيقة لأن البيانات اصطناعية.")

st.markdown("## سير عمل الجودة")
mermaid("""
flowchart LR
  A[Define use & rules] --> B[Measure dimensions] --> C[Diagnose root causes] --> D[Fix at source or clean]
  D --> E[Re-measure] --> F[Monitor over time]
  F -. new issues .-> B
""")

st.markdown("## لوحة جودة البيانات · Data Quality Dashboard")
st.markdown("الدرجة الكلية **متوسط موزون** لدرجات فرعية، كل منها بصيغة معلنة. غيّر الأوزان لترى كيف يتغير الحكم.")
formula(r"Q = \frac{\sum_{i} w_i \, s_i}{\sum_{i} w_i}, \qquad s_i \in [0, 1]", title="صيغة الدرجة الكلية",
        symbols={"s_i": "الدرجة الفرعية للبعد i", "w_i": "وزن البعد i (تحدده أنت حسب الاستخدام)"},
        intuition="رقم واحد مريح لكنه يخفي التفاصيل؛ لذلك نعرض دائمًا الدرجات الفرعية بجانبه.")

version = st.segmented_control("البيانات", ["الخام (التحدي)", "النظيفة (الحقيقة)"], default="الخام (التحدي)",
                               key="qd_ver")
df = customers_raw() if version != "النظيفة (الحقيقة)" else customers_clean()
dataset_card("customers_raw" if version != "النظيفة (الحقيقة)" else "customers_clean")

with st.expander("الأوزان Weights", expanded=True, icon=":material/tune:"):
    cols = st.columns(3)
    weights = {}
    for i, (k, meta) in enumerate(DIMENSIONS.items()):
        weights[k] = cols[i % 3].slider(f"{meta['ar']} ({k})", 0, 5, DEFAULT_WEIGHTS[k], key=f"qd_w_{k}")

sub = quality_report(df, CUSTOMER_RULES, key="customer_id")
score = weighted_score(sub, weights)
c1, c2 = st.columns([1, 2])
with c1:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=round(100 * score, 1), number=dict(suffix="%"),
                                 gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=PALETTE["coral"]),
                                            steps=[dict(range=[0, 70], color="#FFE8CC"),
                                                   dict(range=[70, 90], color="#FFF3BF"),
                                                   dict(range=[90, 100], color="#E6F8FB")])))
    fig.update_layout(height=260, margin=dict(t=30, b=10))
    plot(fig)
with c2:
    subdf = pd.DataFrame({"dimension": [DIMENSIONS[k]["ar"] + f" ({k})" for k in sub],
                          "score": [100 * v for v in sub.values()], "weight": [weights[k] for k in sub]})
    fig = go.Figure(go.Bar(x=subdf["score"], y=subdf["dimension"], orientation="h",
                           marker_color=[PALETTE["coral"] if s < 90 else PALETTE["teal"] for s in subdf["score"]],
                           text=[f"{s:.1f}% · w={w}" for s, w in zip(subdf["score"], subdf["weight"])],
                           textposition="auto"))
    fig.update_layout(height=300, xaxis=dict(range=[0, 100], title="sub-score %"), margin=dict(t=10))
    plot(fig)

with st.expander("كيف تُحسب كل درجة فرعية؟", icon=":material/functions:"):
    for k, meta in DIMENSIONS.items():
        st.markdown(f"**{meta['ar']} · {k}** — {meta['desc']}")
        st.latex(meta["formula"])

weakest = min(sub, key=sub.get)
why(f"ابدأ التحسين من بُعد «{DIMENSIONS[weakest]['ar']}» ({100 * sub[weakest]:.1f}%).",
    "هو الأضعف، وتحسين الأبعاد الضعيفة ذات الوزن العالي يرفع الملاءمة للاستخدام أكثر من تلميع الأبعاد الجيدة.")

if at_least("advanced"):
    st.markdown("## متقدم: حدود المؤشر الواحد")
    st.markdown(
        "- **حساسية الأوزان:** غيّر وزنًا واحدًا وقد ينتقل الحكم من «مقبول» إلى «سيئ».\n"
        "- **عدم الاستقلال:** Sentinel مثل -999 يؤثر على Completeness وValidity معًا.\n"
        "- **قانون Goodhart:** حين يصبح المقياس هدفًا يفقد قيمته؛ قد يملأ الفريق الحقول بقيم افتراضية لرفع الاكتمال."
    )
if at_least("research"):
    researcher_note(["أبلغ عن مقاييس الجودة لكل متغير أساسي في الدراسة، لا لمجمل الملف فقط.",
                     "وثّق قواعد الصلاحية المستخدمة قبل التحليل لتجنب تعديلها بعد رؤية النتائج.",
                     "المراجع: Wang & Strong (1996) لأبعاد الجودة؛ ISO/IEC 25012 لنموذج جودة البيانات."])
real_world(["ما الاستخدام المحدد لهذه البيانات؟ (يحدد الأوزان)", "أي الأبعاد لها تكلفة أعلى عند الخطأ؟",
            "هل يمكن إصلاح المشكلة في المصدر بدل التنظيف المتكرر؟", "هل تتحسن الجودة أم تتدهور عبر الزمن؟"])

page_footer("quality_dimensions",
            takeaways=["الجودة متعددة الأبعاد وتُقاس حسب الاستخدام.", "كل درجة يجب أن تكون بصيغة معلنة وأوزان قابلة للتعديل.",
                       "الدرجة الكلية لا تُعرض أبدًا دون الدرجات الفرعية."],
            mistakes=["نشر Score واحد غامض.", "اعتبار الاكتمال مرادفًا للجودة.", "التنظيف المتكرر بدل إصلاح المصدر."])
