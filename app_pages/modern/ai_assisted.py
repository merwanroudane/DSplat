import re

import pandas as pd
import plotly.express as px
import streamlit as st

from components.callouts import real_world, researcher_note, warning, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import SEQUENCE
from utils.datasets import customers_clean
from utils.plotting import plot

page_header("ai_assisted")

st.markdown("## أنماط استخدام الذكاء الاصطناعي في التحليل")
comparison_table([
    {"النمط": "AI Assistant", "ماذا يفعل": "يجيب أسئلة ويشرح مفاهيم", "مثال": "«ما الفرق بين MAR وMNAR؟»", "المخاطرة": "منخفضة إن تُحقق من الإجابة"},
    {"النمط": "AI Copilot", "ماذا يفعل": "يقترح كودًا داخل المحرر", "مثال": "إكمال كود pandas", "المخاطرة": "كود صحيح نحويًا خاطئ منطقيًا"},
    {"النمط": "NL → Code", "ماذا يفعل": "يحول سؤالًا لغويًا إلى SQL/Python", "مثال": "«متوسط الإنفاق حسب الدولة»", "المخاطرة": "فهم خاطئ للأعمدة أو المقاييس"},
    {"النمط": "NL → Chart", "ماذا يفعل": "يولّد رسمًا من وصف", "مثال": "«ارسم توزيع الأعمار»", "المخاطرة": "رسم مضلل أو نوع غير مناسب"},
    {"النمط": "Data Agent", "ماذا يفعل": "يخطط وينفذ أدوات ويقرأ النتائج في حلقة", "مثال": "«ابحث عن سبب انخفاض المبيعات»", "المخاطرة": "تنفيذ أوامر، صلاحيات، تحويلات خفية"},
    {"النمط": "Automated reports", "ماذا يفعل": "يلخص نتائج التحليل نصيًا", "مثال": "ملخص تنفيذي أسبوعي", "المخاطرة": "مبالغة، أرقام مهلوسة"},
    {"النمط": "AI documentation", "ماذا يفعل": "يكتب Docstrings وقواميس بيانات", "مثال": "وصف أعمدة الجدول", "المخاطرة": "وصف واثق لكنه غير صحيح"},
])

st.markdown("## كيف يعمل وكيل البيانات Data Agent؟")
mermaid("""
flowchart LR
  U[User question] --> P[LLM plans]
  P --> T{Tool call}
  T --> SQL[SQL / pandas tool<br/>sandboxed, read-only]
  T --> CH[Chart tool]
  SQL --> O[Observed result]
  CH --> O
  O --> P
  P --> A[Answer + code + evidence]
  A --> H[Human review]
  SL[Semantic layer:<br/>metric definitions] -.-> P
  LOG[Audit log] -.-> T
""")
st.markdown("الفرق عن Chatbot: الوكيل **ينفذ أدوات** ويقرأ نتائجها ويقرر الخطوة التالية. لذلك يحتاج: صلاحيات للقراءة فقط، "
            "بيئة معزولة Sandbox، سجل تدقيق، وطبقة دلالية Semantic layer تعرّف المقاييس بدل تخمينها من أسماء الأعمدة.")

st.markdown("## محاكاة: من سؤال لغوي إلى كود قابل للتدقيق")
warning("هذا عرض **حتمي بقواعد** يحاكي فكرة NL→Code دون نموذج لغوي ودون إرسال أي بيانات لأي خدمة. "
        "الهدف تعليم **مراجعة** الكود المولّد، لا استبدال التحليل.", title="تنبيه شفافية")
df = customers_clean()
COLS = {"spend": "monthly_spend", "إنفاق": "monthly_spend", "income": "annual_income", "دخل": "annual_income",
        "age": "age", "عمر": "age", "orders": "num_orders", "طلبات": "num_orders", "satisfaction": "satisfaction", "رضا": "satisfaction"}
GROUPS = {"country": "country", "دولة": "country", "membership": "membership", "عضوية": "membership", "gender": "gender", "جنس": "gender"}
AGG = {"average": "mean", "mean": "mean", "متوسط": "mean", "median": "median", "وسيط": "median", "total": "sum", "مجموع": "sum",
       "max": "max", "أعلى": "max", "count": "count", "عدد": "count"}
examples = ["متوسط الإنفاق حسب الدولة", "median income by membership", "عدد الطلبات حسب الجنس", "ارسم توزيع العمر"]
q = st.selectbox("اختر سؤالًا (أو اكتب في الحقل أدناه)", examples, key="ai_q")
q2 = st.text_input("أو اكتب سؤالك", "", key="ai_q2", placeholder="مثال: متوسط الرضا حسب العضوية")
question = (q2 or q).lower()


def find(mapping: dict[str, str]) -> str | None:
    for k, v in mapping.items():
        if re.search(rf"\b{re.escape(k)}\b", question) or k in question:
            return v
    return None


metric, group, agg = find(COLS), find(GROUPS), find(AGG)
is_plot = any(w in question for w in ("ارسم", "plot", "distribution", "توزيع", "histogram"))
if metric is None:
    st.error("لم أتعرف على المقياس المطلوب. وكيل حقيقي قد «يخمّن» عمودًا — وهذا بالضبط الخطر: الأفضل أن يسأل للتوضيح.")
else:
    if is_plot:
        code = f'fig = px.histogram(df, x="{metric}")'
        st.code(code, language="python")
        fig = px.histogram(df, x=metric, nbins=40, color_discrete_sequence=SEQUENCE)
        fig.update_layout(height=320)
        plot(fig)
    else:
        agg = agg or "mean"
        if group:
            code = f'df.groupby("{group}")["{metric}"].{agg}().sort_values(ascending=False)'
            result = getattr(df.groupby(group)[metric], agg)().sort_values(ascending=False)
        else:
            code = f'df["{metric}"].{agg}()'
            result = getattr(df[metric], agg)()
        st.code(code, language="python")
        if isinstance(result, pd.Series):
            st.dataframe(result.round(2))
        else:
            st.metric("النتيجة", f"{result:,.2f}")
        with st.container(border=True):
            st.markdown("**قائمة مراجعة الكود المولّد:**")
            st.checkbox(f"هل «{metric}» هو المقياس الذي قصده السؤال؟", key="ai_c1")
            st.checkbox(f"هل الدالة `{agg}` مناسبة؟ (المتوسط مع بيانات ملتوية قد يضلل)", key="ai_c2")
            st.checkbox("هل تم استبعاد القيم المفقودة/الشاذة بطريقة مقصودة؟", key="ai_c3")
            st.checkbox("هل النتيجة قابلة لإعادة الإنتاج (الكود محفوظ)؟", key="ai_c4")

st.markdown("## مختبر تدقيق مخرجات الذكاء الاصطناعي · AI Output Audit")
st.markdown("فيما يلي «تحليلات» مكتوبة بأسلوب مساعد ذكاء اصطناعي عن بيانات العملاء. **كل واحد يحتوي على خطأ.** "
            "حاول اكتشافه قبل كشف الإجابة؛ كل الأرقام الحقيقية محسوبة من البيانات.")
true_mean = df["monthly_spend"].mean()
true_median = df["monthly_spend"].median()
corr = df[["annual_income", "monthly_spend"]].corr().iloc[0, 1]
churn_rate = df["churned"].mean()
AUDIT = [
    (f"«متوسط الإنفاق الشهري {true_mean * 0.83:.2f} وهذا يمثل العميل النموذجي بدقة.»",
     f"رقم مهلوس + تفسير خاطئ: المتوسط الفعلي {true_mean:.2f}، والتوزيع ملتوٍ (الوسيط {true_median:.2f})، فالمتوسط لا يمثل «العميل النموذجي».",
     "Wrong statistics"),
    (f"«الارتباط بين الدخل والإنفاق {corr:.2f}، مما يثبت أن زيادة الدخل تسبب زيادة الإنفاق.»",
     "الرقم صحيح لكن الاستنتاج سببي من بيانات رصدية: الارتباط لا يثبت السببية، وقد توجد متغيرات مربكة (العمر، العضوية).",
     "Over-claiming"),
    ("«حذفتُ القيم الشاذة في الإنفاق لتحسين جودة التحليل، فأصبحت النتائج أوضح.»",
     "تحويل خفي Hidden transformation غير موثق: أي قيم؟ بأي قاعدة؟ كم صفًا؟ وقد تكون أهم العملاء. يجب الإفصاح والتبرير.",
     "Hidden transformations"),
    ("«لتحسين نموذج المغادرة، أضفت عمود days_until_churn كخاصية فارتفعت الدقة إلى 99%.»",
     "Data leakage: عدد الأيام حتى المغادرة لا يُعرف إلا بعد المغادرة. الأداء المثالي علامة إنذار.", "Data leakage"),
    (f"«معدل المغادرة {churn_rate:.1%}. أرسلتُ ملف العملاء إلى خدمة تحليل خارجية للحصول على رؤى أعمق.»",
     "مخاطرة خصوصية: إرسال بيانات العملاء لخدمة خارجية دون تفويض أو إخفاء هوية مخالف للحوكمة وربما للقانون.", "Privacy"),
    ("«شغّلت التحليل مرة أخرى وحصلت على نتائج مختلفة قليلًا، وهذا طبيعي.»",
     "غياب قابلية إعادة الإنتاج: يجب تثبيت البذور العشوائية وحفظ الكود والإصدارات؛ «مختلفة قليلًا» تحتاج تفسيرًا.", "Reproducibility"),
]
score = 0
for i, (claim, answer, tag) in enumerate(AUDIT):
    with st.container(border=True):
        st.markdown(f"**{i + 1}.** {claim}")
        guess = st.selectbox("ما نوع الخطأ؟", ["—", "Wrong statistics", "Over-claiming", "Hidden transformations", "Data leakage",
                                                "Privacy", "Reproducibility"], key=f"ai_g{i}")
        if st.toggle("اكشف الإجابة", key=f"ai_r{i}"):
            if guess == tag:
                st.success(f"**{tag}** — {answer}")
            else:
                st.info(f"**{tag}** — {answer}")
        score += int(guess == tag)
st.metric("إجاباتك الصحيحة", f"{score} / {len(AUDIT)}")

st.markdown("## المخاطر والضوابط")
comparison_table([
    {"الخطر": "Hallucination", "الوصف": "أرقام أو دوال أو مراجع مخترعة", "الضابط": "كل رقم يأتي من كود مُنفّذ ومحفوظ"},
    {"الخطر": "Wrong statistics", "الوصف": "اختبار غير مناسب أو تفسير خاطئ لـp-value", "الضابط": "مراجعة إحصائية بشرية"},
    {"الخطر": "Data leakage", "الوصف": "خصائص مسربة في النماذج المقترحة", "الضابط": "قائمة فحص التسرب"},
    {"الخطر": "Privacy", "الوصف": "إرسال بيانات حساسة لخدمات خارجية", "الضابط": "أرسل المخطط والإحصاءات المجمعة فقط، أو بيانات اصطناعية"},
    {"الخطر": "Reproducibility", "الوصف": "مخرجات غير حتمية", "الضابط": "احفظ الكود لا النص؛ ثبّت البذور"},
    {"الخطر": "Hidden transformations", "الوصف": "حذف أو تعديل دون إفصاح", "الضابط": "سجل تحويلات إلزامي"},
    {"الخطر": "Over-automation", "الوصف": "قبول المخرجات دون تمحيص (Automation bias)", "الضابط": "مراجعة بنقاط توقف إلزامية"},
])
why("اطلب من الذكاء الاصطناعي **كودًا** لا **أرقامًا**، ثم نفّذ الكود وراجعه بنفسك.",
    "الكود قابل للتدقيق وإعادة التشغيل؛ الرقم في نص المحادثة غير قابل للتحقق وقد يكون مهلوسًا.")
st.markdown("**سياسة هذه المنصة:** لا يوجد أي تكامل خارجي مع نماذج لغوية؛ لا تُرسل بياناتك المرفوعة إلى أي API. "
            "أي تكامل مستقبلي يجب أن يكون اختياريًا وواضحًا ومفاتيحه في `st.secrets` لا في الكود.")

if at_least("advanced"):
    st.markdown("## متقدم: الطبقة الدلالية Semantic layer")
    st.markdown("تعريف مركزي للمقاييس (مثل «الإيراد الصافي = الإيراد − المرتجعات − الخصومات») والكيانات والعلاقات. "
                "عندما يستعلم الوكيل عبرها يستخدم تعريفًا واحدًا متفقًا عليه بدل تخمين الأعمدة، فتقل الأخطاء وتتسق الأرقام بين الفرق.")
if at_least("research"):
    researcher_note(["عند استخدام LLM في البحث: وثّق النموذج والإصدار والتاريخ والتعليمات (Prompts) كجزء من المنهجية.",
                     "لا تستخدم LLM لتوليد بيانات أو نتائج تُقدَّم كحقيقية.",
                     "قيّم دقة الوكيل على مجموعة أسئلة ذات إجابات معروفة قبل الاعتماد عليه."])
real_world(["ما البيانات المسموح مشاركتها مع خدمة AI؟", "هل الكود المولّد محفوظ ومراجع؟", "من المسؤول عن الخطأ؟",
            "هل توجد طبقة دلالية أو قاموس مقاييس رسمي؟"])

page_footer("ai_assisted",
            takeaways=["AI مساعد قوي في الصياغة والكود والتلخيص، لا مصدر للحقيقة.", "الوكيل ينفذ أدوات فيحتاج صلاحيات محدودة وسجل تدقيق.",
                       "المخاطر: هلوسة، إحصاء خاطئ، تسرب، خصوصية، إعادة إنتاج، تحويلات خفية، إفراط في الأتمتة.",
                       "اطلب الكود لا الأرقام."],
            mistakes=["نسخ أرقام من محادثة إلى تقرير.", "رفع بيانات حساسة لخدمة عامة.", "قبول استنتاج سببي من AI دون تصميم مناسب."])
