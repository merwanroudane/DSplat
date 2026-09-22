import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, real_world, researcher_note
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("what_is_ds")

st.markdown("## ما هو علم البيانات؟")
st.markdown(
    "**علم البيانات Data Science** هو مجال يدمج **الإحصاء والرياضيات**، و**الحوسبة والبرمجة**، و**معرفة المجال "
    "Domain knowledge** لتحويل البيانات إلى **معرفة قابلة للاستخدام وقرارات مبررة**. "
    "ليس أداة واحدة ولا مرادفًا لتعلّم الآلة؛ بل دورة كاملة تبدأ بسؤال وتنتهي بتواصل النتائج وحدودها."
)

c1, c2 = st.columns([1, 1])
with c1:
    fig = go.Figure()
    circles = [(0, 0.55, "Statistics &<br>Mathematics", PALETTE["coral"]),
               (-0.5, -0.3, "Computing &<br>Programming", PALETTE["purple"]),
               (0.5, -0.3, "Domain<br>Knowledge", PALETTE["amber"])]
    for x, y, label, color in circles:
        fig.add_shape(type="circle", x0=x - 0.8, y0=y - 0.8, x1=x + 0.8, y1=y + 0.8,
                      fillcolor=color, opacity=0.28, line=dict(color=color, width=2))
        fig.add_annotation(x=x, y=y + (0.45 if y > 0 else -0.45), text=f"<b>{label}</b>", showarrow=False,
                           font=dict(size=12))
    fig.add_annotation(x=0, y=0, text="<b>Data<br>Science</b>", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=-0.62, y=0.3, text="Machine<br>Learning", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.62, y=0.3, text="Traditional<br>Research", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0, y=-0.78, text="Danger<br>zone!", showarrow=False, font=dict(size=10, color="#5F3DC4"))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5])
    fig.update_yaxes(visible=False, range=[-1.25, 1.45], scaleanchor="x")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    plot(fig)
    st.caption("مخطط Venn الشهير (Drew Conway, 2010): تقاطع البرمجة مع المجال دون إحصاء = «منطقة خطر»؛ "
               "كود يعمل وأرقام مقنعة بلا فهم لعدم اليقين.")
with c2:
    intuition(
        "تخيّل طبيبًا: **الإحصاء** يخبره كيف يقرأ نتائج التحاليل بعدم يقين، **البرمجة** تجعله يحلل آلاف الملفات، "
        "و**معرفة المجال** تخبره أن ضغط الدم 300 خطأ قياس لا مريض استثنائي. علم البيانات يحتاج الثلاثة معًا."
    )
    st.markdown("**أنواع الأسئلة التحليلية:**")
    st.markdown(
        "- **Descriptive** — ماذا حدث؟ (مبيعات الشهر)\n"
        "- **Diagnostic** — لماذا حدث؟ (سبب انخفاض المبيعات)\n"
        "- **Predictive** — ماذا سيحدث؟ (الطلب الأسبوع القادم)\n"
        "- **Prescriptive** — ماذا نفعل؟ (أفضل مستوى مخزون)\n"
        "- **Causal** — ماذا لو تدخلنا؟ (أثر خصم 10%) — يحتاج تصميمًا تجريبيًا أو شبه تجريبي"
    )

# ------------------------------------------------------------------ fields
FIELDS = {
    "Data Science": dict(goal="استخراج معرفة وقرارات من البيانات عبر الدورة كاملة", q="ما الذي يحدث؟ لماذا؟ ماذا سيحدث؟",
                         data="كل الأنواع: جداول، نصوص، صور، سلاسل", methods="إحصاء، ML، تصوير، تجارب",
                         tools="Python, R, SQL, pandas, scikit-learn", out="تحليلات، نماذج، تقارير، منتجات بيانات",
                         overlap="يشمل أغلب المجالات الأخرى كأدوات", ex="نظام لتقليل مغادرة العملاء من التشخيص إلى التدخل",
                         radar=[4, 4, 4, 4, 2, 4]),
    "Data Analysis": dict(goal="فحص البيانات للإجابة عن سؤال محدد", q="ماذا تقول البيانات عن هذا السؤال؟",
                          data="جداول منظمة غالبًا", methods="إحصاء وصفي، تجميع، رسوم", tools="Excel, SQL, pandas",
                          out="إجابات، جداول، رسوم", overlap="جزء من Data Science وAnalytics",
                          ex="تحليل أسباب ارتفاع الشكاوى في فرع معين", radar=[3, 2, 4, 1, 1, 4]),
    "Data Analytics": dict(goal="استخدام منهجي ومتكرر للتحليل لدعم قرارات الأعمال", q="كيف نحسن الأداء باستمرار؟",
                           data="بيانات تشغيلية وتاريخية", methods="KPIs، تحليل الاتجاهات، التجزئة، A/B",
                           tools="SQL, BI tools, Python", out="مؤشرات ولوحات وتوصيات", overlap="أوسع من Data Analysis وأقرب للأعمال",
                           ex="نظام مؤشرات أسبوعي لأداء الحملات التسويقية", radar=[3, 3, 4, 2, 2, 3]),
    "Data Mining": dict(goal="اكتشاف أنماط غير معروفة مسبقًا في بيانات كبيرة", q="ما الأنماط المخفية؟",
                        data="قواعد بيانات كبيرة، معاملات", methods="قواعد ارتباط، تجميع، تصنيف، كشف شذوذ",
                        tools="Python, Weka, SQL", out="أنماط وقواعد ومجموعات", overlap="يتداخل بقوة مع ML والإحصاء",
                        ex="اكتشاف المنتجات التي تُشترى معًا", radar=[3, 4, 3, 3, 2, 3]),
    "Statistics": dict(goal="الاستدلال من العينة إلى المجتمع مع قياس عدم اليقين", q="هل الفرق حقيقي؟ ما مقداره وثقتنا به؟",
                       data="عينات مصممة، تجارب، مسوح", methods="تقدير، اختبارات، انحدار، تصميم تجارب",
                       tools="R, Stata, statsmodels", out="تقديرات بفترات ثقة واستنتاجات", overlap="أساس نظري لـML وData Science",
                       ex="تقدير أثر برنامج تدريبي على الأجور", radar=[5, 2, 3, 2, 1, 5]),
    "Machine Learning": dict(goal="بناء خوارزميات تتعلم من البيانات للتنبؤ أو اتخاذ قرار", q="ما التنبؤ الأدق على بيانات جديدة؟",
                             data="بيانات كبيرة موسومة أو غير موسومة", methods="انحدار، أشجار، شبكات، تجميع",
                             tools="scikit-learn, PyTorch, XGBoost", out="نماذج تنبؤية", overlap="أداة مركزية في Data Science",
                             ex="نموذج يتنبأ بتعثر القروض", radar=[3, 5, 2, 5, 2, 2]),
    "Artificial Intelligence": dict(goal="أنظمة تؤدي مهامًا تتطلب ذكاءً بشريًا", q="كيف يدرك النظام ويستنتج ويتصرف؟",
                                    data="نصوص، صور، صوت، بيئات", methods="ML، بحث، استدلال، تخطيط، وكلاء",
                                    tools="PyTorch, LLM APIs", out="أنظمة ذكية ومساعدون", overlap="ML فرع منه",
                                    ex="مساعد يجيب عن أسئلة العملاء", radar=[2, 5, 2, 5, 3, 2]),
    "Business Intelligence": dict(goal="تقديم رؤية واضحة للأداء التاريخي والحالي", q="كيف كان أداؤنا؟",
                                  data="مستودعات بيانات منظمة", methods="تجميع، لوحات، تقارير", tools="Power BI, Tableau, SQL",
                                  out="لوحات وتقارير دورية", overlap="يعتمد على هندسة البيانات، وصفي غالبًا",
                                  ex="لوحة مبيعات يومية للإدارة", radar=[1, 2, 4, 1, 3, 3]),
    "Data Engineering": dict(goal="بناء بنية بيانات موثوقة وقابلة للتوسع", q="كيف تصل البيانات الصحيحة في الوقت المناسب؟",
                             data="كل المصادر وبأحجام كبيرة", methods="ETL/ELT، نمذجة البيانات، التدفق",
                             tools="SQL, Spark, Airflow, dbt", out="Pipelines ومستودعات وجودة", overlap="أساس لكل ما سبق",
                             ex="Pipeline يومي يجمع مبيعات 400 فرع", radar=[1, 5, 2, 1, 5, 1]),
    "Big Data Analytics": dict(goal="تحليل بيانات تتجاوز قدرة جهاز واحد", q="كيف نحلل مليارات السجلات؟",
                               data="حجم/سرعة/تنوع كبير", methods="معالجة موزعة، تقريب، تدفق", tools="Spark, Dask, BigQuery",
                               out="تحليلات على نطاق واسع", overlap="Analytics + Data Engineering",
                               ex="تحليل سجلات نقرات موقع ضخم", radar=[2, 5, 2, 3, 5, 2]),
    "Deep Learning": dict(goal="تعلّم تمثيلات متعددة الطبقات بالشبكات العصبية", q="كيف نتعلم من بيانات غير منظمة؟",
                          data="صور، نصوص، صوت بكميات كبيرة", methods="CNN, RNN, Transformers", tools="PyTorch, TensorFlow, JAX",
                          out="نماذج إدراك وتوليد", overlap="فرع من ML", ex="تصنيف صور الأشعة", radar=[2, 5, 2, 5, 3, 1]),
    "Generative AI": dict(goal="توليد محتوى جديد (نص، صورة، كود)", q="كيف ننتج محتوى يشبه البيانات؟",
                          data="كميات ضخمة غير منظمة", methods="Transformers, diffusion, LLMs", tools="LLM APIs, Hugging Face",
                          out="نصوص وصور وكود مولّد", overlap="Deep Learning + AI", ex="مساعد يكتب كود تحليل من وصف لغوي",
                          radar=[1, 5, 2, 4, 3, 1]),
    "AutoML": dict(goal="أتمتة اختيار النماذج وضبطها", q="ما أفضل نموذج ضمن ميزانية محددة؟",
                   data="جداول غالبًا", methods="بحث في النماذج والمعاملات، Ensembles", tools="FLAML, AutoGluon, TPOT",
                   out="نموذج مختار وLeaderboard", overlap="أداة داخل ML", ex="بحث آلي عن أفضل مصنف للتعثر",
                   radar=[2, 4, 1, 5, 2, 1]),
    "Automated Analytics": dict(goal="أتمتة خطوات التحليل المتكررة (Profiling، تنبيهات، تقارير)", q="ما الذي تغيّر ويستحق الانتباه؟",
                                data="بيانات تشغيلية متجددة", methods="قواعد، إحصاء آلي، LLM للتلخيص",
                                tools="Profilers, alerting, notebooks مجدولة", out="تقارير وتنبيهات آلية",
                                overlap="Analytics + Automation + AI", ex="تقرير أسبوعي يلخص تغيرات المبيعات آليًا",
                                radar=[2, 4, 3, 2, 3, 2]),
}
ATTRS = {"goal": "الهدف الرئيسي Main goal", "q": "الأسئلة النموذجية", "data": "البيانات النموذجية", "methods": "الطرق",
         "tools": "الأدوات", "out": "المخرجات", "overlap": "التداخل", "ex": "مثال مشروع"}
AXES = ["Statistics", "Programming", "Domain", "Prediction", "Infrastructure", "Explanation"]

st.markdown("## مقارنة تفاعلية بين المجالات · Interactive comparison")
st.caption("اختر مجالين أو أكثر للمقارنة. المخطط الراداري تقدير تعليمي للتركيز النسبي (1–5)، وليس قياسًا دقيقًا.")
chosen = st.multiselect("المجالات", list(FIELDS), default=["Data Science", "Statistics", "Machine Learning"],
                        max_selections=4, key="wids_fields")
if chosen:
    fig = go.Figure()
    colors = [PALETTE["coral"], PALETTE["purple"], PALETTE["amber"], PALETTE["sky"]]
    for f, color in zip(chosen, colors):
        r = FIELDS[f]["radar"]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=AXES + AXES[:1], name=f, fill="toself", opacity=0.45,
                                      line=dict(color=color)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5], visible=True)), height=420,
                      legend=dict(orientation="h", y=-0.1))
    plot(fig)
    rows = [{"البعد": ATTRS[a], **{f: FIELDS[f][a] for f in chosen}} for a in ATTRS]
    comparison_table(rows)

with st.expander("الجدول الكامل لكل المجالات (14 مجالًا)", icon=":material/table_chart:"):
    comparison_table([{"المجال": f, **{ATTRS[a]: d[a] for a in ("goal", "methods", "out", "overlap")}}
                      for f, d in FIELDS.items()])

st.markdown("## تمييزات دقيقة يخلط بينها الطلاب")
comparison_table([
    {"المقارنة": "Data Analysis vs Data Analytics",
     "الفرق الجوهري": "Analysis: فحص محدد لسؤال. Analytics: ممارسة منهجية متكررة (غالبًا مؤسسية) تشمل التحليل والأدوات والعمليات."},
    {"المقارنة": "Data Mining vs Machine Learning",
     "الفرق الجوهري": "Mining: هدفه اكتشاف أنماط مفهومة للبشر. ML: هدفه أداء تنبؤي على بيانات جديدة. الأدوات مشتركة كثيرًا."},
    {"المقارنة": "Statistics vs Machine Learning",
     "الفرق الجوهري": "الإحصاء: الاستدلال وعدم اليقين والتفسير. ML: التنبؤ والتعميم. يتقاربان في «Statistical learning»."},
    {"المقارنة": "AI vs ML vs Deep Learning",
     "الفرق الجوهري": "دوائر متداخلة: DL ⊂ ML ⊂ AI. الذكاء الاصطناعي يشمل أيضًا البحث والاستدلال الرمزي."},
    {"المقارنة": "BI vs Data Science",
     "الفرق الجوهري": "BI يصف الماضي والحاضر بلوحات؛ Data Science يضيف التنبؤ والتجريب والنمذجة."},
])

if at_least("advanced"):
    st.markdown("## منظور متقدم: ثقافتا النمذجة")
    st.markdown(
        "صاغ Leo Breiman (2001) الفرق بين **ثقافة نمذجة البيانات** (افتراض نموذج احتمالي وتفسير معاملاته) و**ثقافة "
        "النمذجة الخوارزمية** (معاملة الآلية كصندوق أسود والحكم بالأداء التنبؤي). علم البيانات الحديث يحتاج الاثنتين: "
        "التنبؤ لا يغني عن الاستدلال السببي، والنموذج البسيط القابل للتفسير لا يكفي دائمًا للتنبؤ."
    )
if at_least("research"):
    researcher_note([
        "حدّد في كل دراسة **هدفها المعرفي** بوضوح: وصف، تنبؤ، أم تفسير سببي (Shmueli, 2010 «To Explain or to Predict?»).",
        "النموذج الأعلى دقة تنبؤية ليس بالضرورة الأصلح لاستنتاج الأسباب.",
        "صرّح في التقرير بالقرارات التحليلية التي قد تغيّر النتائج (تنظيف، اختيار متغيرات، معالجة المفقود).",
    ])

real_world([
    "ما القرار الذي سيتخذه أحدهم بناءً على هذا التحليل؟",
    "هل السؤال وصفي أم تنبؤي أم سببي؟ فالطرق تختلف جذريًا.",
    "هل البيانات المتاحة قادرة أصلًا على الإجابة عن السؤال؟",
    "من يملك معرفة المجال التي تفسر الأرقام؟",
])

page_footer(
    "what_is_ds",
    takeaways=["علم البيانات = إحصاء + حوسبة + معرفة مجال، عبر دورة كاملة من السؤال إلى التواصل.",
               "المجالات متداخلة؛ الفرق غالبًا في الهدف ونوع السؤال لا في الأدوات.",
               "التنبؤ والتفسير هدفان مختلفان يتطلبان طرقًا وتقييمًا مختلفين."],
    mistakes=["اختزال علم البيانات في تعلّم الآلة.", "البدء بالأداة قبل تحديد السؤال والقرار.",
              "تفسير نموذج تنبؤي كأنه نموذج سببي."],
)
