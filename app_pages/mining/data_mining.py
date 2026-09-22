import plotly.express as px
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.diagrams import flow, mermaid
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import baskets, transactions
from utils.mining import apriori, association_rules
from utils.plotting import plot

page_header("data_mining")

st.markdown("## ما التنقيب في البيانات؟")
st.markdown("**Data Mining** اكتشاف أنماط **صالحة وجديدة ومفيدة وقابلة للفهم** في بيانات كبيرة (Fayyad et al., 1996). "
            "هو خطوة داخل عملية أوسع لاكتشاف المعرفة KDD:")
flow(["Selection", "Preprocessing", "Transformation", "Data Mining", "Evaluation & Interpretation", "Knowledge"], highlight=3)
comparison_table([
    {"المهمة": "Pattern discovery", "السؤال": "ما الأنماط المتكررة؟", "أمثلة طرق": "Frequent itemsets", "الوحدة": "هذه الصفحة"},
    {"المهمة": "Association rules", "السؤال": "ما الذي يظهر معًا؟", "أمثلة طرق": "Apriori، FP-Growth", "الوحدة": "هذه الصفحة"},
    {"المهمة": "Clustering", "السؤال": "ما المجموعات الطبيعية؟", "أمثلة طرق": "K-Means، DBSCAN", "الوحدة": "Clustering"},
    {"المهمة": "Classification", "السؤال": "لأي فئة تنتمي؟", "أمثلة طرق": "أشجار القرار، Naive Bayes", "الوحدة": "ML"},
    {"المهمة": "Anomaly detection", "السؤال": "ما غير المعتاد؟", "أمثلة طرق": "Isolation Forest، LOF", "الوحدة": "Anomaly detection"},
    {"المهمة": "Sequential patterns", "السؤال": "ما الذي يتبع ماذا؟", "أمثلة طرق": "GSP، PrefixSpan", "الوحدة": "مفهوم"},
    {"المهمة": "Dimensionality reduction", "السؤال": "ما البنية المختصرة؟", "أمثلة طرق": "PCA", "الوحدة": "Dimensionality reduction"},
])
mermaid("""
flowchart LR
  ST[Statistics<br/>inference, uncertainty] --- DM[Data Mining<br/>pattern discovery]
  DM --- ML[Machine Learning<br/>prediction]
  ST --- ML
  DB[Databases<br/>scale & storage] --- DM
  DM --> DS[Data Science]
  ST --> DS
  ML --> DS
""")

st.markdown("## قواعد الارتباط: تحليل سلة المشتريات")
tr = transactions()
b = baskets()
st.markdown(f"{len(b):,} فاتورة، متوسط {sum(map(len, b)) / len(b):.1f} منتج في السلة.")
st.dataframe(tr.groupby("invoice_id")["product"].apply(lambda s: ", ".join(sorted(s))).head(6).rename("basket"))
formula(r"\text{support}(A\Rightarrow B) = P(A \cap B),\quad \text{confidence} = P(B\mid A) = \frac{P(A\cap B)}{P(A)},\quad "
        r"\text{lift} = \frac{P(B\mid A)}{P(B)}", title="المقاييس الثلاثة",
        symbols={r"P(A\cap B)": "نسبة السلال التي تحوي A وB معًا", r"P(B\mid A)": "بين سلال A، نسبة التي تحوي B"},
        intuition="Support: هل القاعدة شائعة بما يكفي؟ Confidence: ما مدى موثوقيتها؟ Lift: هل هي أكثر من الصدفة؟ (Lift > 1)",
        example="200 سلة من 1000 فيها خبز، و120 فيها خبز وزبدة، والزبدة في 250: support = 0.12، confidence = 0.6، lift = 0.6/0.25 = 2.4")
c1, c2, c3 = st.columns(3)
min_sup = c1.slider("الحد الأدنى للـSupport", 0.01, 0.2, 0.03, 0.005, format="%.3f", key="dm_sup")
min_conf = c2.slider("الحد الأدنى للـConfidence", 0.1, 0.9, 0.4, 0.05, key="dm_conf")
min_lift = c3.slider("الحد الأدنى للـLift", 0.5, 5.0, 1.2, 0.1, key="dm_lift")
freq = apriori(b, min_support=min_sup, max_len=3)
rules = association_rules(freq, min_confidence=min_conf)
rules = rules[rules["lift"] >= min_lift]
st.markdown(f"**{len(freq)}** مجموعة متكررة، **{len(rules)}** قاعدة بعد الفلترة.")
if len(rules):
    st.dataframe(rules.round(3), hide_index=True, column_config={
        "lift": st.column_config.ProgressColumn("lift", min_value=0, max_value=float(max(3, rules["lift"].max())), format="%.2f")})
    fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift", hover_data=["antecedent", "consequent"],
                     color_continuous_scale=["#A5D8FF", "#F76707", "#0B4F8A"])
    fig.update_layout(height=380, title="كل نقطة قاعدة (الحجم واللون = Lift)")
    plot(fig)
else:
    st.info("لا توجد قواعد بهذه العتبات؛ خفّف الحدود.")
why("رتّب القواعد بالـLift مع حد أدنى للـSupport، لا بالـConfidence وحدها.",
    "منتج شائع جدًا (مثل الماء) يظهر بثقة عالية بعد أي منتج، لكن Lift ≈ 1 يكشف أن ذلك صدفة لا ارتباط.")
st.markdown("**قاعدة تافهة Trivial:** أي قاعدة نتيجتها منتج شائع جدًا (مثل Water أو Bread) قد تظهر بثقة عالية وLift قريب من 1. "
            "**قاعدة مفيدة:** مثل Pasta → Tomato Sauce إن ظهرت بـLift مرتفع — ابحث عنها في الجدول أعلاه.")

st.markdown("## خوارزمية Apriori")
st.markdown("1. احسب Support لكل منتج منفرد واحتفظ بالمتكرر.\n2. كوّن مرشحين بحجم k+1 من المتكررين بحجم k.\n"
            "3. **التقليم Pruning:** احذف أي مرشح له مجموعة جزئية غير متكررة (خاصية الإغلاق للأسفل Downward closure).\n"
            "4. احسب Support للمرشحين وكرر حتى لا يبقى مرشحون.\n5. ولّد القواعد من المجموعات المتكررة واحسب Confidence وLift.")


def _run(ms: float):
    f = apriori(b, min_support=ms, max_len=2)
    return f.sort_values("support", ascending=False).head(10).round(3)


code_lab("dm_apriori", "Apriori مبني من الصفر في utils/mining.py",
         lambda p: f"freq = apriori(baskets, min_support={p['ms']}, max_len=2)\nfreq.sort_values('support', ascending=False).head(10)",
         _run, params=lambda: {"ms": st.select_slider("min_support", [0.02, 0.05, 0.1], value=0.05, key="dm_ms")},
         explanation="في الإنتاج استخدم مكتبة مُحسّنة (مثل mlxtend أو FP-Growth في Spark)؛ التنفيذ هنا تعليمي وشفاف.")

if at_least("advanced"):
    st.markdown("## متقدم: مقاييس إضافية")
    st.markdown("- **Leverage** = P(A∩B) − P(A)P(B): الفرق المطلق عن الاستقلال.\n"
                "- **Conviction** = (1 − P(B)) / (1 − confidence): كم مرة تخطئ القاعدة أقل مما لو كانا مستقلين.\n"
                "- **FP-Growth**: يتجنب توليد المرشحين عبر شجرة مضغوطة؛ أسرع على بيانات كبيرة.")
if at_least("research"):
    researcher_note(["مع آلاف القواعد تظهر قواعد «مثيرة» بالصدفة؛ استخدم عتبات صارمة أو اختبارات دلالة مع تصحيح المقارنات.",
                     "القواعد وصفية: قياس أثر تدخل (وضع المنتجات متجاورة) يتطلب تجربة A/B."])
real_world(["هل القاعدة قابلة للتنفيذ تجاريًا؟", "هل تختلف القواعد بين الفروع أو المواسم؟", "هل الترويج الحالي سبب القاعدة؟",
            "هل Support كافٍ ليكون للقاعدة أثر؟"])

page_footer("data_mining",
            takeaways=["التنقيب خطوة داخل KDD لاكتشاف أنماط مفيدة.", "Support للشيوع، Confidence للموثوقية، Lift للتميز عن الصدفة.",
                       "Apriori يقلّم المرشحين بخاصية الإغلاق للأسفل.", "القواعد وصفية لا سببية."],
            mistakes=["الترتيب بالـConfidence وحدها.", "عتبات منخفضة جدًا تنتج آلاف القواعد التافهة.", "تفسير القاعدة كعلاقة سببية."])
