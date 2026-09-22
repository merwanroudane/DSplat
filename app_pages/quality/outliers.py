import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import domain, real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.cleaning import clean_customers_reference
from utils.datasets import customers_raw, students
from utils.outliers import (DECISION_QUESTIONS, MODEL_NAMES, iqr_bounds, iqr_mask, model_scores, modified_z,
                            modified_z_mask, outlier_decision, percentile_mask, winsorize, zscore_mask)
from utils.plotting import plot

page_header("outliers")


@st.cache_data(show_spinner=False)
def _clean() -> pd.DataFrame:
    return clean_customers_reference(customers_raw())[0]


data = _clean()

st.markdown("## مصطلحات متقاربة ليست مترادفة")
comparison_table([
    {"المصطلح": "Outlier", "المعنى": "قيمة بعيدة بشكل غير معتاد عن باقي البيانات", "مثال": "إنفاق شهري 8,000 والوسيط 150"},
    {"المصطلح": "Extreme value", "المعنى": "قيمة في طرف التوزيع لكنها متوقعة في توزيع ذي ذيل ثقيل", "مثال": "أعلى دخل في المدينة"},
    {"المصطلح": "Anomaly", "المعنى": "مشاهدة ناتجة عن آلية مختلفة", "مثال": "معاملة احتيالية"},
    {"المصطلح": "Influential observation", "المعنى": "نقطة يغيّر حذفها النموذج كثيرًا", "مثال": "نقطة High leverage في الانحدار"},
    {"المصطلح": "Measurement error", "المعنى": "خطأ في أداة القياس", "مثال": "ميزان غير معاير"},
    {"المصطلح": "Data (entry) error", "المعنى": "خطأ إدخال أو وحدة", "مثال": "طول 1.75 في عمود بالسنتيمتر"},
    {"المصطلح": "Rare event", "المعنى": "حدث حقيقي نادر", "مثال": "زلزال، أزمة مالية"},
    {"المصطلح": "Novelty", "المعنى": "نمط جديد لم يظهر في بيانات التدريب", "مثال": "منتج جديد بسلوك مختلف"},
    {"المصطلح": "Structural break", "المعنى": "تغير دائم في مستوى أو سلوك السلسلة", "مثال": "تغير سياسة تسعير"},
])
domain(["Age = 250 ← مستحيل بيولوجيًا: خطأ.", "Income = 0 ← قد يكون حقيقيًا أو رمزًا لعدم الإجابة.",
        "Temperature = −5°C ← ممكن في الشتاء، مستحيل في غرفة عمليات.",
        "القاعدة: **A statistically unusual value is not automatically a data error.**"])

# ---------------------------------------------------------- statistical
st.markdown("## الطرق الإحصائية")
col = st.selectbox("المتغير", ["monthly_spend", "annual_income", "age", "height_cm", "num_orders"], key="out_col")
x = data[col].astype(float).dropna()
c1, c2, c3 = st.columns(3)
k = c1.slider("مضاعف IQR (k)", 1.0, 4.0, 1.5, 0.1, key="out_k")
zt = c2.slider("عتبة Z-score", 2.0, 5.0, 3.0, 0.1, key="out_z")
mzt = c3.slider("عتبة Modified Z", 2.0, 6.0, 3.5, 0.1, key="out_mz")
lo, hi = iqr_bounds(x, k)
flags = pd.DataFrame({"IQR": iqr_mask(x, k), "Z-score": zscore_mask(x, zt), "Modified Z (MAD)": modified_z_mask(x, mzt),
                      "Percentile 1%/99%": percentile_mask(x)})
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Box(x=x, name=col, boxpoints="outliers", marker_color=PALETTE["purple"], orientation="h"))
    fig.add_vline(x=lo, line_dash="dash", line_color=PALETTE["coral"], annotation_text=f"Q1−{k}·IQR")
    fig.add_vline(x=hi, line_dash="dash", line_color=PALETTE["coral"], annotation_text=f"Q3+{k}·IQR")
    fig.update_layout(height=260, showlegend=False, title="Boxplot مع حدود IQR")
    plot(fig)
    fig = go.Figure(go.Histogram(x=x, nbinsx=60, marker_color=PALETTE["amber"]))
    fig.add_trace(go.Scatter(x=x[flags["IQR"]], y=np.zeros(int(flags["IQR"].sum())), mode="markers",
                             marker=dict(color=PALETTE["coral"], size=8, symbol="line-ns-open"), name="IQR flagged"))
    fig.update_layout(height=260, showlegend=False)
    plot(fig)
with c2:
    counts = flags.sum().rename("flagged").to_frame()
    counts["%"] = (100 * counts["flagged"] / len(x)).round(2)
    st.dataframe(counts)
    st.metric("الالتواء Skewness", f"{x.skew():.2f}")
    st.caption("في التوزيعات الملتوية تعلّم IQR قيمًا كثيرة في الذيل الأيمن رغم أنها حقيقية. "
               "Z-score يتأثر بالقيم المتطرفة نفسها (Masking)؛ Modified Z يعتمد على الوسيط وMAD فيبقى متينًا.")

formula(r"\text{IQR} = Q_3 - Q_1,\qquad \text{flag if } x < Q_1 - k\cdot\text{IQR} \ \text{or}\ x > Q_3 + k\cdot\text{IQR}",
        title="قاعدة Tukey (1977)", symbols={"Q_1, Q_3": "الربيع الأول والثالث", "k": "المضاعف (1.5 شائع، 3 «بعيد جدًا»)"},
        intuition="نقيس الانتشار بالـ50% الوسطى فلا تتأثر القاعدة بالقيم الطرفية نفسها.",
        example=f"لـ{col}: Q1 = {x.quantile(.25):.2f}، Q3 = {x.quantile(.75):.2f}، IQR = {x.quantile(.75) - x.quantile(.25):.2f} "
                f"← الحدود [{lo:.2f}, {hi:.2f}] مع k = {k}.")
formula(r"z_i = \frac{x_i - \bar{x}}{s} \qquad\qquad M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}},\quad "
        r"\text{MAD} = \operatorname{median}(|x_i - \tilde{x}|)",
        title="Z-score مقابل Modified Z-score (Iglewicz & Hoaglin, 1993)",
        symbols={r"\bar{x}, s": "المتوسط والانحراف المعياري", r"\tilde{x}": "الوسيط",
                 "0.6745": "ثابت يجعل MAD مكافئًا لـσ في التوزيع الطبيعي"},
        intuition="z يستخدم مقاييس تتأثر بالقيم الشاذة نفسها؛ M يستخدم مقاييس متينة Robust.",
        example=f"أكبر |M| في {col} = {modified_z(x).abs().max():.1f}؛ أكبر |z| = {((x - x.mean()) / x.std()).abs().max():.1f}.")

with st.expander("رسم متحرك: كيف يسحب Outlier واحد المتوسط والانحراف المعياري", icon=":material/animation:"):
    base = students()["post_score"].to_numpy()
    positions = [100, 150, 200, 300, 450, 600]
    frames = []
    for pval in positions:
        arr = np.append(base, pval)
        frames.append(go.Frame(name=str(pval), data=[
            go.Scatter(x=arr, y=np.random.default_rng(0).uniform(-0.3, 0.3, len(arr)), mode="markers",
                       marker=dict(color=[PALETTE["purple"]] * len(base) + [PALETTE["coral"]], size=[6] * len(base) + [14])),
            go.Scatter(x=[arr.mean()] * 2, y=[-0.6, 0.6], mode="lines", line=dict(color=PALETTE["amber"], width=3)),
            go.Scatter(x=[np.median(arr)] * 2, y=[-0.6, 0.6], mode="lines", line=dict(color=PALETTE["teal"], width=3, dash="dash"))],
            layout=go.Layout(title=f"outlier = {pval}: mean = {arr.mean():.2f} · median = {np.median(arr):.2f} · SD = {arr.std(ddof=1):.2f}")))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(xaxis=dict(range=[0, 620]), yaxis=dict(visible=False, range=[-0.8, 0.8]), showlegend=False,
                      height=330, title=frames[0].layout.title.text)
    from utils.plotting import add_animation_controls
    add_animation_controls(fig, [f.name for f in frames], duration=900, prefix="outlier value: ")
    plot(fig)
    st.caption("الخط الأصفر = المتوسط يتحرك مع القيمة الشاذة؛ الخط المتقطع = الوسيط يبقى ثابتًا تقريبًا.")

# -------------------------------------------------------------- model based
st.markdown("## الطرق المعتمدة على النماذج (متعددة المتغيرات)")
st.markdown("تكشف قيمًا **عادية في كل متغير منفردًا لكنها غير عادية معًا** (مثل دخل منخفض مع إنفاق مرتفع جدًا).")
feats = st.multiselect("الخصائص", ["age", "annual_income", "monthly_spend", "num_orders", "height_cm"],
                       default=["annual_income", "monthly_spend"], key="out_feats")
c1, c2 = st.columns(2)
method = c1.selectbox("الخوارزمية", list(MODEL_NAMES), format_func=MODEL_NAMES.get, key="out_model")
cont = c2.slider("contamination", 0.01, 0.2, 0.03, 0.01, key="out_cont")
if len(feats) >= 2:
    X = data[feats].astype(float).dropna()
    flag, score = model_scores(X, method, cont)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[feats[0]], y=X[feats[1]], mode="markers", name="normal",
                             marker=dict(color=score, colorscale="YlGnBu", size=6, showscale=True,
                                         colorbar=dict(title="anomaly score"))))
    fig.add_trace(go.Scatter(x=X.loc[flag, feats[0]], y=X.loc[flag, feats[1]], mode="markers", name="flagged",
                             marker=dict(color="rgba(0,0,0,0)", size=13, line=dict(color=PALETTE["purple"], width=2))))
    fig.update_layout(xaxis_title=feats[0], yaxis_title=feats[1], height=420, legend=dict(orientation="h", y=1.1))
    plot(fig)
    st.caption(f"{int(flag.sum())} نقطة معلّمة. contamination يحدد العتبة لا الدرجات: هو افتراضك عن نسبة الشذوذ.")
else:
    st.info("اختر خاصيتين على الأقل.")
comparison_table([
    {"الطريقة": "Isolation Forest", "الفكرة": "الشاذ يُعزل بتقسيمات عشوائية أقل", "القوة": "سريع، أبعاد كثيرة", "الضعف": "شذوذ محلي في مناطق كثيفة"},
    {"الطريقة": "Local Outlier Factor", "الفكرة": "كثافة النقطة مقابل كثافة جيرانها", "القوة": "شذوذ محلي", "الضعف": "حساس لـn_neighbors، بطيء"},
    {"الطريقة": "One-Class SVM", "الفكرة": "حدود تحيط بالبيانات العادية", "القوة": "حدود مرنة", "الضعف": "حساس للمعاملات والمقياس"},
    {"الطريقة": "Robust covariance", "الفكرة": "مسافة Mahalanobis بتغاير متين (MCD)", "القوة": "بيانات قريبة من الطبيعية", "الضعف": "يفترض شكلًا إهليلجيًا"},
    {"الطريقة": "Density-based (DBSCAN)", "الفكرة": "نقاط في مناطق منخفضة الكثافة = ضوضاء", "القوة": "أشكال عشوائية", "الضعف": "اختيار eps"},
    {"الطريقة": "Autoencoder (مفهوم)", "الفكرة": "خطأ إعادة بناء كبير = شاذ", "القوة": "بيانات معقدة (صور، سلاسل)", "الضعف": "يحتاج بيانات كثيرة وضبطًا"},
])

# -------------------------------------------------------------- decision tool
st.markdown("## أداة القرار · Outlier Decision Tool")
mermaid("""
flowchart TB
  A[Unusual value] --> B{Logically possible?}
  B -- no --> C{Recoverable from source?}
  C -- yes --> C1[Correct]
  C -- no --> C2[Set to NaN / Remove]
  B -- yes --> D{Unit problem?}
  D -- yes --> D1[Correct the unit]
  D -- no --> E{Rare but valid?}
  E -- yes --> F{Strong influence on model?}
  F -- yes --> F1[Robust model / Transform<br/>+ sensitivity analysis]
  F -- no --> F2[Keep]
  E -- unknown --> G[Investigate]
  G --> H[Cap / Winsorize if influence is high<br/>document thresholds]
""")
answers = {}
cols = st.columns(2)
for i, (key, q) in enumerate(DECISION_QUESTIONS):
    answers[key] = cols[i % 2].radio(q, ["نعم", "لا", "لا أعرف"], index=2, horizontal=True, key=f"out_q_{key}")
action, reason = outlier_decision(answers)
with st.container(border=True):
    st.markdown(f"### القرار المقترح: {action}")
    st.markdown(f"**لماذا؟** {reason}")
st.caption("الأداة قواعد صريحة لا صندوق أسود؛ القرار النهائي لك ولمالك البيانات، ويجب توثيقه.")

st.markdown("## المعالجات الممكنة")
comparison_table([
    {"الإجراء": "Keep", "متى": "حقيقية وغير مؤثرة", "الأثر": "لا تغيير"},
    {"الإجراء": "Investigate", "متى": "غير واضح", "الأثر": "تأجيل القرار حتى التحقق"},
    {"الإجراء": "Correct", "متى": "خطأ قابل للاسترجاع (وحدة، فاصلة)", "الأثر": "يحفظ المعلومة"},
    {"الإجراء": "Transform (log)", "متى": "ذيل طويل طبيعي", "الأثر": "يخفف التأثير ويغير التفسير"},
    {"الإجراء": "Cap / Winsorize", "متى": "مؤثرة وغير مؤكدة", "الأثر": "يحد التأثير ويحفظ الصف"},
    {"الإجراء": "Robust modeling", "متى": "حقيقية ومؤثرة", "الأثر": "نماذج أقل حساسية (Huber، Quantile)"},
    {"الإجراء": "Remove", "متى": "خطأ مؤكد غير قابل للتصحيح", "الأثر": "يفقد الصف — الملاذ الأخير"},
])
w = winsorize(x)
st.markdown(f"**Winsorize 1%/99% على {col}:** المتوسط {x.mean():.2f} ← {w.mean():.2f}، الانحراف {x.std():.2f} ← {w.std():.2f}.")
why("لا تستخدم 1.5×IQR كقاعدة حذف أوتوماتيكية.",
    "صُممت كأداة استكشاف بصري. في بيانات ملتوية (إنفاق، دخل) ستعلّم نسبة كبيرة من الحالات الحقيقية، وحذفها يشوّه الواقع "
    "ويخفض التباين ويحذف أحيانًا أهم العملاء.")

if at_least("advanced"):
    st.markdown("## متقدم: النقاط المؤثرة في الانحدار")
    st.markdown("**Leverage** (بُعد x عن المركز) × **Residual** (بُعد y عن الخط) = **Influence**، تقاس بـCook's distance. "
                "نقطة بعيدة في x لكنها على الخط لا تؤثر كثيرًا؛ نقطة بعيدة في الاثنين تسحب الخط.")
if at_least("research"):
    researcher_note(["حدّد قواعد الشذوذ قبل التحليل، وأبلغ عن النتائج مع وبدون القيم المعالجة.",
                     "Winsorization يغيّر المُقدَّر: أبلغ عن المئينات المستخدمة.",
                     "Leys et al. (2013) يوصون بـMAD بعتبة 2.5 أو 3 بدل الانحراف المعياري."])
real_world(["هل القيمة ممكنة منطقيًا؟ هل تكررت بنفس الرقم (Sentinel)؟", "هل هي في المصدر الأصلي؟",
            "هل تتركز القيم الشاذة في مصدر/فرع/جهاز معين؟", "ماذا يحدث للنتيجة الرئيسية مع وبدونها؟"])

page_footer("outliers",
            takeaways=["القيمة الشاذة إحصائيًا ليست خطأ تلقائيًا.", "الطرق المتينة (MAD) أفضل من Z في البيانات الملوثة.",
                       "النماذج متعددة المتغيرات تكشف تركيبات غير عادية.", "القرار: Keep/Investigate/Correct/Transform/Cap/Robust/Remove مع توثيق."],
            mistakes=["حذف كل ما خارج 1.5×IQR.", "استخدام Z-score على بيانات ملتوية جدًا.", "ضبط contamination دون تبرير.",
                      "عدم تحليل الحساسية."])
