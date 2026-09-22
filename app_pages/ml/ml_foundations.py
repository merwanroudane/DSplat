import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("ml_foundations")

st.markdown("## خريطة مسائل التعلّم")
mermaid("""
flowchart TB
  ML[Machine Learning] --> S[Supervised<br/>labels available]
  ML --> U[Unsupervised<br/>no labels]
  ML --> SS[Semi-supervised<br/>few labels]
  ML --> SSL[Self-supervised<br/>labels from data itself]
  S --> R[Regression<br/>numeric target]
  S --> C[Classification<br/>categorical target]
  U --> CL[Clustering]
  U --> DR[Dimensionality reduction]
  U --> AD[Anomaly detection]
  U --> AR[Association rules]
""")
comparison_table([
    {"النوع": "Regression", "الهدف": "رقمي", "مثال": "سعر منزل، الطلب غدًا", "مقاييس": "MAE، RMSE، R²", "الوحدة": "model_evaluation"},
    {"النوع": "Classification", "الهدف": "فئوي", "مثال": "تعثر/عدم تعثر", "مقاييس": "Precision، Recall، AUC", "الوحدة": "model_evaluation"},
    {"النوع": "Clustering", "الهدف": "لا يوجد", "مثال": "شرائح العملاء", "مقاييس": "Silhouette، الاستقرار", "الوحدة": "clustering"},
    {"النوع": "Dimensionality reduction", "الهدف": "لا يوجد", "مثال": "ضغط 13 خاصية إلى 2", "مقاييس": "التباين المفسر", "الوحدة": "dim_reduction"},
    {"النوع": "Anomaly detection", "الهدف": "نادر/غير موجود", "مثال": "احتيال", "مقاييس": "Precision@k", "الوحدة": "anomaly_detection"},
    {"النوع": "Semi-supervised", "الهدف": "قليل التسميات", "مثال": "تصنيف وثائق بمئة مثال موسوم", "مقاييس": "كالموجّه", "الوحدة": "—"},
    {"النوع": "Self-supervised", "الهدف": "مولّد من البيانات", "مثال": "التنبؤ بكلمة محجوبة (أساس LLMs)", "مقاييس": "مهام لاحقة", "الوحدة": "ai_assisted"},
])
intuition("الإحصاء يسأل غالبًا «ما العلاقة ومدى يقيننا بها؟»، وتعلّم الآلة يسأل «ما أدق تنبؤ على بيانات لم نرها؟». "
          "التنقيب يسأل «ما الأنماط غير المعروفة؟». الأدوات مشتركة، والهدف يحدد طريقة التقييم.")

st.markdown("## الانحياز والتباين · Bias–Variance")
formula(r"\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\text{Bias}[\hat f(x)]^2}_{\text{تبسيط مفرط}} + "
        r"\underbrace{\text{Var}[\hat f(x)]}_{\text{حساسية للعينة}} + \underbrace{\sigma^2}_{\text{ضوضاء لا تُختزل}}",
        title="تفكيك الخطأ المتوقع",
        intuition="نموذج بسيط جدًا يخطئ بشكل منهجي (Bias). نموذج معقد جدًا يتغير كثيرًا مع كل عينة (Variance). "
                  "الهدف: أقل مجموع لا أقل أحدهما.")
c1, c2, c3 = st.columns(3)
degree = c1.slider("تعقيد النموذج (درجة كثيرة الحدود)", 1, 15, 3, key="mlf_deg")
n_train = c2.select_slider("حجم التدريب", [15, 30, 60, 150], value=30, key="mlf_n")
noise = c3.slider("الضوضاء σ", 0.05, 0.6, 0.25, 0.05, key="mlf_noise")
rng = np.random.default_rng(3)
f = lambda x: np.sin(2 * np.pi * x) * 0.8 + 0.3 * x  # noqa: E731
x_tr = np.sort(rng.uniform(0, 1, n_train))
y_tr = f(x_tr) + rng.normal(0, noise, n_train)
x_te = rng.uniform(0, 1, 400)
y_te = f(x_te) + rng.normal(0, noise, 400)
grid = np.linspace(0, 1, 300)


def fit_poly(deg: int):
    coef = np.polyfit(x_tr, y_tr, deg)
    return coef, np.mean((np.polyval(coef, x_tr) - y_tr) ** 2), np.mean((np.polyval(coef, x_te) - y_te) ** 2)


coef, tr_mse, te_mse = fit_poly(degree)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_tr, y=y_tr, mode="markers", name="train", marker=dict(color=PALETTE["purple"], size=7)))
    fig.add_trace(go.Scatter(x=grid, y=f(grid), name="true f(x)", line=dict(color="#999", dash="dot")))
    fig.add_trace(go.Scatter(x=grid, y=np.clip(np.polyval(coef, grid), -3, 3), name=f"degree {degree}",
                             line=dict(color=PALETTE["coral"], width=3)))
    fig.update_layout(height=380, yaxis=dict(range=[-1.8, 1.8]), legend=dict(orientation="h", y=1.12))
    plot(fig)
with c2:
    degs = list(range(1, 16))
    errs = np.array([fit_poly(d)[1:] for d in degs])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degs, y=errs[:, 0], name="train MSE", line=dict(color=PALETTE["purple"])))
    fig.add_trace(go.Scatter(x=degs, y=np.minimum(errs[:, 1], 2), name="test MSE", line=dict(color=PALETTE["coral"])))
    fig.add_vline(x=degree, line_dash="dash")
    fig.update_layout(height=380, xaxis_title="degree", yaxis_title="MSE", legend=dict(orientation="h", y=1.12))
    plot(fig)
st.markdown(f"Train MSE = **{tr_mse:.3f}** · Test MSE = **{te_mse:.3f}**. "
            + ("**Underfitting:** الخطآن مرتفعان." if degree <= 1 else
               "**Overfitting:** خطأ التدريب منخفض جدًا والاختبار أعلى بكثير." if te_mse > 2.2 * tr_mse and degree > 5 else
               "منطقة توازن معقولة."))
why("احكم على النموذج بأدائه على بيانات لم يرها، لا على بيانات التدريب.",
    "خطأ التدريب ينخفض دائمًا مع التعقيد؛ خطأ الاختبار وحده يكشف متى بدأ النموذج بحفظ الضوضاء. زيادة n تسمح بتعقيد أكبر بأمان.")

st.markdown("## العلاقة مع الإحصاء والتنقيب")
comparison_table([
    {"": "الهدف", "Statistics": "الاستدلال وعدم اليقين", "Machine Learning": "التعميم التنبؤي", "Data Mining": "اكتشاف أنماط مفهومة"},
    {"": "التقييم", "Statistics": "افتراضات، فترات ثقة", "Machine Learning": "أداء خارج العينة", "Data Mining": "الفائدة والجِدّة"},
    {"": "النماذج", "Statistics": "بسيطة قابلة للتفسير غالبًا", "Machine Learning": "مرنة ومعقدة أحيانًا", "Data Mining": "قواعد، عناقيد، أشجار"},
    {"": "حجم البيانات", "Statistics": "صغير إلى متوسط", "Machine Learning": "متوسط إلى ضخم", "Data Mining": "كبير غالبًا"},
])

if at_least("advanced"):
    st.markdown("## متقدم: ما وراء Bias–Variance الكلاسيكي")
    st.markdown("في النماذج شديدة التعقيد (شبكات عميقة) لوحظت ظاهرة **Double descent**: خطأ الاختبار يرتفع ثم ينخفض مجددًا "
                "بعد عتبة الاستيفاء. هذا لا يلغي مبدأ التقييم خارج العينة، بل يعقّد الحدس البسيط.")
if at_least("research"):
    researcher_note(["حدد مسبقًا المقياس ومخطط التحقق قبل مقارنة النماذج.",
                     "قارن دائمًا بخط أساس بسيط (Baseline): متوسط، أكثر فئة تكرارًا، أو انحدار خطي."])
real_world(["ما الخطأ الأكثر تكلفة؟", "هل تتوفر تسميات موثوقة؟", "هل يحتاج المستخدم تفسيرًا للقرار؟",
            "هل التوزيع في الإنتاج مشابه للتدريب؟"])

page_footer("ml_foundations",
            takeaways=["نوع الهدف يحدد نوع المسألة.", "Bias–Variance: البساطة المفرطة مقابل الحساسية المفرطة.",
                       "التقييم على بيانات غير مرئية هو المعيار.", "ابدأ دائمًا بخط أساس بسيط."],
            mistakes=["الحكم بخطأ التدريب.", "نموذج معقد دون مقارنة بخط أساس.", "تجاهل تكلفة الأخطاء المختلفة."])
