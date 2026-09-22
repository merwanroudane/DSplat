import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.animation import stepper
from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.mining import isolation_trace
from utils.outliers import MODEL_NAMES, model_scores
from utils.plotting import plot

page_header("anomaly_detection")

rng = np.random.default_rng(12)
normal = np.vstack([rng.normal([0, 0], 0.8, (180, 2)), rng.normal([4, 3], 0.5, (120, 2))])
anomalies = np.array([[2.2, 5.5], [-2.8, 3.5], [6.5, -1.5], [4.1, 4.9], [1.5, 1.2]])
X = np.vstack([normal, anomalies])
is_true_anom = np.r_[np.zeros(len(normal), bool), np.ones(len(anomalies), bool)]

st.markdown("## فكرة العزل Isolation")
st.markdown("Isolation Forest (Liu, Ting & Zhou, 2008) لا يصف «الطبيعي»، بل يقيس **سهولة عزل النقطة**: نختار خاصية "
            "عشوائيًا وقيمة تقسيم عشوائية، ونكرر. النقطة الشاذة (قليلة ومختلفة) تُعزل في خطوات قليلة، والعادية وسط الزحام تحتاج خطوات كثيرة.")
target_name = st.segmented_control("تتبّع عزل نقطة", ["نقطة شاذة", "نقطة عادية"], default="نقطة شاذة", key="ad_target")
target = len(normal) + 1 if target_name != "نقطة عادية" else 5
trace = isolation_trace(X, target, seed=3)


def _render(i: int) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color="#CED4DA", size=6), showlegend=False))
    for s in trace[: i + 1]:
        lo, hi = s["box"]
        fig.add_shape(type="rect", x0=lo[0], y0=lo[1], x1=hi[0], y1=hi[1], line=dict(color=PALETTE["purple"], width=1.5),
                      fillcolor="rgba(112,72,232,0.05)")
    fig.add_trace(go.Scatter(x=[X[target, 0]], y=[X[target, 1]], mode="markers",
                             marker=dict(color=PALETTE["coral"], size=15, symbol="star"), showlegend=False))
    s = trace[i]
    fig.update_layout(height=430, title=f"Split {s['depth']}: feature x{s['feature'] + 1} at {s['split']:.2f} — "
                                        f"points still with target: {s['remaining']}",
                      xaxis=dict(range=[X[:, 0].min() - 0.7, X[:, 0].max() + 0.7]),
                      yaxis=dict(range=[X[:, 1].min() - 0.7, X[:, 1].max() + 0.7]))
    plot(fig)


stepper(f"iso_{target}", len(trace), _render, labels=[f"depth {s['depth']}" for s in trace])
st.caption(f"عُزلت النقطة بعد {len(trace)} تقسيمًا في هذه الشجرة. الغابة تكرر ذلك مع مئات الأشجار وتأخذ متوسط العمق.")
formula(r"s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}", title="درجة الشذوذ في Isolation Forest",
        symbols={"h(x)": "طول المسار حتى عزل x في شجرة", "c(n)": "متوسط طول المسار المتوقع لعينة بحجم n (للتطبيع)"},
        intuition="مسار قصير ← درجة قريبة من 1 (شاذ). مسار بطول المتوسط ← درجة ≈ 0.5 (عادي).")

st.markdown("## قارن الخوارزميات")
c1, c2 = st.columns(2)
cont = c1.slider("contamination", 0.005, 0.1, 0.02, 0.005, format="%.3f", key="ad_cont")
methods = c2.multiselect("الخوارزميات", list(MODEL_NAMES), default=list(MODEL_NAMES), format_func=MODEL_NAMES.get,
                         key="ad_methods")
Xdf = pd.DataFrame(X, columns=["x1", "x2"])
cols = st.columns(2)
rows = []
for i, m in enumerate(methods):
    flag, score = model_scores(Xdf, m, cont)
    tp = int((flag & is_true_anom).sum())
    rows.append({"الخوارزمية": MODEL_NAMES[m], "معلّمة": int(flag.sum()), "شاذة حقيقية مكتشفة (من 5)": tp,
                 "إنذارات كاذبة": int((flag & ~is_true_anom).sum())})
    with cols[i % 2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=score, colorscale="YlGnBu", size=7, line=dict(width=0)), showlegend=False))
        fig.add_trace(go.Scatter(x=X[flag, 0], y=X[flag, 1], mode="markers", showlegend=False,
                                 marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=PALETTE["purple"], width=2))))
        fig.update_layout(title=MODEL_NAMES[m], height=320, margin=dict(t=40, l=10, r=10, b=10))
        plot(fig)
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption("النقطة (1.5, 1.2) بين المجموعتين: شاذة **محليًا** (منطقة خالية) لكنها غير بعيدة عن المركز العام — "
           "LOF يلتقطها غالبًا بينما قد تفوتها الطرق العامة.")
why("اضبط contamination بناءً على معرفة المجال أو ميزانية المراجعة، ثم راجع الحالات الأعلى درجة يدويًا.",
    "contamination يحدد فقط كم نقطة تُعلَّم، لا صحة التعليم. في الواقع نادرًا ما تتوفر تسميات؛ المراجعة البشرية لأعلى k حالة "
    "هي التقييم العملي (Precision@k).")

comparison_table([
    {"الخوارزمية": "Isolation Forest", "النوع": "عام، شجري", "التعقيد": "سريع جدًا", "متى": "بيانات كبيرة وأبعاد كثيرة"},
    {"الخوارزمية": "LOF", "النوع": "محلي، كثافة", "التعقيد": "O(n²) تقريبًا", "متى": "مناطق بكثافات مختلفة"},
    {"الخوارزمية": "One-Class SVM", "النوع": "حدود Kernel", "التعقيد": "بطيء مع n كبير", "متى": "بيانات متوسطة بحدود معقدة"},
    {"الخوارزمية": "Elliptic Envelope", "النوع": "Robust covariance", "التعقيد": "سريع", "متى": "بيانات قريبة من الطبيعية"},
    {"الخوارزمية": "Autoencoder", "النوع": "خطأ إعادة البناء", "التعقيد": "تدريب شبكة", "متى": "صور، سلاسل، بيانات معقدة"},
])

if at_least("advanced"):
    st.markdown("## متقدم: أنواع الشذوذ")
    st.markdown("- **Point anomaly:** قيمة منفردة غير عادية.\n- **Contextual:** عادية عمومًا لكن غير عادية في سياقها "
                "(30°م عادية صيفًا، شاذة شتاءً).\n- **Collective:** سلسلة قيم عادية منفردة لكن نمطها معًا غير عادي "
                "(تسلسل معاملات صغيرة متتالية).\n- **Novelty detection:** التدريب على بيانات نظيفة فقط ثم كشف الجديد "
                "(`LocalOutlierFactor(novelty=True)`).")
if at_least("research"):
    researcher_note(["قيّم كاشفات الشذوذ على بيانات مرجعية بتسميات (مثل ODDS) قبل الاعتماد على نتائجها.",
                     "النتائج حساسة للمقياس والمعاملات؛ أبلغ عن تحليل الحساسية."])
real_world(["من سيراجع الحالات المعلّمة وكم حالة يستطيع يوميًا؟", "ما تكلفة الإنذار الكاذب مقابل الحالة الفائتة؟",
            "هل تتغير «الحالة الطبيعية» مع الزمن؟", "هل الشذوذ خطأ بيانات أم حدث حقيقي؟"])

page_footer("anomaly_detection",
            takeaways=["Isolation Forest: الشاذ يُعزل بمسارات قصيرة.", "LOF يكشف الشذوذ المحلي.",
                       "contamination افتراض عن النسبة لا مقياس للصحة.", "المراجعة البشرية جزء من التقييم."],
            mistakes=["اعتماد contamination افتراضي دون تفكير.", "عدم توحيد المقاييس.", "حذف الحالات المعلّمة آليًا."])
