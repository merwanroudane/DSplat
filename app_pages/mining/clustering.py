import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import customers_clean
from utils.mining import kmeans_trace
from utils.plotting import add_animation_controls, plot

page_header("clustering")

st.markdown("## K-Means خطوة بخطوة")
formula(r"\min_{C_1,\dots,C_k}\ \sum_{j=1}^{k}\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2", title="هدف K-Means (Inertia / WCSS)",
        symbols={r"\mu_j": "مركز العنقود j", "C_j": "نقاط العنقود j"},
        intuition="نبحث عن k مراكز تجعل مجموع مربعات المسافات بين كل نقطة ومركزها أصغر ما يمكن. "
                  "خوارزمية Lloyd تتناوب: (1) إسناد كل نقطة لأقرب مركز، (2) نقل كل مركز لمتوسط نقاطه.")
c1, c2, c3 = st.columns(3)
k = c1.slider("عدد العناقيد k", 2, 8, 4, key="cl_k")
seed = c2.number_input("بذرة التهيئة", 0, 50, 3, key="cl_seed")
true_k = c3.slider("عدد المجموعات الحقيقية في البيانات", 2, 6, 4, key="cl_true")
X, _ = make_blobs(n_samples=360, centers=true_k, cluster_std=1.1, random_state=7)
hist = kmeans_trace(X, k, seed=int(seed))
frames = []
for h_i, h in enumerate(hist):
    colors = [SEQUENCE[l % len(SEQUENCE)] for l in h["labels"]]
    frames.append(go.Frame(name=str(h_i), data=[
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=colors, size=6, opacity=0.65)),
        go.Scatter(x=h["centroids"][:, 0], y=h["centroids"][:, 1], mode="markers",
                   marker=dict(symbol="x", size=18, color="black", line=dict(width=3)))],
        layout=go.Layout(title=f"Iteration {h['iter'] + 1} — {'Assign points' if h['phase'] == 'assign' else 'Update centroids'} "
                               f"— inertia = {h['inertia']:.0f}")))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(height=480, showlegend=False, title=frames[0].layout.title.text,
                  xaxis=dict(range=[X[:, 0].min() - 1, X[:, 0].max() + 1]), yaxis=dict(range=[X[:, 1].min() - 1, X[:, 1].max() + 1]))
add_animation_controls(fig, [f.name for f in frames], duration=700, prefix="step: ")
plot(fig)
st.caption(f"تقارب بعد {hist[-1]['iter'] + 1} تكرار. غيّر البذرة: التهيئة العشوائية قد تنتج حلًا مختلفًا (حد أدنى محلي) — "
           "لذلك تستخدم scikit-learn تهيئة k-means++ وتشغيلات متعددة n_init.")

st.markdown("## كم عنقودًا؟ Elbow وSilhouette")
ks = list(range(2, 10))


@st.cache_data(show_spinner=False)
def elbow_scores(n_centers: int) -> tuple[list[float], list[float]]:
    data, _ = make_blobs(n_samples=360, centers=n_centers, cluster_std=1.1, random_state=7)
    fits = [KMeans(kk, n_init=10, random_state=0).fit(data) for kk in ks]
    return [m.inertia_ for m in fits], [silhouette_score(data, m.labels_) for m in fits]


inertias, sils = elbow_scores(true_k)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers", line=dict(color=PALETTE["coral"])))
    fig.update_layout(title="Elbow: Inertia مقابل k", xaxis_title="k", height=300)
    plot(fig)
with c2:
    fig = go.Figure(go.Scatter(x=ks, y=sils, mode="lines+markers", line=dict(color=PALETTE["purple"])))
    fig.update_layout(title="Silhouette مقابل k (الأعلى أفضل)", xaxis_title="k", height=300)
    plot(fig)
formula(r"s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}", title="Silhouette",
        symbols={"a(i)": "متوسط المسافة إلى نقاط عنقودها", "b(i)": "متوسط المسافة إلى أقرب عنقود آخر"},
        intuition="قريب من 1: النقطة في مكانها. قرب 0: على الحدود. سالب: ربما في العنقود الخطأ.")

st.markdown("## مقارنة الخوارزميات على أشكال مختلفة")
shape = st.segmented_control("شكل البيانات", ["Blobs", "Moons", "Different densities"], default="Moons", key="cl_shape")
if shape == "Moons":
    Xs, _ = make_moons(400, noise=0.07, random_state=0)
elif shape == "Different densities":
    Xs = np.vstack([make_blobs(250, centers=[[0, 0]], cluster_std=0.3, random_state=1)[0],
                    make_blobs(120, centers=[[3, 3]], cluster_std=1.3, random_state=2)[0]])
else:
    Xs, _ = make_blobs(400, centers=3, random_state=4)
Xs = StandardScaler().fit_transform(Xs)
c1, c2 = st.columns(2)
eps = c1.slider("DBSCAN eps", 0.05, 1.0, 0.25, 0.05, key="cl_eps")
kk = c2.slider("k لـK-Means وHierarchical", 2, 5, 2, key="cl_kk")
algos = {"K-Means": KMeans(kk, n_init=10, random_state=0).fit_predict(Xs),
         "Hierarchical (Ward)": AgglomerativeClustering(kk).fit_predict(Xs),
         "DBSCAN": DBSCAN(eps=eps, min_samples=5).fit_predict(Xs)}
cols = st.columns(3)
for col, (name, labels) in zip(cols, algos.items()):
    with col:
        fig = go.Figure(go.Scatter(x=Xs[:, 0], y=Xs[:, 1], mode="markers",
                                   marker=dict(color=["#BBB" if l == -1 else SEQUENCE[l % len(SEQUENCE)] for l in labels], size=5)))
        noise = int((labels == -1).sum())
        fig.update_layout(title=f"{name}" + (f" · noise={noise}" if name == "DBSCAN" else ""), height=300, showlegend=False,
                          margin=dict(l=10, r=10, t=40, b=10))
        plot(fig)
comparison_table([
    {"الخوارزمية": "K-Means", "الشكل": "كروي بأحجام متقاربة", "يحتاج k؟": "نعم", "الضوضاء": "لا يعالجها", "المقياس": "حساس"},
    {"الخوارزمية": "Hierarchical", "الشكل": "حسب طريقة الربط", "يحتاج k؟": "يُقطع الشجرة لاحقًا", "الضوضاء": "لا", "المقياس": "حساس"},
    {"الخوارزمية": "DBSCAN", "الشكل": "أي شكل، كثافة متقاربة", "يحتاج k؟": "لا (eps وmin_samples)", "الضوضاء": "يعلّمها (−1)", "المقياس": "حساس"},
    {"الخوارزمية": "Gaussian Mixture", "الشكل": "إهليلجي، احتمالي", "يحتاج k؟": "نعم (BIC للاختيار)", "الضوضاء": "لا", "المقياس": "حساس"},
])

st.markdown("## تطبيق: شرائح العملاء")
cust = customers_clean()
feats = st.multiselect("الخصائص", ["age", "annual_income", "monthly_spend", "num_orders", "satisfaction"],
                       default=["annual_income", "monthly_spend", "num_orders"], key="cl_feats")
scale = st.toggle("طبّق Standardization + log للإنفاق والدخل", True, key="cl_scale")
if len(feats) >= 2:
    Z = cust[feats].astype(float).copy()
    if scale:
        for c in ("annual_income", "monthly_spend"):
            if c in Z:
                Z[c] = np.log1p(Z[c])
        Z = pd.DataFrame(StandardScaler().fit_transform(Z), columns=feats)
    k_seg = st.slider("عدد الشرائح", 2, 6, 3, key="cl_seg")
    lab = KMeans(k_seg, n_init=10, random_state=0).fit_predict(Z)
    prof = cust[feats].assign(segment=lab).groupby("segment").agg(["mean"]).round(1)
    prof.columns = [c[0] for c in prof.columns]
    prof["size"] = pd.Series(lab).value_counts().sort_index()
    prof["churn rate"] = cust.assign(segment=lab).groupby("segment")["churned"].mean().round(3)
    st.dataframe(prof)
    fig = px.scatter(cust.assign(segment=lab.astype(str)), x=feats[0], y=feats[1], color="segment",
                     color_discrete_sequence=SEQUENCE, opacity=0.7, log_x=feats[0] in ("annual_income", "monthly_spend"))
    fig.update_layout(height=380)
    plot(fig)
    st.caption(f"Silhouette = {silhouette_score(Z, lab):.3f}. معدل المغادرة لم يُستخدم في التجميع — اختلافه بين الشرائح "
               "دليل صلاحية خارجية External validation على أن الشرائح ذات معنى.")
why("وحّد المقاييس قبل التجميع المعتمد على المسافة.",
    "بدون Scaling يهيمن الدخل (عشرات الآلاف) على المسافة ويُهمل عدد الطلبات تمامًا؛ جرّب إيقاف الخيار أعلاه ولاحظ الفرق.")

if at_least("advanced"):
    st.markdown("## متقدم: التجميع الهرمي والـDendrogram")
    import plotly.figure_factory as ff
    sample = StandardScaler().fit_transform(cust[["annual_income", "monthly_spend", "num_orders"]].astype(float).sample(40, random_state=0))
    fig = ff.create_dendrogram(sample, color_threshold=4)
    fig.update_layout(height=360, title="Dendrogram (Ward) لعينة 40 عميلًا")
    plot(fig)
if at_least("research"):
    researcher_note(["العناقيد تُكتشف دائمًا حتى في بيانات بلا بنية؛ اختبر الاستقرار بـBootstrap ومقارنة الخوارزميات.",
                     "Silhouette مرتفعة لا تعني عناقيد «حقيقية» بالمعنى العلمي؛ الصلاحية الخارجية أهم.",
                     "Hennig (2015): «What are the true clusters?» — التعريف يعتمد على هدف التحليل."])
real_world(["هل الشرائح قابلة للتنفيذ (حملة مختلفة لكل شريحة)؟", "هل هي مستقرة الشهر القادم؟",
            "هل يفهمها فريق الأعمال ويستطيع تسميتها؟", "هل حجم كل شريحة كافٍ؟"])

page_footer("clustering",
            takeaways=["K-Means يتناوب بين الإسناد وتحديث المراكز لتقليل Inertia.", "Elbow وSilhouette والصلاحية الخارجية لاختيار k.",
                       "DBSCAN للأشكال غير المنتظمة والضوضاء.", "Scaling ضروري."],
            mistakes=["التجميع دون Scaling.", "اختيار k بمعيار واحد.", "افتراض أن كل عنقود «حقيقي»."])
