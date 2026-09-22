import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from components.callouts import intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import wine
from utils.plotting import add_animation_controls, plot

page_header("dim_reduction")
w = wine()
feats = [c for c in w.columns if c != "cultivar"]

st.markdown("## لماذا نقلل الأبعاد؟")
st.markdown("- **التصوير:** 13 خاصية لا تُرسم؛ مكونان رئيسيان يُرسمان.\n- **الضوضاء والتعدد الخطي:** دمج المتغيرات المترابطة.\n"
            "- **لعنة الأبعاد Curse of dimensionality:** في الأبعاد العالية تتشابه المسافات فتضعف الطرق المعتمدة عليها.\n"
            "- **الكفاءة:** نماذج أسرع وأقل ذاكرة.")

st.markdown("## PCA هندسيًا: رسم متحرك")
intuition("تخيّل سحابة نقاط بيضاوية. PCA يدير المحاور حتى يمتد المحور الأول على طول أطول اتجاه للسحابة (أكبر تباين)، "
          "والثاني عموديًا عليه. الإسقاط على المحور الأول يحفظ أكبر قدر ممكن من «الانتشار».")
rng = np.random.default_rng(0)
cloud = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2.2]], 250)
pca2 = PCA(2).fit(cloud)
v1 = pca2.components_[0]
angles = np.linspace(0, np.arctan2(v1[1], v1[0]), 10)
frames = []
for i, a in enumerate(angles):
    d = np.array([np.cos(a), np.sin(a)])
    proj = cloud @ d
    pts = np.outer(proj, d)
    frames.append(go.Frame(name=str(i), data=[
        go.Scatter(x=cloud[:, 0], y=cloud[:, 1], mode="markers", marker=dict(color=PALETTE["purple"], size=5, opacity=0.5)),
        go.Scatter(x=[-5 * d[0], 5 * d[0]], y=[-5 * d[1], 5 * d[1]], mode="lines", line=dict(color=PALETTE["coral"], width=3)),
        go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(color=PALETTE["amber"], size=4))],
        layout=go.Layout(title=f"angle = {np.degrees(a):.0f}° · variance of projection = {proj.var():.2f}")))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(height=460, showlegend=False, xaxis=dict(range=[-6, 6]), yaxis=dict(range=[-5, 5], scaleanchor="x"),
                  title=frames[0].layout.title.text)
add_animation_controls(fig, [f.name for f in frames], duration=600, prefix="rotation: ")
plot(fig)
st.caption(f"التباين المسقط يزداد حتى يبلغ الحد الأقصى ({pca2.explained_variance_[0]:.2f}) عند اتجاه PC1.")

formula(r"\Sigma = \frac{1}{n-1}Z^\top Z,\qquad \Sigma\,v_j = \lambda_j v_j,\qquad \text{explained}_j = \frac{\lambda_j}{\sum_k \lambda_k}",
        title="PCA رياضيًا", symbols={"Z": "البيانات بعد التمركز (والتوحيد عادة)", "v_j": "المتجه الذاتي = اتجاه المكون j",
                                       r"\lambda_j": "القيمة الذاتية = التباين على المكون j"},
        intuition="المكونات هي المتجهات الذاتية لمصفوفة التغاير، مرتبة حسب قيمها الذاتية.")

st.markdown("## PCA على بيانات Wine")
scale = st.toggle("وحّد المقاييس قبل PCA (StandardScaler)", True, key="dr_scale")
Z = StandardScaler().fit_transform(w[feats]) if scale else (w[feats] - w[feats].mean()).to_numpy()
pca = PCA().fit(Z)
ev = pca.explained_variance_ratio_
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, marker_color=PALETTE["purple"], name="individual"))
    fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=np.cumsum(ev), name="cumulative",
                             line=dict(color=PALETTE["coral"])))
    fig.add_hline(y=0.8, line_dash="dash", annotation_text="80%")
    fig.update_layout(title="Scree plot", yaxis_tickformat=".0%", height=360, legend=dict(orientation="h", y=1.12))
    plot(fig)
with c2:
    pcs = pca.transform(Z)[:, :2]
    fig = px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=w["cultivar"].astype(str), color_discrete_sequence=SEQUENCE,
                     labels={"x": f"PC1 ({ev[0]:.0%})", "y": f"PC2 ({ev[1]:.0%})", "color": "cultivar"})
    fig.update_layout(height=360, title="الإسقاط على PC1 × PC2")
    plot(fig)
n80 = int(np.searchsorted(np.cumsum(ev), 0.8) + 1)
st.markdown(f"**{n80}** مكونات تفسر 80% من التباين من أصل {len(feats)} خاصية.")
if not scale:
    st.warning("بدون توحيد: PC1 يفسر ≈ كل التباين لأنه ببساطة proline (أكبر مقياس رقمي). هذا «اكتشاف» زائف للوحدات لا للبنية.",
               icon=":material/warning:")

st.markdown("### Loadings: ماذا يعني كل مكون؟")
load = pd.DataFrame(pca.components_[:3].T, index=feats, columns=["PC1", "PC2", "PC3"])
fig = go.Figure(go.Heatmap(z=load.to_numpy(), x=load.columns, y=load.index, zmid=0, text=load.round(2).to_numpy(),
                           texttemplate="%{text}", colorscale=[[0, PALETTE["purple"]], [0.5, "#FFFCF5"], [1, PALETTE["coral"]]]))
fig.update_layout(height=460)
plot(fig)
st.caption("الـLoading الكبير (موجبًا أو سالبًا) يعني أن الخاصية تساهم بقوة في المكون. التسمية التفسيرية للمكون تحتاج خبير مجال.")

st.markdown("## t-SNE وUMAP (مقدمة)")
perp = st.slider("t-SNE perplexity", 5, 50, 30, key="dr_perp")


@st.cache_data(show_spinner="تشغيل t-SNE…")
def run_tsne(perplexity: int) -> np.ndarray:
    return TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca").fit_transform(
        StandardScaler().fit_transform(w[feats]))


emb = run_tsne(perp)
fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=w["cultivar"].astype(str), color_discrete_sequence=SEQUENCE,
                 labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "cultivar"})
fig.update_layout(height=380)
plot(fig)
comparison_table([
    {"": "النوع", "PCA": "خطي", "t-SNE": "غير خطي", "UMAP": "غير خطي"},
    {"": "يحفظ", "PCA": "التباين العام والمسافات الكبيرة", "t-SNE": "الجوار المحلي", "UMAP": "الجوار المحلي + بعض البنية العامة"},
    {"": "قابل للتفسير", "PCA": "نعم (Loadings)", "t-SNE": "لا", "UMAP": "لا"},
    {"": "بيانات جديدة", "PCA": "transform مباشر", "t-SNE": "لا يدعم", "UMAP": "يدعم"},
    {"": "الاستخدام", "PCA": "ضغط، معالجة مسبقة، تصوير", "t-SNE": "تصوير فقط", "UMAP": "تصوير، أحيانًا معالجة"},
])
why("لا تفسر المسافات بين العناقيد ولا أحجامها في t-SNE/UMAP.",
    "هذه الطرق تشوّه البنية العامة عمدًا لحفظ الجوار المحلي؛ غيّر perplexity ولاحظ كيف تتغير الأشكال والمسافات.")
st.caption("UMAP يتطلب مكتبة umap-learn (اعتمادية اختيارية غير مثبتة هنا)؛ الفكرة مماثلة لـt-SNE مع سرعة أكبر.")

if at_least("advanced"):
    st.markdown("## متقدم: PCA داخل Pipeline")
    st.code("pipe = make_pipeline(StandardScaler(), PCA(n_components=0.9), LogisticRegression())\n"
            "cross_val_score(pipe, X, y, cv=5)   # PCA fitted inside each fold", language="python")
    st.markdown("`n_components=0.9` يختار عدد المكونات الذي يفسر 90% من التباين تلقائيًا.")
if at_least("research"):
    researcher_note(["PCA غير موجه: المكونات ذات التباين الكبير ليست بالضرورة الأكثر تنبؤًا بالهدف (انظر PLS).",
                     "أبلغ عن معالجة ما قبل PCA (التوحيد) لأنها تغيّر النتائج جذريًا."])
real_world(["هل تحتاج تفسيرًا للمكونات أم ضغطًا فقط؟", "هل وحّدت المقاييس؟", "هل تستخدم t-SNE كدليل على عناقيد؟ (لا تفعل)",
            "كم معلومة تفقد مع كل مكون محذوف؟"])

page_footer("dim_reduction",
            takeaways=["PCA يدير المحاور نحو اتجاهات التباين الأكبر.", "Scree plot وLoadings لاختيار المكونات وتفسيرها.",
                       "التوحيد قبل PCA ضروري مع وحدات مختلفة.", "t-SNE/UMAP للتصوير لا للتفسير الكمي."],
            mistakes=["PCA بلا Scaling.", "تفسير مسافات t-SNE.", "افتراض أن PC1 هو الأكثر تنبؤًا."])
