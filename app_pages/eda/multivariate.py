import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from components.callouts import real_world, researcher_note, why
from components.dataset_viewer import dataset_picker
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.plotting import heatmap, plot
from utils.statistics import vif
from utils.types import categorical_columns, numeric_columns

page_header("multivariate")
name, df = dataset_picker("mv_ds", default="wine", allowed=["wine", "iris", "customers_clean", "students", "credit", "survey"])
nums = [c for c in numeric_columns(df) if df[c].nunique() > 2]
cats = [c for c in categorical_columns(df) if 1 < df[c].nunique() <= 10]

st.markdown("## مصفوفة الارتباط Correlation matrix")
c1, c2 = st.columns([1, 3])
method = c1.radio("الطريقة", ["pearson", "spearman"], key="mv_method")
chosen = c2.multiselect("المتغيرات", nums, default=nums[: min(10, len(nums))], key="mv_vars")
if len(chosen) >= 2:
    corr = df[chosen].corr(method=method)
    if st.toggle("رتّب حسب التشابه (Hierarchical order)", True, key="mv_order"):
        from scipy.cluster.hierarchy import leaves_list, linkage
        order = leaves_list(linkage(1 - corr.abs().to_numpy()[np.triu_indices(len(corr), 1)], "average"))
        corr = corr.iloc[order, order]
    plot(heatmap(corr, f"{method.title()} correlation"), height=520)
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values(key=np.abs, ascending=False)
    st.markdown("**أقوى الأزواج:** " + " · ".join(f"{a}–{b}: {v:.2f}" for (a, b), v in pairs.head(5).items()))
    st.caption("الترتيب الهرمي يجمع المتغيرات المترابطة في كتل فتظهر البنية بوضوح.")

st.markdown("## العلاقات الزوجية Pairplot")
pp_vars = st.multiselect("حتى 5 متغيرات", nums, default=nums[:4], max_selections=5, key="mv_pp")
hue = st.selectbox("لوّن حسب", ["(none)"] + cats, index=1 if cats else 0, key="mv_hue")
if len(pp_vars) >= 2:
    fig = px.scatter_matrix(df, dimensions=pp_vars, color=None if hue == "(none)" else hue,
                            color_discrete_sequence=SEQUENCE, opacity=0.55)
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    fig.update_layout(height=620)
    plot(fig)

st.markdown("## الإحداثيات المتوازية Parallel coordinates")
if len(chosen) >= 3:
    d = df[chosen[:7]].dropna()
    z = (d - d.mean()) / d.std()
    color_vals = df.loc[d.index, cats[0]].astype("category").cat.codes if cats else z.iloc[:, 0]
    fig = go.Figure(go.Parcoords(line=dict(color=color_vals, colorscale=[[0, PALETTE["purple"]], [0.5, PALETTE["amber"]],
                                                                        [1, PALETTE["coral"]]]),
                                 dimensions=[dict(label=c, values=z[c]) for c in z.columns]))
    fig.update_layout(height=420, title="قيم موحّدة z؛ كل خط مشاهدة" + (f" (اللون: {cats[0]})" if cats else ""))
    plot(fig)
    st.caption("اسحب على أي محور لتصفية الخطوط تفاعليًا. الخطوط المتوازية المتقاربة = مشاهدات بملف متشابه.")

st.markdown("## PCA للتصوير")
if len(chosen) >= 3:
    d = df[chosen].dropna()
    Z = StandardScaler().fit_transform(d)
    pca = PCA(n_components=2).fit(Z)
    pcs = pca.transform(Z)
    pdf = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=d.index)
    if cats:
        pdf[cats[0]] = df.loc[d.index, cats[0]].astype(str)
    fig = px.scatter(pdf, x="PC1", y="PC2", color=cats[0] if cats else None, color_discrete_sequence=SEQUENCE, opacity=0.7)
    fig.update_layout(height=420, title=f"PC1 + PC2 تفسران {pca.explained_variance_ratio_.sum():.0%} من التباين")
    plot(fig)
    st.caption("إسقاط كل المتغيرات على مستوى واحد يحفظ أكبر قدر من التباين. التفاصيل في وحدة «تقليل الأبعاد».")

    st.markdown("## الشذوذ متعدد المتغيرات")
    center = Z.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(Z, rowvar=False))
    md = np.sqrt(np.einsum("ij,jk,ik->i", Z - center, cov_inv, Z - center))
    from scipy import stats as sst
    cutoff = np.sqrt(sst.chi2.ppf(0.99, Z.shape[1]))
    st.markdown(f"مسافة Mahalanobis تقيس البعد عن المركز مع مراعاة الارتباطات. {int((md > cutoff).sum())} مشاهدة تتجاوز "
                f"العتبة √χ²₀.₉₉({Z.shape[1]}) = {cutoff:.2f}.")
    fig = go.Figure(go.Histogram(x=md, nbinsx=50, marker_color=PALETTE["amber"]))
    fig.add_vline(x=cutoff, line_color=PALETTE["coral"], line_dash="dash", annotation_text="99% cutoff")
    fig.update_layout(height=280)
    plot(fig)

st.markdown("## التعدد الخطي Multicollinearity")
formula(r"\text{VIF}_j = \frac{1}{1 - R_j^2}", title="Variance Inflation Factor",
        symbols={"R_j^2": "معامل التحديد عند انحدار المتغير j على كل المتغيرات الأخرى"},
        intuition="إن أمكن التنبؤ بمتغير من البقية، فمعامله في الانحدار غير مستقر: يتضخم تباينه بعامل VIF.",
        example="R²_j = 0.9 ← VIF = 10: الخطأ المعياري لمعامله أكبر ≈ √10 ≈ 3.2 مرة مما لو كان مستقلًا.")
if len(chosen) >= 2:
    v = vif(df[chosen])
    v["flag"] = np.where(v["VIF"] > 10, "🔴 > 10", np.where(v["VIF"] > 5, "🟠 > 5", "🟢"))
    st.dataframe(v.sort_values("VIF", ascending=False), hide_index=True)
why("لا تحذف متغيرًا لمجرد VIF مرتفع إن كان هدفك التنبؤ.",
    "التعدد الخطي يضر تفسير المعاملات واستقرارها، لكنه يؤثر قليلًا على دقة التنبؤ. للتفسير: ادمج، اختر، أو استخدم Ridge.")

if at_least("advanced"):
    st.markdown("## متقدم: الارتباط الجزئي")
    st.markdown("الارتباط بين X وY بعد إزالة أثر Z: نأخذ بواقي انحدار X على Z وبواقي Y على Z ونحسب ارتباطهما. "
                "يكشف ما إذا كانت علاقة ثنائية مجرد انعكاس لمتغير ثالث.")
if at_least("research"):
    researcher_note(["مصفوفة ارتباط كبيرة = مقارنات متعددة؛ لا تختر «الأزواج الدالة» للتقرير دون تصحيح.",
                     "Mahalanobis يفترض توزيعًا إهليلجيًا؛ مع قيم ملوثة استخدم تقديرًا متينًا (MCD)."])
real_world(["أي المتغيرات تقيس الشيء نفسه تقريبًا؟", "هل البنية مختلفة بين المجموعات؟", "هل هناك مشاهدات غير عادية في التركيب؟",
            "ما الغرض: تفسير أم تنبؤ؟ (يحدد التعامل مع التعدد الخطي)"])

page_footer("multivariate",
            takeaways=["Heatmap مرتبة تكشف كتل المتغيرات.", "Pairplot وParallel coordinates وPCA للاستكشاف البصري.",
                       "Mahalanobis للشذوذ متعدد المتغيرات.", "VIF يقيس التعدد الخطي."],
            mistakes=["قراءة المصفوفة كعلاقات سببية.", "حذف المتغيرات بـVIF في مهام التنبؤ دون حاجة.", "PCA بلا توحيد المقاييس."])
