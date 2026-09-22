import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import why
from components.dataset_viewer import dataset_card, dataset_picker
from core.page import footer, page_header
from core.state import log_decision
from core.theme import PALETTE
from utils.outliers import MODEL_NAMES, iqr_mask, model_scores, modified_z_mask, percentile_mask, zscore_mask
from utils.plotting import heatmap, plot
from utils.types import numeric_columns

page_header("outlier_lab")
name, df = dataset_picker("ol_ds", default="customers_clean", allowed=["customers_clean", "credit", "students", "wine", "survey",
                                                                       "daily_sales"])
dataset_card(name)
nums = [c for c in numeric_columns(df) if df[c].nunique() > 5]
c1, c2 = st.columns(2)
xcol = c1.selectbox("المتغير الأساسي", nums, key="ol_x")
ycol = c2.selectbox("المتغير الثاني (للطرق متعددة المتغيرات)", [c for c in nums if c != xcol], key="ol_y")
d = df[[xcol, ycol]].dropna().astype(float)
c1, c2, c3 = st.columns(3)
k = c1.slider("IQR k", 1.0, 3.0, 1.5, 0.1, key="ol_k")
zt = c2.slider("Z threshold", 2.0, 4.0, 3.0, 0.1, key="ol_z")
cont = c3.slider("contamination", 0.005, 0.1, 0.03, 0.005, format="%.3f", key="ol_c")

flags = pd.DataFrame(index=d.index)
flags["IQR"] = iqr_mask(d[xcol], k)
flags["Z-score"] = zscore_mask(d[xcol], zt)
flags["Modified Z"] = modified_z_mask(d[xcol])
flags["Percentile"] = percentile_mask(d[xcol])
for m in ("isolation_forest", "lof"):
    flags[MODEL_NAMES[m]] = model_scores(d, m, cont)[0]

st.markdown("### كم نقطة تعلّمها كل طريقة؟")
counts = flags.sum().sort_values()
fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=PALETTE["coral"]))
fig.update_layout(height=280)
plot(fig)

st.markdown("### مصفوفة الاتفاق (Jaccard)")
cols = flags.columns
jac = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
for a in cols:
    for b in cols:
        inter = (flags[a] & flags[b]).sum()
        union = (flags[a] | flags[b]).sum()
        jac.loc[a, b] = inter / union if union else 1.0
plot(heatmap(jac, "Jaccard agreement between methods", zmin=0, zmax=1,
             colorscale=[[0, "#FFFBEA"], [1, PALETTE["coral_deep"]]]), height=420)
st.caption("اتفاق منخفض = الطرق تعرّف «الشذوذ» بشكل مختلف: أحادي المتغير مقابل متعدد، عام مقابل محلي.")

st.markdown("### أين تقع النقاط المعلّمة؟")
method = st.selectbox("اعرض", list(cols), key="ol_show")
votes = flags.sum(axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=d[xcol], y=d[ycol], mode="markers", name="all",
                         marker=dict(color=votes, colorscale=[[0, "#E9ECEF"], [1, PALETTE["softred"]]], size=6,
                                     showscale=True, colorbar=dict(title="votes"))))
sel = flags[method]
fig.add_trace(go.Scatter(x=d.loc[sel, xcol], y=d.loc[sel, ycol], mode="markers", name=method,
                         marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=PALETTE["purple"], width=2))))
fig.update_layout(xaxis_title=xcol, yaxis_title=ycol, height=440, legend=dict(orientation="h", y=1.1))
plot(fig)
st.markdown("### الحالات التي اتفقت عليها أغلب الطرق (للمراجعة البشرية)")
top = df.loc[votes[votes >= 4].index]
st.dataframe(top.head(30))
why("راجع أولًا الحالات التي تتفق عليها طرق متعددة.",
    "الإجماع يزيد احتمال أن الحالة غير عادية فعلًا؛ لكن حتى هذه قد تكون حقيقية ونادرة — قرر بأداة القرار في وحدة القيم الشاذة.")
note = st.text_input("قرارك بشأن هذه الحالات", placeholder="Keep: عملاء حقيقيون ذوو دخل مرتفع", key="ol_note")
if st.button("سجّل القرار", key="ol_log", icon=":material/bookmark:", disabled=not note):
    log_decision("outlier_lab", f"{name}.{xcol}", f"{len(top)} cases reviewed", note)
    st.toast("سُجّل القرار.")
footer()
