import plotly.graph_objects as go
import streamlit as st

from components.callouts import why
from components.dataset_viewer import dataset_card, dataset_picker
from core.page import footer, page_header
from core.state import log_decision
from core.theme import PALETTE
from utils.missing import IMPUTERS, compare_by_missingness, impute, masked_evaluation, missing_patterns
from utils.plotting import missing_matrix, plot
from utils.types import numeric_columns, semantic_missing_mask

page_header("missing_lab")
st.markdown("**الخطة:** (1) حوّل القيم الدلالية، (2) شخّص، (3) أخفِ قيمًا **معروفة** وقِس خطأ كل طريقة في استرجاعها، "
            "(4) اختر الطريقة ووثّق القرار.")

name, df = dataset_picker("ml_ds", default="survey", allowed=["survey", "customers_clean", "students", "credit", "panel", "wine"])
dataset_card(name)
num_cols = numeric_columns(df)

st.markdown("### 1. القيم الدلالية")
conv = []
for c in num_cols:
    m = semantic_missing_mask(df[c])
    if m.any():
        conv.append(c)
        df[c] = df[c].where(~m)  # where() upcasts ints safely
st.markdown(f"حُوّلت القيم الدلالية إلى NaN في: {', '.join(conv) if conv else 'لا شيء'}.")

st.markdown("### 2. التشخيص")
c1, c2 = st.columns([3, 2])
with c1:
    plot(missing_matrix(df[num_cols]), height=380)
with c2:
    pat = missing_patterns(df[num_cols])
    if pat.empty:
        st.success("لا توجد قيم مفقودة في الأعمدة الرقمية — التجربة أدناه ستُخفي قيمًا عشوائيًا لتقييم الطرق.")
    else:
        st.dataframe(pat, hide_index=True)
target = st.selectbox("المتغير المستهدف بالتعويض", num_cols,
                      index=num_cols.index("stress_score") if "stress_score" in num_cols else 0, key="ml_target")
if df[target].isna().any():
    cmp = compare_by_missingness(df, target)
    if not cmp.empty:
        sig = cmp[cmp["p_value"] < 0.05]["variable"].tolist()
        st.markdown(f"متغيرات تختلف دالًا حين يُفقد **{target}**: {', '.join(sig) if sig else 'لا شيء'} "
                    + ("← الفقد ليس MCAR على الأرجح؛ فضّل طرقًا تستخدم هذه المتغيرات." if sig else "← لا دليل ضد MCAR (وهذا ليس إثباتًا له)."))

st.markdown("### 3. التقييم بإخفاء قيم معروفة")
c1, c2 = st.columns(2)
frac = c1.slider("نسبة القيم المعروفة التي تُخفى", 0.05, 0.4, 0.15, 0.05, key="ml_frac")
methods = c2.multiselect("الطرق", [k for k in IMPUTERS if k not in ("listwise",)], default=["mean", "median", "knn", "regression",
                                                                                           "iterative"], format_func=IMPUTERS.get,
                         key="ml_methods")


@st.cache_data(show_spinner="جارٍ التقييم…")
def evaluate(data, target: str, frac: float, methods: tuple[str, ...]):
    return masked_evaluation(data, target, list(methods), frac=frac, seed=0)


if methods:
    res = evaluate(df[num_cols], target, frac, tuple(methods))
    st.dataframe(res, hide_index=True)
    fig = go.Figure(go.Bar(x=res["RMSE"], y=res["method"], orientation="h", marker_color=PALETTE["purple"]))
    fig.update_layout(height=300, xaxis_title="RMSE on hidden values (lower is better)", yaxis=dict(autorange="reversed"))
    plot(fig)
    why("لا تختر بأقل RMSE وحده؛ انظر عمود «SD after / SD before».",
        "الطرق الحتمية تحقق RMSE جيدًا لكنها تقلص التباين (النسبة < 1)، فتجعل الاستدلال واثقًا أكثر من اللازم. "
        "وتذكّر: الإخفاء هنا عشوائي (MCAR)، أما الفقد الحقيقي فقد يكون MAR أو MNAR.")

st.markdown("### 4. طبّق ووثّق")
chosen = st.selectbox("الطريقة المختارة", [k for k in IMPUTERS if k != "listwise"], format_func=IMPUTERS.get, index=8, key="ml_chosen")
filled = impute(df, chosen, cols=[target])
fig = go.Figure()
fig.add_trace(go.Histogram(x=df[target], name="observed", opacity=0.55, marker_color=PALETTE["purple"], nbinsx=40))
fig.add_trace(go.Histogram(x=filled.loc[df[target].isna(), target], name="imputed values", opacity=0.75,
                           marker_color=PALETTE["coral"], nbinsx=40))
fig.update_layout(barmode="overlay", height=320, legend=dict(orientation="h", y=1.12))
plot(fig)
reason = st.text_input("سبب الاختيار (للتوثيق)", placeholder="الفقد MAR على العمر؛ KNN يحافظ على العلاقات", key="ml_reason")
c1, c2 = st.columns(2)
if c1.button("سجّل القرار", icon=":material/bookmark:", key="ml_log", disabled=not reason):
    log_decision("missing_lab", f"{name}.{target}", IMPUTERS[chosen], reason)
    st.toast("سُجّل في سجل القرارات.")
c2.download_button("حمّل البيانات بعد التعويض", filled.to_csv(index=False).encode("utf-8-sig"), f"{name}_imputed.csv",
                   "text/csv", icon=":material/download:")
footer()
