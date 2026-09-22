import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from components.dataset_viewer import dataset_card, dataset_picker
from core.page import footer, page_header
from core.theme import SEQUENCE
from utils.plotting import heatmap, plot, sample_for_plot
from utils.profiling import normality_hint
from utils.types import categorical_columns, numeric_columns

page_header("eda_lab")
name, df = dataset_picker("el_ds", default="customers_clean")
dataset_card(name)
nums = [c for c in numeric_columns(df) if df[c].nunique() > 2]
cats = [c for c in categorical_columns(df) if 1 < df[c].nunique() <= 15]

mode = st.segmented_control("نوع التحليل", ["أحادي", "ثنائي", "متعدد", "حسب المجموعة", "عبر الزمن"], default="أحادي", key="el_mode")

if mode == "أحادي":
    col = st.selectbox("المتغير", nums + cats, key="el_uni")
    if col in nums:
        x = df[col].dropna()
        c1, c2 = st.columns([3, 2])
        with c1:
            plot(px.histogram(df, x=col, nbins=50, marginal="box", color_discrete_sequence=SEQUENCE), height=400)
        with c2:
            st.dataframe(x.describe().round(3))
            st.markdown(f"**تفسير تلقائي:** {normality_hint(x)} الالتواء = {stats.skew(x):.2f}. "
                        f"{'المتوسط أكبر من الوسيط بوضوح ← ذيل أيمن.' if x.mean() > 1.1 * x.median() else ''} "
                        f"{int(((x - x.median()).abs() > 3.5 * stats.median_abs_deviation(x, scale='normal')).sum())} قيمة بعيدة (Modified Z > 3.5).")
    else:
        vc = df[col].value_counts(dropna=False).head(20)
        plot(px.bar(x=vc.values, y=vc.index.astype(str), orientation="h", color_discrete_sequence=SEQUENCE,
                    labels={"x": "count", "y": col}), height=400)
        st.markdown(f"**تفسير تلقائي:** {df[col].nunique()} فئة؛ الأكثر تكرارًا «{vc.index[0]}» بنسبة {vc.iloc[0] / len(df):.1%}.")
elif mode == "ثنائي":
    c1, c2 = st.columns(2)
    a = c1.selectbox("المتغير الأول", nums + cats, key="el_a")
    b = c2.selectbox("المتغير الثاني", [c for c in nums + cats if c != a], key="el_b")
    if a in nums and b in nums:
        d, _ = sample_for_plot(df[[a, b]].dropna())
        plot(px.scatter(d, x=a, y=b, trendline="lowess", opacity=0.5, color_discrete_sequence=SEQUENCE), height=420)
        r, p = stats.spearmanr(d[a], d[b])
        st.markdown(f"**تفسير تلقائي:** Spearman ρ = {r:.2f} (p = {p:.2g}) ← علاقة "
                    f"{'قوية' if abs(r) > 0.5 else 'متوسطة' if abs(r) > 0.3 else 'ضعيفة'} "
                    f"{'موجبة' if r > 0 else 'سالبة'}. منحنى LOWESS يكشف إن كانت غير خطية. الارتباط لا يعني السببية.")
    elif (a in nums) != (b in nums):
        numv, catv = (a, b) if a in nums else (b, a)
        plot(px.box(df, x=catv, y=numv, color=catv, points="outliers", color_discrete_sequence=SEQUENCE), height=420)
        groups = [g.dropna() for _, g in df.groupby(catv, observed=True)[numv] if len(g.dropna()) > 1]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            st.markdown(f"**تفسير تلقائي:** Kruskal–Wallis H = {h:.1f}, p = {p:.2g} ← "
                        f"{'دليل على اختلاف التوزيعات بين المجموعات' if p < 0.05 else 'لا دليل كافٍ على الاختلاف'} "
                        "(استكشافي: لا تبنِ عليه استنتاجًا تأكيديًا).")
    else:
        ct = pd.crosstab(df[a], df[b], normalize="index")
        plot(px.imshow(ct, text_auto=".0%", color_continuous_scale=["#FFFBEA", "#F76707"], aspect="auto"), height=420)
        chi2, p, *_ = stats.chi2_contingency(pd.crosstab(df[a], df[b]))
        st.markdown(f"**تفسير تلقائي:** χ² = {chi2:.1f}, p = {p:.2g}. النسب في كل صف تُقرأ كتوزيع {b} داخل فئة {a}.")
elif mode == "متعدد":
    chosen = st.multiselect("المتغيرات", nums, default=nums[: min(6, len(nums))], key="el_multi")
    if len(chosen) >= 2:
        plot(heatmap(df[chosen].corr(method="spearman"), "Spearman correlation"), height=460)
        hue = st.selectbox("لوّن حسب", ["(none)"] + cats, key="el_hue")
        fig = px.scatter_matrix(df, dimensions=chosen[:5], color=None if hue == "(none)" else hue,
                                color_discrete_sequence=SEQUENCE, opacity=0.5)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
        plot(fig, height=600)
elif mode == "حسب المجموعة":
    if cats and nums:
        g = st.selectbox("المجموعة", cats, key="el_g")
        m = st.multiselect("المقاييس", nums, default=nums[:3], key="el_m")
        if m:
            summ = df.groupby(g, observed=True)[m].agg(["count", "mean", "median"]).round(2)
            st.dataframe(summ)
            plot(px.bar(df.groupby(g, observed=True)[m[0]].median().reset_index(), x=g, y=m[0], color=g,
                        color_discrete_sequence=SEQUENCE, title=f"Median {m[0]} by {g}"), height=360)
    else:
        st.info("تحتاج متغيرًا فئويًا ورقميًا.")
else:
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols and nums:
        tcol = st.selectbox("عمود الزمن", dt_cols, key="el_t")
        v = st.selectbox("المقياس", nums, key="el_v")
        freq = st.segmented_control("التجميع", ["D", "W", "MS"], default="W", key="el_f")
        s = df.set_index(tcol)[v].resample(freq or "W").mean()
        plot(px.line(s.reset_index(), x=tcol, y=v, color_discrete_sequence=SEQUENCE), height=380)
        chg = s.pct_change().dropna()
        st.markdown(f"**تفسير تلقائي:** أكبر ارتفاع بين فترتين {chg.max():.1%} وأكبر انخفاض {chg.min():.1%}. "
                    "تحقق من الأحداث (عروض، أعطال) عند هذه النقاط.")
    else:
        st.info("لا يوجد عمود تاريخ في هذه البيانات؛ جرّب «مبيعات يومية» أو «معاملات متجر».")
st.caption("التفسيرات التلقائية قواعد بسيطة تساعد على صياغة الأسئلة؛ لا تغني عن فحص الرسوم والسياق.")
footer()
