import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from components.callouts import intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.dataset_viewer import dataset_picker
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.plotting import plot, sample_for_plot
from utils.types import categorical_columns, numeric_columns

page_header("bivariate")

comparison_table([
    {"الزوج": "رقمي × رقمي", "الرسم": "Scatter (+ خط اتجاه)", "الملخص": "Pearson / Spearman", "الحذر": "علاقات غير خطية، قيم مؤثرة"},
    {"الزوج": "رقمي × فئوي", "الرسم": "Boxplot / Violin / Strip", "الملخص": "ملخصات حسب المجموعة", "الحذر": "أحجام مجموعات مختلفة"},
    {"الزوج": "فئوي × فئوي", "الرسم": "Stacked / grouped bar، Heatmap", "الملخص": "Crosstab بنسب شرطية", "الحذر": "اختيار اتجاه النسبة"},
])

name, df = dataset_picker("bi_ds", default="students",
                          allowed=["students", "customers_clean", "survey", "credit", "iris", "wine"])
nums = [c for c in numeric_columns(df) if df[c].nunique() > 2]
cats = [c for c in categorical_columns(df) if 1 < df[c].nunique() <= 12]

tab1, tab2, tab3 = st.tabs(["رقمي × رقمي", "رقمي × فئوي", "فئوي × فئوي"])
with tab1:
    c1, c2, c3 = st.columns(3)
    xa = c1.selectbox("X", nums, key="bi_x")
    ya = c2.selectbox("Y", nums, index=min(1, len(nums) - 1), key="bi_y")
    color = c3.selectbox("لوّن حسب", ["(none)"] + cats, key="bi_color")
    d = df[[xa, ya] + ([color] if color != "(none)" else [])].dropna()
    d, sampled = sample_for_plot(d)
    fig = px.scatter(d, x=xa, y=ya, color=None if color == "(none)" else color, opacity=0.6,
                     color_discrete_sequence=SEQUENCE, trendline="ols" if st.toggle("خط الانحدار", True, key="bi_tl") else None)
    fig.update_layout(height=440)
    plot(fig)
    if sampled:
        st.caption("عُرضت عينة عشوائية لتسريع الرسم.")
    pr, pp = stats.pearsonr(d[xa], d[ya])
    sr, sp = stats.spearmanr(d[xa], d[ya])
    c1, c2 = st.columns(2)
    c1.metric("Pearson r", f"{pr:.3f}", f"p = {pp:.2g}", delta_color="off")
    c2.metric("Spearman ρ", f"{sr:.3f}", f"p = {sp:.2g}", delta_color="off")
    why("اعرض الـScatter دائمًا بجانب معامل الارتباط.",
        "نفس قيمة r يمكن أن تنتج عن أشكال مختلفة جذريًا (رباعية Anscombe): علاقة خطية، منحنى، أو نقطة شاذة واحدة.")

with tab2:
    if cats:
        c1, c2 = st.columns(2)
        num = c1.selectbox("المتغير الرقمي", nums, key="bi_num")
        grp = c2.selectbox("المجموعة", cats, key="bi_grp")
        kind = st.segmented_control("الرسم", ["Boxplot", "Violin", "Strip"], default="Violin", key="bi_kind")
        d = df[[num, grp]].dropna()
        if kind == "Boxplot":
            fig = px.box(d, x=grp, y=num, color=grp, color_discrete_sequence=SEQUENCE, points="outliers")
        elif kind == "Violin":
            fig = px.violin(d, x=grp, y=num, color=grp, box=True, color_discrete_sequence=SEQUENCE)
        else:
            fig = px.strip(d, x=grp, y=num, color=grp, color_discrete_sequence=SEQUENCE)
        fig.update_layout(height=430, showlegend=False)
        plot(fig)
        summ = d.groupby(grp, observed=True)[num].agg(["count", "mean", "median", "std"]).round(2)
        st.dataframe(summ)
        st.code(f'df.groupby("{grp}")["{num}"].agg(["count", "mean", "median", "std"])', language="python")
    else:
        st.info("لا توجد متغيرات فئوية مناسبة.")

with tab3:
    if len(cats) >= 2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("الصفوف", cats, key="bi_a")
        b = c2.selectbox("الأعمدة", [c for c in cats if c != a], key="bi_b")
        norm = st.radio("النسبة", ["عدد", "نسب الصفوف (index)", "نسب الأعمدة (columns)", "من الكل (all)"],
                        horizontal=True, key="bi_norm")
        nmap = {"عدد": False, "نسب الصفوف (index)": "index", "نسب الأعمدة (columns)": "columns", "من الكل (all)": "all"}
        ct = pd.crosstab(df[a], df[b], normalize=nmap[norm], margins=nmap[norm] is False)
        st.dataframe((ct * (100 if nmap[norm] else 1)).round(1))
        ct2 = pd.crosstab(df[a], df[b], normalize="index")
        fig = go.Figure()
        for i, colname in enumerate(ct2.columns):
            fig.add_trace(go.Bar(x=[str(v) for v in ct2.index], y=ct2[colname], name=str(colname),
                                 marker_color=SEQUENCE[i % len(SEQUENCE)]))
        fig.update_layout(barmode="stack", title=f"توزيع {b} داخل كل فئة من {a} (100%)", height=380, yaxis_tickformat=".0%")
        plot(fig)
        st.caption("نسب الصفوف تجيب: «بين من هم في فئة A، كم نسبة كل فئة من B؟» — عادة الاتجاه الصحيح عندما يكون A "
                   "المتغير المفسِّر.")
    else:
        st.info("تحتاج متغيرين فئويين على الأقل.")

st.markdown("## الارتباط رياضيًا")
formula(r"r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\,\sqrt{\sum_i (y_i-\bar{y})^2}}",
        title="معامل Pearson", symbols={"r": "بين −1 و+1"},
        intuition="متوسط حاصل ضرب الانحرافات المعيارية: موجب إذا كانت القيم الكبيرة لـx تقترن بالكبيرة لـy.",
        example="Spearman = Pearson على **رتب** البيانات بدل قيمها؛ لذلك يلتقط أي علاقة رتيبة ويتحمل القيم المتطرفة.")

st.markdown("## رباعية Anscombe: نفس الإحصاءات، قصص مختلفة")
xs = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ans = pd.DataFrame({"dataset": np.repeat(["I", "II", "III", "IV"], 11),
                    "x": xs * 3 + [8] * 7 + [19] + [8] * 3,
                    "y": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68,
                          9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74,
                          7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73,
                          6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]})  # Anscombe (1973)
fig = px.scatter(ans, x="x", y="y", facet_col="dataset", facet_col_wrap=4, trendline="ols",
                 color_discrete_sequence=[PALETTE["coral"]])
fig.update_layout(height=320)
plot(fig)
st.caption("المجموعات الأربع لها تقريبًا نفس المتوسطات والتباينات وr ≈ 0.82 وخط الانحدار. الرسم وحده يكشف الفرق.")

intuition("الارتباط لا يعني السببية: مبيعات المثلجات وحالات الغرق مرتبطتان لأن الصيف يرفع الاثنين (متغير مربك Confounder).")
if at_least("advanced"):
    st.markdown("## متقدم: مفارقة Simpson")
    rng = np.random.default_rng(1)
    rows = []
    for g, (x0, y0) in enumerate([(2, 8), (5, 5), (8, 2)]):
        xx = rng.normal(x0, 0.8, 60)
        rows.append(pd.DataFrame({"x": xx, "y": y0 + 0.9 * (xx - x0) + rng.normal(0, 0.5, 60), "group": f"G{g + 1}"}))
    sim = pd.concat(rows)
    fig = px.scatter(sim, x="x", y="y", color="group", trendline="ols", color_discrete_sequence=SEQUENCE)
    fig.add_trace(go.Scatter(x=[0, 10], y=np.polyval(np.polyfit(sim["x"], sim["y"], 1), [0, 10]), mode="lines",
                             name="overall trend", line=dict(color="black", dash="dash")))
    fig.update_layout(height=380)
    plot(fig)
    st.caption(f"داخل كل مجموعة العلاقة موجبة، لكن إجمالًا r = {sim[['x', 'y']].corr().iloc[0, 1]:.2f} سالب.")
if at_least("research"):
    researcher_note(["أبلغ عن فترة ثقة للارتباط، لا قيمته فقط (انظر وحدة الاختبارات).",
                     "الارتباط في بيانات مجمّعة (دول، مدن) لا يُنقل للأفراد (Ecological fallacy)."])
real_world(["هل العلاقة متسقة عبر المجموعات؟", "هل يوجد متغير ثالث يفسرها؟", "هل تقودها نقاط قليلة؟",
            "هل العلاقة مستقرة عبر الزمن؟"])

page_footer("bivariate",
            takeaways=["اختر الأداة حسب نوعي المتغيرين.", "الرسم قبل المعامل دائمًا.", "Spearman للعلاقات الرتيبة والقيم المتطرفة.",
                       "حدد اتجاه النسب في Crosstab بوعي."],
            mistakes=["r = 0 يعني «لا علاقة».", "الاستنتاج السببي من الارتباط.", "نسب Crosstab في الاتجاه الخاطئ."])
