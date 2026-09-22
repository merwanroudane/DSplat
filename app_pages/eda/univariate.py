import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import explain_code
from components.dataset_viewer import dataset_picker
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import ecdf, plot
from utils.types import categorical_columns, numeric_columns

page_header("univariate")

name, df = dataset_picker("uni_ds", default="customers_clean",
                          allowed=["customers_clean", "survey", "students", "credit", "daily_sales", "wine", "iris"])

st.markdown("## المتغيرات الرقمية")
nums = [c for c in numeric_columns(df) if df[c].nunique() > 2]
col = st.selectbox("المتغير", nums, key="uni_col")
x = df[col].dropna().astype(float)
q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
mode_val = x.mode().iloc[0] if not x.mode().empty else np.nan
stats_tbl = pd.DataFrame({
    "المقياس": ["Mean", "Median", "Mode", "Variance", "Std. deviation", "Range", "Q1", "Q3", "IQR", "Skewness", "Kurtosis (excess)", "n"],
    "القيمة": [x.mean(), med, mode_val, x.var(), x.std(), x.max() - x.min(), q1, q3, q3 - q1, stats.skew(x),
               stats.kurtosis(x), len(x)],
    "المعنى": ["مركز الثقل", "القيمة الوسطى", "الأكثر تكرارًا", "متوسط مربع الانحراف", "التشتت بوحدة المتغير",
               "Max − Min", "25% أقل منها", "75% أقل منها", "انتشار النصف الأوسط", "عدم التماثل", "ثقل الذيول مقارنة بالطبيعي",
               "عدد القيم غير المفقودة"],
})
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(stats_tbl.assign(**{"القيمة": stats_tbl["القيمة"].map(lambda v: f"{v:,.3f}")}), hide_index=True, height=460)
with c2:
    view = st.segmented_control("الرسم", ["Histogram", "Density", "Boxplot", "ECDF"], default="Histogram", key="uni_view")
    if view == "Histogram":
        bins = st.slider("عدد الفئات Bins", 5, 120, 40, key="uni_bins")
        fig = go.Figure(go.Histogram(x=x, nbinsx=bins, marker_color=PALETTE["coral"]))
        fig.add_vline(x=x.mean(), line_color=PALETTE["purple"], annotation_text="mean")
        fig.add_vline(x=med, line_color=PALETTE["teal"], line_dash="dash", annotation_text="median")
    elif view == "Density":
        grid = np.linspace(x.min(), x.max(), 300)
        bw = st.slider("عرض النطاق (مضاعف قاعدة Scott)", 0.2, 3.0, 1.0, 0.1, key="uni_bw")
        kde = stats.gaussian_kde(x, bw_method=lambda k: k.scotts_factor() * bw)
        fig = go.Figure(go.Scatter(x=grid, y=kde(grid), fill="tozeroy", line=dict(color=PALETTE["purple"])))
    elif view == "Boxplot":
        fig = go.Figure(go.Box(x=x, boxpoints="outliers", marker_color=PALETTE["purple"], name=col))
    else:
        fig = ecdf({col: x})
    fig.update_layout(height=420, title=f"{view}: {col}")
    plot(fig)

sk = stats.skew(x)
why(f"صف {col} بـ{'الوسيط وIQR' if abs(sk) > 1 else 'المتوسط والانحراف المعياري'}.",
    f"الالتواء = {sk:.2f}؛ " + ("التوزيع ملتوٍ بوضوح فالمتوسط يُسحب نحو الذيل والوسيط أصدق تمثيلًا للقيمة النموذجية."
                                if abs(sk) > 1 else "التوزيع قريب من التماثل فالمتوسط والانحراف ملخصان مناسبان."))

st.markdown("## الرياضيات")
formula(r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i", title="المتوسط Mean", symbols={"n": "عدد المشاهدات", "x_i": "المشاهدة i"},
        intuition="نقطة التوازن: مجموع الانحرافات عنه صفر. قيمة متطرفة واحدة تحركه.",
        example="[2, 3, 4, 100] ← المتوسط 27.25 والوسيط 3.5.")
formula(r"s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2,\qquad s = \sqrt{s^2}", title="التباين والانحراف المعياري",
        symbols={"n-1": "تصحيح Bessel: يجعل s² مقدِّرًا غير منحاز لتباين المجتمع"},
        intuition="متوسط مربع البعد عن المتوسط؛ الجذر يعيده لوحدة المتغير.",
        example="[2, 4, 6] ← المتوسط 4، الانحرافات [−2, 0, 2]، المربعات 8، s² = 8/2 = 4، s = 2.")
formula(r"g_1 = \frac{\frac{1}{n}\sum (x_i-\bar{x})^3}{\left(\frac{1}{n}\sum (x_i-\bar{x})^2\right)^{3/2}}", title="الالتواء Skewness",
        intuition="المكعب يحفظ الإشارة: ذيل أيمن طويل ← قيمة موجبة. الصفر يعني تماثلًا (وليس بالضرورة طبيعية).")
formula(r"\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{x_i \le x\}", title="ECDF",
        intuition="نسبة المشاهدات ≤ x. لا تحتاج اختيار bins، وتُظهر كل المئينات مباشرة.")

st.markdown("## المتغيرات الفئوية")
cats = [c for c in categorical_columns(df) if df[c].nunique() <= 40]
if cats:
    ccol = st.selectbox("المتغير الفئوي", cats, key="uni_cat")
    vc = df[ccol].value_counts(dropna=False)
    tbl = pd.DataFrame({"frequency": vc, "percent": (100 * vc / vc.sum()).round(2)})
    tbl["cumulative %"] = tbl["percent"].cumsum().round(2)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(tbl)
    c1.markdown(f"**المنوال Mode:** {vc.index[0]}")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[str(i) for i in tbl.index], y=tbl["frequency"], marker_color=PALETTE["purple"], name="frequency"))
        fig.add_trace(go.Scatter(x=[str(i) for i in tbl.index], y=tbl["cumulative %"], yaxis="y2", name="cumulative %",
                                 line=dict(color=PALETTE["coral"])))
        fig.update_layout(title="Pareto chart", yaxis2=dict(overlaying="y", side="right", range=[0, 105]), height=380,
                          legend=dict(orientation="h", y=1.1))
        plot(fig)
    st.caption("مبدأ Pareto: غالبًا تغطي فئات قليلة معظم المشاهدات — مفيد لتحديد الأولويات أو تجميع الفئات النادرة.")
else:
    st.info("لا توجد متغيرات فئوية مناسبة في هذه البيانات.")

explain_code("df['monthly_spend'].describe()",
             [("df['monthly_spend']", "Series للعمود"), (".describe()", "count, mean, std, min, 25%, 50%, 75%, max")],
             output="Series بثمانية ملخصات",
             interpretation="قارن mean بـ50% (الوسيط): فرق كبير = التواء. قارن max بـ75%: قفزة كبيرة = ذيل أو قيم طرفية.")

if at_least("advanced"):
    st.markdown("## متقدم: مقاييس متينة")
    comparison_table([
        {"المقياس": "Trimmed mean (10%)", "القيمة": f"{stats.trim_mean(x, 0.1):,.3f}", "الفكرة": "متوسط بعد حذف 10% من كل طرف"},
        {"المقياس": "MAD", "القيمة": f"{stats.median_abs_deviation(x):,.3f}", "الفكرة": "وسيط الانحرافات المطلقة"},
        {"المقياس": "IQR / 1.349", "القيمة": f"{(q3 - q1) / 1.349:,.3f}", "الفكرة": "تقدير متين لـσ تحت الطبيعية"},
        {"المقياس": "Std. deviation", "القيمة": f"{x.std():,.3f}", "الفكرة": "للمقارنة: يتأثر بالذيول"},
    ])
if at_least("research"):
    researcher_note(["أبلغ عن الوسيط وIQR للمتغيرات الملتوية، والمتوسط وSD للمتماثلة، مع n دائمًا.",
                     "Kurtosis في scipy يعيد «Excess kurtosis» (الطبيعي = 0) افتراضيًا؛ حدد ذلك في التقرير.",
                     "عدد الـBins يغيّر الانطباع؛ جرّب عدة قيم أو استخدم ECDF."])
real_world(["هل الوحدة والمدى منطقيان؟", "هل هناك تكدس عند قيم مستديرة؟", "هل التوزيع ثنائي القمة (مجموعتان مختلطتان)؟",
            "هل المتوسط يمثل «العميل النموذجي» فعلًا؟"])

page_footer("univariate",
            takeaways=["المركز + التشتت + الشكل + الأطراف = وصف كامل.", "الوسيط وIQR للتوزيعات الملتوية.",
                       "ECDF بديل قوي للـHistogram دون اختيار bins.", "Pareto للفئوي."],
            mistakes=["الاكتفاء بالمتوسط.", "الحكم على الشكل من Histogram واحد.", "تجاهل n والقيم المفقودة."])
