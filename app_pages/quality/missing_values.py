import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import domain, intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import explain_code
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import customers_raw, daily_sales, panel, survey
from utils.missing import (compare_by_missingness, impact_table, littles_mcar_test, make_missing, missing_by_group,
                           missing_patterns, missing_table, missingness_correlation, simulate_bivariate)
from utils.plotting import add_animation_controls, heatmap, missing_matrix, plot
from utils.types import semantic_missing_mask

page_header("missing_values")

# ------------------------------------------------------------ representations
st.markdown("## 1. كيف يظهر «المفقود»؟")
st.markdown("القيمة المفقودة ليست شكلًا واحدًا. بعضها يكتشفه `isna()` وبعضها **مفقود دلاليًا** فقط.")
demo = pd.DataFrame({
    "representation": ["np.nan", "None", "pd.NA", "pd.NaT", "'' (empty string)", "'N/A'", "'Unknown'", "'?'",
                       "-999", "9999", "0 (income)"],
    "value": [np.nan, None, pd.NA, pd.NaT, "", "N/A", "Unknown", "?", -999, 9999, 0],
})
demo["isna() يكتشفه؟"] = demo["value"].map(lambda v: "✅" if pd.isna(v) else "❌")
demo["مفقود دلاليًا؟"] = ["✅"] * 10 + ["⚠️ حسب السياق"]
st.dataframe(demo.drop(columns="value"), hide_index=True)
comparison_table([
    {"الشكل": "NaN", "المعنى": "Not a Number — تمثيل عشري للمفقود في NumPy/pandas", "ملاحظة": "NaN ≠ NaN دائمًا"},
    {"الشكل": "None", "المعنى": "كائن Python الفارغ", "ملاحظة": "يتحول إلى NaN في الأعمدة الرقمية"},
    {"الشكل": "pd.NA / NaT", "المعنى": "المفقود في الأنواع القابلة للقيم الفارغة Nullable / التواريخ", "ملاحظة": "pd.NA يُنتج منطقًا ثلاثي القيم"},
    {"الشكل": "Null", "المعنى": "المفقود في SQL", "ملاحظة": "NULL = NULL ليست True في SQL"},
    {"الشكل": "Sentinel / Placeholder", "المعنى": "قيمة اصطلاحية مثل -999، 9999، N/A، ?", "ملاحظة": "خطيرة: تدخل الحسابات كقيم حقيقية"},
])
domain("Income = 0: قد يكون دخلًا صفريًا حقيقيًا (طالب) أو رمزًا لـ«رفض الإجابة». لا يحسم ذلك إلا قاموس البيانات "
       "أو صاحب المصدر. الإحصاء وحده لا يكفي.")

st.markdown("### اكتشف القيم الدلالية في بيانات حقيقية")
choice = st.segmented_control("البيانات", ["الاستبيان", "العملاء الخام"], default="الاستبيان", key="mv_sem_ds")
d = survey() if choice == "الاستبيان" else customers_raw()
rows = []
for c in d.columns:
    m = semantic_missing_mask(d[c])
    rows.append({"column": c, "isna()": int(d[c].isna().sum()), "semantic missing": int(m.sum()),
                 "examples": ", ".join(repr(v) for v in d.loc[m, c].astype(str).unique()[:4])})
tbl = pd.DataFrame(rows)
st.dataframe(tbl[(tbl["isna()"] > 0) | (tbl["semantic missing"] > 0)], hide_index=True)
if choice == "الاستبيان":
    x = d["work_hours"]
    c1, c2 = st.columns(2)
    c1.metric("متوسط ساعات العمل (مع -99)", f"{x.mean():.2f}")
    c2.metric("المتوسط بعد تحويل -99 إلى NaN", f"{x.replace(-99, np.nan).mean():.2f}")
    st.caption("14 قيمة Sentinel فقط غيّرت المتوسط بوضوح. هذا سبب تحويلها قبل أي حساب.")

explain_code("df.isna().sum()", [("df", "الجدول"), (".isna()", "DataFrame منطقي: True حيث القيمة مفقودة (NaN/None/NA/NaT)"),
                                 (".sum()", "يجمع كل عمود؛ True = 1 فيعطي عدد المفقود")],
             output="Series: اسم العمود ← عدد القيم المفقودة",
             interpretation="الأرقام تشمل المفقود الصريح فقط. أضف فحص القيم الدلالية، واقسم على len(df) لتحصل على النسبة.")

# ---------------------------------------------------------------- mechanisms
st.markdown("## 2. آليات الفقد · Missingness mechanisms (Rubin, 1976)")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**MCAR** — Missing Completely At Random")
    st.markdown("الفقد لا يعتمد على أي شيء. *مثال:* ضاعت استمارات بسبب عطل عشوائي في الخادم.")
with c2:
    st.markdown("**MAR** — Missing At Random")
    st.markdown("الفقد يعتمد على متغيرات **مرصودة**. *مثال:* الشباب يتركون سؤال الضغط أكثر، والعمر معروف للجميع.")
with c3:
    st.markdown("**MNAR** — Missing Not At Random")
    st.markdown("الفقد يعتمد على **القيمة المفقودة نفسها**. *مثال:* أصحاب الدخل المرتفع يرفضون الإفصاح عن دخلهم.")
mermaid("""
flowchart LR
  subgraph MCAR
    X1[X observed] ~~~ Y1[Y value]
    R1[Missing indicator R]
  end
  subgraph MAR
    X2[X observed] --> R2[R]
    Y2[Y value]
  end
  subgraph MNAR
    Y3[Y value] --> R3[R]
    X3[X observed]
  end
""")
intuition("السؤال الجوهري: **هل تعرف لماذا القيمة مفقودة؟** إن كان السبب مرصودًا في البيانات (MAR) يمكن تصحيحه "
          "باستخدام ذلك المتغير. إن كان السبب هو القيمة نفسها (MNAR) فالبيانات وحدها لا تكفي.")

st.markdown("## 3. محاكاة تفاعلية: كيف يشوّه كل نوع التقديرات؟")
mech = st.segmented_control("اختر الآلية", ["Simulate MCAR", "Simulate MAR", "Simulate MNAR"],
                            default="Simulate MAR", key="mv_mech")
mech = (mech or "Simulate MAR").split()[-1]
c1, c2, c3 = st.columns(3)
rate = c1.slider("نسبة الفقد في y", 0.05, 0.7, 0.3, 0.05, key="mv_rate")
rho = c2.slider("الارتباط الحقيقي بين x وy", 0.0, 0.95, 0.6, 0.05, key="mv_rho")
n = c3.select_slider("حجم العينة n", [200, 500, 1000, 2000], value=1000, key="mv_n")
full = simulate_bivariate(n, rho, seed=7)
mask = make_missing(full, mech, rate, seed=11)
obs = full.copy()
obs.loc[mask, "y"] = np.nan

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full.loc[~mask, "x"], y=full.loc[~mask, "y"], mode="markers", name="observed",
                             marker=dict(color=PALETTE["purple"], size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=full.loc[mask, "x"], y=full.loc[mask, "y"], mode="markers", name="missing (hidden truth)",
                             marker=dict(color=PALETTE["coral"], size=5, opacity=0.55, symbol="x")))
    fig.update_layout(title=f"{mech}: القيم الحقيقية المخفية", xaxis_title="x (always observed)", yaxis_title="y",
                      height=380, legend=dict(orientation="h", y=1.12))
    plot(fig)
with c2:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=full["y"], name="full data (truth)", opacity=0.5, histnorm="probability density",
                               marker_color=PALETTE["amber"], nbinsx=40))
    fig.add_trace(go.Histogram(x=obs["y"].dropna(), name="observed only", opacity=0.55, histnorm="probability density",
                               marker_color=PALETTE["purple"], nbinsx=40))
    fig.add_vline(x=full["y"].mean(), line_color=PALETTE["amber"], line_dash="dash")
    fig.add_vline(x=obs["y"].mean(), line_color=PALETTE["purple"], line_dash="dash")
    fig.update_layout(barmode="overlay", title="التوزيع: الحقيقة مقابل المرصود", height=380,
                      legend=dict(orientation="h", y=1.12))
    plot(fig)
st.markdown("**أثر تحليل الحالات المكتملة Complete-case analysis:**")
st.dataframe(impact_table(full, obs), width="stretch")
explain = {
    "MCAR": "الانحياز قريب من الصفر في كل المقاييس: العينة المتبقية عينة عشوائية. الخسارة في الدقة (n أصغر) لا في الانحياز.",
    "MAR": "المتوسط والتباين منحازان لأن القيم المفقودة تتركز حيث x كبير. لكن ميل الانحدار y~x شبه سليم: "
           "المشروطية على x تصحح الانحياز — لهذا تنجح طرق التعويض التي تستخدم x.",
    "MNAR": "كل المقاييس منحازة، حتى الميل: القيم الكبيرة من y نفسها هي التي اختفت. لا يوجد تصحيح من البيانات المرصودة وحدها.",
}
why("اختر طريقة المعالجة بعد تشخيص الآلية المحتملة، لا قبله.", explain[mech])

with st.expander("رسم متحرك: كيف يتغير المتوسط المرصود مع زيادة نسبة الفقد؟", icon=":material/animation:"):
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    frames, means = [], []
    for r in rates:
        m = make_missing(full, mech, max(r, 1e-6), seed=11) if r > 0 else pd.Series(False, index=full.index)
        yo = full.loc[~m, "y"]
        means.append(yo.mean())
        frames.append(go.Frame(name=f"{int(r * 100)}%", data=[
            go.Histogram(x=yo, nbinsx=40, marker_color=PALETTE["purple"], opacity=0.75, histnorm="probability density"),
            go.Scatter(x=[yo.mean(), yo.mean()], y=[0, 0.05], mode="lines", line=dict(color=PALETTE["coral"], width=3))]))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.add_vline(x=full["y"].mean(), line_dash="dash", line_color=PALETTE["amber"], annotation_text="true mean")
    fig.update_layout(title=f"{mech}: توزيع y المرصود (الخط البرتقالي = المتوسط المرصود)", showlegend=False,
                      xaxis=dict(range=[full['y'].min(), full['y'].max()]), yaxis=dict(range=[0, 0.05]), height=420)
    add_animation_controls(fig, [f.name for f in frames], duration=700, prefix="missing rate: ")
    plot(fig)
    st.caption(" · ".join(f"{int(r * 100)}% → mean {mu:.2f}" for r, mu in zip(rates, means)))

# --------------------------------------------------------------- diagnostics
st.markdown("## 4. تشخيص الفقد · Missingness diagnostics")
sv = survey()
sv["work_hours"] = sv["work_hours"].replace(-99, np.nan)
st.caption("على بيانات الاستبيان بعد تحويل Sentinel -99 إلى NaN.")
t1, t2, t3, t4, t5 = st.tabs(["العدد والنسبة", "المصفوفة Matrix", "الأنماط Patterns", "حسب المجموعة", "الارتباط بالفقد"])
with t1:
    mt = missing_table(sv)
    st.dataframe(mt[mt["missing"] > 0])
    fig = go.Figure(go.Bar(x=mt["missing_%"], y=mt.index, orientation="h", marker_color=PALETTE["coral"]))
    fig.update_layout(height=380, xaxis_title="% missing", yaxis=dict(autorange="reversed"))
    plot(fig)
with t2:
    plot(missing_matrix(sv.sort_values("age")), height=430)
    st.caption("الصفوف مرتبة حسب العمر: لاحظ تركز فقد stress_score في أعلى المصفوفة (الأصغر سنًا) ← دليل على MAR.")
with t3:
    st.markdown("كل صف نمط فقد مختلف (1 = مفقود). **Joint missingness**: هل تُفقد أعمدة معًا؟")
    st.dataframe(missing_patterns(sv), hide_index=True)
    mc = missingness_correlation(sv)
    if not mc.empty:
        plot(heatmap(mc, "ارتباط مؤشرات الفقد"), height=360)
        st.caption("ارتباطات قريبة من الصفر: الأعمدة تُفقد باستقلال تقريبي. ارتباط قوي يعني أن سببًا مشتركًا يحذفها معًا.")
    if at_least("advanced"):
        st.markdown("**Monotone missingness:** إن كان فقد عمود يعني فقد كل الأعمدة التالية (مثل انسحاب مشارك من دراسة "
                    "طولية)، فالنمط «رتيب» ويسمح بطرق تعويض تتابعية أبسط.")
with t4:
    g = st.selectbox("المتغير المجمِّع", ["region", "education", "gender"], key="mv_grp")
    tgt = st.selectbox("المتغير المفقود", ["income", "stress_score", "q2"], key="mv_tgt")
    mg = missing_by_group(sv, tgt, g)
    fig = go.Figure(go.Bar(x=mg.index.astype(str), y=mg["missing_%"], marker_color=PALETTE["purple"],
                           text=mg["n"].map(lambda v: f"n={v}"), textposition="outside"))
    fig.update_layout(yaxis_title=f"% missing in {tgt}", height=340)
    plot(fig)
with t5:
    tgt2 = st.selectbox("هل يختلف الآخرون حين يُفقد…", ["stress_score", "income", "q2"], key="mv_cmp")
    st.dataframe(compare_by_missingness(sv, tgt2), hide_index=True)
    st.markdown("فرق كبير ودال (|d| متوسط أو أكبر) ← الفقد **ليس MCAR** ويتسق مع MAR على ذلك المتغير. "
                "غياب الفرق **لا يثبت** MCAR، ولا يمكنه استبعاد MNAR.")

if at_least("advanced"):
    st.markdown("### اختبار Little لـMCAR")
    cols = ["age", "income", "work_hours", "stress_score", "job_satisfaction", "q2", "q4"]
    res = littles_mcar_test(sv[cols])
    c1, c2, c3 = st.columns(3)
    c1.metric("χ² (d²)", f"{res['statistic']:.1f}")
    c2.metric("df", res["df"])
    c3.metric("p-value", f"{res['p_value']:.2g}")
    st.markdown("**H0:** البيانات MCAR. رفض H0 يعطي دليلًا ضد MCAR. **القيود:** يفترض الطبيعية متعددة المتغيرات، "
                "قوته ضعيفة مع عينات صغيرة، ولا يستطيع تمييز MAR من MNAR.")

st.markdown("## 5. التشخيص حسب تصميم البيانات")
design = st.segmented_control("التصميم", ["Cross-sectional", "Time series", "Panel"], default="Time series",
                              key="mv_design")
if design == "Time series":
    ts = daily_sales()
    full_idx = pd.date_range(ts["date"].min(), ts["date"].max(), freq="D")
    missing_days = full_idx.difference(ts["date"])
    st.markdown(f"في السلاسل الزمنية نوعان من الفقد: **قيم مفقودة** ({int(ts['sales'].isna().sum())}) و**طوابع زمنية مفقودة** "
                f"({len(missing_days)} يومًا غير موجود أصلًا في الجدول — لن يكشفها `isna()`).")
    re = ts.set_index("date").reindex(full_idx)
    fig = go.Figure(go.Scatter(x=re.index, y=re["sales"], mode="lines", line=dict(color=PALETTE["purple"], width=1)))
    for dday in missing_days:
        fig.add_vrect(x0=dday, x1=dday + pd.Timedelta(days=1), fillcolor=PALETTE["coral"], opacity=0.35, line_width=0)
    fig.update_layout(title="الفجوات الحمراء = أيام غائبة من الفهرس الزمني", height=320)
    plot(fig)
    st.code('full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")\n'
            'missing_days = full.difference(df["date"])\ndf = df.set_index("date").reindex(full)', language="python")
elif design == "Panel":
    p = panel()
    status = (p.assign(miss=p["rnd_spend"].isna().astype(int))
              .pivot_table(index="firm_id", columns="year", values="miss", aggfunc="max"))
    z = status.fillna(-1).to_numpy()  # -1 = firm not present that year
    fig = go.Figure(go.Heatmap(z=z, x=status.columns, y=status.index,
                               colorscale=[[0, "#E9ECEF"], [0.5, "#E7F5FF"], [1, PALETTE["coral_deep"]]], zmin=-1, zmax=1,
                               showscale=False))
    fig.update_layout(title="شركة × سنة: رمادي = الشركة غير موجودة، برتقالي داكن = rnd_spend مفقود", height=520)
    plot(fig)
    yearly = p.groupby("year")["rnd_spend"].apply(lambda s: s.isna().mean() * 100)
    st.markdown(f"نسبة الفقد ترتفع من {yearly.iloc[0]:.0f}% إلى {yearly.iloc[-1]:.0f}% عبر السنوات (إرهاق الإبلاغ). "
                "في Panel ميّز بين: **فقد عنصر** (قيمة غائبة لشركة موجودة) و**فقد وحدة/انسحاب Attrition** (الشركة خرجت).")
else:
    st.markdown("في البيانات المقطعية نركز على: النسبة لكل متغير، الأنماط المشتركة، والفروق بين من فُقدت قيمته ومن لم تُفقد "
                "(الأدوات في القسم 4). لا يوجد ترتيب زمني يُستغل في التعويض.")

if at_least("research"):
    researcher_note(["MCAR/MAR/MNAR افتراضات عن آلية لا نراها؛ البيانات قد تدعم أو تضعف فرضية، لكنها لا تثبت MAR ولا تنفي MNAR.",
                     "أبلغ عن نسب الفقد لكل متغير، ومقارنة من أجاب بمن لم يُجب، والطريقة المستخدمة وافتراضها.",
                     "لـMNAR أجرِ تحليل حساسية (δ-adjustment أو Selection models) — انظر Little & Rubin (2019)، van Buuren (2018)."])
real_world(["هل الفقد منهجي؟ (يعتمد على شيء)", "هل يرتبط بمجموعة معينة (منطقة، فرع، موظف إدخال)؟",
            "هل سببه عملية الجمع (سؤال حساس، حقل اختياري، عطل)؟", "هل يعتمد على الزمن؟",
            "هل الفقد نفسه يحمل معلومة؟ (عدم الإفصاح عن الدخل قد يتنبأ بالسلوك)"])

page_footer("missing_values",
            takeaways=["الفقد صريح (NaN) أو دلالي (-999, N/A)؛ افحص الاثنين.", "MCAR لا ينحاز، MAR قابل للتصحيح بالمتغيرات المرصودة، MNAR يحتاج افتراضات خارجية.",
                       "التشخيص بالأنماط والمجموعات والمقارنات يسبق اختيار العلاج.", "في السلاسل الزمنية ابحث عن الطوابع المفقودة، وفي Panel عن الانسحاب."],
            mistakes=["حساب المتوسط قبل تحويل Sentinels.", "افتراض MCAR دون فحص.", "الاعتماد على اختبار Little وحده كإثبات.",
                      "تجاهل الأيام الغائبة من فهرس السلسلة الزمنية."])
