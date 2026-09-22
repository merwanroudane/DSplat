import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.diagrams import mermaid
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import daily_sales, survey
from utils.missing import IMPUTERS, impute, make_missing, multiple_imputation_mean, simulate_bivariate
from utils.plotting import add_animation_controls, plot

page_header("missing_treatment")

st.markdown("## شجرة قرار: كيف أتعامل مع القيم المفقودة؟")
mermaid("""
flowchart TB
  A[Missing values found] --> B{Semantic sentinels converted to NaN?}
  B -- no --> B1[Convert -999 / N/A / ? first] --> C
  B -- yes --> C{Share missing?}
  C -- "> 60% and not essential" --> C1[Consider dropping the variable<br/>document why]
  C -- otherwise --> D{Likely mechanism?}
  D -- MCAR, small share --> E[Listwise deletion acceptable<br/>or simple imputation]
  D -- MAR --> F[Model-based: regression / KNN / iterative / MI<br/>include the variables that explain missingness]
  D -- MNAR suspected --> G[Sensitivity analysis<br/>+ domain information<br/>+ missingness indicator]
  D -- Time series --> H[Interpolation / ffill within limits<br/>respect time order]
  E --> V[Validate: distribution, variance, relationships]
  F --> V
  G --> V
  H --> V
""")

st.markdown("## مقارنة الطرق")
METHODS = [
    ("Listwise deletion", "MCAR", "بسيط، غير منحاز تحت MCAR", "يفقد عينة وقوة؛ منحاز تحت MAR/MNAR", "فقد قليل وMCAR", "لا يغيّره (للمتبقي)", "يحافظ", "كبير إن لم يكن MCAR"),
    ("Pairwise deletion", "MCAR", "يستخدم كل البيانات المتاحة لكل زوج", "مصفوفات غير متسقة (قد لا تكون موجبة التعريف)", "ارتباطات وصفية", "—", "قد يتناقض", "كالحذف"),
    ("Mean", "MCAR", "سريع وبسيط", "يخفض التباين ويضعف الارتباطات", "خط أساس فقط", "قمة عند المتوسط", "يُضعفها", "في التباين والعلاقات"),
    ("Median", "MCAR", "متين أمام الالتواء", "كالمتوسط في خفض التباين", "متغيرات ملتوية وفقد قليل", "قمة عند الوسيط", "يُضعفها", "في التباين"),
    ("Mode", "MCAR", "للفئوي", "يضخم الفئة الأكثر تكرارًا", "فئوي بفقد قليل", "يضخم فئة", "يُضعفها", "نحو الفئة الشائعة"),
    ("Constant / 'Missing'", "—", "يحفظ معلومة «مفقود»", "قيمة اصطناعية تشوه الرقمي", "فئوي، أو مع مؤشر", "كتلة عند الثابت", "يشوّه", "كبير للرقمي"),
    ("Forward / Backward fill", "بطء التغير زمنيًا", "يحترم الزمن", "يطيل القيم القديمة؛ bfill يسرب المستقبل", "سلاسل بطيئة التغير", "تكرار قيم", "يرفع الارتباط الذاتي", "في الفجوات الطويلة"),
    ("Interpolation", "استمرارية", "سلس ومنطقي زمنيًا", "يفترض الخطية، يسرب المستقبل في التنبؤ", "سلاسل بفجوات قصيرة", "ينعّم", "يحافظ نسبيًا", "صغير لفجوات قصيرة"),
    ("KNN", "MAR", "يستخدم التشابه بين الصفوف", "حساس للمقياس، بطيء على بيانات كبيرة", "علاقات غير خطية محلية", "جيد", "يحافظ جزئيًا", "متوسط"),
    ("Regression", "MAR", "يستخدم العلاقات", "يبالغ في قوة العلاقات (قيم على الخط تمامًا)", "علاقات خطية واضحة", "ضيق", "يضخمها", "في التباين"),
    ("Stochastic regression", "MAR", "يضيف ضوضاء فيحفظ التباين", "يتجاهل عدم يقين المعاملات", "خطوة نحو MI", "جيد", "يحافظ", "صغير"),
    ("Iterative (MICE-style)", "MAR", "متعدد المتغيرات وتكراري", "أبطأ، يحتاج مواصفة", "فقد في عدة أعمدة", "جيد", "يحافظ", "صغير تحت MAR"),
    ("Multiple imputation", "MAR", "يعكس عدم اليقين في الأخطاء المعيارية", "أعقد، يتطلب دمج النتائج", "الاستدلال الإحصائي والبحث", "الأفضل", "يحافظ", "الأقل تحت MAR"),
]
comparison_table([{"الطريقة": m[0], "الافتراض": m[1], "المزايا": m[2], "المخاطر": m[3], "متى تناسب": m[4],
                   "أثر التوزيع": m[5], "أثر العلاقات": m[6], "الانحياز المحتمل": m[7]} for m in METHODS])

# --------------------------------------------------------------- experiment
st.markdown("## تجربة تفاعلية · Interactive Missing Data Experiment")
st.caption("نملك الحقيقة كاملة (بيانات محاكاة)، نحذف قيمًا بآلية تختارها، ثم نعوّضها ونقارن بالحقيقة.")
c1, c2, c3 = st.columns(3)
mech = c1.selectbox("آلية الفقد", ["MCAR", "MAR", "MNAR"], index=1, key="mt_mech")
rate = c2.slider("نسبة الفقد", 0.05, 0.6, 0.3, 0.05, key="mt_rate")
method = c3.selectbox("طريقة التعويض", [k for k in IMPUTERS if k not in ("ffill", "bfill", "interpolate", "constant")],
                      format_func=IMPUTERS.get, index=6, key="mt_method")
full = simulate_bivariate(800, 0.7, seed=3)
full["z"] = 0.5 * full["x"] + np.random.default_rng(4).normal(0, 6, len(full))
mask = make_missing(full, mech, rate, seed=5)
obs = full.copy()
obs.loc[mask, "y"] = np.nan
filled = impute(obs, method, cols=["y"], seed=1)
imputed_idx = mask[mask].index.intersection(filled.index)

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    keep = filled.index.difference(imputed_idx)
    fig.add_trace(go.Scatter(x=filled.loc[keep, "x"], y=filled.loc[keep, "y"], mode="markers", name="observed",
                             marker=dict(color=PALETTE["purple"], size=5, opacity=0.5)))
    if len(imputed_idx):
        fig.add_trace(go.Scatter(x=filled.loc[imputed_idx, "x"], y=filled.loc[imputed_idx, "y"], mode="markers",
                                 name="imputed", marker=dict(color=PALETTE["coral"], size=6, symbol="diamond")))
    fig.update_layout(title=f"{IMPUTERS[method]} تحت {mech}", height=380, legend=dict(orientation="h", y=1.12))
    plot(fig)
with c2:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=full["y"], name="truth", opacity=0.45, histnorm="probability density",
                               marker_color=PALETTE["amber"], nbinsx=40))
    fig.add_trace(go.Histogram(x=filled["y"], name="after imputation", opacity=0.55, histnorm="probability density",
                               marker_color=PALETTE["coral"], nbinsx=40))
    fig.update_layout(barmode="overlay", title="التوزيع بعد التعويض مقابل الحقيقة", height=380,
                      legend=dict(orientation="h", y=1.12))
    plot(fig)


def _metrics(d: pd.DataFrame, label: str) -> dict:
    dd = d[["x", "y"]].dropna()
    return {"النسخة": label, "Mean(y)": d["y"].mean(), "SD(y)": d["y"].std(), "Corr(x,y)": dd.corr().iloc[0, 1],
            "Slope y~x": np.polyfit(dd["x"], dd["y"], 1)[0], "n": len(dd)}


res = pd.DataFrame([_metrics(full, "الحقيقة"), _metrics(obs, "الحالات المكتملة"), _metrics(filled, IMPUTERS[method])])
if len(imputed_idx):
    rmse = float(np.sqrt(((filled.loc[imputed_idx, "y"] - full.loc[imputed_idx, "y"]) ** 2).mean()))
    res["RMSE on hidden values"] = [np.nan, np.nan, rmse]
st.dataframe(res.set_index("النسخة").round(3), width="stretch")

with st.expander("قارن كل الطرق دفعة واحدة", icon=":material/leaderboard:"):
    rows = []
    for m in [k for k in IMPUTERS if k not in ("ffill", "bfill", "interpolate", "constant", "listwise")]:
        f = impute(obs, m, cols=["y"], seed=1)
        met = _metrics(f, IMPUTERS[m])
        met["Mean bias"] = met["Mean(y)"] - full["y"].mean()
        met["SD ratio"] = met["SD(y)"] / full["y"].std()
        met["RMSE"] = float(np.sqrt(((f.loc[mask, "y"] - full.loc[mask, "y"]) ** 2).mean()))
        rows.append(met)
    st.dataframe(pd.DataFrame(rows).set_index("النسخة")[["Mean bias", "SD ratio", "Corr(x,y)", "Slope y~x", "RMSE"]].round(3),
                 width="stretch")
    st.caption(f"الحقيقة: Corr = {full[['x', 'y']].corr().iloc[0, 1]:.3f} · Slope = {np.polyfit(full['x'], full['y'], 1)[0]:.3f}. "
               "لاحظ: RMSE الأقل لا يعني الأفضل للاستدلال — Regression يعطي RMSE منخفضًا لكنه يخفض SD ويضخم الارتباط.")

why("للتحليل الاستدلالي تحت MAR فضّل Multiple Imputation أو Iterative مع ضوضاء، وليس الطريقة ذات أقل RMSE.",
    "الطرق الحتمية (Mean، Regression) تضع القيم «في المكان المتوقع تمامًا» فتخفض التباين وتجعل الأخطاء المعيارية "
    "أصغر من الحقيقة، أي ثقة زائفة.")

# ------------------------------------------------------------ mean animation
st.markdown("## رسم متحرك: أثر Mean imputation مع زيادة الفقد")
rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
frames = []
for r in rates:
    mk = make_missing(full, "MCAR", max(r, 1e-6), seed=9) if r else pd.Series(False, index=full.index)
    y = full["y"].copy()
    y[mk] = y[~mk].mean()
    frames.append(go.Frame(name=f"{int(r * 100)}%", data=[go.Histogram(x=y, xbins=dict(start=-15, end=55, size=1.5),
                                                                        marker_color=PALETTE["coral"], opacity=0.8)],
                           layout=go.Layout(title=f"فقد {int(r * 100)}% — SD = {y.std():.2f} (الحقيقي {full['y'].std():.2f})")))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(xaxis=dict(range=[-15, 55]), yaxis=dict(range=[0, 420]), height=420, title=frames[0].layout.title.text)
add_animation_controls(fig, [f.name for f in frames], duration=800, prefix="missing: ")
plot(fig)
st.caption("قمة تنمو عند المتوسط وتباين يتقلص: هذا التشوه يُغذّى لاحقًا إلى كل نموذج واختبار.")

# ---------------------------------------------------------------- time series
st.markdown("## السلاسل الزمنية: ffill وbfill وInterpolation")
ts = daily_sales().set_index("date").asfreq("D")
win = ts.loc["2024-02-01":"2024-04-15", "sales"].copy()
gap = win.copy()
gap.loc["2024-03-01":"2024-03-06"] = np.nan
opts = st.multiselect("الطرق", ["ffill", "bfill", "interpolate (linear)", "interpolate (time)"],
                      default=["ffill", "interpolate (linear)"], key="mt_ts")
fig = go.Figure()
fig.add_trace(go.Scatter(x=win.index, y=win, name="truth", line=dict(color="#BBB", dash="dot")))
fig.add_trace(go.Scatter(x=gap.index, y=gap, name="with gap", line=dict(color=PALETTE["purple"], width=2)))
fill = {"ffill": gap.ffill(), "bfill": gap.bfill(), "interpolate (linear)": gap.interpolate("linear"),
        "interpolate (time)": gap.interpolate("time")}
for i, o in enumerate(opts):
    part = fill[o].loc["2024-02-28":"2024-03-08"]
    fig.add_trace(go.Scatter(x=part.index, y=part, name=o, line=dict(color=SEQUENCE[(i + 2) % len(SEQUENCE)], width=3)))
fig.update_layout(height=360, legend=dict(orientation="h", y=1.12))
plot(fig)
st.markdown("`bfill` و`interpolate` يستخدمان قيمًا **مستقبلية**: مقبولان للوصف، لكنهما **Leakage** في مهام التنبؤ. "
            "استخدم `limit=` لمنع ملء فجوات طويلة: `s.interpolate(limit=3)`.")

# --------------------------------------------------------------------- MI
st.markdown("## التعويض المتعدد · Multiple Imputation")
formula(r"\bar{Q} = \frac{1}{m}\sum_{j=1}^{m}\hat{Q}_j,\quad \bar{U} = \frac{1}{m}\sum_{j}U_j,\quad "
        r"B = \frac{1}{m-1}\sum_j(\hat{Q}_j-\bar{Q})^2,\quad T = \bar{U} + \left(1+\frac{1}{m}\right)B",
        title="قواعد Rubin لدمج النتائج",
        symbols={r"\hat{Q}_j": "التقدير من النسخة j", "U_j": "تباين التقدير داخل النسخة j", "B": "التباين بين النسخ",
                 "T": "التباين الكلي (يُستخدم للخطأ المعياري وفترة الثقة)", "m": "عدد النسخ المعوّضة"},
        intuition="كل نسخة «تخمين معقول» مختلف. الاختلاف بين النسخ يقيس عدم اليقين الناتج عن الفقد نفسه، ويُضاف إلى الخطأ المعياري.")
m_count = st.slider("عدد النسخ m", 3, 20, 5, key="mt_m")
if st.button("شغّل Multiple Imputation لمتوسط الدخل (الاستبيان)", icon=":material/play_arrow:", key="mt_mi"):
    s_all = survey(include_truth=True)
    cols = ["age", "income", "work_hours", "stress_score", "job_satisfaction"]
    sv = s_all[cols].copy()
    sv["work_hours"] = sv["work_hours"].replace(-99, np.nan)
    r = multiple_imputation_mean(sv, "income", m=m_count, seed=2)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pooled mean", f"{r['pooled_mean']:,.0f}")
    c2.metric("SE (Rubin)", f"{r['se']:,.0f}")
    c3.metric("Complete-case mean", f"{sv['income'].mean():,.0f}")
    c4.metric("True mean (hidden)", f"{s_all['income_true'].mean():,.0f}")
    fig = go.Figure(go.Bar(x=[f"imp {i + 1}" for i in range(len(r["estimates"]))], y=r["estimates"],
                           marker_color=PALETTE["purple"]))
    fig.add_hline(y=r["pooled_mean"], line_color=PALETTE["coral"], annotation_text="pooled")
    fig.update_layout(height=300, yaxis=dict(range=[min(r["estimates"]) * 0.98, max(r["estimates"]) * 1.02]))
    plot(fig)
    st.warning("الدخل هنا **MNAR** (أصحاب الدخل المرتفع أقل إفصاحًا)، لذا حتى MI تحت افتراض MAR تبقى منحازة للأسفل. "
               "قارن بالمتوسط الحقيقي: هذا بالضبط سبب الحاجة لتحليل الحساسية في MNAR.", icon=":material/warning:")

st.markdown("## كود قابل للتشغيل")


def _run(strategy: str):
    from sklearn.impute import SimpleImputer
    d = survey()[["age", "income", "work_hours"]].replace(-99, np.nan)
    before = d.describe().loc[["count", "mean", "std"]]
    after = pd.DataFrame(SimpleImputer(strategy=strategy).fit_transform(d), columns=d.columns).describe().loc[["count", "mean", "std"]]
    return pd.concat({"before": before, "after": after})


code_lab("mt_simple", "SimpleImputer من scikit-learn",
         lambda p: ("from sklearn.impute import SimpleImputer\n"
                    'd = survey[["age", "income", "work_hours"]].replace(-99, np.nan)\n'
                    f'imp = SimpleImputer(strategy="{p["strategy"]}")\n'
                    "filled = pd.DataFrame(imp.fit_transform(d), columns=d.columns)\n"
                    "filled.describe()"),
         _run, params=lambda: {"strategy": st.radio("strategy", ["mean", "median", "most_frequent"], horizontal=True,
                                                      key="mt_strat")},
         explanation="count يصبح كاملًا، والمتوسط يبقى تقريبًا، لكن std ينخفض — العلامة المميزة للتعويض بقيمة واحدة. "
                     "في النمذجة ضع Imputer داخل Pipeline ليتعلم من بيانات التدريب فقط.")

if at_least("advanced"):
    st.markdown("## متقدم: مؤشر الفقد Missing indicator")
    st.markdown("إضافة عمود `income_was_missing` تحفظ معلومة الغياب التي قد تكون تنبؤية (خاصة تحت MNAR). "
                "في scikit-learn: `SimpleImputer(add_indicator=True)`. الحذر: في الاستدلال السببي قد يسبب انحيازًا.")
if at_least("research"):
    researcher_note(["اذكر في المنهجية: نسبة الفقد، الآلية المفترضة ومبررها، الطريقة، m، والمتغيرات في نموذج التعويض.",
                     "ضمّن في نموذج التعويض كل متغيرات التحليل والهدف (Congeniality) والمتغيرات المساعدة المرتبطة بالفقد.",
                     "قاعدة إرشادية: m ≥ نسبة الفقد المئوية (White, Royston & Wood, 2011).",
                     "قارن نتائجك مع Complete-case كتحليل حساسية."])
real_world(["هل حوّلت كل القيم الدلالية أولًا؟", "هل الطريقة تناسب الآلية المحتملة؟", "هل وضعت التعويض داخل Pipeline التدريب؟",
            "هل قارنت التوزيعات والتباين قبل وبعد؟", "هل وثّقت كم قيمة عُوّضت ولماذا؟"])

page_footer("missing_treatment",
            takeaways=["لا توجد طريقة مثلى عامة؛ الاختيار يعتمد على الآلية والهدف.", "التعويض بقيمة واحدة يخفض التباين ويضعف العلاقات.",
                       "Multiple Imputation مع قواعد Rubin للاستدلال تحت MAR.", "في السلاسل الزمنية احترم الترتيب واحذر التسرب."],
            mistakes=["Mean imputation بشكل أعمى.", "اختيار الطريقة بأقل RMSE للاستدلال.", "التعويض قبل تقسيم التدريب/الاختبار.",
                      "bfill في مهام التنبؤ."])
