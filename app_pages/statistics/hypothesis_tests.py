import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils import statistics as S
from utils.datasets import students
from utils.plotting import plot

page_header("hypothesis_tests")
df = students()

st.markdown("## نظام اختيار الاختبار · Test Decision System")
mermaid("""
flowchart LR
  Q[Research question] --> V[Variable types] --> D[Design:<br/>independent / paired] --> A[Assumptions] --> T[Candidate test] --> I[Interpretation:<br/>estimate + CI + effect size]
""")
c1, c2, c3, c4, c5 = st.columns(5)
goal = c1.selectbox("الهدف", ["compare", "relationship", "normality", "variance"],
                    format_func={"compare": "مقارنة مجموعات", "relationship": "علاقة", "normality": "فحص الطبيعية",
                                 "variance": "مقارنة التباين"}.get, key="ht_goal")
outcome = c2.selectbox("المتغير الناتج", ["numeric", "categorical"], format_func={"numeric": "رقمي/ترتيبي", "categorical": "فئوي"}.get,
                       key="ht_out")
groups = c3.selectbox("عدد المجموعات", ["one", "two", "three+"], index=1, format_func={"one": "واحدة", "two": "اثنتان", "three+": "ثلاث+"}.get,
                      key="ht_groups")
design = c4.selectbox("التصميم", ["independent", "paired"], format_func={"independent": "مستقل", "paired": "مزدوج/متكرر"}.get,
                      key="ht_design")
normal = c5.selectbox("قريب من الطبيعي؟", ["yes", "no"], format_func={"yes": "نعم / n كبير", "no": "لا / ترتيبي / ملتوٍ"}.get,
                      key="ht_normal")
test_key, reasoning = S.select_test(goal, outcome, groups, design, normal)
with st.container(border=True):
    st.markdown(f"### الاختبار المقترح: {S.TEST_NAMES[test_key]}")
    st.markdown(f"**لماذا؟** {reasoning}")

st.markdown("## شغّل الاختبار على بيانات الطلاب")
st.caption("طرق تدريس (Lecture / Flipped / Project-based)، اختبار قبلي وبعدي، ساعات دراسة، نوع المدرسة.")
chosen = st.selectbox("الاختبار", list(S.TEST_NAMES), index=list(S.TEST_NAMES).index(test_key),
                      format_func=S.TEST_NAMES.get, key="ht_run")
res = None
fig = None
if chosen == "one_sample_t":
    mu0 = st.number_input("القيمة المرجعية μ₀ للاختبار البعدي", value=65.0, key="ht_mu0")
    res = S.one_sample_t(df["post_score"], mu0)
    fig = px.histogram(df, x="post_score", nbins=30, color_discrete_sequence=SEQUENCE)
    fig.add_vline(x=mu0, line_dash="dash", annotation_text="μ₀")
elif chosen in ("independent_t", "mann_whitney"):
    g = df.groupby("school_type")["post_score"]
    a, b = g.get_group("Private"), g.get_group("Public")
    res = S.independent_t(a, b) if chosen == "independent_t" else S.mann_whitney(a, b)
    fig = px.violin(df, x="school_type", y="post_score", box=True, color="school_type", color_discrete_sequence=SEQUENCE)
    st.markdown("**السؤال:** هل تختلف درجة الاختبار البعدي بين المدارس الخاصة والعامة؟")
elif chosen in ("paired_t", "wilcoxon"):
    res = S.paired_t(df["pre_score"], df["post_score"]) if chosen == "paired_t" else S.wilcoxon(df["pre_score"], df["post_score"])
    diff = df["post_score"] - df["pre_score"]
    fig = px.histogram(diff, nbins=30, color_discrete_sequence=SEQUENCE, labels={"value": "post − pre"})
    fig.add_vline(x=0, line_dash="dash")
    st.markdown("**السؤال:** هل تحسنت درجات نفس الطلاب بين الاختبار القبلي والبعدي؟")
elif chosen in ("anova", "kruskal", "levene"):
    groups_d = {k: v for k, v in df.groupby("teaching_method")["post_score"]}
    res = {"anova": S.anova, "kruskal": S.kruskal, "levene": S.levene}[chosen](groups_d)
    fig = px.box(df, x="teaching_method", y="post_score", color="teaching_method", points="all", color_discrete_sequence=SEQUENCE)
    st.markdown("**السؤال:** هل تختلف الدرجات البعدية باختلاف طريقة التدريس؟")
elif chosen == "chi_square":
    table = pd.crosstab(df["teaching_method"], df["result"])
    res = S.chi_square(table)
    st.markdown("**السؤال:** هل ترتبط نتيجة النجاح/الرسوب بطريقة التدريس؟")
    c1, c2 = st.columns(2)
    c1.markdown("**المرصود Observed**")
    c1.dataframe(table)
    c2.markdown("**المتوقع تحت الاستقلال Expected**")
    c2.dataframe(res.extra["expected"])
    ct = pd.crosstab(df["teaching_method"], df["result"], normalize="index")
    fig = px.bar(ct, barmode="stack", color_discrete_sequence=SEQUENCE)
elif chosen in ("pearson", "spearman"):
    res = S.correlation(df["study_hours"], df["post_score"], chosen)
    fig = px.scatter(df, x="study_hours", y="post_score", trendline="ols", color_discrete_sequence=SEQUENCE, opacity=0.6)
    st.markdown("**السؤال:** هل ترتبط ساعات الدراسة بالدرجة البعدية؟")
elif chosen == "shapiro":
    var = st.selectbox("المتغير", ["post_score", "study_hours", "attendance_pct"], key="ht_shap")
    res = S.shapiro(df[var])
    x = np.sort(df[var].dropna())
    theo = stats.norm.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x))
    fig = go.Figure(go.Scatter(x=theo, y=x, mode="markers", marker=dict(color=PALETTE["purple"])))
    fig.add_trace(go.Scatter(x=theo, y=x.mean() + x.std() * theo, mode="lines", line=dict(color=PALETTE["coral"])))
    fig.update_layout(title="Q-Q plot مقابل التوزيع الطبيعي", xaxis_title="theoretical quantiles", yaxis_title="sample quantiles",
                      showlegend=False)
if fig is not None:
    fig.update_layout(height=380)
    plot(fig)

if res is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"الإحصاءة {res.statistic_name}", f"{res.statistic:.3f}", f"df: {res.df}" if res.df else None, delta_color="off")
    c2.metric("p-value", f"{res.p_value:.3g}")
    c3.metric(res.effect_name or "Effect", f"{res.effect:.3f}" if res.effect is not None else "—", res.effect_label, delta_color="off")
    c4.metric("CI 95%", f"[{res.ci[0]:.2f}, {res.ci[1]:.2f}]" if res.ci else "—")
    comparison_table([{"H0": res.h0, "H1": res.h1, "الاختبار": res.name}])
    alpha = st.select_slider("مستوى الدلالة α", [0.01, 0.05, 0.10], value=0.05, key="ht_alpha")
    st.markdown(f"**التفسير:** {res.interpretation(alpha)}")
    st.markdown("**الافتراضات:**\n" + "\n".join(f"- {a}" for a in res.assumptions))

why("أبلغ عن التقدير وفترة الثقة وحجم الأثر معًا، لا p-value وحدها.",
    "p-value تخلط حجم الأثر مع حجم العينة. التقدير + CI يقولان «كم» و«بأي دقة»، وهما ما يحتاجه صاحب القرار.")

st.markdown("## ملخص الاختبارات")
comparison_table([
    {"الاختبار": "One-sample t", "البيانات": "رقمي، عينة واحدة", "H0": "μ = μ₀", "حجم الأثر": "d", "البديل اللامعلمي": "Wilcoxon signed-rank"},
    {"الاختبار": "Independent t (Welch)", "البيانات": "رقمي، مجموعتان مستقلتان", "H0": "μ₁ = μ₂", "حجم الأثر": "d / Hedges g", "البديل اللامعلمي": "Mann–Whitney U"},
    {"الاختبار": "Paired t", "البيانات": "رقمي، قياسان لنفس الوحدات", "H0": "μ_d = 0", "حجم الأثر": "d_z", "البديل اللامعلمي": "Wilcoxon signed-rank"},
    {"الاختبار": "One-way ANOVA", "البيانات": "رقمي، 3+ مجموعات", "H0": "μ₁ = … = μ_k", "حجم الأثر": "η² / ω²", "البديل اللامعلمي": "Kruskal–Wallis"},
    {"الاختبار": "Chi-square", "البيانات": "فئويان", "H0": "استقلال", "حجم الأثر": "Cramér's V", "البديل اللامعلمي": "Fisher exact (جداول صغيرة)"},
    {"الاختبار": "Pearson", "البيانات": "رقميان، علاقة خطية", "H0": "ρ = 0", "حجم الأثر": "r", "البديل اللامعلمي": "Spearman / Kendall"},
    {"الاختبار": "Shapiro–Wilk", "البيانات": "رقمي", "H0": "طبيعي", "حجم الأثر": "—", "البديل اللامعلمي": "Q-Q plot (بصري)"},
    {"الاختبار": "Levene", "البيانات": "رقمي، مجموعات", "H0": "تساوي التباين", "حجم الأثر": "نسبة التباين", "البديل اللامعلمي": "Fligner–Killeen"},
])

if at_least("advanced"):
    st.markdown("## متقدم: المقارنات المتعددة واختبارات Post-hoc")
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tk = pairwise_tukeyhsd(df["post_score"], df["teaching_method"])
    tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
    st.dataframe(tk_df, hide_index=True)
    st.caption("Tukey HSD يقارن كل الأزواج مع ضبط معدل الخطأ العائلي. عند إجراء m اختبار مستقل: Bonferroni يستخدم α/m، "
               "وBenjamini–Hochberg يضبط معدل الاكتشافات الكاذبة FDR.")
if at_least("research"):
    researcher_note(["اختر الاختبار وα قبل رؤية البيانات؛ تغيير الاختبار بعد رؤية p-value شكل من p-hacking.",
                     "اختبارات الطبيعية مع n كبير ترفض لانحرافات تافهة؛ اعتمد على Q-Q plot ومتانة الاختبار.",
                     "Welch t-test هو الخيار الافتراضي الموصى به (Delacre et al., 2017).",
                     "للتصميمات المعقدة (قياسات متكررة، تجمعات): النماذج المختلطة Mixed models بدل الاختبارات البسيطة."])
real_world(["هل الوحدات مستقلة فعلًا؟ (طلاب في نفس الفصل ليسوا مستقلين تمامًا)", "هل حجم الأثر ذو معنى تربوي/تجاري؟",
            "هل أجريت اختبارات كثيرة واخترت الدال منها؟", "هل التصميم يسمح باستنتاج سببي أم ارتباطي فقط؟"])

page_footer("hypothesis_tests",
            takeaways=["السؤال ← الأنواع ← التصميم ← الافتراضات ← الاختبار ← التفسير.",
                       "H0 وH1 والإحصاءة وp وCI وحجم الأثر معًا.", "البدائل اللامعلمية للبيانات الملتوية/الترتيبية.",
                       "المقارنات المتعددة تحتاج تصحيحًا."],
            mistakes=["اختزال التحليل في p < 0.05.", "اختبار مستقل لبيانات مزدوجة.", "تجاهل الافتراضات.",
                      "«يثبت الاختبار أن…» بدل «يوفر دليلًا تحت افتراضات…»."])
