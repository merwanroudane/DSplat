import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from components.animation import stepper
from components.callouts import intuition, real_world, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("stat_foundations")

st.markdown("## المجتمع والعينة")
comparison_table([
    {"المفهوم": "Population المجتمع", "المعنى": "كل الوحدات التي نريد الاستدلال عنها", "مثال": "كل عملاء البنك"},
    {"المفهوم": "Sample العينة", "المعنى": "الجزء الذي نرصده فعلًا", "مثال": "500 عميل اخترناهم"},
    {"المفهوم": "Parameter المعلمة", "المعنى": "قيمة ثابتة تصف المجتمع (مجهولة غالبًا)", "مثال": "μ متوسط دخل كل العملاء"},
    {"المفهوم": "Statistic الإحصاءة", "المعنى": "قيمة محسوبة من العينة لتقدير المعلمة", "مثال": "x̄ متوسط دخل العينة"},
    {"المفهوم": "Random variable متغير عشوائي", "المعنى": "كمية تأخذ قيمًا باحتمالات", "مثال": "دخل عميل يُختار عشوائيًا"},
    {"المفهوم": "Distribution التوزيع", "المعنى": "كيف تتوزع الاحتمالات على القيم", "مثال": "Normal، Binomial، Log-normal"},
])
intuition("لو سحبنا عينة أخرى لحصلنا على متوسط مختلف قليلًا. **الإحصاء الاستدلالي** كله يدور حول سؤال: "
          "كم يمكن أن تتغير الإحصاءة من عينة لأخرى؟ هذا التغير هو **Sampling variability**.")

st.markdown("## توزيع المعاينة ونظرية النهاية المركزية (CLT)")
c1, c2, c3 = st.columns(3)
pop_shape = c1.selectbox("شكل المجتمع", ["Log-normal (ملتوٍ)", "Uniform", "Bimodal", "Normal"], key="sf_shape")
n = c2.select_slider("حجم العينة n", [2, 5, 10, 30, 100], value=10, key="sf_n")
reps = c3.select_slider("عدد العينات", [200, 1000, 3000], value=1000, key="sf_reps")
rng = np.random.default_rng(11)
POP = {
    "Log-normal (ملتوٍ)": lambda k: rng.lognormal(3, 0.8, k),
    "Uniform": lambda k: rng.uniform(0, 100, k),
    "Bimodal": lambda k: np.where(rng.random(k) < 0.5, rng.normal(30, 6, k), rng.normal(75, 6, k)),
    "Normal": lambda k: rng.normal(50, 12, k),
}
population = POP[pop_shape](200_000)
mu, sigma = population.mean(), population.std()
means = population[rng.integers(0, len(population), (reps, n))].mean(axis=1)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Histogram(x=population[:20000], nbinsx=60, marker_color=PALETTE["amber"]))
    fig.update_layout(title="المجتمع", height=320)
    plot(fig)
with c2:
    fig = go.Figure(go.Histogram(x=means, nbinsx=50, histnorm="probability density", marker_color=PALETTE["purple"], name="sample means"))
    grid = np.linspace(means.min(), means.max(), 200)
    fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, mu, sigma / np.sqrt(n)), name="N(μ, σ/√n)",
                             line=dict(color=PALETTE["coral"], width=3)))
    fig.update_layout(title=f"توزيع متوسطات {reps} عينة بحجم n = {n}", height=320, legend=dict(orientation="h", y=1.15))
    plot(fig)
c1, c2, c3 = st.columns(3)
c1.metric("SD للمجتمع σ", f"{sigma:.2f}")
c2.metric("SE النظري σ/√n", f"{sigma / np.sqrt(n):.2f}")
c3.metric("SD الفعلي لمتوسطات العينات", f"{means.std():.2f}")
st.caption("مع زيادة n يقترب توزيع المتوسطات من الطبيعي مهما كان شكل المجتمع (بشرط تباين محدود)، ويضيق بمعدل 1/√n.")

formula(r"\text{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}}", title="الخطأ المعياري Standard Error",
        symbols={r"\sigma": "انحراف المجتمع", "s": "انحراف العينة (تقدير لـσ)", "n": "حجم العينة"},
        intuition="SD يصف تشتت **البيانات**؛ SE يصف تشتت **التقدير**. لمضاعفة الدقة (نصف SE) تحتاج 4 أضعاف العينة.",
        example="s = 20، n = 100 ← SE = 2. ولـn = 400 ← SE = 1.")

st.markdown("## فترة الثقة Confidence Interval")
formula(r"\bar{x} \pm t_{1-\alpha/2,\;n-1}\cdot\frac{s}{\sqrt{n}}", title="فترة ثقة للمتوسط",
        symbols={r"t_{1-\alpha/2,\,n-1}": "القيمة الحرجة من توزيع t (≈ 1.96 لعينات كبيرة و95%)"},
        intuition="الإجراء يلتقط μ الحقيقية في 95% من العينات المتكررة. الفترة الواحدة إما تحتويها أو لا.")
st.markdown("### رسم متحرك: تغطية فترات الثقة")
c1, c2 = st.columns(2)
conf = c1.select_slider("مستوى الثقة", [0.80, 0.90, 0.95, 0.99], value=0.95, key="sf_conf")
n_ci = c2.select_slider("حجم كل عينة", [10, 30, 100], value=30, key="sf_nci")
K = 40
rng2 = np.random.default_rng(5)
samples = rng2.normal(50, 12, (K, n_ci))
m = samples.mean(axis=1)
se = samples.std(axis=1, ddof=1) / np.sqrt(n_ci)
h = stats.t.ppf((1 + conf) / 2, n_ci - 1) * se
lo, hi = m - h, m + h
cover = (lo <= 50) & (hi >= 50)


def _ci_frame(k: int) -> None:
    k = k + 1
    fig = go.Figure()
    for i in range(k):
        color = PALETTE["purple"] if cover[i] else PALETTE["coral"]
        fig.add_trace(go.Scatter(x=[lo[i], hi[i]], y=[i, i], mode="lines", line=dict(color=color, width=3), showlegend=False))
        fig.add_trace(go.Scatter(x=[m[i]], y=[i], mode="markers", marker=dict(color=color, size=6), showlegend=False))
    fig.add_vline(x=50, line_dash="dash", line_color=PALETTE["amber"], annotation_text="μ = 50")
    fig.update_layout(xaxis=dict(range=[36, 64]), yaxis=dict(range=[-1, K], title="sample #"), height=460,
                      title=f"{int(cover[:k].sum())} من {k} فترة تحتوي μ ({cover[:k].mean():.0%}) — المستوى الاسمي {conf:.0%}")
    plot(fig)


stepper("ci_cov", K, _ci_frame)
st.caption("الفترات الحمراء فاتتها μ. مع عينات كثيرة تقترب النسبة من مستوى الثقة المختار.")
why("لا تقل «احتمال أن μ في هذه الفترة 95%».",
    "في الإطار التكراري μ ثابتة وليست عشوائية؛ العشوائي هو الفترة. الصحيح: «إجراء بناء الفترة يلتقط μ في 95% من العينات». "
    "(التفسير الاحتمالي المباشر ممكن في الإطار البايزي بفترة Credible interval.)")

st.markdown("## الدلالة الإحصائية وp-value")
st.markdown("**p-value** = احتمال الحصول على نتيجة بهذا التطرف أو أكثر **لو كانت H0 صحيحة وافتراضات الاختبار متحققة**. "
            "ليست احتمال صحة H0، وليست حجم الأثر.")
c1, c2, c3 = st.columns(3)
effect = c1.slider("الأثر الحقيقي (فرق المتوسطات بوحدات SD)", 0.0, 1.0, 0.0, 0.05, key="sf_eff")
n_g = c2.select_slider("n لكل مجموعة", [10, 30, 100, 500], value=30, key="sf_ng")
sims = 2000
rng3 = np.random.default_rng(8)
a = rng3.normal(0, 1, (sims, n_g))
b = rng3.normal(effect, 1, (sims, n_g))
pvals = stats.ttest_ind(b, a, axis=1).pvalue
fig = go.Figure(go.Histogram(x=pvals, xbins=dict(start=0, end=1, size=0.05), marker_color=PALETTE["purple"]))
fig.add_vline(x=0.05, line_color=PALETTE["coral"], line_dash="dash", annotation_text="α = 0.05")
fig.update_layout(title=f"توزيع p-values في {sims} تجربة", height=320, xaxis_title="p-value")
plot(fig)
rej = (pvals < 0.05).mean()
c3.metric("نسبة p < 0.05", f"{rej:.1%}", "= α (إيجابيات كاذبة)" if effect == 0 else "= القوة Power", delta_color="off")
st.markdown("عندما الأثر = 0: p-values موزعة **بانتظام**، و5% منها أقل من 0.05 بالصدفة (خطأ من النوع الأول). "
            "مع أثر حقيقي: نسبة الرفض هي **القوة الإحصائية** وتزيد مع n وحجم الأثر.")

st.markdown("## حجم الأثر Effect size")
formula(r"d = \frac{\bar{x}_1 - \bar{x}_2}{s_p},\qquad s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}",
        title="Cohen's d", symbols={"s_p": "الانحراف المعياري المجمّع"},
        intuition="الفرق بوحدات الانحراف المعياري؛ مستقل عن n. إرشادات Cohen: 0.2 صغير، 0.5 متوسط، 0.8 كبير — لكن المعنى العملي يحدده المجال.")
n_big = st.select_slider("مثال: n لكل مجموعة", [50, 500, 5000, 50000], value=50000, key="sf_big")
rng4 = np.random.default_rng(2)
g1, g2 = rng4.normal(100, 15, n_big), rng4.normal(100.6, 15, n_big)
t = stats.ttest_ind(g2, g1)
d = (g2.mean() - g1.mean()) / np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
c1, c2 = st.columns(2)
c1.metric("p-value", f"{t.pvalue:.2g}")
c2.metric("Cohen's d", f"{d:.3f}")
st.caption("مع n ضخم يصبح فرق تافه (0.6 نقطة على مقياس انحرافه 15) «دالًا جدًا». الدلالة ≠ الأهمية العملية.")

if at_least("advanced"):
    st.markdown("## متقدم: Bootstrap")
    st.markdown("عندما لا توجد صيغة لـSE (الوسيط، نسبة، مقياس معقد): أعد المعاينة من العينة **مع الإرجاع** آلاف المرات "
                "واحسب الإحصاءة في كل مرة؛ انتشار النتائج يقدّر SE وفترة الثقة (Percentile CI).")
    sample = np.random.default_rng(3).lognormal(3, 0.8, 80)
    boots = np.median(sample[np.random.default_rng(4).integers(0, 80, (3000, 80))], axis=1)
    lo_b, hi_b = np.percentile(boots, [2.5, 97.5])
    fig = go.Figure(go.Histogram(x=boots, nbinsx=50, marker_color=PALETTE["amber"]))
    fig.add_vrect(x0=lo_b, x1=hi_b, fillcolor=PALETTE["purple"], opacity=0.15, line_width=0)
    fig.update_layout(title=f"Bootstrap للوسيط: 95% CI ≈ [{lo_b:.1f}, {hi_b:.1f}]", height=300)
    plot(fig)
if at_least("research"):
    researcher_note(["بيان ASA حول p-values (Wasserstein & Lazar, 2016): p-value لا تقيس احتمال صحة الفرضية ولا حجم الأثر.",
                     "أبلغ دائمًا: التقدير، فترة الثقة، حجم الأثر، n، والاختبار المستخدم.",
                     "خطط لحجم العينة مسبقًا بتحليل القوة Power analysis بناءً على أصغر أثر ذي معنى."])
real_world(["هل العينة ممثلة للمجتمع المستهدف؟", "ما حجم الأثر الذي يهم عمليًا؟", "هل n كافٍ لاكتشافه؟",
            "هل عرضت عدم اليقين في التقرير؟"])

page_footer("stat_foundations",
            takeaways=["الإحصاءة تتغير من عينة لأخرى؛ SE يقيس ذلك.", "CLT: متوسطات العينات قريبة من الطبيعي مع n كافٍ.",
                       "فترة الثقة خاصية الإجراء لا الفترة المفردة.", "p-value ليست حجم الأثر ولا احتمال صحة H0."],
            mistakes=["الخلط بين SD وSE.", "تفسير CI كاحتمال لمعلمة ثابتة.", "الاحتفاء بـp صغيرة مع أثر تافه.", "«لا دلالة» = «لا أثر»."])
