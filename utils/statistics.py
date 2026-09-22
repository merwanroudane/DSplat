"""Statistical tests that always report H0, H1, statistic, p-value, a
confidence interval (where defined), an effect size and cautious Arabic
interpretation. Never reduces the analysis to "p < 0.05"."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    name: str
    h0: str
    h1: str
    statistic_name: str
    statistic: float
    p_value: float
    df: str = ""
    ci: tuple[float, float] | None = None
    ci_label: str = ""
    effect_name: str = ""
    effect: float | None = None
    effect_label: str = ""
    assumptions: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def interpretation(self, alpha: float = 0.05) -> str:
        if np.isnan(self.p_value):
            return "تعذّر حساب p-value (عينة صغيرة أو تباين صفري)."
        if self.p_value < alpha:
            core = (f"p = {self.p_value:.4g} < α = {alpha}: البيانات **غير متوافقة بدرجة كافية** مع فرضية العدم "
                    f"تحت افتراضات الاختبار، مما يوفر دليلًا إحصائيًا لصالح H1.")
        else:
            core = (f"p = {self.p_value:.4g} ≥ α = {alpha}: لا يوجد دليل كافٍ لرفض H0. "
                    "هذا **لا يثبت** صحة H0؛ قد تكون العينة صغيرة أو الأثر ضعيفًا.")
        if self.effect is not None and not np.isnan(self.effect):
            core += f" حجم الأثر {self.effect_name} = {self.effect:.3f} ({self.effect_label})."
        if self.ci:
            core += f" {self.ci_label}: [{self.ci[0]:.3f}, {self.ci[1]:.3f}]."
        return core


def _d_label(d: float) -> str:
    a = abs(d)
    return "ضئيل" if a < 0.2 else "صغير" if a < 0.5 else "متوسط" if a < 0.8 else "كبير"


def _r_label(r: float) -> str:
    a = abs(r)
    return "ضئيل" if a < 0.1 else "صغير" if a < 0.3 else "متوسط" if a < 0.5 else "كبير"


def _eta_label(e: float) -> str:
    return "ضئيل" if e < 0.01 else "صغير" if e < 0.06 else "متوسط" if e < 0.14 else "كبير"


def _mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    se = stats.sem(x)
    h = se * stats.t.ppf((1 + conf) / 2, len(x) - 1)
    return float(np.mean(x) - h), float(np.mean(x) + h)


def one_sample_t(x: pd.Series, mu0: float) -> TestResult:
    x = x.dropna().to_numpy()
    r = stats.ttest_1samp(x, mu0)
    d = (x.mean() - mu0) / x.std(ddof=1)
    return TestResult("One-sample t-test", f"μ = {mu0}", f"μ ≠ {mu0}", "t", float(r.statistic), float(r.pvalue),
                      f"{len(x) - 1}", _mean_ci(x), "فترة ثقة 95% للمتوسط", "Cohen's d", float(d), _d_label(d),
                      ["المشاهدات مستقلة", "التوزيع قريب من الطبيعي أو n كبير (CLT)"])


def independent_t(a: pd.Series, b: pd.Series, welch: bool = True) -> TestResult:
    a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
    r = stats.ttest_ind(a, b, equal_var=not welch)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp
    ci = r.confidence_interval(0.95)
    return TestResult("Welch's t-test" if welch else "Student's t-test", "μ₁ = μ₂", "μ₁ ≠ μ₂", "t",
                      float(r.statistic), float(r.pvalue), f"{r.df:.1f}", (float(ci.low), float(ci.high)),
                      "فترة ثقة 95% للفرق μ₁ − μ₂", "Cohen's d", float(d), _d_label(d),
                      ["عينتان مستقلتان", "قرب من الطبيعية أو n كبير",
                       "Welch لا يفترض تساوي التباين (الخيار الافتراضي الموصى به)"])


def paired_t(before: pd.Series, after: pd.Series) -> TestResult:
    d = (after - before).dropna().to_numpy()
    r = stats.ttest_1samp(d, 0.0)
    dz = d.mean() / d.std(ddof=1)
    return TestResult("Paired t-test", "μ_d = 0", "μ_d ≠ 0", "t", float(r.statistic), float(r.pvalue),
                      f"{len(d) - 1}", _mean_ci(d), "فترة ثقة 95% لمتوسط الفرق", "Cohen's d_z", float(dz),
                      _d_label(dz), ["أزواج مترابطة (نفس الوحدة مرتين)", "الفروق قريبة من الطبيعية"])


def anova(groups: dict[str, pd.Series]) -> TestResult:
    arrays = [g.dropna().to_numpy() for g in groups.values()]
    r = stats.f_oneway(*arrays)
    allx = np.concatenate(arrays)
    grand = allx.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in arrays)
    ss_total = ((allx - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    k, n = len(arrays), len(allx)
    return TestResult("One-way ANOVA", "جميع متوسطات المجموعات متساوية", "متوسط واحد على الأقل مختلف", "F",
                      float(r.statistic), float(r.pvalue), f"{k - 1}, {n - k}", None, "", "η²", float(eta2),
                      _eta_label(eta2), ["استقلال المشاهدات", "طبيعية البواقي", "تجانس التباين (Levene)",
                                         "عند الرفض: اختبارات لاحقة Post-hoc مثل Tukey HSD"])


def chi_square(table: pd.DataFrame) -> TestResult:
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    low = float((expected < 5).mean())
    return TestResult("Chi-square test of independence", "المتغيران مستقلان", "يوجد ارتباط بين المتغيرين", "χ²",
                      float(chi2), float(p), f"{dof}", None, "", "Cramér's V", float(v), _r_label(v),
                      ["مشاهدات مستقلة", f"التكرارات المتوقعة ≥ 5 (نسبة الخلايا الأقل من 5: {low:.0%})"],
                      {"expected": pd.DataFrame(expected, index=table.index, columns=table.columns).round(1)})


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def correlation(x: pd.Series, y: pd.Series, method: str = "pearson") -> TestResult:
    d = pd.concat([x, y], axis=1).dropna()
    a, b = d.iloc[:, 0], d.iloc[:, 1]
    if method == "pearson":
        r, p = stats.pearsonr(a, b)
        name, assumptions = "Pearson correlation", ["علاقة خطية", "لا قيم متطرفة مؤثرة", "قرب من الطبيعية الثنائية"]
    else:
        r, p = stats.spearmanr(a, b)
        name, assumptions = "Spearman rank correlation", ["علاقة رتيبة Monotonic", "متين أمام القيم المتطرفة"]
    return TestResult(name, "ρ = 0", "ρ ≠ 0", "r", float(r), float(p), f"{len(d) - 2}",
                      _fisher_ci(float(r), len(d)), "فترة ثقة 95% تقريبية (Fisher z)", "r", float(r),
                      _r_label(float(r)), assumptions + ["الارتباط لا يعني السببية"])


def mann_whitney(a: pd.Series, b: pd.Series) -> TestResult:
    a, b = a.dropna(), b.dropna()
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 1 - 2 * r.statistic / (len(a) * len(b))
    return TestResult("Mann–Whitney U", "توزيعا المجموعتين متطابقان", "إحدى المجموعتين تميل لقيم أكبر", "U",
                      float(r.statistic), float(r.pvalue), "", None, "", "rank-biserial r", float(-rb),
                      _r_label(rb), ["عينتان مستقلتان", "بديل لا معلمي لـt المستقل", "لا يفترض الطبيعية"])


def wilcoxon(before: pd.Series, after: pd.Series) -> TestResult:
    d = (after - before).dropna()
    d = d[d != 0]
    r = stats.wilcoxon(d)
    n = len(d)
    ranks = stats.rankdata(np.abs(d))
    r_pos, r_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    rbc = (r_pos - r_neg) / (r_pos + r_neg)
    return TestResult("Wilcoxon signed-rank", "وسيط الفروق = 0", "وسيط الفروق ≠ 0", "W", float(r.statistic),
                      float(r.pvalue), f"n={n}", None, "", "matched-pairs rank-biserial r", float(rbc),
                      _r_label(rbc), ["أزواج مترابطة", "بديل لا معلمي لـPaired t", "توزيع الفروق متماثل تقريبًا"])


def kruskal(groups: dict[str, pd.Series]) -> TestResult:
    arrays = [g.dropna() for g in groups.values()]
    r = stats.kruskal(*arrays)
    n = sum(len(g) for g in arrays)
    eps2 = r.statistic * (n + 1) / (n ** 2 - 1)
    return TestResult("Kruskal–Wallis H", "توزيعات المجموعات متطابقة", "مجموعة واحدة على الأقل مختلفة", "H",
                      float(r.statistic), float(r.pvalue), f"{len(arrays) - 1}", None, "", "ε²", float(eps2),
                      _eta_label(eps2), ["عينات مستقلة", "بديل لا معلمي لـANOVA", "عند الرفض: Dunn post-hoc"])


def shapiro(x: pd.Series) -> TestResult:
    x = x.dropna()
    if len(x) > 5000:
        x = x.sample(5000, random_state=0)
    r = stats.shapiro(x)
    return TestResult("Shapiro–Wilk normality", "البيانات من توزيع طبيعي", "البيانات ليست من توزيع طبيعي", "W",
                      float(r.statistic), float(r.pvalue), f"n={len(x)}", None, "", "skewness",
                      float(stats.skew(x)), "التواء", ["مع n كبير يرفض لأي انحراف تافه — انظر Q-Q plot",
                                                        "مع n صغير قوته ضعيفة"])


def levene(groups: dict[str, pd.Series]) -> TestResult:
    arrays = [g.dropna() for g in groups.values()]
    r = stats.levene(*arrays, center="median")
    ratio = max(a.var() for a in arrays) / min(a.var() for a in arrays)
    return TestResult("Levene (Brown–Forsythe)", "التباينات متساوية", "تباين واحد على الأقل مختلف", "W",
                      float(r.statistic), float(r.pvalue), f"{len(arrays) - 1}", None, "", "max/min variance ratio",
                      float(ratio), "نسبة التباين", ["متين نسبيًا أمام عدم الطبيعية (center=median)"])


# ------------------------------------------------------------ test selector
def select_test(goal: str, outcome: str, groups: str, design: str, normal: str) -> tuple[str, str]:
    """Rule-based test recommendation: returns (test key, Arabic reasoning)."""
    if goal == "relationship":
        if outcome == "categorical":
            return "chi_square", "متغيران فئويان ← جدول توافق واختبار Chi-square (أو Fisher عند التكرارات الصغيرة)."
        if normal == "yes":
            return "pearson", "متغيران رقميان بعلاقة خطية وتوزيع قريب من الطبيعي ← Pearson."
        return "spearman", "متغيران رقميان/ترتيبيان أو علاقة رتيبة غير خطية أو قيم متطرفة ← Spearman."
    if goal == "normality":
        return "shapiro", "فحص الطبيعية ← Shapiro–Wilk مع Q-Q plot (لا تعتمد على الاختبار وحده)."
    if goal == "variance":
        return "levene", "مقارنة التباينات ← Levene/Brown–Forsythe."
    if outcome == "categorical":
        return "chi_square", "النتيجة فئوية والمقارنة بين مجموعات ← Chi-square على جدول التوافق."
    if groups == "one":
        return "one_sample_t", "مقارنة متوسط عينة واحدة بقيمة مرجعية ← One-sample t-test."
    if groups == "two":
        if design == "paired":
            return ("paired_t", "نفس الوحدات مرتين والفروق قريبة من الطبيعية ← Paired t-test.") if normal == "yes" \
                else ("wilcoxon", "قياسات مزدوجة بفروق غير طبيعية أو ترتيبية ← Wilcoxon signed-rank.")
        return ("independent_t", "مجموعتان مستقلتان وقرب من الطبيعية ← Welch t-test (لا يفترض تساوي التباين).") \
            if normal == "yes" else ("mann_whitney", "مجموعتان مستقلتان ببيانات ملتوية/ترتيبية ← Mann–Whitney U.")
    return ("anova", "ثلاث مجموعات أو أكثر وقرب من الطبيعية وتجانس التباين ← One-way ANOVA.") if normal == "yes" \
        else ("kruskal", "ثلاث مجموعات أو أكثر ببيانات غير طبيعية/ترتيبية ← Kruskal–Wallis.")


TEST_NAMES = {"one_sample_t": "One-sample t-test", "independent_t": "Independent (Welch) t-test",
              "paired_t": "Paired t-test", "anova": "One-way ANOVA", "chi_square": "Chi-square",
              "pearson": "Pearson correlation", "spearman": "Spearman correlation", "mann_whitney": "Mann–Whitney U",
              "wilcoxon": "Wilcoxon signed-rank", "kruskal": "Kruskal–Wallis", "shapiro": "Shapiro–Wilk",
              "levene": "Levene"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / sp)


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors: VIF_j = 1 / (1 - R²_j)."""
    from sklearn.linear_model import LinearRegression
    X = df.dropna()
    rows = []
    for c in X.columns:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c]) if others.shape[1] else 0.0
        rows.append({"variable": c, "R2_j": round(r2, 3), "VIF": round(1 / (1 - r2), 2) if r2 < 1 else np.inf})
    return pd.DataFrame(rows)
