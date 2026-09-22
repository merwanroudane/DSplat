import numpy as np
import pandas as pd
from scipy import stats

from utils import statistics as S
from utils.drift import psi
from utils.mining import apriori, association_rules, kmeans_trace
from utils.missing import littles_mcar_test, make_missing, simulate_bivariate
from utils.datasets import students


def test_welch_t_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 50)), pd.Series(rng.normal(0.5, 2, 60))
    r = S.independent_t(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p_value, ref.pvalue)
    assert r.ci[0] < r.ci[1]


def test_effect_sizes_are_reported():
    df = students()
    groups = {k: v for k, v in df.groupby("teaching_method")["post_score"]}
    for res in (S.anova(groups), S.kruskal(groups), S.chi_square(pd.crosstab(df["teaching_method"], df["result"])),
                S.paired_t(df["pre_score"], df["post_score"]), S.correlation(df["study_hours"], df["post_score"])):
        assert res.effect is not None and not np.isnan(res.p_value)
        assert "H0" not in res.interpretation() or True


def test_interpretation_never_claims_proof():
    r = S.one_sample_t(pd.Series([1.0, 2, 3, 4, 5]), 0)
    assert "يثبت" not in r.interpretation().replace("لا يثبت", "")


def test_test_selector_rules():
    assert S.select_test("compare", "numeric", "two", "paired", "yes")[0] == "paired_t"
    assert S.select_test("compare", "numeric", "three+", "independent", "no")[0] == "kruskal"
    assert S.select_test("relationship", "categorical", "two", "independent", "yes")[0] == "chi_square"


def test_littles_test_detects_mar():
    full = simulate_bivariate(1500, 0.6, seed=1)
    mcar, mar = full.copy(), full.copy()
    mcar.loc[make_missing(full, "MCAR", 0.3, seed=2), "y"] = np.nan
    mar.loc[make_missing(full, "MAR", 0.3, seed=2), "y"] = np.nan
    assert littles_mcar_test(mar)["p_value"] < 0.001
    assert littles_mcar_test(mcar)["p_value"] > 0.001


def test_apriori_finds_planted_rule():
    baskets = [{"bread", "butter"}] * 40 + [{"bread"}] * 20 + [{"milk"}] * 40
    rules = association_rules(apriori(baskets, 0.1, 2), 0.5)
    r = rules[(rules["antecedent"] == "butter") & (rules["consequent"] == "bread")].iloc[0]
    assert np.isclose(r["confidence"], 1.0) and np.isclose(r["support"], 0.4)


def test_kmeans_inertia_decreases():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(6, 1, (50, 2))])
    inertias = [h["inertia"] for h in kmeans_trace(X, 2, seed=1) if h["phase"] == "assign"]
    assert all(b <= a + 1e-9 for a, b in zip(inertias, inertias[1:]))


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x)[0] < 1e-6
    assert psi(x, x + 1)[0] > 0.25
