import numpy as np
import pandas as pd
import pytest

from utils.cleaning import CUSTOMER_STEPS, apply_action, apply_customer_steps, canonical_labels, clean_customers_reference, compare_to_truth
from utils.datasets import customers_clean, customers_raw
from utils.missing import IMPUTERS, impute, make_missing, simulate_bivariate
from utils.outliers import iqr_mask, modified_z_mask, outlier_decision
from utils.pipeline import PipelineConfig, generate_pandas_code, generate_sklearn_code, run_pipeline
from utils.profiling import detect_issues


def test_reference_cleaning_improves_match_with_truth():
    raw = customers_raw()
    cleaned, log = clean_customers_reference(raw)
    cols = ["gender", "country", "annual_income", "height_cm"]
    before = compare_to_truth(raw.drop_duplicates("customer_id"), customers_clean(), "customer_id", cols)
    after = compare_to_truth(cleaned, customers_clean(), "customer_id", cols)
    assert (after["matches_truth_%"].to_numpy() >= before["matches_truth_%"].to_numpy()).all()
    assert cleaned["customer_id"].is_unique
    assert len(log) == len(CUSTOMER_STEPS)


def test_any_subset_of_steps_runs():
    keys = [s[0] for s in CUSTOMER_STEPS]
    for i in range(len(keys) + 1):
        apply_customer_steps(customers_raw(), keys[:i])
        apply_customer_steps(customers_raw(), keys[i:])


def test_duplicate_detection_and_actions():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "v": [5, 5, 7, 8]})
    out, n = apply_action(df, "(all)", "drop_exact_duplicates")
    assert n == 1 and len(out) == 3
    out, n = apply_action(df, "id", "dedupe_key")
    assert n == 2 and out["id"].is_unique


def test_label_standardization():
    s = pd.Series(["USA", "usa", " U.S.A", "United States", "Egypt", "egypt "])
    out = canonical_labels(s, {"usa": "United States", "u.s.a": "United States", "united states": "United States"})
    assert set(out) == {"United States", "Egypt"}


def test_issue_detector_never_mutates():
    raw = customers_raw()
    snapshot = raw.copy()
    issues = detect_issues(raw)
    pd.testing.assert_frame_equal(raw, snapshot)
    kinds = {i.issue for i in issues}
    assert any("Wrong type" in k for k in kinds)
    assert any("Sentinel" in k for k in kinds)


@pytest.mark.parametrize("method", [m for m in IMPUTERS if m != "listwise"])
def test_imputers_fill_values(method):
    full = simulate_bivariate(300, 0.6, seed=1)
    obs = full.copy()
    obs.loc[make_missing(full, "MCAR", 0.2, seed=2), "y"] = np.nan
    out = impute(obs, method, cols=["y"])
    assert out["y"].isna().sum() == 0


def test_mean_imputation_shrinks_variance():
    full = simulate_bivariate(1000, 0.6, seed=1)
    obs = full.copy()
    obs.loc[make_missing(full, "MCAR", 0.4, seed=2), "y"] = np.nan
    assert impute(obs, "mean", cols=["y"])["y"].std() < full["y"].std()


def test_outlier_rules():
    x = pd.Series([10, 11, 12, 11, 10, 13, 12, 11, 95])
    assert iqr_mask(x).iloc[-1] and modified_z_mask(x).iloc[-1]
    assert not iqr_mask(x).iloc[:-1].any()
    action, _ = outlier_decision({"unit": "نعم"})
    assert action.startswith("Correct")


def test_pipeline_runs_and_generates_code():
    df = customers_clean()
    cfg = PipelineConfig(exclude=("customer_id", "churned", "email"))
    out, log = run_pipeline(df, cfg)
    assert len(log) == 6 and out.select_dtypes("number").shape[1] > 0
    code = generate_pandas_code(cfg, "customers", ["age"], ["country"]) + generate_sklearn_code(cfg, ["age"], ["country"], "churned")
    compile(code, "<generated>", "exec")  # generated code must be valid Python
