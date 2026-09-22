import pandas as pd

from utils.datasets import customers_clean, customers_raw
from utils.quality import DEFAULT_WEIGHTS, quality_report, weighted_score
from utils.types import infer_semantic_type, semantic_missing_mask, to_number
from utils.validation import CUSTOMER_RULES, Rule, check, run_rules


def test_clean_data_passes_all_rules():
    res = run_rules(customers_clean(), CUSTOMER_RULES)
    assert (res["failed"] == 0).all()


def test_raw_data_fails_expected_rules():
    res = run_rules(customers_raw(), CUSTOMER_RULES).set_index("kind")
    assert res["failed"].sum() > 0


def test_range_rule_evidence():
    df = pd.DataFrame({"age": [20, 250, -3, None, 40]})
    r = check(df, Rule("range", "age", {"min": 18, "max": 100}))
    assert r.failed == 2 and sorted(r.failing_index) == [1, 2]


def test_temporal_rule():
    df = pd.DataFrame({"a": ["2024-01-10", "2024-01-01"], "b": ["2024-01-05", "2024-01-05"]})
    r = check(df, Rule("temporal", "a", {"other": "b", "op": ">="}))
    assert r.failed == 1


def test_quality_score_bounds():
    raw = quality_report(customers_raw(), CUSTOMER_RULES, "customer_id")
    clean = quality_report(customers_clean(), CUSTOMER_RULES, "customer_id")
    assert all(0 <= v <= 1 for v in raw.values())
    assert weighted_score(clean, DEFAULT_WEIGHTS) == 1.0
    assert weighted_score(raw, DEFAULT_WEIGHTS) < 1.0


def test_type_inference():
    assert infer_semantic_type(pd.Series(["$1,200", "300", "45"]), "income")[0] == "numeric-as-text"
    assert infer_semantic_type(pd.Series(["2024-01-01", "2024-02-03"]), "d")[0] == "date-as-text"
    assert infer_semantic_type(pd.Series(["a@b.com", "c@d.org"]), "email")[0] == "email"
    assert infer_semantic_type(pd.Series([0, 1, 1, 0]), "flag")[0] == "binary"


def test_semantic_missing_and_parsing():
    s = pd.Series(["12", "unknown", "-999", "N/A", "$1,000"])
    assert semantic_missing_mask(s).tolist() == [False, True, True, True, False]
    assert to_number(pd.Series(["$1,000", "2.5%"])).tolist() == [1000.0, 2.5]
