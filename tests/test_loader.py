import json

import pandas as pd

from utils.data_loader import load_bytes
from utils.datasets import DATASET_INFO, load_dataset


def test_all_builtin_datasets_load():
    for name in DATASET_INFO:
        df = load_dataset(name)
        assert isinstance(df, pd.DataFrame) and len(df) > 50, name


def test_datasets_are_deterministic_and_copies():
    a, b = load_dataset("customers_raw"), load_dataset("customers_raw")
    pd.testing.assert_frame_equal(a, b)
    a.loc[0, "age"] = "changed"
    assert load_dataset("customers_raw").loc[0, "age"] != "changed"


def test_csv_semicolon_and_arabic_encoding():
    text = "id;مدينة;value\n1;الرباط;10\n2;القاهرة;20\n"
    res = load_bytes(text.encode("cp1256"), "x.csv")
    assert res.ok
    assert res.df.shape == (2, 3)
    assert res.info["delimiter"] == ";"
    assert "الرباط" in res.df["مدينة"].tolist()


def test_malformed_rows_are_counted_not_hidden():
    text = "a,b\n1,2\n3,4,5,6\n7,8\n"
    res = load_bytes(text.encode("utf-8"), "x.csv")
    assert res.ok and len(res.df) == 2
    assert any("معطوب" in w for w in res.warnings)


def test_rejects_bad_extension_and_empty_file():
    assert not load_bytes(b"abc", "x.exe").ok
    assert not load_bytes(b"   ", "x.csv").ok


def test_duplicate_headers_warning():
    res = load_bytes(b"a,a,b\n1,2,3\n", "x.csv")
    assert any("مكررة" in w for w in res.warnings)


def test_nested_json_is_flattened():
    data = [{"id": 1, "c": {"city": "Rabat"}}, {"id": 2, "c": {"city": "Cairo"}}]
    res = load_bytes(json.dumps(data).encode(), "x.json")
    assert res.ok and "c.city" in res.df.columns


def test_json_list_columns_become_text():
    from utils.profiling import detect_issues, overview
    data = [{"id": 1, "tags": ["promo"]}, {"id": 2, "tags": []}]
    res = load_bytes(json.dumps(data).encode(), "x.json")
    assert res.ok and isinstance(res.df.loc[0, "tags"], str)
    overview(res.df)
    detect_issues(res.df)
