"""Cleaning operations used by the labs.

Every function returns a *new* DataFrame plus a log entry; nothing is
modified in place, so any step can be undone by re-running from the source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.types import is_numeric, semantic_missing_mask, text_values, to_number

COUNTRY_SYNONYMS = {
    "usa": "United States", "us": "United States", "u.s.a": "United States", "united states": "United States",
    "morocco": "Morocco", "maroc": "Morocco", "ksa": "Saudi Arabia", "saudi arabia": "Saudi Arabia",
    "egypt": "Egypt",
}
GENDER_SYNONYMS = {"f": "Female", "female": "Female", "m": "Male", "male": "Male"}


def canonical_labels(s: pd.Series, synonyms: dict[str, str] | None = None) -> pd.Series:
    """strip → lookup synonyms → otherwise most frequent spelling within a
    case/punctuation-insensitive group."""
    out = s.copy()
    mask = s.notna()
    stripped = s[mask].astype(str).str.strip()
    key = stripped.str.lower().str.replace(r"[^a-z0-9؀-ۿ ]", "", regex=True).str.strip()
    canon = stripped.groupby(key).agg(lambda v: v.value_counts().index[0])
    mapped = key.map(canon)
    if synonyms:
        syn = stripped.str.lower().map(synonyms)
        syn2 = key.map({k.replace(".", ""): v for k, v in synonyms.items()})
        mapped = syn.fillna(syn2).fillna(mapped)
    out = out.astype(object)
    out.loc[mask] = mapped.to_numpy()
    return out


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Parse ISO, dd/mm/yyyy and yyyy/mm/dd explicitly; anything else → NaT."""
    txt = s.astype(str).str.strip()
    iso = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(txt, format="%d/%m/%Y", errors="coerce")
    ymd = pd.to_datetime(txt, format="%Y/%m/%d", errors="coerce")
    return iso.fillna(dmy).fillna(ymd)


# ------------------------------------------------ customer challenge steps
# Each step: (key, Arabic label, why, function(df) -> (df, affected), code snippet)
def _s_dups(df):
    n = len(df)
    df = df.drop_duplicates()
    return df, n - len(df)


def _s_key(df):
    n = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    return df, n - len(df)


def _s_numeric(df):
    df = df.copy()
    before = int(df[["age", "annual_income", "satisfaction"]].isna().sum().sum())
    for c in ["age", "annual_income", "satisfaction"]:
        df[c] = to_number(df[c].where(~semantic_missing_mask(df[c]), np.nan))
    return df, int(df[["age", "annual_income", "satisfaction"]].isna().sum().sum()) - before


def _s_ranges(df):
    df = df.copy()
    age = pd.to_numeric(df["age"], errors="coerce")
    sat = pd.to_numeric(df["satisfaction"], errors="coerce")
    orders = pd.to_numeric(df["num_orders"], errors="coerce")
    bad_age = ~age.between(18, 100) & age.notna()
    bad_sat = ~sat.between(1, 5) & sat.notna()
    bad_ord = orders < 0
    df["age"] = age.where(~bad_age)
    df["satisfaction"] = sat.where(~bad_sat)
    df["num_orders"] = orders.where(~bad_ord)
    return df, int(bad_age.sum() + bad_sat.sum() + bad_ord.sum())


def _s_units(df):
    df = df.copy()
    inc = pd.to_numeric(df["annual_income"], errors="coerce")
    thousands = inc.between(1, 1000)
    inc = inc.where(~thousands, inc * 1000)
    bad = (inc == 0) | (inc >= 9_000_000)
    df["annual_income"] = inc.where(~bad)
    h = df["height_cm"]
    metres = h < 3
    df["height_cm"] = h.where(~metres, h * 100)
    return df, int(thousands.sum() + bad.sum() + metres.sum())


def _s_labels(df):
    df = df.copy()
    before = sum(df[c].nunique() for c in ["gender", "country", "membership"])
    df["gender"] = canonical_labels(df["gender"], GENDER_SYNONYMS)
    df["country"] = canonical_labels(df["country"], COUNTRY_SYNONYMS)
    df["membership"] = df["membership"].astype(str).str.strip().str.title()
    return df, int(before - sum(df[c].nunique() for c in ["gender", "country", "membership"]))


def _s_dates(df):
    df = df.copy()
    df["signup_date"] = parse_mixed_dates(df["signup_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    bad = df["last_purchase_date"] < df["signup_date"]
    df.loc[bad, "last_purchase_date"] = pd.NaT
    return df, int(df["signup_date"].isna().sum() + bad.sum())


def _s_email(df):
    df = df.copy()
    ok = df["email"].astype(str).str.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
    df["email_valid"] = ok
    return df, int((~ok).sum())


CUSTOMER_STEPS = [
    ("dups", "حذف التكرار التام", "نسخ مطابقة لنفس السجل تضخم الأوزان.", _s_dups,
     "df = df.drop_duplicates()"),
    ("key", "حذف تكرار customer_id (الإبقاء على الأول)",
     "نفس العميل بقيم متعارضة؛ في مشروع حقيقي نرجع للمصدر لمعرفة السجل الأحدث.", _s_key,
     'df = df.drop_duplicates(subset="customer_id", keep="first")'),
    ("numeric", "تحويل age/annual_income/satisfaction إلى أرقام مع Sentinels ← NaN",
     "النصوص مثل unknown/N/A/?/-999 تعني «غير معروف»، والرموز $ و, تمنع التحويل.", _s_numeric,
     'for c in ["age", "annual_income", "satisfaction"]:\n'
     '    s = df[c].replace(["unknown", "N/A", "?", "-999", -999], np.nan)\n'
     '    df[c] = pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")'),
    ("ranges", "القيم المستحيلة ← NaN (age ∉ [18,100]، satisfaction ∉ [1,5]، orders < 0)",
     "قواعد مجال: لا عمر 250، ولا رضا 9 في مقياس 1–5، ولا طلبات سالبة.", _s_ranges,
     'df.loc[~df["age"].between(18, 100), "age"] = np.nan\n'
     'df.loc[~df["satisfaction"].between(1, 5), "satisfaction"] = np.nan\n'
     'df.loc[df["num_orders"] < 0, "num_orders"] = np.nan'),
    ("units", "تصحيح الوحدات (دخل بالآلاف، طول بالمتر) والـSentinels الرقمية",
     "قيم الدخل بين 1 و1000 مسجلة بالآلاف؛ الطول < 3 بالمتر؛ 0 و9,999,999 رموز لا قيم.", _s_units,
     'inc = df["annual_income"]\n'
     'df.loc[inc.between(1, 1000), "annual_income"] = inc * 1000\n'
     'df.loc[(inc == 0) | (inc >= 9_000_000), "annual_income"] = np.nan\n'
     'df.loc[df["height_cm"] < 3, "height_cm"] *= 100'),
    ("labels", "توحيد التسميات (gender, country, membership)",
     "قاموس توحيد موثق + الصيغة الأكثر تكرارًا داخل كل مجموعة متشابهة.", _s_labels,
     'df["country"] = df["country"].str.strip().str.lower().map(country_mapping)\n'
     'df["gender"] = df["gender"].str.strip().str.lower().map({"f": "Female", "female": "Female", "m": "Male", "male": "Male"})\n'
     'df["membership"] = df["membership"].str.strip().str.title()'),
    ("dates", "تحويل التواريخ بصيغ صريحة + تصحيح الترتيب الزمني المستحيل",
     "«01/05/2023» غامضة بدون صيغة؛ آخر شراء قبل التسجيل مستحيل.", _s_dates,
     'iso = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="coerce")\n'
     'dmy = pd.to_datetime(df["signup_date"], format="%d/%m/%Y", errors="coerce")\n'
     'ymd = pd.to_datetime(df["signup_date"], format="%Y/%m/%d", errors="coerce")\n'
     'df["signup_date"] = iso.fillna(dmy).fillna(ymd)\n'
     'df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])\n'
     'df.loc[df["last_purchase_date"] < df["signup_date"], "last_purchase_date"] = pd.NaT'),
    ("email", "إضافة مؤشر email_valid (دون حذف)", "البريد غير الصالح لا يبرر حذف العميل؛ نضع علامة للمتابعة.", _s_email,
     r'df["email_valid"] = df["email"].str.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")'),
]


def apply_customer_steps(raw: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    df = raw.copy()
    log: list[dict] = []
    for key, label, why, fn, _code in CUSTOMER_STEPS:
        if key in keys:
            df, n = fn(df)
            log.append({"الخطوة": label, "صفوف/قيم متأثرة": int(n), "لماذا؟": why})
    return df.reset_index(drop=True), log


def clean_customers_reference(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """The documented reference cleaning of the customer challenge dataset (all steps)."""
    df, log = apply_customer_steps(raw, [s[0] for s in CUSTOMER_STEPS])
    for c in ("num_orders", "age", "satisfaction"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    return df, log


# -------------------------------------------------------- hybrid-lab actions
def apply_action(df: pd.DataFrame, column: str, action: str) -> tuple[pd.DataFrame, int]:
    """Apply one issue action; returns (new_df, number of affected cells/rows)."""
    out = df.copy()
    if action == "drop_exact_duplicates":
        n = len(out)
        out = out.drop_duplicates()
        return out, n - len(out)
    if action == "dedupe_key":
        n = len(out)
        out = out.drop_duplicates(subset=column, keep="first")
        return out, n - len(out)
    if column not in out.columns:
        return out, 0
    s = out[column]
    if action == "sentinel_to_nan":
        m = semantic_missing_mask(s)
        out[column] = s.where(~m, np.nan)
        return out, int(m.sum())
    if action == "coerce_numeric":
        before = s.isna().sum()
        out[column] = to_number(s.where(~semantic_missing_mask(s), np.nan))
        return out, int(out[column].isna().sum() - before)
    if action == "parse_dates":
        out[column] = parse_mixed_dates(s)
        return out, int(out[column].notna().sum())
    if action == "strip_whitespace":
        changed = (s.dropna().astype(str) != s.dropna().astype(str).str.strip()).sum()
        out[column] = s.where(s.isna(), s.astype(str).str.strip())
        return out, int(changed)
    if action == "normalize_case":
        syn = COUNTRY_SYNONYMS if "country" in column.lower() else GENDER_SYNONYMS if "gender" in column.lower() else None
        new = canonical_labels(s, syn)
        return out.assign(**{column: new}), int((new.astype(str) != s.astype(str)).sum())
    if action in ("negative_to_nan", "inf_to_nan"):
        x = to_number(s)
        m = (x < 0) if action == "negative_to_nan" else np.isinf(x)
        out[column] = x.where(~m, np.nan)
        return out, int(m.sum())
    if action == "cap_iqr":
        x = to_number(s)
        q1, q3 = x.quantile([0.25, 0.75])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        m = (x < lo) | (x > hi)
        out[column] = x.clip(lo, hi)
        return out, int(m.sum())
    if action == "log_transform":
        x = to_number(s)
        out[f"log1p_{column}"] = np.log1p(x.clip(lower=0))
        return out, int(x.notna().sum())
    if action == "impute_median":
        x = to_number(s) if not is_numeric(s) else s
        m = x.isna()
        out[f"{column}_was_missing"] = m.astype(int)
        out[column] = x.fillna(x.median())
        return out, int(m.sum())
    if action == "impute_missing_label":
        m = s.isna()
        out[column] = s.astype(object).where(~m, "Missing")
        return out, int(m.sum())
    if action == "group_rare":
        vals = text_values(s)
        freq = vals.value_counts(normalize=True)
        rare = set(freq[freq < 0.01].index)
        m = s.astype(str).str.strip().isin(rare)
        out[column] = s.astype(object).where(~m, "Other")
        return out, int(m.sum())
    if action == "drop_column":
        return out.drop(columns=column), len(out)
    return out, 0


def compare_to_truth(cleaned: pd.DataFrame, truth: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Accuracy of a cleaned dataset against the ground truth, column by column."""
    merged = cleaned.merge(truth, on=key, suffixes=("_cleaned", "_truth"))
    rows = []
    for c in cols:
        a, b = merged[f"{c}_cleaned"], merged[f"{c}_truth"]
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            a = pd.to_numeric(a, errors="coerce")
            ok = (a - b).abs() <= 0.01 * np.maximum(1, b.abs())  # within 1% (source rounding)
        elif pd.api.types.is_datetime64_any_dtype(b):
            ok = pd.to_datetime(a, errors="coerce") == b
        else:
            ok = a.astype(str).str.strip() == b.astype(str)
        ok = pd.Series(ok).fillna(False).astype(bool)  # a missing value never counts as a match
        rows.append({"column": c, "matches_truth_%": round(100 * ok.mean(), 2),
                     "missing_after_cleaning": int(a.isna().sum()), "rows_compared": len(merged)})
    return pd.DataFrame(rows)
