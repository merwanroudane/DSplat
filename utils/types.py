"""Column type helpers that behave the same on pandas 2.x and 3.x, plus
semantic type inference (what a column *means*, not only its dtype)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SENTINEL_NUMBERS = (-999, -99, -9, 999, 9999, 99999, -1)
SENTINEL_STRINGS = ("", "na", "n/a", "nan", "null", "none", "unknown", "?", "-", "--", "missing", "not available",
                    "#n/a", "prefer not to say")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_NUMERIC_TEXT = re.compile(r"^\s*[-+]?[$€£]?\s*[\d,]*\.?\d+\s*%?\s*$")
_DATE_TEXT = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def is_textlike(s: pd.Series) -> bool:
    return (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_textlike(df[c]) or pd.api.types.is_bool_dtype(df[c])]


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_datetime(df[c])]


def text_values(s: pd.Series) -> pd.Series:
    """Non-null values as stripped Python strings."""
    return s.dropna().astype(str).str.strip()


def infer_semantic_type(s: pd.Series, name: str = "") -> tuple[str, str]:
    """Return (semantic_type, reason_ar)."""
    n = len(s)
    nonnull = s.dropna()
    lname = name.lower()
    if n == 0 or nonnull.empty:
        return "empty", "العمود فارغ بالكامل."
    nunique = nonnull.nunique()
    if is_datetime(s):
        return "datetime", "نوع البيانات datetime."
    if pd.api.types.is_bool_dtype(s):
        return "binary", "نوع منطقي Boolean."
    if is_numeric(s):
        if nunique == 2:
            return "binary", "عمود رقمي بقيمتين فقط (مثل 0/1)."
        if (lname.endswith("_id") or lname == "id") and nunique == len(nonnull):
            return "identifier", "اسم ينتهي بـ id وكل القيم فريدة."
        is_int = np.allclose(nonnull, np.round(nonnull))
        if is_int and nunique <= 10:
            return "ordinal/discrete", f"أعداد صحيحة بعدد قيم قليل ({nunique}) — قد يكون ترتيبيًا (Likert) أو عدًّا."
        if is_int:
            return "numeric-discrete", "أعداد صحيحة كثيرة القيم (عدّ Count)."
        return "numeric-continuous", "أرقام عشرية كثيرة القيم."
    vals = text_values(s)
    sample = vals.head(300)
    if sample.map(lambda v: bool(_EMAIL.match(v))).mean() > 0.8:
        return "email", "أغلب القيم تطابق نمط البريد الإلكتروني."
    if sample.map(lambda v: bool(_DATE_TEXT.match(v))).mean() > 0.7:
        return "date-as-text", "أغلب القيم تشبه تواريخ لكنها مخزنة نصًا."
    numeric_share = sample.map(lambda v: bool(_NUMERIC_TEXT.match(v))).mean()
    if numeric_share > 0.7:
        return "numeric-as-text", f"{numeric_share:.0%} من القيم أرقام مخزنة كنص (ربما مع رموز أو فواصل)."
    if (lname.endswith("_id") or lname == "id") and nunique >= 0.9 * len(nonnull):
        return "identifier", "معرّف: اسم ينتهي بـ id وقيم شبه فريدة."
    avg_len = sample.str.len().mean()
    if avg_len > 25 and nunique > 0.5 * len(sample):
        return "free-text", f"نص حر: متوسط الطول {avg_len:.0f} حرفًا وقيم متنوعة."
    if nunique == 2:
        return "binary", "فئتان فقط."
    if nunique <= max(20, 0.05 * n):
        return "categorical", f"فئوي: {nunique} فئة."
    return "high-cardinality", f"فئوي عالي التعدد: {nunique} قيمة مختلفة."


def to_number(s: pd.Series) -> pd.Series:
    """Parse numbers stored as text: strips currency, thousands separators, %."""
    if is_numeric(s):
        return s.astype(float)
    cleaned = (s.astype(str).str.strip().str.replace(r"[$€£,%\s]", "", regex=True)
               .replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    return pd.to_numeric(cleaned, errors="coerce")


def semantic_missing_mask(s: pd.Series) -> pd.Series:
    """True where a value *looks* present but means 'missing' (sentinels, placeholders)."""
    if is_numeric(s):
        return s.isin([v for v in SENTINEL_NUMBERS if v in (-999, -99, 9999, 99999)])
    lowered = s.astype(str).str.strip().str.lower()
    mask = lowered.isin(SENTINEL_STRINGS) & s.notna()
    numeric_like = pd.to_numeric(lowered, errors="coerce")
    return mask | numeric_like.isin([-999, -99, 9999, 99999])
