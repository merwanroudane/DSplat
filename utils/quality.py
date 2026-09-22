"""Transparent Data Quality Score.

Score = Σ wᵢ·sᵢ / Σ wᵢ, where each sub-score sᵢ ∈ [0, 1] has an explicit
formula (documented in DIMENSIONS) and the weights wᵢ are user-editable.
"""

from __future__ import annotations

import pandas as pd

from utils.types import categorical_columns, infer_semantic_type, semantic_missing_mask, text_values
from utils.validation import Rule, check

DIMENSIONS: dict[str, dict[str, str]] = {
    "completeness": {"ar": "الاكتمال", "formula": r"1 - \frac{\text{missing cells (incl. sentinels)}}{\text{all cells}}",
                     "desc": "نسبة الخلايا غير المفقودة، مع احتساب القيم الدلالية (-999, unknown) كمفقودة."},
    "uniqueness": {"ar": "التفرّد", "formula": r"1 - \frac{\text{duplicate rows} + \text{duplicate keys}}{2n}",
                   "desc": "متوسط غياب الصفوف المكررة وغياب تكرار المفتاح الأساسي."},
    "validity": {"ar": "الصلاحية", "formula": r"\frac{1}{R}\sum_{r=1}^{R} \text{pass rate}_r",
                 "desc": "متوسط نسب النجاح في قواعد النوع والمدى والصيغة (Type/Range/Regex)."},
    "consistency": {"ar": "الاتساق", "formula": r"1 - \frac{\text{label variants}}{\text{distinct labels}}",
                    "desc": "نسبة الفئات غير المكررة بصيغ مختلفة (USA/usa/U.S.A) في الأعمدة الفئوية."},
    "conformity": {"ar": "المطابقة للصيغة", "formula": r"\frac{\text{values in expected format}}{\text{values}}",
                   "desc": "نسبة القيم المطابقة لصيغ متوقعة (تواريخ ISO، بريد صحيح)."},
    "integrity": {"ar": "السلامة المنطقية", "formula": r"\frac{1}{C}\sum_{c} \text{pass rate}_c",
                  "desc": "متوسط نجاح قواعد Cross-field/Temporal (مثل: آخر شراء بعد التسجيل)."},
}

DEFAULT_WEIGHTS = {"completeness": 3, "uniqueness": 2, "validity": 3, "consistency": 2, "conformity": 1,
                   "integrity": 2}


def completeness(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    missing = df.isna().sum().sum() + sum(int(semantic_missing_mask(df[c]).sum()) for c in df.columns)
    return float(1 - missing / df.size)


def uniqueness(df: pd.DataFrame, key: str | None) -> float:
    n = max(1, len(df))
    dup_rows = df.duplicated().sum() / n
    dup_key = df[key].dropna().duplicated().sum() / n if key and key in df.columns else dup_rows
    return float(1 - (dup_rows + dup_key) / 2)


def consistency(df: pd.DataFrame) -> float:
    total, variants = 0, 0
    for c in categorical_columns(df):
        sem, _ = infer_semantic_type(df[c], c)
        if sem not in ("categorical", "binary"):
            continue
        vals = text_values(df[c])
        if vals.empty:
            continue
        norm = vals.str.lower().str.replace(r"[^a-z0-9؀-ۿ]", "", regex=True)
        groups = vals.groupby(norm).nunique()
        total += int(vals.nunique())
        variants += int((groups - 1).clip(lower=0).sum())
    return 1.0 if total == 0 else float(1 - variants / total)


def _mean_pass(df: pd.DataFrame, rules: list[Rule]) -> float:
    rates = [check(df, r).pass_rate for r in rules if r.column in df.columns or r.kind == "schema"]
    return float(sum(rates) / len(rates)) if rates else 1.0


def quality_report(df: pd.DataFrame, rules: list[Rule], key: str | None = None) -> dict[str, float]:
    validity_rules = [r for r in rules if r.kind in ("type", "range", "category")]
    conformity_rules = [r for r in rules if r.kind in ("regex",) or (r.kind == "type" and
                                                                   r.params.get("dtype") == "datetime")]
    integrity_rules = [r for r in rules if r.kind in ("cross_field", "temporal")]
    return {
        "completeness": completeness(df),
        "uniqueness": uniqueness(df, key),
        "validity": _mean_pass(df, validity_rules),
        "consistency": consistency(df),
        "conformity": _mean_pass(df, conformity_rules),
        "integrity": _mean_pass(df, integrity_rules),
    }


def weighted_score(sub: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(k, 0) for k in sub)
    if total_w == 0:
        return 0.0
    return float(sum(sub[k] * weights.get(k, 0) for k in sub) / total_w)
