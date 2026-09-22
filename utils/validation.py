"""A small, transparent validation engine (no external dependency).

Each rule returns the number of rows checked, the number failing and the
index of failing rows, so the UI can show evidence. The same rules are
also shown as optional Pandera / Pydantic code on the Data Validation page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils.types import to_number


@dataclass
class Rule:
    kind: str  # schema | type | range | category | regex | cross_field | unique | referential | temporal | not_null
    column: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class RuleResult:
    rule: Rule
    checked: int
    failed: int
    failing_index: list
    message: str = ""

    @property
    def pass_rate(self) -> float:
        return 1.0 if self.checked == 0 else 1 - self.failed / self.checked


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(dtype=object)


def check(df: pd.DataFrame, rule: Rule) -> RuleResult:
    k, c, p = rule.kind, rule.column, rule.params
    if k == "schema":
        missing = [col for col in p.get("required", []) if col not in df.columns]
        extra = [col for col in df.columns if col not in p.get("required", [])] if p.get("strict") else []
        failed = len(missing) + len(extra)
        msg = (f"أعمدة مفقودة: {missing}" if missing else "") + (f" أعمدة غير متوقعة: {extra}" if extra else "")
        return RuleResult(rule, len(p.get("required", [])), failed, [], msg or "المخطط مطابق")
    if c not in df.columns:
        return RuleResult(rule, 0, 0, [], f"العمود {c} غير موجود")
    s = df[c]
    nonnull = s.dropna()
    if k == "not_null":
        bad = s.isna()
        return RuleResult(rule, len(s), int(bad.sum()), list(s.index[bad]))
    if k == "type":
        expected = p["dtype"]
        if expected == "numeric":
            parsed = to_number(nonnull) if not pd.api.types.is_numeric_dtype(nonnull) else nonnull
            bad_idx = nonnull.index[parsed.isna().to_numpy()]
            return RuleResult(rule, len(nonnull), len(bad_idx), list(bad_idx),
                              f"النوع المخزّن: {s.dtype}")
        if expected == "datetime":
            parsed = pd.to_datetime(nonnull.astype(str), errors="coerce", format=p.get("format"))
            bad_idx = nonnull.index[parsed.isna().to_numpy()]
            return RuleResult(rule, len(nonnull), len(bad_idx), list(bad_idx))
        return RuleResult(rule, len(nonnull), 0, [])
    if k == "range":
        x = to_number(nonnull)
        bad = (x < p.get("min", -float("inf"))) | (x > p.get("max", float("inf")))
        bad = bad.fillna(False)
        return RuleResult(rule, int(x.notna().sum()), int(bad.sum()), list(x.index[bad]))
    if k == "category":
        allowed = set(p["allowed"])
        bad = ~nonnull.astype(str).isin(allowed)
        return RuleResult(rule, len(nonnull), int(bad.sum()), list(nonnull.index[bad]))
    if k == "regex":
        pat = re.compile(p["pattern"])
        bad = ~nonnull.astype(str).map(lambda v: bool(pat.fullmatch(v)))
        return RuleResult(rule, len(nonnull), int(bad.sum()), list(nonnull.index[bad]))
    if k == "unique":
        bad = nonnull.duplicated(keep=False)
        return RuleResult(rule, len(nonnull), int(bad.sum()), list(nonnull.index[bad]))
    if k == "referential":
        ref = set(p["reference"])
        bad = ~nonnull.isin(ref)
        return RuleResult(rule, len(nonnull), int(bad.sum()), list(nonnull.index[bad]))
    if k in ("cross_field", "temporal"):
        other = p["other"]
        a = pd.to_datetime(df[c], errors="coerce", format="mixed") if k == "temporal" else to_number(df[c])
        b = pd.to_datetime(df[other], errors="coerce", format="mixed") if k == "temporal" else to_number(df[other])
        both = a.notna() & b.notna()
        op = p.get("op", ">=")
        ok = {">=": a >= b, ">": a > b, "<=": a <= b, "<": a < b}[op]
        bad = both & ~ok
        return RuleResult(rule, int(both.sum()), int(bad.sum()), list(df.index[bad]))
    raise ValueError(f"Unknown rule kind: {k}")


def run_rules(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    rows = []
    for r in rules:
        res = check(df, r)
        rows.append({
            "rule": r.description or f"{r.kind}({r.column})",
            "kind": r.kind,
            "column": r.column,
            "checked": res.checked,
            "failed": res.failed,
            "pass_rate_%": round(100 * res.pass_rate, 2),
            "status": "✅ ناجح" if res.failed == 0 else "❌ فاشل",
            "note": res.message,
        })
    return pd.DataFrame(rows)


CUSTOMER_RULES: list[Rule] = [
    Rule("schema", "*", {"required": ["customer_id", "signup_date", "age", "country", "annual_income", "email",
                                      "churned"]}, "Schema: الأعمدة المطلوبة موجودة"),
    Rule("unique", "customer_id", {}, "Uniqueness: customer_id فريد"),
    Rule("not_null", "customer_id", {}, "Completeness: customer_id غير فارغ"),
    Rule("type", "age", {"dtype": "numeric"}, "Type: age رقمي"),
    Rule("range", "age", {"min": 18, "max": 100}, "Range: 18 ≤ age ≤ 100"),
    Rule("type", "annual_income", {"dtype": "numeric"}, "Type: annual_income رقمي"),
    Rule("range", "num_orders", {"min": 0}, "Range: num_orders ≥ 0"),
    Rule("range", "satisfaction", {"min": 1, "max": 5}, "Range: satisfaction بين 1 و5"),
    Rule("category", "membership", {"allowed": ["Bronze", "Silver", "Gold", "Platinum"]},
         "Category: membership من القائمة المعتمدة"),
    Rule("category", "gender", {"allowed": ["Female", "Male"]}, "Category: gender موحّد"),
    Rule("regex", "email", {"pattern": r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}"}, "Regex: صيغة البريد صحيحة"),
    Rule("type", "signup_date", {"dtype": "datetime", "format": "%Y-%m-%d"}, "Type: signup_date بصيغة ISO"),
    Rule("temporal", "last_purchase_date", {"other": "signup_date", "op": ">="},
         "Temporal: آخر شراء بعد التسجيل"),
    Rule("range", "height_cm", {"min": 120, "max": 230}, "Range: 120 ≤ height_cm ≤ 230 (يكشف خطأ الوحدة)"),
]
