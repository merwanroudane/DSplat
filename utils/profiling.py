"""In-house profiler: the engine behind the Dataset Explorer, the Automated
EDA page, the issue detector and the automated report.

It is intentionally transparent: every finding carries its evidence and a
"why", and findings are classified as detected / potential / recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from utils.types import (categorical_columns, infer_semantic_type, is_numeric, numeric_columns,
                         semantic_missing_mask, text_values, to_number)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
SEVERITY_AR = {"high": "مرتفعة", "medium": "متوسطة", "low": "منخفضة", "info": "معلومة"}
KIND_AR = {"detected": "مشكلة مكتشفة", "potential": "مشكلة محتملة", "recommendation": "توصية"}


@dataclass
class Issue:
    column: str
    issue: str
    kind: str  # detected | potential | recommendation
    severity: str  # high | medium | low | info
    evidence: str
    action: str
    why: str
    action_code: str = ""  # machine-readable action id used by the hybrid lab

    def as_row(self) -> dict:
        d = asdict(self)
        d["kind_ar"] = KIND_AR[self.kind]
        d["severity_ar"] = SEVERITY_AR[self.severity]
        return d


def overview(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float(df.isna().mean().mean() * 100) if df.size else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric": len(numeric_columns(df)),
        "categorical": len(categorical_columns(df)),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        sem, reason = infer_semantic_type(s, c)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "semantic_type": sem,
            "non_null": int(s.notna().sum()),
            "missing_%": round(float(s.isna().mean() * 100), 2),
            "semantic_missing": int(semantic_missing_mask(s).sum()),
            "unique": int(s.nunique(dropna=True)),
            "example": str(s.dropna().iloc[0]) if s.notna().any() else "",
            "reason": reason,
        })
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    out = df[cols].describe().T
    out["skew"] = df[cols].skew()
    out["kurtosis"] = df[cols].kurt()
    out["zeros"] = (df[cols] == 0).sum()
    out["negatives"] = (df[cols] < 0).sum()
    return out.round(3)


def categorical_summary(df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    rows = []
    for c in categorical_columns(df):
        s = df[c]
        vc = s.value_counts(dropna=True)
        rows.append({
            "column": c,
            "unique": int(s.nunique()),
            "top_values": ", ".join(f"{k} ({v})" for k, v in vc.head(top).items()),
            "rare_(<1%)": int((vc / max(1, s.notna().sum()) < 0.01).sum()),
            "missing_%": round(float(s.isna().mean() * 100), 2),
        })
    return pd.DataFrame(rows)


def iqr_outlier_counts(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    rows = []
    for c in numeric_columns(df):
        x = df[c].dropna()
        if x.nunique() < 5:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        rows.append({"column": c, "lower": round(lo, 3), "upper": round(hi, 3), "outliers": n_out,
                     "outliers_%": round(100 * n_out / len(x), 2)})
    return pd.DataFrame(rows)


def top_correlations(df: pd.DataFrame, k: int = 8, method: str = "spearman") -> pd.DataFrame:
    cols = [c for c in numeric_columns(df) if df[c].nunique() > 2]
    if len(cols) < 2:
        return pd.DataFrame(columns=["var_1", "var_2", "r"])
    corr = df[cols].corr(method=method)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                pairs.append((a, b, float(r)))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pd.DataFrame(pairs[:k], columns=["var_1", "var_2", "r"]).round(3)


# ------------------------------------------------------------------ issues
def detect_issues(df: pd.DataFrame) -> list[Issue]:
    """Rule-based + statistical issue detection. Never modifies ``df``."""
    issues: list[Issue] = []
    n = len(df)
    if n == 0:
        return [Issue("(dataset)", "Dataset فارغة", "detected", "high", "0 صفوف", "تحقق من الملف المصدر",
                      "لا يمكن إجراء أي تحليل على بيانات فارغة.")]

    dups = int(df.duplicated().sum())
    if dups:
        issues.append(Issue("(all columns)", "صفوف مكررة تمامًا Exact duplicates", "detected",
                            "medium" if dups / n < 0.05 else "high", f"{dups} صفًا ({dups / n:.1%})",
                            "راجع المصدر ثم احذف النسخ المطابقة مع الإبقاء على الأولى",
                            "التكرار التام يضخم الأوزان ويشوه الإحصاءات، لكنه قد يكون مشروعًا في بيانات المعاملات.",
                            "drop_exact_duplicates"))

    for c in df.columns:
        s = df[c]
        sem, _ = infer_semantic_type(s, c)
        miss = float(s.isna().mean())
        if miss > 0:
            sev = "high" if miss > 0.3 else "medium" if miss > 0.05 else "low"
            action = ("احذف العمود أو اجمع بيانات إضافية" if miss > 0.6 else
                      "Median imputation مع مؤشر فقد" if is_numeric(s) else "فئة صريحة «Missing»")
            why = ("نسبة الفقد مرتفعة جدًا؛ أي تعويض سيكون تخمينًا في الغالب." if miss > 0.6 else
                   "الوسيط متين أمام الالتواء والقيم المتطرفة، ومؤشر الفقد يحفظ معلومة «أن القيمة غائبة»." if is_numeric(s) else
                   "إبقاء الفقد كفئة مستقلة لا يفترض شيئًا عن القيمة الحقيقية ويسمح للنموذج باستخدام نمط الفقد.")
            issues.append(Issue(c, "قيم مفقودة Missing values", "detected", sev, f"{s.isna().sum()} ({miss:.1%})",
                                action, why + " افحص آلية الفقد (MCAR/MAR/MNAR) قبل التطبيق.",
                                "impute_median" if is_numeric(s) else "impute_missing_label"))
        sm = int(semantic_missing_mask(s).sum())
        if sm:
            examples = ", ".join(repr(v) if v.strip() == "" else v
                                 for v in s[semantic_missing_mask(s)].astype(str).unique()[:4])
            issues.append(Issue(c, "قيم مفقودة دلاليًا Sentinel/placeholder", "detected", "medium",
                                f"{sm} قيمة مثل: {examples}", "حوّلها إلى NaN قبل أي حساب",
                                "هذه القيم تبدو موجودة برمجيًا لكنها تعني «غير معروف»؛ تركها يفسد المتوسطات.",
                                "sentinel_to_nan"))
        if sem == "numeric-as-text":
            parsed = to_number(s)
            failed = int((parsed.isna() & s.notna()).sum())
            issues.append(Issue(c, "نوع خاطئ: أرقام مخزنة نصًا Wrong type", "detected", "high",
                                f"dtype={s.dtype}; {failed} قيمة لا تتحول لرقم",
                                "حوّل بـ pd.to_numeric(errors='coerce') بعد إزالة الرموز والفواصل",
                                "لا يمكن حساب المتوسط أو الارتباط على نص؛ والتحويل القسري يكشف القيم غير القابلة للتحويل.",
                                "coerce_numeric"))
        if sem == "date-as-text":
            parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
            bad = int((parsed.isna() & s.notna() & (s.astype(str).str.strip() != "")).sum())
            issues.append(Issue(c, "تاريخ مخزن نصًا Date as text", "detected", "medium",
                                f"{bad} قيمة غير قابلة للتحويل", "حوّل بـ pd.to_datetime مع صيغة صريحة ثم راجع الفشل",
                                "الصيغ المختلطة (يوم/شهر مقابل شهر/يوم) قد تُقرأ خطأ دون أي رسالة.",
                                "parse_dates"))
        if not is_numeric(s) and sem in ("categorical", "binary", "high-cardinality"):
            vals = text_values(s)
            if len(vals):
                ws = int((s.dropna().astype(str) != vals).sum())
                norm = vals.str.lower().str.replace(r"[^a-z0-9؀-ۿ]", "", regex=True)
                collisions = vals.groupby(norm).nunique()
                variants = int((collisions > 1).sum())
                if ws:
                    issues.append(Issue(c, "مسافات زائدة Whitespace", "detected", "low", f"{ws} قيمة",
                                        "طبّق str.strip()", "« Egypt» و«Egypt» ستُعدّان فئتين مختلفتين.",
                                        "strip_whitespace"))
                if variants:
                    ex = collisions[collisions > 1].index[:3].tolist()
                    issues.append(Issue(c, "تسميات غير متسقة Inconsistent labels", "detected", "medium",
                                        f"{variants} مجموعة متغايرة الكتابة (مثل: {', '.join(ex)})",
                                        "وحّد حالة الأحرف ثم طبّق قاموس توحيد Mapping يراجعه إنسان",
                                        "التهجئات المختلفة لنفس الكيان تقسّم المجموعات وتضلل التكرارات.",
                                        "normalize_case"))
                vc = vals.value_counts(normalize=True)
                rare = vc[vc < 0.01]
                if 0 < len(rare) < len(vc):
                    issues.append(Issue(c, "فئات نادرة Rare categories", "potential", "low",
                                        f"{len(rare)} فئة بنسبة < 1% (مثل: {', '.join(map(str, rare.index[:3]))})",
                                        "تحقق إن كانت خطأ إدخال أو فئة حقيقية؛ يمكن دمجها في «Other» للنمذجة",
                                        "الفئة النادرة قد تكون خطأً إملائيًا أو حالة حقيقية مهمة؛ لا تُدمج قبل الفحص.",
                                        "group_rare"))
        num = s if is_numeric(s) else (to_number(s) if sem == "numeric-as-text" else None)
        if num is not None and num.notna().sum() > 10:
            x = num[~semantic_missing_mask(num)].dropna()
            if np.isinf(x).any():
                issues.append(Issue(c, "قيم لا نهائية Infinity", "detected", "high", f"{int(np.isinf(x).sum())} قيمة",
                                    "استبدلها بـ NaN وابحث عن القسمة على صفر في مصدرها",
                                    "Infinity تكسر المتوسطات والنماذج دون تحذير واضح.", "inf_to_nan"))
                x = x[np.isfinite(x)]
            neg = int((x < 0).sum())
            lname = c.lower()
            if neg and any(k in lname for k in ("age", "order", "count", "height", "price", "qty", "quantity",
                                                "income", "spend", "weight", "hours")):
                issues.append(Issue(c, "قيم سالبة مستحيلة منطقيًا Impossible negatives", "detected", "high",
                                    f"{neg} قيمة سالبة", "اعتبرها NaN ثم ارجع للمصدر",
                                    "هذا المتغير لا يمكن أن يكون سالبًا حسب معناه (Domain rule).", "negative_to_nan"))
            if x.nunique() > 10:
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    far = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
                    if far:
                        issues.append(Issue(c, "قيم متطرفة جدًا Extreme outliers (3×IQR)", "potential", "medium",
                                            f"{far} قيمة خارج [Q1−3·IQR, Q3+3·IQR]",
                                            "حقّق يدويًا: خطأ وحدة؟ خطأ إدخال؟ أم حالة نادرة حقيقية؟ ثم قرر (Keep/Cap/Correct)",
                                            "القيمة غير المعتادة إحصائيًا ليست بالضرورة خطأ؛ الحذف الآلي قد يزيل أهم الحالات.",
                                            "cap_iqr"))
                sk = float(stats.skew(x))
                if abs(sk) > 1.5:
                    issues.append(Issue(c, "التواء شديد Strong skewness", "recommendation", "info",
                                        f"skewness = {sk:.2f}",
                                        "فكّر في log1p أو Yeo-Johnson للنماذج الحساسة، واستخدم الوسيط في الوصف",
                                        "الالتواء الشديد يجعل المتوسط مضللًا ويؤثر على النماذج الخطية والمسافات.",
                                        "log_transform"))
            if x.nunique() == 1:
                issues.append(Issue(c, "عمود ثابت Constant column", "detected", "low", "قيمة واحدة فقط",
                                    "احذفه من النمذجة", "عمود بلا تباين لا يحمل معلومات.", "drop_column"))
        if sem == "identifier":
            dup_ids = int(s.dropna().duplicated().sum())
            if dup_ids:
                issues.append(Issue(c, "معرّفات مكررة Duplicate keys", "potential", "high", f"{dup_ids} تكرار",
                                    "افحص هل هي تكرارات خاطئة أم تصميم مشروع (Panel/معاملات)",
                                    "المفتاح الأساسي يُفترض أن يكون فريدًا؛ تكراره قد يعني دمجًا خاطئًا أو إدخالًا مزدوجًا.",
                                    "dedupe_key"))
    for c in df.columns:
        if df[c].isna().mean() == 0 and infer_semantic_type(df[c], c)[0] == "identifier":
            issues.append(Issue(c, "استبعد المعرّف من النمذجة", "recommendation", "info", "معرّف فريد",
                                "لا تستخدمه كـFeature", "المعرّف لا يحمل علاقة سببية ويسبب Overfitting أو Leakage.",
                                "none"))
    issues.sort(key=lambda i: (SEVERITY_ORDER[i.severity], i.column))
    return issues


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["column", "issue", "kind_ar", "severity_ar", "evidence", "action", "why"])
    return pd.DataFrame([i.as_row() for i in issues])


def recommended_next_steps(df: pd.DataFrame, issues: list[Issue]) -> list[str]:
    steps = []
    kinds = {i.issue for i in issues}
    if any("Wrong type" in k for k in kinds):
        steps.append("ابدأ بتصحيح الأنواع (Types): التحويل الخاطئ يفسد كل ما بعده.")
    if any("Sentinel" in k for k in kinds):
        steps.append("حوّل القيم الدلالية (-999, unknown, ?) إلى NaN قبل حساب الفقد الحقيقي.")
    if any("Missing" in k for k in kinds):
        steps.append("شخّص آلية الفقد في صفحة «القيم المفقودة» قبل اختيار طريقة التعويض.")
    if any("duplicates" in k.lower() or "Duplicate" in k for k in kinds):
        steps.append("قرر: هل التكرار خطأ أم تصميم مشروع؟ ثم عالجه بـ subset مناسب.")
    if any("labels" in k for k in kinds):
        steps.append("وحّد التسميات الفئوية بقاموس Mapping موثّق.")
    if any("outliers" in k.lower() for k in kinds):
        steps.append("استخدم أداة قرار القيم الشاذة؛ لا تحذف تلقائيًا.")
    steps.append("وثّق كل قرار (ماذا، لماذا، كم صفًا تأثر) لضمان قابلية إعادة الإنتاج.")
    return steps


def normality_hint(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 8:
        return "عينة صغيرة جدًا للحكم."
    sk = stats.skew(x)
    if abs(sk) < 0.5:
        return "قريب من التماثل."
    return "ملتوٍ لليمين (ذيل طويل للقيم الكبيرة)." if sk > 0 else "ملتوٍ لليسار (ذيل طويل للقيم الصغيرة)."
