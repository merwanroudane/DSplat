"""Outlier detection: statistical rules and model-based detectors, plus the
rule-based Outlier Decision Tool."""

from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_mask(x: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = x.quantile([0.25, 0.75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def zscore_mask(x: pd.Series, threshold: float = 3.0) -> pd.Series:
    z = (x - x.mean()) / x.std(ddof=1)
    return z.abs() > threshold


def modified_z(x: pd.Series) -> pd.Series:
    """Iglewicz & Hoaglin (1993): M = 0.6745 (x - median) / MAD."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def modified_z_mask(x: pd.Series, threshold: float = 3.5) -> pd.Series:
    return modified_z(x).abs() > threshold


def percentile_mask(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = x.quantile([lower, upper])
    return (x < lo) | (x > hi)


def winsorize(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = x.quantile([lower, upper])
    return x.clip(lo, hi)


def model_scores(X: pd.DataFrame, method: str, contamination: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_outlier bool array, anomaly score where higher = more anomalous)."""
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(contamination=contamination, random_state=seed).fit(Z)
        return m.predict(Z) == -1, -m.score_samples(Z)
    if method == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        m = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        labels = m.fit_predict(Z)
        return labels == -1, -m.negative_outlier_factor_
    if method == "ocsvm":
        from sklearn.svm import OneClassSVM
        m = OneClassSVM(nu=contamination, gamma="scale").fit(Z)
        return m.predict(Z) == -1, -m.score_samples(Z)
    if method == "robust_cov":
        from sklearn.covariance import EllipticEnvelope
        m = EllipticEnvelope(contamination=contamination, random_state=seed).fit(Z)
        return m.predict(Z) == -1, -m.score_samples(Z)
    raise ValueError(method)


MODEL_NAMES = {"isolation_forest": "Isolation Forest", "lof": "Local Outlier Factor",
               "ocsvm": "One-Class SVM", "robust_cov": "Robust covariance (Elliptic Envelope)"}


# ----------------------------------------------------------- decision tool
DECISION_QUESTIONS = [
    ("possible", "هل القيمة ممكنة منطقيًا؟ (Is the value logically possible?)"),
    ("rules", "هل تخالف قواعد المجال؟ (Does it violate domain rules?)"),
    ("unit", "هل يمكن أن تكون مشكلة وحدة قياس؟ (Could it be a unit problem?)"),
    ("measurement", "هل يمكن أن تكون خطأ قياس أو إدخال؟ (Measurement/entry problem?)"),
    ("source", "هل هي موجودة كما هي في المصدر الأصلي؟ (Present in the original source?)"),
    ("rare_valid", "هل هي حالة نادرة لكنها حقيقية؟ (Rare but valid?)"),
    ("influence", "هل تؤثر بقوة على النموذج أو التقدير؟ (Strongly affects the model?)"),
]


def outlier_decision(a: dict[str, str]) -> tuple[str, str]:
    """Map answers ('نعم' / 'لا' / 'لا أعرف') to (action, explanation)."""
    yes = lambda k: a.get(k) == "نعم"  # noqa: E731
    no = lambda k: a.get(k) == "لا"  # noqa: E731
    unknown = [k for k, v in a.items() if v == "لا أعرف"]
    if yes("unit"):
        return ("Correct — صحّح الوحدة",
                "القيمة تبدو صحيحة بعد تحويل الوحدة (مثل 1.75 م ← 175 سم). صحّحها ووثّق قاعدة التحويل.")
    if no("possible") or yes("rules"):
        if yes("measurement") or not yes("source"):
            return ("Correct / Remove — صحّح أو احذف",
                    "القيمة مستحيلة أو تخالف قاعدة مجال. إن أمكن استرجاع القيمة الصحيحة من المصدر فصحّحها، "
                    "وإلا اعتبرها مفقودة (NaN) بدل حذف الصف كاملًا.")
        return ("Investigate — تحقّق",
                "القيمة مستحيلة لكنها موجودة في المصدر: الخطأ في الجمع نفسه. أبلغ مالك البيانات قبل أي تعديل.")
    if unknown:
        return ("Investigate — تحقّق",
                "هناك أسئلة بلا إجابة. لا يجوز اتخاذ قرار نهائي قبل التحقق من المصدر ومعنى المتغير.")
    if yes("rare_valid"):
        if yes("influence"):
            return ("Robust modeling / Transform — نمذجة متينة أو تحويل",
                    "الحالة حقيقية فلا تُحذف. استخدم مقدرات متينة (Huber, Quantile) أو تحويل log، "
                    "وقارن النتائج مع وبدون القيمة (Sensitivity analysis).")
        return ("Keep — احتفظ بها",
                "حالة نادرة حقيقية ولا تؤثر كثيرًا. حذفها يشوّه تمثيل الواقع ويخفض التباين الحقيقي.")
    if yes("influence"):
        return ("Cap / Winsorize — قصّ الأطراف",
                "القيمة ممكنة لكن تأثيرها كبير وحقيقتها غير مؤكدة: القص عند مئين (مثل 1%/99%) يحد التأثير "
                "مع إبقاء الصف. وثّق الحدود المستخدمة.")
    return ("Keep & document — احتفظ ووثّق",
            "لا يوجد دليل على خطأ ولا تأثير كبير؛ الإبقاء هو الخيار الافتراضي الأسلم.")
