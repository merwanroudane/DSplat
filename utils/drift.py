"""Drift and schema-change measures used by the observability module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Population Stability Index with quantile bins from the reference.

    PSI = Σ (cur% − ref%) · ln(cur% / ref%). Rule of thumb: < 0.1 stable,
    0.1–0.25 moderate shift, > 0.25 large shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    eps = 1e-6
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    contrib = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    table = pd.DataFrame({"bin": range(1, len(contrib) + 1), "reference_%": 100 * ref_pct, "current_%": 100 * cur_pct,
                          "contribution": contrib})
    return float(contrib.sum()), table


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    r = stats.ks_2samp(reference, current)
    return float(r.statistic), float(r.pvalue)


def psi_label(value: float) -> str:
    return "مستقر" if value < 0.1 else "تغير متوسط" if value < 0.25 else "تغير كبير"


def schema_diff(old: dict[str, str], new: dict[str, str]) -> pd.DataFrame:
    rows = []
    for col in sorted(set(old) | set(new)):
        if col not in new:
            rows.append({"column": col, "change": "حُذف Removed", "old": old[col], "new": "—"})
        elif col not in old:
            rows.append({"column": col, "change": "أُضيف Added", "old": "—", "new": new[col]})
        elif old[col] != new[col]:
            rows.append({"column": col, "change": "تغيّر النوع Type changed", "old": old[col], "new": new[col]})
    return pd.DataFrame(rows, columns=["column", "change", "old", "new"])
