"""Missing data: simulation of mechanisms, diagnostics, imputation, and
Little's MCAR test (EM-based, following Little, 1988)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from utils.types import numeric_columns


# ---------------------------------------------------------------- simulate
def simulate_bivariate(n: int = 500, rho: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """x (always observed) and y (to be made missing), bivariate normal."""
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    y = 20 + rho * (x - 50) + np.sqrt(1 - rho ** 2) * rng.normal(0, 10, n)
    return pd.DataFrame({"x": x, "y": y})


def make_missing(df: pd.DataFrame, mechanism: str, rate: float, seed: int = 0, col: str = "y",
                 driver: str = "x") -> pd.Series:
    """Return a boolean mask of values to remove from ``col``.

    MCAR: independent of everything. MAR: depends on the observed ``driver``.
    MNAR: depends on the unobserved value of ``col`` itself.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if mechanism == "MCAR":
        p = np.full(n, rate)
    else:
        z = df[driver] if mechanism == "MAR" else df[col]
        z = (z - z.mean()) / z.std()
        # logistic probabilities calibrated to the requested average rate
        lo, hi = -10.0, 10.0
        for _ in range(60):
            b = (lo + hi) / 2
            if (1 / (1 + np.exp(-(b + 2.0 * z)))).mean() > rate:
                hi = b
            else:
                lo = b
        p = 1 / (1 + np.exp(-(b + 2.0 * z)))
    return pd.Series(rng.random(n) < p, index=df.index)


def impact_table(full: pd.DataFrame, observed: pd.DataFrame, col: str = "y", driver: str = "x") -> pd.DataFrame:
    """Compare statistics on the full data vs complete-case analysis."""
    def slope(d: pd.DataFrame) -> float:
        d = d[[driver, col]].dropna()
        return float(np.polyfit(d[driver], d[col], 1)[0]) if len(d) > 2 else np.nan

    rows = []
    for label, d in (("البيانات الكاملة (الحقيقة)", full), ("الحالات المكتملة فقط", observed)):
        rows.append({
            "النسخة": label,
            "Mean(y)": d[col].mean(),
            "Var(y)": d[col].var(),
            "Corr(x,y)": d[[driver, col]].dropna().corr().iloc[0, 1],
            "Slope y~x": slope(d),
            "n": int(d[col].notna().sum()),
        })
    out = pd.DataFrame(rows).set_index("النسخة")
    bias = (out.iloc[1] - out.iloc[0]).rename("الانحياز Bias")
    return pd.concat([out, bias.to_frame().T]).round(3)


# ------------------------------------------------------------- diagnostics
def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"missing": df.isna().sum(), "missing_%": (df.isna().mean() * 100).round(2)})
    return out.sort_values("missing", ascending=False)


def missing_patterns(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Rows grouped by their missingness pattern (1 = missing)."""
    m = df.isna().astype(int)
    cols = [c for c in df.columns if m[c].any()]
    if not cols:
        return pd.DataFrame()
    pat = m[cols].astype(str).agg("".join, axis=1)
    counts = pat.value_counts().head(top)
    rows = []
    for p, cnt in counts.items():
        row = {c: int(ch) for c, ch in zip(cols, p)}
        row["n_rows"] = int(cnt)
        row["%"] = round(100 * cnt / len(df), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between missingness indicators (joint missingness)."""
    m = df.isna().astype(int)
    m = m.loc[:, m.std() > 0]
    return m.corr().round(2) if m.shape[1] >= 2 else pd.DataFrame()


def missing_by_group(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    g = df.groupby(group, observed=True)[target].apply(lambda s: s.isna().mean() * 100)
    n = df.groupby(group, observed=True)[target].size()
    return pd.DataFrame({"missing_%": g.round(2), "n": n})


def compare_by_missingness(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """For each other numeric variable: does it differ when ``target`` is missing?

    A significant difference is evidence *against* MCAR (consistent with MAR
    or MNAR). No difference does not prove MCAR.
    """
    ind = df[target].isna()
    rows = []
    for c in numeric_columns(df):
        if c == target:
            continue
        a, b = df.loc[ind, c].dropna(), df.loc[~ind, c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        t = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"variable": c, "mean_if_missing": a.mean(), "mean_if_observed": b.mean(),
                     "cohen_d": (a.mean() - b.mean()) / pooled if pooled else np.nan,
                     "welch_t": t.statistic, "p_value": t.pvalue})
    return pd.DataFrame(rows).round(4)


def _em_mvn(x: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """EM estimates of mean and covariance for multivariate normal data with NaNs.

    Rows are processed per missingness pattern, so each iteration costs one
    regression per pattern rather than one per row.
    """
    n, p = x.shape
    miss = np.isnan(x)
    patterns, inverse = np.unique(miss, axis=0, return_inverse=True)
    groups = [(pat, x[inverse.ravel() == j]) for j, pat in enumerate(patterns)]
    mu = np.nanmean(x, axis=0)
    sigma = np.diag(np.nanvar(x, axis=0)) + 1e-6 * np.eye(p)
    for _ in range(max_iter):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for pat, rows in groups:
            m, o = pat, ~pat
            xhat = rows.copy()
            c = np.zeros((p, p))
            if m.any():
                if o.any():
                    s_mo = sigma[np.ix_(m, o)]
                    reg = s_mo @ np.linalg.pinv(sigma[np.ix_(o, o)])
                    xhat[:, m] = mu[m] + (rows[:, o] - mu[o]) @ reg.T
                    c[np.ix_(m, m)] = sigma[np.ix_(m, m)] - reg @ s_mo.T
                else:
                    xhat[:, m] = mu[m]
                    c[np.ix_(m, m)] = sigma[np.ix_(m, m)]
            sum_x += xhat.sum(axis=0)
            sum_xx += xhat.T @ xhat + len(rows) * c
        new_mu = sum_x / n
        new_sigma = sum_xx / n - np.outer(new_mu, new_mu)
        done = np.max(np.abs(new_mu - mu)) < tol and np.max(np.abs(new_sigma - sigma)) < tol
        mu, sigma = new_mu, new_sigma
        if done:
            break
    return mu, sigma


def littles_mcar_test(df: pd.DataFrame) -> dict:
    """Little's (1988) chi-square test of MCAR for numeric columns.

    H0: data are MCAR. Rejection is evidence against MCAR; non-rejection is
    not proof of MCAR (low power, assumes multivariate normality).
    """
    x = df[numeric_columns(df)].to_numpy(dtype=float)
    x = x[:, ~np.all(np.isnan(x), axis=0)]
    n, p = x.shape
    mu, sigma = _em_mvn(x)
    miss = np.isnan(x)
    patterns, inverse = np.unique(miss, axis=0, return_inverse=True)
    d2, df_total = 0.0, 0
    for j, pat in enumerate(patterns):
        o = ~pat
        if not o.any():
            continue
        rows = x[inverse.ravel() == j][:, o]
        nj = rows.shape[0]
        diff = rows.mean(axis=0) - mu[o]
        d2 += nj * diff @ np.linalg.pinv(sigma[np.ix_(o, o)]) @ diff
        df_total += int(o.sum())
    df_total -= p
    p_value = float(stats.chi2.sf(d2, df_total)) if df_total > 0 else float("nan")
    return {"statistic": float(d2), "df": int(df_total), "p_value": p_value, "n_patterns": int(len(patterns))}


# --------------------------------------------------------------- imputation
IMPUTERS = {
    "listwise": "حذف الصفوف Listwise deletion",
    "mean": "المتوسط Mean",
    "median": "الوسيط Median",
    "mode": "المنوال Mode",
    "constant": "قيمة ثابتة Constant",
    "ffill": "Forward fill",
    "bfill": "Backward fill",
    "interpolate": "Linear interpolation",
    "knn": "KNN imputation",
    "regression": "Regression imputation",
    "stochastic_regression": "Stochastic regression",
    "iterative": "Iterative (MICE-style)",
}


def impute(df: pd.DataFrame, method: str, cols: list[str] | None = None, constant: float = 0.0,
           k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Impute numeric ``cols`` (default: all numeric). Returns a new frame."""
    out = df.copy()
    cols = cols or numeric_columns(df)
    if method == "listwise":
        return out.dropna(subset=cols)
    if method in ("mean", "median"):
        for c in cols:
            out[c] = out[c].fillna(getattr(out[c], method)())
        return out
    if method == "mode":
        for c in cols:
            m = out[c].mode()
            out[c] = out[c].fillna(m.iloc[0] if len(m) else np.nan)
        return out
    if method == "constant":
        out[cols] = out[cols].fillna(constant)
        return out
    if method == "ffill":
        out[cols] = out[cols].ffill()
        return out
    if method == "bfill":
        out[cols] = out[cols].bfill()
        return out
    if method == "interpolate":
        out[cols] = out[cols].interpolate(limit_direction="both")
        return out
    num = numeric_columns(df)
    if method == "knn":
        from sklearn.impute import KNNImputer
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        z = sc.fit_transform(out[num])
        z = KNNImputer(n_neighbors=k).fit_transform(z)
        out[num] = sc.inverse_transform(z)
        return out
    if method in ("regression", "stochastic_regression"):
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(seed)
        for c in cols:
            preds = [p for p in num if p != c]
            base = out[preds].fillna(out[preds].median())
            obs = out[c].notna()
            if obs.sum() < 3 or (~obs).sum() == 0:
                continue
            model = LinearRegression().fit(base[obs], out.loc[obs, c])
            pred = model.predict(base[~obs])
            if method == "stochastic_regression":
                resid_sd = np.std(out.loc[obs, c] - model.predict(base[obs]))
                pred = pred + rng.normal(0, resid_sd, len(pred))
            out.loc[~obs, c] = pred
        return out
    if method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        out[num] = IterativeImputer(random_state=seed, max_iter=15).fit_transform(out[num])
        return out
    raise ValueError(method)


def multiple_imputation_mean(df: pd.DataFrame, col: str, m: int = 5, seed: int = 0) -> dict:
    """Multiple imputation of ``col`` with posterior draws, pooled by Rubin's rules."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    num = numeric_columns(df)
    estimates, variances = [], []
    for i in range(m):
        imp = IterativeImputer(sample_posterior=True, random_state=seed + i, max_iter=10)
        filled = pd.DataFrame(imp.fit_transform(df[num]), columns=num)
        x = filled[col]
        estimates.append(x.mean())
        variances.append(x.var(ddof=1) / len(x))
    q_bar = float(np.mean(estimates))
    u_bar = float(np.mean(variances))
    b = float(np.var(estimates, ddof=1))
    t = u_bar + (1 + 1 / m) * b
    return {"estimates": estimates, "pooled_mean": q_bar, "within_var": u_bar, "between_var": b,
            "total_var": t, "se": float(np.sqrt(t))}


def masked_evaluation(df: pd.DataFrame, col: str, methods: list[str], frac: float = 0.15,
                      seed: int = 0) -> pd.DataFrame:
    """Hide known values, impute, and score each method against the truth."""
    rng = np.random.default_rng(seed)
    base = df.copy()
    known = base.index[base[col].notna()]
    hide = rng.choice(known, int(len(known) * frac), replace=False)
    truth = base.loc[hide, col]
    masked = base.copy()
    masked.loc[hide, col] = np.nan
    rows = []
    for m in methods:
        if m == "listwise":
            continue
        filled = impute(masked, m, cols=[col], seed=seed)
        est = filled.loc[hide, col]
        rows.append({"method": IMPUTERS[m], "RMSE": float(np.sqrt(np.mean((est - truth) ** 2))),
                     "MAE": float(np.mean(np.abs(est - truth))),
                     "SD after / SD before": float(filled[col].std() / df[col].std()),
                     "mean shift": float(filled[col].mean() - df[col].mean())})
    return pd.DataFrame(rows).round(3).sort_values("RMSE")
