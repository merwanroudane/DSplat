"""Pipeline Builder engine: configuration → execution log → generated code.

Two code styles are generated from the same configuration:
* a sequential pandas script (readable, beginner-friendly), and
* a leakage-safe scikit-learn Pipeline (parameters learned on training data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.types import categorical_columns, numeric_columns

OPTIONS = {
    "duplicates": {"none": "لا شيء", "exact": "حذف التكرار التام", "key": "حذف تكرار المفتاح (الإبقاء على الأول)"},
    "missing_num": {"none": "لا شيء", "drop_rows": "حذف الصفوف", "mean": "Mean", "median": "Median",
                    "median_indicator": "Median + مؤشر فقد", "knn": "KNN", "iterative": "Iterative"},
    "missing_cat": {"none": "لا شيء", "mode": "المنوال Mode", "label": "فئة «Missing»"},
    "outliers": {"none": "لا شيء", "flag": "إضافة مؤشر فقط (Flag)", "cap_iqr": "قصّ عند 1.5×IQR",
                 "winsorize": "Winsorize 1%/99%", "remove_iqr": "حذف الصفوف خارج 1.5×IQR"},
    "transform": {"none": "لا شيء", "log1p": "log1p للأعمدة الملتوية", "yeo_johnson": "Yeo-Johnson"},
    "scaling": {"none": "لا شيء", "standard": "Standardization (z)", "minmax": "Min-Max [0,1]", "robust": "Robust (median/IQR)"},
    "encoding": {"none": "لا شيء", "onehot": "One-Hot", "ordinal": "Ordinal codes", "frequency": "Frequency"},
}


@dataclass
class PipelineConfig:
    duplicates: str = "exact"
    key: str | None = None
    missing_num: str = "median"
    missing_cat: str = "label"
    outliers: str = "flag"
    transform: str = "none"
    scaling: str = "standard"
    encoding: str = "onehot"
    exclude: tuple[str, ...] = ()


def _num(df: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    return [c for c in numeric_columns(df) if c not in cfg.exclude and df[c].nunique() > 2]


def _cat(df: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    return [c for c in categorical_columns(df) if c not in cfg.exclude and df[c].nunique() <= 50]


def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    log: list[dict] = []

    def step(stage: str, choice: str, detail: str, shape: tuple) -> None:
        log.append({"stage": stage, "choice": choice, "detail": detail, "shape_after": f"{shape[0]} × {shape[1]}"})

    # 1. duplicates
    n = len(out)
    if cfg.duplicates == "exact":
        out = out.drop_duplicates()
    elif cfg.duplicates == "key" and cfg.key and cfg.key in out.columns:
        out = out.drop_duplicates(subset=cfg.key, keep="first")
    step("1. Duplicates", OPTIONS["duplicates"][cfg.duplicates], f"حُذف {n - len(out)} صفًا", out.shape)

    num, cat = _num(out, cfg), _cat(out, cfg)

    # Numeric columns become float so later steps (capping, transforms, scaling) can write decimals.
    if num:
        out[num] = out[num].astype(float)

    # 2. missing values
    miss_before = int(out[num].isna().sum().sum()) if num else 0
    if cfg.missing_num == "drop_rows" and num:
        n = len(out)
        out = out.dropna(subset=num)
        detail = f"حُذف {n - len(out)} صفًا فيه قيم رقمية مفقودة"
    elif cfg.missing_num in ("mean", "median", "median_indicator") and num:
        for c in num:
            if cfg.missing_num == "median_indicator" and out[c].isna().any():
                out[f"{c}_was_missing"] = out[c].isna().astype(int)
            fill = out[c].mean() if cfg.missing_num == "mean" else out[c].median()
            out[c] = out[c].fillna(fill)
        detail = f"عُوّضت {miss_before} قيمة رقمية"
    elif cfg.missing_num in ("knn", "iterative") and num:
        from utils.missing import impute
        out = impute(out, cfg.missing_num, cols=num)
        detail = f"عُوّضت {miss_before} قيمة رقمية بنموذج"
    else:
        detail = "دون تغيير"
    if cfg.missing_cat != "none" and cat:
        cat_miss = int(out[cat].isna().sum().sum())
        for c in cat:
            if cfg.missing_cat == "mode":
                m = out[c].mode()
                out[c] = out[c].fillna(m.iloc[0] if len(m) else "Missing")
            else:
                out[c] = out[c].astype(object).fillna("Missing")
        detail += f"، و{cat_miss} قيمة فئوية"
    step("2. Missing values", f"{OPTIONS['missing_num'][cfg.missing_num]} / {OPTIONS['missing_cat'][cfg.missing_cat]}",
         detail, out.shape)

    # 3. outliers
    affected = 0
    for c in num:
        x = out[c]
        q1, q3 = x.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mask = (x < lo) | (x > hi)
        affected += int(mask.sum())
        if cfg.outliers == "flag":
            out[f"{c}_outlier"] = mask.astype(int)
        elif cfg.outliers == "cap_iqr":
            out[c] = x.clip(lo, hi)
        elif cfg.outliers == "winsorize":
            out[c] = x.clip(*x.quantile([0.01, 0.99]))
    if cfg.outliers == "remove_iqr" and num:
        keep = pd.Series(True, index=out.index)
        for c in num:
            x = out[c]
            q1, q3 = x.quantile([0.25, 0.75])
            keep &= x.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)) | x.isna()
        n = len(out)
        out = out[keep]
        step("3. Outliers", OPTIONS["outliers"][cfg.outliers], f"حُذف {n - len(out)} صفًا ⚠️ قرار غير قابل للتراجع في البيانات الأصلية", out.shape)
    else:
        step("3. Outliers", OPTIONS["outliers"][cfg.outliers], f"{affected} قيمة خارج 1.5×IQR", out.shape)

    # 4. transformation
    transformed = []
    if cfg.transform == "log1p":
        for c in num:
            if out[c].min() >= 0 and abs(out[c].skew()) > 1:
                out[c] = np.log1p(out[c])
                transformed.append(c)
    elif cfg.transform == "yeo_johnson" and num:
        from sklearn.preprocessing import PowerTransformer
        out[num] = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(out[num])
        transformed = num
    step("4. Transformation", OPTIONS["transform"][cfg.transform],
         f"الأعمدة: {', '.join(transformed) if transformed else '—'}", out.shape)

    # 5. scaling
    if cfg.scaling != "none" and num:
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[cfg.scaling]
        valid = out[num].notna().all(axis=1)
        out.loc[valid, num] = scaler.fit_transform(out.loc[valid, num])
    step("5. Scaling", OPTIONS["scaling"][cfg.scaling], f"{len(num)} عمودًا رقميًا", out.shape)

    # 6. encoding
    if cfg.encoding == "onehot" and cat:
        out = pd.get_dummies(out, columns=cat, dtype=int)
    elif cfg.encoding == "ordinal" and cat:
        for c in cat:
            out[c] = out[c].astype("category").cat.codes
    elif cfg.encoding == "frequency" and cat:
        for c in cat:
            out[c] = out[c].map(out[c].value_counts(normalize=True))
    step("6. Encoding", OPTIONS["encoding"][cfg.encoding], f"{len(cat)} عمودًا فئويًا", out.shape)
    return out, log


def mermaid_diagram(cfg: PipelineConfig) -> str:
    labels = [
        ("Raw data", ""),
        ("Duplicates", OPTIONS["duplicates"][cfg.duplicates]),
        ("Missing", f"{OPTIONS['missing_num'][cfg.missing_num]} / {OPTIONS['missing_cat'][cfg.missing_cat]}"),
        ("Outliers", OPTIONS["outliers"][cfg.outliers]),
        ("Transform", OPTIONS["transform"][cfg.transform]),
        ("Scaling", OPTIONS["scaling"][cfg.scaling]),
        ("Encoding", OPTIONS["encoding"][cfg.encoding]),
        ("Model-ready data", ""),
    ]
    lines = ["flowchart LR"]
    for i, (title, choice) in enumerate(labels):
        text = f"<b>{title}</b>" + (f"<br/>{choice}" if choice else "")
        lines.append(f'  N{i}["{text}"]')
    for i in range(len(labels) - 1):
        lines.append(f"  N{i} --> N{i + 1}")
    skipped = [i for i, (_, c) in enumerate(labels) if c == "لا شيء"]
    lines.append("  classDef off fill:#F1F3F5,color:#868E96,stroke:#CED4DA,stroke-dasharray:4 3;")
    lines.append("  classDef ends fill:#7048E8,color:#fff,stroke:#5F3DC4;")
    if skipped:
        lines.append(f"  class {','.join(f'N{i}' for i in skipped)} off;")
    lines.append(f"  class N0,N{len(labels) - 1} ends;")
    return "\n".join(lines)


def generate_pandas_code(cfg: PipelineConfig, dataset_hint: str, num: list[str], cat: list[str]) -> str:
    L = ["# ============================================================",
         "# Data cleaning pipeline — generated by Data Science Interactive Academy",
         "# Sequential pandas version: easy to read, run top to bottom.",
         "# NOTE: statistics (median, quantiles, scaler) are computed on the whole",
         "# dataset here. Before modelling, learn them on the TRAINING split only",
         "# (see the scikit-learn version) to avoid preprocessing leakage.",
         "# ============================================================",
         "import numpy as np",
         "import pandas as pd",
         "",
         f'df = pd.read_csv("{dataset_hint}.csv")  # replace with your file',
         f"num_cols = {num!r}",
         f"cat_cols = {cat!r}",
         ""]
    L.append("# --- 1. Duplicates ------------------------------------------")
    if cfg.duplicates == "exact":
        L += ["n_before = len(df)", "df = df.drop_duplicates()  # identical rows only",
              'print(f"Removed {n_before - len(df)} exact duplicates")']
    elif cfg.duplicates == "key":
        L += [f'df = df.drop_duplicates(subset="{cfg.key}", keep="first")  # one row per key']
    else:
        L.append("# (skipped)")
    L += ["", "# --- 2. Missing values --------------------------------------"]
    if cfg.missing_num == "drop_rows":
        L.append("df = df.dropna(subset=num_cols)  # listwise deletion: valid only if MCAR")
    elif cfg.missing_num in ("mean", "median"):
        L += ["for col in num_cols:",
              f"    df[col] = df[col].fillna(df[col].{cfg.missing_num}())"]
    elif cfg.missing_num == "median_indicator":
        L += ["for col in num_cols:",
              "    if df[col].isna().any():",
              '        df[f"{col}_was_missing"] = df[col].isna().astype(int)  # keep the information',
              "    df[col] = df[col].fillna(df[col].median())"]
    elif cfg.missing_num == "knn":
        L += ["from sklearn.impute import KNNImputer",
              "df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])  # scale first in practice"]
    elif cfg.missing_num == "iterative":
        L += ["from sklearn.experimental import enable_iterative_imputer  # noqa: F401",
              "from sklearn.impute import IterativeImputer",
              "df[num_cols] = IterativeImputer(random_state=42).fit_transform(df[num_cols])"]
    else:
        L.append("# numeric: (skipped)")
    if cfg.missing_cat == "mode":
        L += ["for col in cat_cols:", "    df[col] = df[col].fillna(df[col].mode().iloc[0])"]
    elif cfg.missing_cat == "label":
        L += ["for col in cat_cols:", '    df[col] = df[col].astype(object).fillna("Missing")  # explicit category']
    L += ["", "# --- 3. Outliers (1.5 x IQR rule) ---------------------------"]
    if cfg.outliers == "none":
        L.append("# (skipped)")
    else:
        L += ["for col in num_cols:",
              "    q1, q3 = df[col].quantile([0.25, 0.75])",
              "    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)"]
        if cfg.outliers == "flag":
            L.append('    df[f"{col}_outlier"] = (~df[col].between(low, high)).astype(int)  # flag, do not delete')
        elif cfg.outliers == "cap_iqr":
            L.append("    df[col] = df[col].clip(low, high)  # cap instead of delete")
        elif cfg.outliers == "winsorize":
            L.append("    df[col] = df[col].clip(*df[col].quantile([0.01, 0.99]))  # winsorize 1%/99%")
        elif cfg.outliers == "remove_iqr":
            L.append("    df = df[df[col].between(low, high)]  # WARNING: deletes rows; justify this choice")
    L += ["", "# --- 4. Transformation --------------------------------------"]
    if cfg.transform == "log1p":
        L += ["for col in num_cols:",
              "    if df[col].min() >= 0 and abs(df[col].skew()) > 1:",
              "        df[col] = np.log1p(df[col])  # log(1 + x) handles zeros"]
    elif cfg.transform == "yeo_johnson":
        L += ["from sklearn.preprocessing import PowerTransformer",
              'df[num_cols] = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(df[num_cols])']
    else:
        L.append("# (skipped)")
    L += ["", "# --- 5. Scaling ---------------------------------------------"]
    if cfg.scaling != "none":
        cls = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}[cfg.scaling]
        L += [f"from sklearn.preprocessing import {cls}", f"df[num_cols] = {cls}().fit_transform(df[num_cols])"]
    else:
        L.append("# (skipped)")
    L += ["", "# --- 6. Encoding --------------------------------------------"]
    if cfg.encoding == "onehot":
        L.append("df = pd.get_dummies(df, columns=cat_cols, dtype=int)")
    elif cfg.encoding == "ordinal":
        L += ["for col in cat_cols:", '    df[col] = df[col].astype("category").cat.codes  # arbitrary order!']
    elif cfg.encoding == "frequency":
        L += ["for col in cat_cols:", "    df[col] = df[col].map(df[col].value_counts(normalize=True))"]
    else:
        L.append("# (skipped)")
    L += ["", 'df.to_csv("cleaned.csv", index=False)', 'print(df.shape)']
    return "\n".join(L)


def generate_sklearn_code(cfg: PipelineConfig, num: list[str], cat: list[str], target: str | None) -> str:
    imp_num = {"mean": 'SimpleImputer(strategy="mean")', "median": 'SimpleImputer(strategy="median")',
               "median_indicator": 'SimpleImputer(strategy="median", add_indicator=True)',
               "knn": "KNNImputer(n_neighbors=5)", "iterative": "IterativeImputer(random_state=42)"}.get(cfg.missing_num)
    imp_cat = {"mode": 'SimpleImputer(strategy="most_frequent")',
               "label": 'SimpleImputer(strategy="constant", fill_value="Missing")'}.get(cfg.missing_cat)
    scaler = {"standard": "StandardScaler()", "minmax": "MinMaxScaler()", "robust": "RobustScaler()"}.get(cfg.scaling)
    enc = {"onehot": 'OneHotEncoder(handle_unknown="ignore")',
           "ordinal": 'OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)'}.get(cfg.encoding)
    num_steps = []
    if imp_num:
        num_steps.append(f'("impute", {imp_num})')
    if cfg.transform == "yeo_johnson":
        num_steps.append('("power", PowerTransformer(method="yeo-johnson"))')
    elif cfg.transform == "log1p":
        num_steps.append('("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))')
    if scaler:
        num_steps.append(f'("scale", {scaler})')
    cat_steps = []
    if imp_cat:
        cat_steps.append(f'("impute", {imp_cat})')
    if enc:
        cat_steps.append(f'("encode", {enc})')
    t = target or "target"
    L = ["# Leakage-safe version: every statistic is learned on X_train only.",
         "import numpy as np",
         "import pandas as pd",
         "from sklearn.compose import ColumnTransformer",
         "from sklearn.experimental import enable_iterative_imputer  # noqa: F401",
         "from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer",
         "from sklearn.model_selection import train_test_split",
         "from sklearn.pipeline import Pipeline",
         "from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,",
         "                                   PowerTransformer, RobustScaler, StandardScaler)",
         "",
         'df = pd.read_csv("data.csv").drop_duplicates()' if cfg.duplicates != "none" else 'df = pd.read_csv("data.csv")',
         f"num_cols = {num!r}",
         f"cat_cols = {cat!r}",
         f'X, y = df[num_cols + cat_cols], df["{t}"]',
         "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
         "",
         "numeric = Pipeline([" + ", ".join(num_steps) + "])" if num_steps else 'numeric = "passthrough"',
         "categorical = Pipeline([" + ", ".join(cat_steps) + "])" if cat_steps else 'categorical = "passthrough"',
         'preprocess = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])',
         "",
         "X_train_ready = preprocess.fit_transform(X_train)  # fit on train ONLY",
         "X_test_ready = preprocess.transform(X_test)        # reuse train statistics",
         "# Outlier rules and row deletions are data decisions: apply them after review,",
         "# and never let them use information from the test set."]
    return "\n".join(L)
