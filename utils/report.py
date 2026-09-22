"""Automated analysis report (Markdown and standalone HTML).

The report separates *detected issues*, *potential issues* and
*recommendations*, and every recommendation carries its "why".
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from config import APP_AUTHOR_EN, APP_NAME_EN
from utils.profiling import (categorical_summary, column_summary, detect_issues, iqr_outlier_counts, numeric_summary,
                             overview, recommended_next_steps, top_correlations)


def build_report(df: pd.DataFrame, name: str) -> dict:
    issues = detect_issues(df)
    return {
        "name": name,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overview": overview(df),
        "columns": column_summary(df),
        "numeric": numeric_summary(df),
        "categorical": categorical_summary(df),
        "outliers": iqr_outlier_counts(df),
        "correlations": top_correlations(df),
        "missing": df.isna().sum().rename("missing").to_frame().assign(pct=lambda t: (100 * t["missing"] / max(1, len(df))).round(2)),
        "detected": [i for i in issues if i.kind == "detected"],
        "potential": [i for i in issues if i.kind == "potential"],
        "recommendations": [i for i in issues if i.kind == "recommendation"],
        "next_steps": recommended_next_steps(df, issues),
    }


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    d = df.head(max_rows).copy()
    d = d.reset_index() if not isinstance(d.index, pd.RangeIndex) else d
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in d.iterrows():
        vals = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in vals) + " |")
    return "\n".join(out) + "\n"


def _issues_md(items) -> str:
    if not items:
        return "_None._\n"
    return "\n".join(f"- **{i.column}** — {i.issue} (severity: {i.severity}). Evidence: {i.evidence}. "
                     f"Suggested action: {i.action}. **Why?** {i.why}" for i in items) + "\n"


def to_markdown(r: dict) -> str:
    o = r["overview"]
    parts = [
        f"# Automated Data Analysis Report — {r['name']}",
        f"_Generated {r['generated']} by {APP_NAME_EN} ({APP_AUTHOR_EN}). Automated findings require human review._",
        "## 1. Overview",
        f"- Rows: {o['rows']:,}\n- Columns: {o['columns']}\n- Numeric / categorical: {o['numeric']} / {o['categorical']}\n"
        f"- Missing cells: {o['missing_cells']:,} ({o['missing_pct']:.2f}%)\n- Exact duplicate rows: {o['duplicate_rows']}\n"
        f"- Memory: {o['memory_mb']:.2f} MB",
        "## 2. Types (stored vs semantic)", _md_table(r["columns"][["column", "dtype", "semantic_type", "missing_%", "unique"]]),
        "## 3. Missingness", _md_table(r["missing"][r["missing"]["missing"] > 0]),
        "## 4. Duplicates", f"{o['duplicate_rows']} exact duplicate rows.",
        "## 5. Numeric summary", _md_table(r["numeric"]),
        "## 6. Categorical summary", _md_table(r["categorical"]),
        "## 7. Outliers (1.5×IQR, for inspection only)", _md_table(r["outliers"]),
        "## 8. Strongest correlations (Spearman)", _md_table(r["correlations"]),
        "## 9. Detected issues", _issues_md(r["detected"]),
        "## 10. Potential issues (investigate)", _issues_md(r["potential"]),
        "## 11. Recommendations", _issues_md(r["recommendations"]),
        "## 12. Suggested next steps", "\n".join(f"{k + 1}. {s}" for k, s in enumerate(r["next_steps"])),
    ]
    return "\n\n".join(parts) + "\n"


def to_html(r: dict) -> str:
    """Self-contained HTML (no external resources)."""
    def table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "<p><em>None</em></p>"
        return df.head(40).to_html(classes="t", border=0, float_format=lambda v: f"{v:.4g}")

    def issues(items) -> str:
        if not items:
            return "<p><em>None.</em></p>"
        li = "".join(f"<li><b>{html.escape(str(i.column))}</b> — {html.escape(i.issue)} "
                     f"<span class='sev {i.severity}'>{i.severity}</span><br>Evidence: {html.escape(i.evidence)}<br>"
                     f"Action: {html.escape(i.action)}<br><b>Why?</b> {html.escape(i.why)}</li>" for i in items)
        return f"<ul>{li}</ul>"

    o = r["overview"]
    css = ("body{font-family:'Segoe UI',Tahoma,sans-serif;background:#FFFCF5;color:#212529;max-width:1100px;margin:auto;padding:24px}"
           "h1{color:#1971C2}h2{border-bottom:2px solid #C5D5F0;padding-bottom:4px;margin-top:32px}"
           ".t{border-collapse:collapse;font-size:13px}.t th,.t td{border:1px solid #DDE3EE;padding:4px 8px}.t th{background:#F1F3F9}"
           ".sev{border-radius:8px;padding:1px 8px;font-size:12px;color:#fff}.high{background:#F08C00}.medium{background:#F76707}"
           ".low{background:#F59F00}.info{background:#1C7ED6}li{margin-bottom:10px}.note{background:#FFF9DB;padding:10px;border-radius:8px}")
    body = [
        f"<h1>Automated Data Analysis Report — {html.escape(r['name'])}</h1>",
        f"<p class='note'>Generated {r['generated']} by {APP_NAME_EN} ({APP_AUTHOR_EN}). "
        "Automated findings are <b>suggestions</b> that require human review; nothing was modified.</p>",
        "<h2>1. Overview</h2>",
        f"<p>Rows: {o['rows']:,} · Columns: {o['columns']} · Missing cells: {o['missing_cells']:,} ({o['missing_pct']:.2f}%) · "
        f"Duplicate rows: {o['duplicate_rows']} · Memory: {o['memory_mb']:.2f} MB</p>",
        "<h2>2. Types</h2>", table(r["columns"][["column", "dtype", "semantic_type", "missing_%", "unique"]]),
        "<h2>3. Missingness</h2>", table(r["missing"][r["missing"]["missing"] > 0]),
        "<h2>4. Numeric summary</h2>", table(r["numeric"]),
        "<h2>5. Categorical summary</h2>", table(r["categorical"]),
        "<h2>6. Outliers (1.5×IQR)</h2>", table(r["outliers"]),
        "<h2>7. Correlations</h2>", table(r["correlations"]),
        "<h2>8. Detected issues</h2>", issues(r["detected"]),
        "<h2>9. Potential issues</h2>", issues(r["potential"]),
        "<h2>10. Recommendations</h2>", issues(r["recommendations"]),
        "<h2>11. Next steps</h2><ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in r["next_steps"]) + "</ol>",
    ]
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Data report</title><style>{css}</style></head>" \
           f"<body>{''.join(body)}</body></html>"
