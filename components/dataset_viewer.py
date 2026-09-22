"""Dataset picker and quick overview, shared by labs and lessons."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from utils.datasets import DATASET_INFO, load_dataset
from utils.profiling import overview

UPLOADED = "__uploaded__"


def _label(name: str) -> str:
    if name == UPLOADED:
        return f"ملفك المرفوع: {st.session_state.get('uploaded_name') or ''}"
    return DATASET_INFO[name]["title"]


def dataset_picker(key: str, default: str = "customers_raw", allowed: Sequence[str] | None = None,
                   allow_uploaded: bool = True, label: str = "اختر مجموعة البيانات") -> tuple[str, pd.DataFrame]:
    options = list(allowed or DATASET_INFO)
    if allow_uploaded and st.session_state.get("uploaded_df") is not None:
        options = [UPLOADED] + options
    index = options.index(default) if default in options else 0
    name = st.selectbox(label, options, index=index, format_func=_label, key=key)
    if name == UPLOADED:
        return name, st.session_state["uploaded_df"].copy()
    return name, load_dataset(name)


def dataset_card(name: str) -> None:
    if name == UPLOADED:
        st.caption("بيانات مرفوعة من المستخدم — تبقى داخل جلستك فقط ولا تُرسل إلى أي خدمة خارجية.")
        return
    info = DATASET_INFO[name]
    with st.expander(f"عن هذه البيانات: {info['title']}", icon=":material/info:"):
        st.markdown(f"**الغرض:** {info['purpose']}  \n**التصميم:** {info['design']}")
        st.markdown("**المتغيرات:**\n" + "\n".join(f"- `{k}`: {v}" for k, v in info["variables"].items()))
        st.markdown("**المشكلات المقصودة:**\n" + "\n".join(f"- {i}" for i in info["issues"]))


def overview_metrics(df: pd.DataFrame) -> None:
    o = overview(df)
    with st.container(horizontal=True):
        st.metric("الصفوف", f"{o['rows']:,}", border=True)
        st.metric("الأعمدة", o["columns"], border=True)
        st.metric("خلايا مفقودة", f"{o['missing_cells']:,}", f"{o['missing_pct']:.1f}%", delta_color="off",
                  border=True)
        st.metric("صفوف مكررة", o["duplicate_rows"], border=True)
        st.metric("الذاكرة", f"{o['memory_mb']:.2f} MB", border=True)
