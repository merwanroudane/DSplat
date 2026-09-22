"""Entry point: page config, theme, state, sidebar controls and navigation.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from config import APP_AUTHOR_AR, APP_NAME_AR, LEVELS
from core.navigation import build_pages
from core.state import init_state, progress_pct
from core.theme import inject_css, register_plotly_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title=APP_NAME_AR, page_icon=":material/school:", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
register_plotly_template()
init_state()

page = st.navigation(build_pages(), position="sidebar")

with st.sidebar:
    st.markdown(f"**{APP_NAME_AR}**")
    st.caption(f"تطوير: {APP_AUTHOR_AR}")
    st.segmented_control(
        "مستوى الشرح",
        options=list(LEVELS),
        format_func=LEVELS.get,
        key="level",
        required=True,
        help="مبتدئ: شرح مبسّط. متقدم: افتراضات وحالات حدّية. بحثي: ملاحظات منهجية وتوصيات للتقارير.",
    )
    st.toggle("تقليل الحركة (Reduced motion)", key="reduce_motion",
              help="يعطّل التشغيل التلقائي للرسوم المتحركة.")

page.run()

# Rendered after the page so completion toggles on this run are reflected.
st.sidebar.progress(progress_pct() / 100, text=f"التقدّم في المقرر: {progress_pct():.0f}%")
