import streamlit as st

from components.dataset_viewer import dataset_card, dataset_picker
from core.page import footer, page_header
from utils.report import build_report, to_html, to_markdown

page_header("auto_report")
st.markdown("تقرير تحليلي آلي يميّز بوضوح بين **مشكلة مكتشفة** و**مشكلة محتملة** و**توصية**، وكل توصية معها «لماذا؟». "
            "التقرير لا يعدّل البيانات.")
name, df = dataset_picker("ar_ds", default="customers_raw")
dataset_card(name)


@st.cache_data(show_spinner="توليد التقرير…")
def make(data, label: str) -> tuple[str, str, dict]:
    r = build_report(data, label)
    return to_markdown(r), to_html(r), {"detected": len(r["detected"]), "potential": len(r["potential"]),
                                         "recommendations": len(r["recommendations"])}


label = st.session_state.get("uploaded_name") if name == "__uploaded__" else name
md, html_doc, counts = make(df, label or name)
with st.container(horizontal=True):
    st.metric("مشكلات مكتشفة Detected", counts["detected"], border=True)
    st.metric("مشكلات محتملة Potential", counts["potential"], border=True)
    st.metric("توصيات Recommendations", counts["recommendations"], border=True)
c1, c2 = st.columns(2)
c1.download_button("حمّل التقرير Markdown", md.encode("utf-8"), f"report_{name}.md", "text/markdown", icon=":material/download:",
                   type="primary")
c2.download_button("حمّل التقرير HTML", html_doc.encode("utf-8"), f"report_{name}.html", "text/html", icon=":material/download:")
st.caption("التقرير بالإنجليزية ليُتداول بسهولة في الفرق التقنية والمستودعات؛ الواجهة والشرح بالعربية.")
with st.container(border=True, key="ltr-report"):
    st.markdown(md)
footer()
