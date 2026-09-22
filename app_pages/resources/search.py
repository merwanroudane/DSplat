import re

import streamlit as st

from content.glossary import GLOSSARY
from core.curriculum import MODULES, get_module
from core.page import footer, page_header

page_header("search")

_DIAC = re.compile(r"[ً-ْـ]")


def norm(t: str) -> str:
    t = _DIAC.sub("", t.lower())
    return re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")


q = st.text_input("ابحث عن موضوع أو مصطلح", placeholder="مثال: Missing Values، القيم الشاذة، Scaling، Leakage، ترميز",
                  key="search_q")
st.caption("اقتراحات: " + " · ".join(["Missing Values", "Outliers", "Scaling", "Encoding", "Data Mining", "Leakage", "p-value", "تسرب"]))
if q.strip():
    terms = [norm(t) for t in q.split() if t.strip()]
    results = []
    for m in MODULES:
        fields = {"title": norm(m.title_ar + " " + m.title_en), "keywords": norm(" ".join(m.keywords)),
                  "desc": norm(m.description + " " + " ".join(m.objectives))}
        score = sum(3 * fields["title"].count(t) + 2 * fields["keywords"].count(t) + fields["desc"].count(t) for t in terms)
        if score:
            results.append((score, m))
    results.sort(key=lambda r: -r[0])
    gl = [g for g in GLOSSARY if all(t in norm(" ".join(g[:3])) for t in terms)]
    st.markdown(f"### الدروس والمختبرات ({len(results)})")
    for _, m in results[:15]:
        with st.container(border=True):
            st.page_link(m.file, label=f"{m.title_ar} · {m.title_en}", icon=m.icon)
            st.caption(m.description)
    if not results:
        st.info("لا نتائج في الدروس. جرّب مصطلحًا بالإنجليزية أو بالعربية بصيغة أخرى.")
    st.markdown(f"### المسرد ({len(gl)})")
    for en, ar, definition, mid in gl[:20]:
        st.markdown(f"**{ar}** · *{en}* — {definition}")
        st.page_link(get_module(mid).file, label=f"الوحدة: {get_module(mid).title_ar}", icon=":material/arrow_back:")
footer()
