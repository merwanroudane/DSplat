import pandas as pd
import streamlit as st

from content.glossary import GLOSSARY
from core.curriculum import get_module
from core.page import footer, page_header

page_header("glossary")
df = pd.DataFrame(GLOSSARY, columns=["English", "العربية", "التعريف", "module"])
df["الوحدة"] = df["module"].map(lambda m: get_module(m).title_ar)
q = st.text_input("ابحث في المسرد (عربي أو إنجليزي)", key="gl_q", placeholder="Imputation، التسرب، Cardinality…")
letters = sorted({e[0].upper() for e in df["English"]})
letter = st.pills("الحرف الأول", letters, key="gl_letter")
view = df
if q.strip():
    ql = q.strip().lower()
    view = view[view.apply(lambda r: ql in (r["English"] + " " + r["العربية"] + " " + r["التعريف"]).lower(), axis=1)]
if letter:
    view = view[view["English"].str.upper().str.startswith(letter)]
st.caption(f"{len(view)} من {len(df)} مصطلحًا")
st.dataframe(view[["English", "العربية", "التعريف", "الوحدة"]], hide_index=True, height=560,
             column_config={"التعريف": st.column_config.TextColumn(width="large")})
st.download_button("حمّل المسرد CSV", df.drop(columns="module").to_csv(index=False).encode("utf-8-sig"), "glossary_ar_en.csv",
                   "text/csv", icon=":material/download:")
footer()
