import pandas as pd
import streamlit as st

from content.quizzes import QUIZZES
from core.curriculum import GROUPS, MODULES, get_module, tracked_modules
from core.page import footer, page_header
from core.state import export_progress, import_progress, progress_pct

page_header("progress")
completed = st.session_state.get("completed", set())
with st.container(horizontal=True):
    st.metric("نسبة الإنجاز", f"{progress_pct():.0f}%", border=True)
    st.metric("وحدات مكتملة", f"{len(completed)} / {len(tracked_modules())}", border=True)
    st.metric("اختبارات", f"{len(st.session_state.get('quiz_scores', {}))} / {len(QUIZZES)}", border=True)
    st.metric("تمارين نموذجية اطلعت عليها", len(st.session_state.get("exercises", set())), border=True)
    st.metric("مختبرات زرتها", len(st.session_state.get("labs_visited", set())), border=True)

st.markdown("## حسب المجموعة")
rows = []
for g, label in GROUPS.items():
    mods = [m for m in MODULES if m.group == g and m.tracked]
    if mods:
        rows.append({"المجموعة": label, "مكتمل": sum(m.id in completed for m in mods), "المجموع": len(mods)})
df = pd.DataFrame(rows)
df["%"] = (100 * df["مكتمل"] / df["المجموع"]).round(0)
st.dataframe(df, hide_index=True, column_config={"%": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})

st.markdown("## نتائج الاختبارات")
qs = st.session_state.get("quiz_scores", {})
if qs:
    st.dataframe(pd.DataFrame([{"الوحدة": get_module(k).title_ar, "النتيجة": f"{v['score']} / {v['total']}", "الوقت (UTC)": v["at"]}
                               for k, v in qs.items()]), hide_index=True)
else:
    st.info("لم تُجرِ أي اختبار بعد. الاختبارات في نهاية كل وحدة.")

st.markdown("## المفضلة")
marks = st.session_state.get("bookmarks", set())
if marks:
    for mid in sorted(marks):
        m = get_module(mid)
        st.page_link(m.file, label=m.title_ar, icon=":material/bookmark:")
else:
    st.caption("أضف وحدات إلى المفضلة من زر «أضف إلى المفضلة» في رأس كل وحدة.")

st.markdown("## حفظ واستعادة التقدم")
st.caption("التقدم يُحفظ في جلستك فقط. صدّره كملف JSON واستورده لاحقًا على أي جهاز — لا حسابات ولا خوادم.")
c1, c2 = st.columns(2)
c1.download_button("صدّر التقدم JSON", export_progress().encode("utf-8"), "ds_academy_progress.json", "application/json",
                   icon=":material/download:")
up = c2.file_uploader("استورد ملف تقدم", type=["json"], key="prog_up")
if up is not None and st.session_state.get("prog_loaded") != up.file_id:
    ok, msg = import_progress(up.getvalue().decode("utf-8", errors="replace"))
    st.session_state["prog_loaded"] = up.file_id
    (st.success if ok else st.error)(msg)
footer()
