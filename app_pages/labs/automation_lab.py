import pandas as pd
import streamlit as st

from components.callouts import why
from components.cards import comparison_table
from components.dataset_viewer import dataset_card, dataset_picker
from core.page import footer, page_header
from core.state import log_decision
from utils.cleaning import apply_action, compare_to_truth
from utils.datasets import customers_clean
from utils.profiling import detect_issues, issues_frame
from utils.quality import DEFAULT_WEIGHTS, quality_report, weighted_score
from utils.validation import CUSTOMER_RULES

page_header("automation_lab")
name, df = dataset_picker("al_ds", default="customers_raw", allowed=["customers_raw", "survey", "credit", "students"])
dataset_card(name)
mode = st.segmented_control("النمط", ["Manual", "Automated", "Hybrid"], default="Hybrid", key="al_mode")
comparison_table([
    {"النمط": "Manual", "من يكتشف؟": "أنت", "من يقرر؟": "أنت", "من ينفذ؟": "أنت (إجراءً إجراءً)"},
    {"النمط": "Automated", "من يكتشف؟": "المنصة", "من يقرر؟": "المنصة", "من ينفذ؟": "المنصة (كل الاقتراحات)"},
    {"النمط": "Hybrid", "من يكتشف؟": "المنصة", "من يقرر؟": "أنت (تعتمد أو ترفض)", "من ينفذ؟": "المنصة + سجل قرارات"},
])

ACTIONS = {"sentinel_to_nan": "Sentinels ← NaN", "coerce_numeric": "تحويل إلى رقم", "parse_dates": "تحويل إلى تاريخ",
           "strip_whitespace": "حذف المسافات", "normalize_case": "توحيد التسميات", "negative_to_nan": "السالب ← NaN",
           "cap_iqr": "قص عند 3×IQR", "log_transform": "إضافة log1p", "impute_median": "تعويض بالوسيط + مؤشر",
           "impute_missing_label": "فئة «Missing»", "group_rare": "دمج النادر في Other", "drop_exact_duplicates": "حذف التكرار التام",
           "dedupe_key": "حذف تكرار المفتاح", "drop_column": "حذف العمود"}
ORDER = ["drop_exact_duplicates", "dedupe_key", "sentinel_to_nan", "coerce_numeric", "parse_dates", "strip_whitespace",
         "normalize_case", "negative_to_nan", "cap_iqr", "impute_median", "impute_missing_label", "group_rare", "log_transform",
         "drop_column"]


def apply_many(data: pd.DataFrame, actions: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[dict]]:
    log = []
    for code in ORDER:  # a sensible execution order: structure → types → values → imputation
        for col, act in actions:
            if act == code:
                data, n = apply_action(data, col, act)
                log.append({"column": col, "action": ACTIONS.get(act, act), "affected": n})
    return data, log


def summary(data: pd.DataFrame) -> dict:
    out = {"rows": len(data), "missing cells": int(data.isna().sum().sum()), "issues": len(detect_issues(data))}
    if name == "customers_raw":
        q = quality_report(data, CUSTOMER_RULES, "customer_id")
        out["quality %"] = round(100 * weighted_score(q, DEFAULT_WEIGHTS), 1)
        if "customer_id" in data:
            acc = compare_to_truth(data.drop_duplicates("customer_id"), customers_clean(), "customer_id",
                                   [c for c in ["age", "country", "gender", "annual_income", "monthly_spend", "height_cm"] if c in data])
            out["match truth %"] = round(acc["matches_truth_%"].mean(), 1)
    return out


issues = detect_issues(df)
f = issues_frame(issues)
f = f[f["action_code"].isin(ACTIONS)].reset_index(drop=True)
before = summary(df)

if mode == "Manual":
    st.markdown("### أنت تفحص وتقرر وتنفذ")
    st.caption("لا اقتراحات آلية هنا: اختر العمود والإجراء بنفسك بناءً على فحصك (استخدم مستكشف البيانات أو وحدات الجودة).")
    st.session_state.setdefault(f"al_manual_{name}", [])
    c1, c2, c3 = st.columns([2, 2, 1])
    col = c1.selectbox("العمود", ["(all columns)"] + list(df.columns), key="al_mcol")
    act = c2.selectbox("الإجراء", list(ACTIONS), format_func=ACTIONS.get, key="al_mact")
    if c3.button("أضف", key="al_madd", icon=":material/add:"):
        st.session_state[f"al_manual_{name}"].append((col, act))
    chosen = st.session_state[f"al_manual_{name}"]
    if chosen:
        st.dataframe(pd.DataFrame(chosen, columns=["column", "action"]).assign(action=lambda t: t["action"].map(ACTIONS)),
                     hide_index=True)
        if st.button("امسح القائمة", key="al_mclear"):
            st.session_state[f"al_manual_{name}"] = []
            st.rerun()
    result, log = apply_many(df, chosen)
elif mode == "Automated":
    st.markdown("### المنصة تكتشف وتقرر وتنفذ كل شيء")
    chosen = [(r.column, r.action_code) for r in f.itertuples() if r.action_code != "none"]
    result, log = apply_many(df, chosen)
    st.warning("النمط الآلي طبّق **كل** الاقتراحات دون تمييز، بما فيها قص القيم المتطرفة الحقيقية ودمج الفئات النادرة "
               "والتعويض قبل فهم آلية الفقد. قارن النتائج بالنمط الهجين.", icon=":material/warning:")
else:
    st.markdown("### المنصة تكتشف وتقترح — أنت تعتمد")
    st.caption("ضع علامة على الإجراءات التي تعتمدها. الافتراضي: اعتماد المشكلات «المكتشفة» فقط، وترك «المحتملة» للتحقيق.")
    table = f[["column", "issue", "kind_ar", "severity_ar", "evidence", "why", "action_code"]].copy()
    table.insert(0, "approve", f["kind"].eq("detected"))
    table["action"] = table["action_code"].map(ACTIONS)
    edited = st.data_editor(table.drop(columns=["action_code"]), hide_index=True, key=f"al_editor_{name}",
                            disabled=["column", "issue", "kind_ar", "severity_ar", "evidence", "why", "action"],
                            column_config={"approve": st.column_config.CheckboxColumn("اعتماد"), "kind_ar": "النوع",
                                           "severity_ar": "الخطورة", "evidence": "الدليل", "why": "لماذا؟"}, height=380)
    approved = edited["approve"].to_numpy()
    chosen = [(c, a) for c, a, ok in zip(f["column"], f["action_code"], approved) if ok]
    result, log = apply_many(df, chosen)
    if st.button("سجّل قراراتي", key="al_hlog", icon=":material/bookmark:"):
        for r, ok in zip(f.itertuples(), approved):
            log_decision("automation_lab", f"{r.column}: {r.issue}", "approved" if ok else "rejected / investigate", r.evidence)
        st.toast("سُجّلت القرارات في سجل الإنسان في الحلقة.")

st.markdown("### النتيجة")
if log:
    st.dataframe(pd.DataFrame(log), hide_index=True)
after = summary(result)
cmp = pd.DataFrame({"قبل": before, "بعد": after})
st.dataframe(cmp)
st.session_state.setdefault("al_scores", {})
st.session_state["al_scores"][f"{name}:{mode}"] = after
scores = {k.split(":")[1]: v for k, v in st.session_state["al_scores"].items() if k.startswith(f"{name}:")}
if len(scores) > 1:
    st.markdown("**مقارنة الأنماط التي جربتها على هذه البيانات:**")
    st.dataframe(pd.DataFrame(scores))
why("قارن «match truth %» بين الأنماط الثلاثة على بيانات التحدي.",
    "الآلي الكامل يرفع درجات الجودة الظاهرية (لا فقد، لا قيم متطرفة) لكنه قد يبتعد عن الحقيقة؛ الهجين يصحح الأخطاء الواضحة "
    "ويترك الحالات الغامضة للتحقيق.")
st.download_button("حمّل النتيجة CSV", result.to_csv(index=False).encode("utf-8-sig"), f"{name}_{(mode or 'hybrid').lower()}.csv",
                   "text/csv", icon=":material/download:")
footer()
