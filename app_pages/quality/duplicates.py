import difflib

import pandas as pd
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import customers_raw, panel, transactions

page_header("duplicates")
raw = customers_raw()

st.markdown("## ليس كل تكرار خطأ")
comparison_table([
    {"النوع": "Exact duplicates", "الوصف": "صفوف متطابقة في كل الأعمدة", "خطأ؟": "غالبًا (إدخال مزدوج، دمج خاطئ)", "المعالجة": "drop_duplicates()"},
    {"النوع": "Key-level duplicates", "الوصف": "نفس المفتاح بقيم مختلفة", "خطأ؟": "تعارض يجب حله", "المعالجة": "قاعدة: الأحدث/الأكمل/مصدر الحقيقة"},
    {"النوع": "Partial duplicates", "الوصف": "تطابق في أعمدة مختارة", "خطأ؟": "حسب السياق", "المعالجة": "subset= مع فهم المعنى"},
    {"النوع": "Near duplicates", "الوصف": "اختلافات إملائية/تنسيقية", "خطأ؟": "غالبًا", "المعالجة": "تطبيع + مقاييس تشابه + مراجعة"},
    {"النوع": "Repeated measures", "الوصف": "نفس الوحدة تُقاس مرات", "خطأ؟": "لا — تصميم", "المعالجة": "مفتاح مركب (id, time)"},
    {"النوع": "Transaction repetition", "الوصف": "نفس العميل يشتري مرات", "خطأ؟": "لا — سلوك", "المعالجة": "لا حذف؛ تجميع حسب الحاجة"},
    {"النوع": "Duplicate IDs", "الوصف": "معرّف واحد لكيانين مختلفين", "خطأ؟": "نعم — خلل في توليد المعرّف", "المعالجة": "إعادة ترقيم بعد التحقيق"},
])

st.markdown("## التطبيق على بيانات العملاء الخام")
c1, c2, c3 = st.columns(3)
c1.metric("صفوف", len(raw))
c2.metric("تكرار تام", int(raw.duplicated().sum()))
c3.metric("تكرار customer_id", int(raw["customer_id"].duplicated().sum()))

st.markdown("### جرّب المعاملات: `subset` و`keep`")
cols = st.multiselect("subset (فارغ = كل الأعمدة)", list(raw.columns), default=["customer_id"], key="dup_subset")
keep = st.segmented_control("keep", ["first", "last", "False"], default="False", key="dup_keep")
keep_val = False if keep == "False" else (keep or "first")
mask = raw.duplicated(subset=cols or None, keep=keep_val)
st.markdown(f"`df.duplicated(subset={cols or None}, keep={keep_val!r})` ← **{int(mask.sum())}** صفًا معلّمًا بـTrue")
if keep_val is False and mask.any():
    grp = raw[mask].sort_values(cols or list(raw.columns))
    st.dataframe(grp.head(20), hide_index=False)
    st.caption("keep=False يعلّم كل النسخ، فتظهر المجموعات كاملة للمراجعة قبل الحذف. قارن monthly_spend داخل كل مجموعة: "
               "بعضها متطابق (تكرار تام) وبعضها متعارض (تكرار مفتاح).")
comparison_table([
    {"keep": "'first'", "المعنى": "علّم كل النسخ عدا الأولى (هي التي تبقى)"},
    {"keep": "'last'", "المعنى": "علّم كل النسخ عدا الأخيرة"},
    {"keep": "False", "المعنى": "علّم كل النسخ المكررة — للمراجعة لا للحذف"},
])

st.markdown("### التكرارات المتعارضة: أيها نُبقي؟")
dup_keys = raw.loc[raw["customer_id"].duplicated(keep=False), "customer_id"]
conflict = raw[raw["customer_id"].isin(dup_keys)].groupby("customer_id").filter(
    lambda g: g.drop(columns="customer_id").nunique(dropna=False).max() > 1)
if len(conflict):
    ex = conflict["customer_id"].iloc[0]
    view = conflict[conflict["customer_id"] == ex]
    diff_cols = [c for c in view.columns if view[c].nunique(dropna=False) > 1]
    st.dataframe(view[["customer_id"] + diff_cols], hide_index=True)
    why("لا تختر keep='first' عشوائيًا للتكرارات المتعارضة.",
        "الترتيب في الملف ليس بالضرورة زمنيًا. ارجع لمصدر الحقيقة أو لعمود «آخر تحديث»؛ وإن تعذر، وثّق القاعدة المختارة.")

st.markdown("## التكرار المشروع")
t1, t2 = st.tabs(["Transactions", "Panel"])
with t1:
    tr = transactions()
    top = tr.groupby(["customer_id", "product"]).size().sort_values(ascending=False).head(5)
    st.markdown(f"في بيانات المعاملات {len(tr):,} صفًا؛ `customer_id` يتكرر {int(tr['customer_id'].duplicated().sum()):,} مرة — "
                "**وهذا طبيعي**: العميل يشتري مرات عديدة.")
    st.dataframe(top.rename("purchases").reset_index(), hide_index=True)
with t2:
    p = panel()
    st.markdown(f"`firm_id` يتكرر {int(p['firm_id'].duplicated().sum())} مرة، لكن المفتاح الصحيح `(firm_id, year)` "
                f"مكرر {int(p.duplicated(subset=['firm_id', 'year']).sum())} مرة فقط.")
    st.code('assert not df.duplicated(subset=["firm_id", "year"]).any(), "duplicate firm-year!"', language="python")

st.markdown("## Near duplicates: التشابه بدل التطابق")
names = ["Ahmed Ali", "Ahmad Ali", "AHMED  ALI", "Mohamed Salah", "Mohammed Salah", "Sara Ben", "Sarah Ben", "Omar Idrissi"]
thr = st.slider("عتبة التشابه", 0.5, 1.0, 0.85, 0.01, key="dup_thr")


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


pairs = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        r = difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()
        if r >= thr:
            pairs.append({"A": a, "B": b, "similarity": round(r, 3)})
st.dataframe(pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["A", "B", "similarity"]), hide_index=True)
st.caption("`difflib.SequenceMatcher` بعد تطبيع المسافات وحالة الأحرف. عتبة منخفضة جدًا تدمج أشخاصًا مختلفين؛ "
           "عالية جدًا تفوّت التكرارات. الحالات قرب العتبة تحتاج مراجعة بشرية.")


def _dedupe(subset: str, keep_opt: str):
    d = customers_raw()
    sub = None if subset == "None" else [subset]
    out = d.drop_duplicates(subset=sub, keep=keep_opt)
    return f"الصفوف: {len(d)} ← {len(out)} (حُذف {len(d) - len(out)})"


code_lab("dup_lab", "drop_duplicates",
         lambda p: f"clean = df.drop_duplicates(subset={None if p['subset'] == 'None' else [p['subset']]}, keep='{p['keep_opt']}')",
         _dedupe, params=lambda: {"subset": st.radio("subset", ["None", "customer_id", "email"], horizontal=True, key="dup_s2"),
                                  "keep_opt": st.radio("keep", ["first", "last"], horizontal=True, key="dup_k2")},
         explanation="subset=None يحذف التكرار التام فقط. subset=['customer_id'] يحذف أيضًا التكرارات المتعارضة — قرار أقوى يحتاج تبريرًا.")

if at_least("advanced"):
    st.markdown("## متقدم: Record linkage على نطاق واسع")
    st.markdown("مقارنة كل الأزواج تكلف O(n²). الحل: **Blocking** (مقارنة داخل مجموعات مثل نفس الرمز البريدي)، ثم مقاييس "
                "تشابه (Jaro-Winkler، Levenshtein)، ثم تصنيف احتمالي للأزواج (نموذج Fellegi–Sunter).")
if at_least("research"):
    researcher_note(["أبلغ عن عدد التكرارات المحذوفة وقاعدة الحذف.", "في الاستبيانات: التكرار قد يعني مشاركة احتيالية (Bots).",
                     "في الدمج بين قواعد بيانات: قيّم دقة الربط بعينة مراجعة يدوية."])
real_world(["ما وحدة التحليل؟ (تحدد ما هو التكرار)", "هل يوجد عمود زمني يحدد النسخة الأحدث؟", "هل التكرار ناتج عن Join خاطئ؟",
            "كم صفًا تأثر، وهل يتركز في مصدر معين؟"])

page_footer("duplicates",
            takeaways=["التكرار يُعرّف بالنسبة لوحدة التحليل والمفتاح.", "keep=False للمراجعة، first/last للحذف بقاعدة موثقة.",
                       "التكرار المشروع (معاملات، Panel) لا يُحذف."],
            mistakes=["drop_duplicates على كل البيانات دون فهم.", "حذف التكرارات المتعارضة عشوائيًا.", "إغفال Near duplicates."])
