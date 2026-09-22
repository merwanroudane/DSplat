import io
import json

import pandas as pd
import streamlit as st

from components.callouts import real_world, why
from components.code_lab import explain_code
from components.dataset_viewer import overview_metrics
from config import MAX_UPLOAD_MB
from core.page import page_footer, page_header
from utils.data_loader import initial_warnings, load_bytes
from utils.datasets import customers_raw, survey
from utils.missing import missing_table

page_header("importing_data")

st.markdown("## ارفع ملفك أو استخدم مثالًا")
st.caption(f"الصيغ: CSV, TXT, XLSX, JSON · الحد الأقصى {MAX_UPLOAD_MB} MB · الملف يبقى في جلستك فقط ولا يُرسل لأي مكان.")
source = st.segmented_control("المصدر", ["رفع ملف", "مثال: CSV بترميز cp1256 وفاصل ;", "مثال: JSON متداخل"],
                              default="مثال: CSV بترميز cp1256 وفاصل ;", key="imp_src")

raw, fname = None, None
if source == "رفع ملف":
    up = st.file_uploader("اختر ملفًا", type=["csv", "txt", "xlsx", "xls", "json"], key="imp_file")
    if up is not None:
        raw, fname = up.getvalue(), up.name
elif source and source.startswith("مثال: CSV"):
    demo = survey().head(40)[["respondent_id", "age", "region", "income", "work_hours"]].copy()
    demo.insert(2, "ملاحظة", ["عميل جديد" if i % 3 == 0 else "متابعة" for i in range(len(demo))])
    buf = io.StringIO()
    demo.to_csv(buf, sep=";", index=False)
    text = buf.getvalue() + "R9999;broken;row;with;too;many;fields;x\n"
    raw, fname = text.encode("cp1256"), "survey_export.csv"
elif source:
    records = [{"order_id": f"O{i}", "customer": {"id": f"C{i % 5}", "city": ["Rabat", "Cairo", "Amman"][i % 3]},
                "amount": round(20 + i * 3.5, 2), "tags": ["promo"] if i % 4 == 0 else []} for i in range(1, 25)]
    raw, fname = json.dumps(records, ensure_ascii=False).encode("utf-8"), "orders.json"

if raw is not None:
    res = load_bytes(raw, fname)
    for e in res.errors:
        st.error(e, icon=":material/cancel:")
    if res.ok:
        df = res.df
        st.success(f"قُرئ «{fname}» بنجاح.", icon=":material/check_circle:")
        info_bits = [f"{k}: {v}" for k, v in res.info.items() if k != "sheets"]
        st.caption(" · ".join(info_bits))
        for w in res.warnings:
            st.warning(w, icon=":material/warning:")
        overview_metrics(df)
        t1, t2, t3, t4 = st.tabs(["معاينة Preview", "الأعمدة والأنواع", "الفقد والتكرار", "تحذيرات أولية"])
        with t1:
            st.dataframe(df.head(20))
        with t2:
            st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(),
                                       "unique": df.nunique(), "memory_KB": (df.memory_usage(deep=True, index=False) / 1024).round(1)}))
        with t3:
            st.dataframe(missing_table(df))
            st.metric("صفوف مكررة تمامًا", int(df.duplicated().sum()))
        with t4:
            warns = initial_warnings(df)
            if warns:
                for w in warns:
                    st.markdown(f"- {w}")
            else:
                st.markdown("لا تحذيرات أولية.")
        if st.button("استخدم هذه البيانات في مستكشف البيانات والمختبرات", icon=":material/upload:", type="primary"):
            st.session_state["uploaded_df"] = df
            st.session_state["uploaded_name"] = fname
            st.toast("أصبحت البيانات متاحة في قوائم الاختيار كـ«ملفك المرفوع».", icon=":material/check_circle:")

st.markdown("## ماذا تفحص المنصة عند الاستيراد؟")
st.markdown(
    "1. **الامتداد والحجم** قبل القراءة.\n"
    "2. **الملف الفارغ**.\n"
    "3. **الترميز Encoding**: UTF-8 ← UTF-8-SIG ← Windows-1256 (عربي) ← Latin-1.\n"
    "4. **الفاصل Delimiter** بـ`csv.Sniffer` (`,` `;` TAB `|`).\n"
    "5. **الأسطر المعطوبة Malformed rows** (عدد حقول مختلف) وتُعدّ ولا تُخفى.\n"
    "6. **الترويسة**: أعمدة مكررة، بلا اسم، أو ترويسة مفقودة.\n"
    "7. **JSON**: تسطيح الحقول المتداخلة واختيار القائمة داخل الكائن."
)
why("حدد الأنواع والقيم المفقودة صراحة عند القراءة في المشاريع الحقيقية (dtype, na_values, parse_dates).",
    "الاستنتاج التلقائي يتغير بين ملف وآخر، ويحذف الأصفار البادئة من الرموز، ولا يعرف أن -999 تعني «مفقود».")

st.markdown("## الكود المكافئ")
explain_code(
    'df = pd.read_csv("file.csv", sep=";", encoding="cp1256", na_values=["-999", "N/A"])',
    [("pd.read_csv", "دالة قراءة الملفات النصية المفصولة"), ('sep=";"', "الفاصل بين الحقول"),
     ('encoding="cp1256"', "ترميز Windows العربي للملفات القديمة"),
     ("na_values", "قيم نصية تُعامل كمفقودة عند القراءة")],
    output="DataFrame", interpretation="تحقق فورًا بـdf.shape وdf.dtypes وdf.isna().sum() قبل أي تحليل.")
explain_code("df.info(memory_usage='deep')", [("info", "ملخص: عدد الصفوف، الأعمدة، غير المفقود، الأنواع"),
                                              ("memory_usage='deep'", "يحسب الذاكرة الفعلية للنصوص")],
             output="نص مطبوع (لا يعيد DataFrame)")

with st.expander("جرّب: بيانات التحدي الخام", icon=":material/science:"):
    raw_df = customers_raw()
    overview_metrics(raw_df)
    st.markdown("**تحذيرات أولية:**\n" + "\n".join(f"- {w}" for w in initial_warnings(raw_df)))

real_world(["هل عدد الصفوف يطابق ما يتوقعه مالك البيانات؟", "هل ظهرت النصوص العربية بشكل صحيح؟",
            "هل الأعمدة الرقمية قُرئت أرقامًا؟", "هل تجاهلت القراءة أسطرًا بصمت؟"])

page_footer("importing_data",
            takeaways=["تحقق من الترميز والفاصل والترويسة قبل الوثوق بأي رقم.", "عدّ الأسطر المعطوبة ولا تتجاهلها بصمت.",
                       "الأنواع المستنتجة تلقائيًا تحتاج مراجعة."],
            mistakes=["قراءة ملف عربي قديم بـUTF-8 دون تحقق.", "تجاهل التحذيرات الأولية.", "افتراض أن الصف الأول دائمًا ترويسة."])
