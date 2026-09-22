import io
import json

import pandas as pd
import streamlit as st

from components.callouts import real_world, researcher_note, warning
from components.cards import comparison_table
from components.code_lab import code_lab
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import customers_clean

page_header("data_sources")

st.markdown("## مقارنة المصادر")
aspect = st.segmented_control("قارن من حيث", ["Structure", "Reliability", "Typical problems", "Privacy"],
                              default="Typical problems", key="ds_aspect")
ROWS = [
    ("CSV", "جدول نصي مسطح", "متوسطة؛ لا أنواع", "فواصل، ترميز، أصفار بادئة تُحذف", "قد يحوي PII بلا حماية"),
    ("Excel", "جداول بأوراق وتنسيق", "متوسطة", "خلايا مدمجة، تواريخ كأرقام، ترويسات متعددة", "يُتداول بالبريد كثيرًا"),
    ("JSON", "شبه منظم متداخل", "جيدة بمخطط", "حقول اختيارية، تداخل عميق", "حسب المحتوى"),
    ("Parquet", "عمودي مضغوط بأنواع", "عالية", "إصدارات مخطط مختلفة", "حسب المحتوى"),
    ("SQL database", "جداول مترابطة بمفاتيح", "عالية", "دمج خاطئ، استعلام يغيّر العينة", "صلاحيات الوصول"),
    ("APIs", "JSON عبر HTTP", "حسب المزود", "حدود الطلبات، ترقيم الصفحات، تغيّر الإصدار", "مفاتيح API سرية"),
    ("Web scraping", "HTML غير منظم", "منخفضة-متوسطة", "تغيّر بنية الصفحة، بيانات ناقصة", "شروط الاستخدام وحقوق النشر"),
    ("Surveys", "استجابات بشرية", "تتأثر بالتحيز", "عدم الاستجابة، تحيز الرغبة الاجتماعية", "موافقة المشاركين"),
    ("Experiments", "بيانات مصممة عشوائيًا", "عالية للاستدلال السببي", "تسرب المشاركين، عينة صغيرة", "موافقة ولجان أخلاقيات"),
    ("Administrative data", "سجلات مؤسسية", "عالية التغطية", "تعريفات تتغير، أُنشئت لغرض آخر", "حساسة غالبًا"),
    ("Sensors / IoT", "سلاسل زمنية عالية التردد", "متغيرة", "انجراف المعايرة، انقطاع، ضوضاء", "بيانات موقع"),
    ("Public / open data", "متنوعة", "جيدة غالبًا", "تحديثات ومراجعات، توثيق ناقص", "مجهولة عادة"),
    ("Cloud warehouses", "جداول ضخمة موزعة", "عالية", "تكلفة الاستعلام، نسخ متعددة", "حوكمة الوصول"),
    ("Logs", "أحداث نصية/JSON", "عالية الحجم", "صيغ متعددة، مناطق زمنية", "عناوين IP ومعرّفات"),
    ("Streaming", "تدفق مستمر", "حسب النظام", "ترتيب الأحداث، التأخر، التكرار", "حسب المحتوى"),
]
idx = {"Structure": 1, "Reliability": 2, "Typical problems": 3, "Privacy": 4}[aspect or "Typical problems"]
label = {"Structure": "البنية", "Reliability": "الموثوقية", "Typical problems": "المشكلات النموذجية",
         "Privacy": "اعتبارات الخصوصية"}[aspect or "Typical problems"]
comparison_table([{"المصدر": r[0], label: r[idx]} for r in ROWS])

st.markdown("## الحجم وتكرار التحديث")
comparison_table([
    {"الحجم": "< 1 GB", "الأداة المناسبة": "pandas على جهاز واحد", "ملاحظة": "أغلب مشاريع التعليم والبحث"},
    {"الحجم": "1–100 GB", "الأداة المناسبة": "Parquet + Polars/DuckDB، أو قراءة على دفعات", "ملاحظة": "اقرأ الأعمدة المطلوبة فقط"},
    {"الحجم": "> 100 GB", "الأداة المناسبة": "Spark، مستودعات سحابية", "ملاحظة": "المعالجة الموزعة"},
])
st.markdown("تكرار التحديث: **دفعي Batch** (يومي/شهري) أو **لحظي Streaming**. البيانات الأحدث ليست دائمًا "
            "الأدق: بعض المصادر الحكومية تُراجَع لاحقًا (Revisions).")

st.markdown("## القراءة في pandas")
fmt = st.selectbox("الصيغة", ["CSV", "Excel", "JSON", "Parquet", "SQL"], key="ds_fmt")
snippets = {
    "CSV": 'df = pd.read_csv("data.csv", sep=",", encoding="utf-8",\n                 na_values=["-999", "N/A"], dtype={"zip_code": "string"})',
    "Excel": 'df = pd.read_excel("data.xlsx", sheet_name="Sales", skiprows=2)  # needs openpyxl',
    "JSON": 'import json\nwith open("data.json", encoding="utf-8") as f:\n    records = json.load(f)\ndf = pd.json_normalize(records)  # flatten nested fields',
    "Parquet": 'df = pd.read_parquet("data.parquet", columns=["id", "amount"])  # needs pyarrow',
    "SQL": 'import sqlite3\ncon = sqlite3.connect("shop.db")\ndf = pd.read_sql_query(\n    "SELECT * FROM orders WHERE order_date >= ?", con, params=("2024-01-01",))',
}
st.code(snippets[fmt], language="python")
if fmt == "SQL":
    warning("استخدم دائمًا الاستعلامات ذات المعاملات Parameterized queries (`?`) ولا تدمج مدخلات المستخدم في نص SQL "
            "مباشرة؛ ذلك يمنع SQL injection.")


def _roundtrip(fmt_choice: str):
    df = customers_clean().head(5)[["customer_id", "age", "country", "signup_date"]]
    if fmt_choice == "CSV":
        text = df.to_csv(index=False)
        back = pd.read_csv(io.StringIO(text))
        return [f"```\n{text}\n```", back.dtypes.astype(str).to_frame("dtype after reading")]
    text = df.to_json(orient="records", date_format="iso", force_ascii=False)
    back = pd.read_json(io.StringIO(text))
    return [f"```json\n{json.dumps(json.loads(text)[:2], ensure_ascii=False, indent=1)}\n```",
            back.dtypes.astype(str).to_frame("dtype after reading")]


code_lab("ds_rt", "رحلة ذهاب وعودة: ماذا يحدث للأنواع؟",
         lambda p: f"text = df.to_{p['fmt_choice'].lower()}(...)\nback = pd.read_{p['fmt_choice'].lower()}(...)\nback.dtypes",
         _roundtrip, params=lambda: {"fmt_choice": st.radio("الصيغة", ["CSV", "JSON"], horizontal=True, key="ds_rt_f")},
         explanation="CSV لا يحفظ الأنواع: التاريخ يعود نصًا ما لم تحدد parse_dates. هذا سبب تفضيل Parquet لتبادل البيانات بين الخطوات.")

st.markdown("## الجمع بالتصميم: الاستبيانات والتجارب")
st.markdown(
    "- **الاستبيان Survey:** يقيس ما يقوله الناس؛ عرضة لتحيز الاختيار، وعدم الاستجابة، وصياغة السؤال.\n"
    "- **التجربة Experiment:** توزيع عشوائي للمعالجة يسمح باستدلال سببي؛ مكلفة وقد تكون محدودة التعميم.\n"
    "- **البيانات الرصدية Observational:** متاحة بكثرة لكن الارتباط فيها لا يعني السببية."
)
if at_least("research"):
    researcher_note(["وثّق إطار المعاينة Sampling frame ومعدل الاستجابة لأي استبيان.",
                     "البيانات الإدارية أُنشئت لغرض آخر: تحقق من تغير التعريفات عبر السنوات قبل تحليل الاتجاهات.",
                     "للبيانات المجموعة بـscraping احفظ تاريخ الجمع ونسخة من الصفحات لإعادة الإنتاج."])
real_world(["من جمع البيانات ولأي غرض؟", "ما الذي لا تغطيه البيانات (من غائب)؟", "كم عمر البيانات وكم مرة تُحدَّث؟",
            "هل يُسمح قانونيًا وأخلاقيًا باستخدامها لهذا الغرض؟"])

page_footer("data_sources",
            takeaways=["لكل مصدر مشكلات نموذجية يجب توقعها قبل التحليل.", "CSV لا يحفظ الأنواع؛ Parquet يفعل.",
                       "طريقة الجمع تحدد نوع الاستنتاج الممكن (وصفي، تنبؤي، سببي)."],
            mistakes=["الوثوق بالأنواع المستنتجة تلقائيًا من CSV.", "تجاهل تحيز الاختيار في الاستبيانات الإلكترونية.",
                      "دمج مدخلات المستخدم في نص SQL."])
