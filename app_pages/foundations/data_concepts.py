import pandas as pd
import streamlit as st

from components.callouts import intuition, real_world, researcher_note
from components.cards import comparison_table
from components.code_lab import code_lab, explain_code
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import customers_clean

page_header("data_concepts")
df = customers_clean()

st.markdown("## تشريح مجموعة البيانات · Anatomy of a dataset")
st.markdown(
    "**Dataset** مجموعة منظمة من **المشاهدات Observations** (الصفوف) تُقاس عليها **متغيرات Variables** (الأعمدة). "
    "كل مشاهدة تمثل **وحدة التحليل Unit of analysis**: عميل، معاملة، شركة-سنة، مريض-زيارة..."
)
view = df.head(6)[["customer_id", "age", "country", "monthly_spend", "num_orders", "churned"]]
role = st.segmented_control(
    "ظلّل دور العمود", ["Identifier", "Features", "Target"], default="Features", key="dc_role")
cols_map = {"Identifier": ["customer_id"], "Features": ["age", "country", "monthly_spend", "num_orders"],
            "Target": ["churned"]}
highlight = cols_map.get(role or "Features", [])
st.dataframe(view.style.map(lambda _: "background-color: #D3F9D8", subset=highlight), hide_index=True)
st.caption("صف = Observation / Record · عمود = Variable · خلية = Measurement (قيمة متغير لمشاهدة).")

st.markdown("## المصطلحات الأساسية")
comparison_table([
    {"المصطلح": "Observation / Record / Row", "المعنى": "وحدة واحدة مرصودة", "مثال": "العميل C0001"},
    {"المصطلح": "Variable / Column / Attribute", "المعنى": "خاصية تُقاس لكل مشاهدة", "مثال": "age"},
    {"المصطلح": "Feature / Predictor", "المعنى": "متغير مدخل للنموذج", "مثال": "monthly_spend"},
    {"المصطلح": "Target / Label / Response", "المعنى": "ما نتنبأ به أو نفسره", "مثال": "churned"},
    {"المصطلح": "Identifier (ID)", "المعنى": "يميّز السجل؛ لا يُستخدم كخاصية", "مثال": "customer_id"},
    {"المصطلح": "Index", "المعنى": "تسميات الصفوف في pandas (قد تكون المعرّف أو أرقامًا)", "مثال": "0, 1, 2 ..."},
    {"المصطلح": "Primary key", "المعنى": "عمود/أعمدة فريدة لكل سجل", "مثال": "(firm_id, year) في Panel"},
    {"المصطلح": "Entity", "المعنى": "الكيان الذي تصفه البيانات", "مثال": "العميل، المنتج"},
    {"المصطلح": "Schema", "المعنى": "بنية البيانات: الأعمدة والأنواع والقيود", "مثال": "age: int, 18–100"},
    {"المصطلح": "Metadata", "المعنى": "بيانات عن البيانات", "مثال": "المصدر، تاريخ الاستخراج، المالك"},
    {"المصطلح": "Data dictionary", "المعنى": "توثيق معنى ووحدة ونطاق كل عمود", "مثال": "الجدول أدناه"},
])

intuition("فكّر في جدول البيانات كاستمارة: الاستمارة (Schema) تحدد الحقول وقواعدها، وكل استمارة معبأة "
          "مشاهدة، وقاموس البيانات هو دليل تعبئة الاستمارة.")

st.markdown("## العلاقات بين الكيانات")
mermaid("""
erDiagram
  CUSTOMER ||--o{ ORDER : places
  ORDER ||--|{ ORDER_LINE : contains
  PRODUCT ||--o{ ORDER_LINE : "appears in"
  CUSTOMER {
    string customer_id PK
    int age
    string country
  }
  ORDER {
    string order_id PK
    string customer_id FK
    date order_date
  }
""")
st.caption("المفتاح الأساسي PK يميّز السجل، والمفتاح الأجنبي FK يربط الجداول. دمج جداول بمفاتيح غير فريدة "
           "يضاعف الصفوف بصمت — مصدر شائع لأخطاء التحليل.")

st.markdown("## قاموس البيانات Data Dictionary")
st.markdown("ولّد قاموسًا أوليًا تلقائيًا ثم أكمله بالمعرفة البشرية (المعنى والوحدة):")
dd = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "non_null": df.notna().sum().to_numpy(),
    "unique": df.nunique().to_numpy(),
    "example": [str(df[c].iloc[0]) for c in df.columns],
})
dd["meaning (human)"] = ""
dd["unit (human)"] = ""
edited = st.data_editor(dd, hide_index=True, disabled=["column", "dtype", "non_null", "unique", "example"],
                        key="dc_dict")
st.download_button("حمّل قاموس البيانات CSV", edited.to_csv(index=False).encode("utf-8-sig"),
                   "data_dictionary.csv", "text/csv", icon=":material/download:")

st.markdown("## الكود: فحص البنية")
explain_code("df.dtypes", [("df", "كائن DataFrame يحوي الجدول"),
                           ("dtypes", "خاصية Attribute (بلا أقواس) تعيد نوع كل عمود")],
             output="Series: اسم العمود ← نوعه (int64, float64, str, datetime64...)",
             interpretation="قارن النوع المخزن بالمعنى: عمر مخزن نصًا أو تاريخ مخزن نصًا يحتاج تصحيحًا.")


def _run(n: int):
    return df.sample(n, random_state=1)[["customer_id", "age", "country", "churned"]]


code_lab("dc_sample", "عيّنة عشوائية من المشاهدات",
         lambda p: f"df.sample(n={p['n']}, random_state=1)[['customer_id', 'age', 'country', 'churned']]",
         _run, params=lambda: {"n": st.slider("عدد الصفوف n", 3, 15, 5, key="dc_n")},
         explanation="random_state يثبت العشوائية فتحصل على نفس العينة كل مرة (قابلية إعادة الإنتاج).")

if at_least("advanced"):
    st.markdown("## متقدم: البيانات المرتبة Tidy data")
    st.markdown(
        "مبادئ Wickham (2014): (1) كل متغير عمود، (2) كل مشاهدة صف، (3) كل نوع وحدة تحليل جدول. "
        "جدول بأعمدة `sales_2022, sales_2023` ليس مرتبًا؛ نحوله بـ`melt` إلى `year, sales`."
    )
    wide = pd.DataFrame({"store": ["A", "B"], "sales_2023": [120, 90], "sales_2024": [135, 99]})
    c1, c2 = st.columns(2)
    c1.markdown("**Wide (غير مرتب)**")
    c1.dataframe(wide, hide_index=True)
    long = wide.melt(id_vars="store", var_name="year", value_name="sales")
    long["year"] = long["year"].str.replace("sales_", "").astype(int)
    c2.markdown("**Long (مرتب)**")
    c2.dataframe(long, hide_index=True)
    st.code('long = wide.melt(id_vars="store", var_name="year", value_name="sales")', language="python")
if at_least("research"):
    researcher_note(["عرّف وحدة التحليل قبل أي نموذج: خلط مستوى المعاملة بمستوى العميل يكسر فرضية الاستقلال.",
                     "وثّق Metadata الاستخراج (التاريخ والاستعلام والإصدار) لضمان قابلية إعادة الإنتاج."])

real_world(["ما الذي يمثله الصف بالضبط؟", "هل المفتاح الأساسي فريد فعلًا؟", "هل يوجد قاموس بيانات رسمي؟ من يحدّثه؟",
            "هل تغيّر تعريف أي عمود عبر الزمن؟"])

page_footer("data_concepts",
            takeaways=["الصف مشاهدة لوحدة التحليل، والعمود متغير.", "الهدف يُفصل عن الخصائص، والمعرّف لا يُستخدم كخاصية.",
                       "المخطط وقاموس البيانات أساس التحقق والجودة."],
            mistakes=["استخدام customer_id كخاصية.", "دمج جداول بمفاتيح غير فريدة.", "الخلط بين وحدة التسجيل ووحدة التحليل."])
