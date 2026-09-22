import importlib.util

import pandas as pd
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.dataset_viewer import dataset_card
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import customers_raw
from utils.validation import CUSTOMER_RULES, Rule, check, run_rules

page_header("data_validation")
df = customers_raw()

st.markdown("## ما التحقق من صحة البيانات؟")
st.markdown(
    "**Data Validation** فحص البيانات مقابل **قواعد صريحة مكتوبة مسبقًا**. الفرق عن الاستكشاف: "
    "الاستكشاف يسأل «ماذا يوجد؟»، والتحقق يسأل «هل هذا ما يجب أن يوجد؟». القواعد نفسها توثيق حي للبيانات."
)
comparison_table([
    {"نوع القاعدة": "Schema", "تتحقق من": "وجود الأعمدة المطلوبة (وعدم وجود غيرها)", "مثال": "customer_id, age, email موجودة"},
    {"نوع القاعدة": "Type", "تتحقق من": "نوع القيم", "مثال": "age رقمي"},
    {"نوع القاعدة": "Range", "تتحقق من": "الحدود الدنيا والعليا", "مثال": "18 ≤ age ≤ 100"},
    {"نوع القاعدة": "Category", "تتحقق من": "قيم من قائمة معتمدة", "مثال": "membership ∈ {Bronze, Silver, Gold, Platinum}"},
    {"نوع القاعدة": "Regex", "تتحقق من": "الصيغة النصية", "مثال": "بريد إلكتروني صالح"},
    {"نوع القاعدة": "Uniqueness", "تتحقق من": "عدم التكرار", "مثال": "customer_id فريد"},
    {"نوع القاعدة": "Cross-field", "تتحقق من": "منطق بين حقلين", "مثال": "سعر البيع ≥ التكلفة"},
    {"نوع القاعدة": "Referential integrity", "تتحقق من": "وجود المفتاح في جدول مرجعي", "مثال": "كل طلب لعميل موجود"},
    {"نوع القاعدة": "Business rules", "تتحقق من": "قواعد المؤسسة", "مثال": "الخصم ≤ 30%"},
    {"نوع القاعدة": "Temporal", "تتحقق من": "الترتيب والتسلسل الزمني", "مثال": "آخر شراء بعد التسجيل"},
])

st.markdown("## شغّل حزمة القواعد على بيانات التحدي")
dataset_card("customers_raw")
results = run_rules(df, CUSTOMER_RULES)
st.dataframe(results, hide_index=True, column_config={
    "pass_rate_%": st.column_config.ProgressColumn("pass rate", min_value=0, max_value=100, format="%.1f%%")})
failing = results[results["failed"] > 0]
st.markdown(f"**{len(failing)} من {len(results)} قاعدة فشلت.** اختر قاعدة لعرض الصفوف المخالفة (الدليل):")
pick = st.selectbox("القاعدة", [r.description for r in CUSTOMER_RULES if r.kind != "schema"], key="dv_rule", index=8)
rule = next(r for r in CUSTOMER_RULES if r.description == pick)
res = check(df, rule)
cols = [rule.column] + ([rule.params["other"]] if "other" in rule.params else [])
st.dataframe(df.loc[res.failing_index[:50], ["customer_id"] + cols], hide_index=False)
st.caption(f"{res.failed} صفًا فاشلًا من {res.checked} (تُعرض أول 50).")

st.markdown("## ابنِ قاعدتك · Rule Builder")
c1, c2, c3 = st.columns(3)
kind = c1.selectbox("النوع", ["range", "category", "regex", "not_null", "unique", "temporal"], key="dv_kind")
col = c2.selectbox("العمود", list(df.columns), key="dv_col", index=list(df.columns).index("monthly_spend"))
params: dict = {}
if kind == "range":
    lo = c3.number_input("الحد الأدنى", value=0.0, key="dv_lo")
    hi = c3.number_input("الحد الأعلى", value=2000.0, key="dv_hi")
    params = {"min": lo, "max": hi}
elif kind == "category":
    vals = sorted(df[col].dropna().astype(str).unique().tolist())[:30]
    params = {"allowed": c3.multiselect("القيم المسموحة", vals, default=vals[:3], key="dv_allowed")}
elif kind == "regex":
    params = {"pattern": c3.text_input("النمط Regex", r"C\d{4}", key="dv_regex")}
elif kind == "temporal":
    params = {"other": c3.selectbox("مقارنة مع", [c for c in df.columns if c != col], key="dv_other"), "op": ">="}
try:
    custom = check(df, Rule(kind, col, params, "custom"))
    st.metric("نتيجة القاعدة", f"{100 * custom.pass_rate:.2f}% ناجح", f"{custom.failed} مخالفة", delta_color="inverse")
    if custom.failed:
        st.dataframe(df.loc[custom.failing_index[:20], [col]])
except Exception as exc:  # e.g. an invalid regex typed by the learner
    st.error(f"القاعدة غير صالحة: {type(exc).__name__}. تحقق من النمط أو المعاملات.")

why("عند فشل قاعدة: حقق، ثم صحح أو علّم أو اعزل؛ ولا تحذف تلقائيًا.",
    "القاعدة نفسها قد تكون خاطئة (مثل حد أعلى صارم جدًا)، والصفوف المخالفة قد تحوي معلومات مهمة عن عملية الجمع.")

st.markdown("## الأدوات الحديثة")
comparison_table([
    {"الأداة": "Pandera", "المستوى": "DataFrame (pandas, polars…)", "القوة": "مخطط تصريحي، lazy validation، اختبارات Hypothesis",
     "متى؟": "تحقق التحليل وخطوط المعالجة في Python"},
    {"الأداة": "Pydantic", "المستوى": "سجل/كائن واحد", "القوة": "تحقق أنواع صارم، تحويل تلقائي، مثالي لـAPIs",
     "متى؟": "مدخلات API، ملفات إعداد، سجلات JSON"},
    {"الأداة": "Great Expectations (GX Core 1.x)", "المستوى": "جداول ومستودعات", "القوة": "Expectations، توثيق Data Docs، تكامل Pipelines",
     "متى؟": "فرق بيانات ومراقبة مستمرة"},
    {"الأداة": "قواعد مخصصة (هذه المنصة)", "المستوى": "DataFrame", "القوة": "شفافة وبلا اعتماديات", "متى؟": "التعليم والنماذج الأولية"},
])
st.caption("حالة الصيانة (تحقق حتى سبتمبر 2026): Pandera 0.29 يدعم Python 3.10–3.14؛ GX Core 1.16 (أبريل 2026). "
           "انظر docs/research_notes.md.")

tab_pa, tab_pd = st.tabs(["Pandera", "Pydantic"])
with tab_pa:
    code = '''import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "customer_id": pa.Column(str, unique=True, nullable=False),
    "age": pa.Column(float, pa.Check.in_range(18, 100), nullable=True, coerce=True),
    "membership": pa.Column(str, pa.Check.isin(["Bronze", "Silver", "Gold", "Platinum"])),
    "email": pa.Column(str, pa.Check.str_matches(r"[^@\\s]+@[^@\\s]+\\.[A-Za-z]{2,}")),
})
try:
    schema.validate(df, lazy=True)        # lazy=True collects ALL failures
except pa.errors.SchemaErrors as err:
    print(err.failure_cases.head())       # one row per failing value'''
    st.code(code, language="python")
    if importlib.util.find_spec("pandera"):
        if st.button("شغّل Pandera على البيانات", key="dv_run_pa", icon=":material/play_arrow:"):
            import pandera.pandas as pa  # noqa: PLC0415
            schema = pa.DataFrameSchema({
                "customer_id": pa.Column(str, unique=True, nullable=False),
                "membership": pa.Column(str, pa.Check.isin(["Bronze", "Silver", "Gold", "Platinum"])),
            })
            try:
                schema.validate(df, lazy=True)
                st.success("لا مخالفات.")
            except pa.errors.SchemaErrors as err:
                st.dataframe(err.failure_cases.head(30))
    else:
        st.info("Pandera غير مثبتة في هذه البيئة (اعتمادية اختيارية). النتائج المكافئة معروضة في جدول القواعد أعلاه. "
                "للتثبيت: `pip install pandera`.", icon=":material/info:")
with tab_pd:
    st.code('''from pydantic import BaseModel, EmailStr, Field, ValidationError

class Customer(BaseModel):
    customer_id: str = Field(pattern=r"^C\\d{4}$")
    age: int = Field(ge=18, le=100)
    membership: str

try:
    Customer(customer_id="C0001", age=250, membership="Gold")
except ValidationError as e:
    print(e)''', language="python")
    if st.button("شغّل مثال Pydantic", key="dv_run_pyd", icon=":material/play_arrow:"):
        try:
            from pydantic import BaseModel, Field, ValidationError  # noqa: PLC0415

            class Customer(BaseModel):
                customer_id: str = Field(pattern=r"^C\d{4}$")
                age: int = Field(ge=18, le=100)
                membership: str

            rows = df.head(12)
            report = []
            for rec in rows.to_dict("records"):
                try:
                    Customer(customer_id=rec["customer_id"], age=rec["age"], membership=rec["membership"])
                    report.append({"customer_id": rec["customer_id"], "valid": True, "errors": ""})
                except ValidationError as e:
                    report.append({"customer_id": rec["customer_id"], "valid": False,
                                   "errors": "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())})
            st.dataframe(pd.DataFrame(report), hide_index=True)
        except ImportError:
            st.info("Pydantic غير متاحة في هذه البيئة.")

if at_least("research"):
    researcher_note(["اكتب قواعد التحقق **قبل** تحليل النتائج، واحفظها مع الكود.",
                     "ميّز بين القواعد الصلبة (توقف المعالجة) والناعمة (تحذير وعزل).",
                     "أبلغ في التقرير عن عدد الصفوف المستبعدة لكل قاعدة."])
real_world(["من يملك تعريف القواعد: المحلل أم مالك البيانات؟", "ماذا يحدث عند فشل قاعدة في الإنتاج؟",
            "هل تُراجع القواعد عند تغير العمل؟", "هل تُحفظ تقارير التحقق لكل تشغيل؟"])

page_footer("data_validation",
            takeaways=["التحقق يقارن البيانات بقواعد صريحة، والقواعد توثيق حي.", "عشرة أنواع قواعد تغطي معظم الحالات.",
                       "Pandera للـDataFrames، Pydantic للسجلات، GX للمؤسسات."],
            mistakes=["حذف كل صف فاشل تلقائيًا.", "كتابة القواعد بعد رؤية النتائج.", "تحقق من الأنواع فقط دون المنطق بين الحقول."])
