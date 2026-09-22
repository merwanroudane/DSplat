import pandas as pd
import streamlit as st

from components.animation import stepper
from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import flow
from core.page import page_footer, page_header
from core.state import at_least, log_decision

page_header("human_in_loop")

st.markdown("## المبدأ")
st.markdown("### Automation should assist judgment, not erase judgment.")
st.markdown("**الأتمتة تساعد الحكم البشري ولا تلغيه.** الآلة ممتازة في المسح الشامل والحساب المتكرر، والإنسان مسؤول عن المعنى "
            "والسياق والقرارات غير القابلة للتراجع.")

STEPS = [
    ("Machine Detection", "الآلة تكتشف: 12 قيمة -999 في age، و15 صفًا مكررًا، وقيم دخل بالآلاف."),
    ("Human Review", "المحلل يراجع الدليل: عينة من الصفوف، التوزيع، مكان تركز المشكلة."),
    ("Domain Check", "خبير المجال يؤكد: -999 رمز «غير معروف» في نظام CRM القديم؛ الدخل يُدخل أحيانًا بالآلاف في فرع معين."),
    ("Decision", "القرار: -999 ← NaN، الدخل بين 1 و1000 ← ×1000، التكرار التام يُحذف، المتعارض يُحال للمصدر."),
    ("Transformation", "تطبيق القرارات على نسخة من البيانات بكود محفوظ، دون المساس بالأصل."),
    ("Validation", "إعادة تشغيل قواعد التحقق: هل اختفت المخالفات؟ هل ظهرت مشكلات جديدة؟"),
    ("Documentation", "سجل: ماذا، لماذا، من قرر، كم صفًا تأثر، وإصدار الكود والبيانات."),
]


def _render(i: int) -> None:
    flow([s[0] for s in STEPS], highlight=i)
    with st.container(border=True):
        st.markdown(f"**{i + 1}. {STEPS[i][0]}**  \n{STEPS[i][1]}")


stepper("hitl", len(STEPS), _render, labels=[s[0] for s in STEPS])

st.markdown("## أين يجب أن يتدخل الإنسان؟")
comparison_table([
    {"النقطة": "قبل حذف صفوف أو أعمدة", "لماذا": "غير قابل للتراجع في التحليل اللاحق", "مستوى التدخل": "إلزامي"},
    {"النقطة": "تغيير تعريف متغير أو وحدة", "لماذا": "يغيّر معنى كل النتائج", "مستوى التدخل": "إلزامي"},
    {"النقطة": "فشل قاعدة تحقق حرجة", "لماذا": "قد يعني خللًا في المصدر", "مستوى التدخل": "إلزامي"},
    {"النقطة": "توحيد تسميات فئوية", "لماذا": "دمج خاطئ يخلط كيانات مختلفة", "مستوى التدخل": "مراجعة القاموس مرة ثم أتمتة"},
    {"النقطة": "تعويض فقد منخفض النسبة", "لماذا": "أثر محدود وموثق", "مستوى التدخل": "أتمتة مع مراجعة دورية"},
    {"النقطة": "حساب الملخصات والتقارير", "لماذا": "حتمي وقابل للتحقق", "مستوى التدخل": "أتمتة كاملة"},
])
why("اجعل المراجعة البشرية مركزة على القرارات عالية المخاطر، لا على كل شيء.",
    "مراجعة كل شيء تُتعب المراجع فيقبل كل شيء آليًا (Automation bias). مراجعة القليل المهم بعمق أنجح.")

st.markdown("## سجل القرارات · Decision log")
st.caption("سجل مشترك بين المختبرات في جلستك. أضف قرارًا يدويًا أو من المختبرات (التنظيف، الهجين، الفئوي).")
with st.form("hitl_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    item = c1.text_input("العنصر (عمود/قاعدة)", placeholder="age")
    decision = c2.selectbox("القرار", ["Keep", "Correct", "Set to NaN", "Remove", "Transform", "Investigate", "Escalate to data owner"])
    reason = st.text_area("السبب والدليل", placeholder="12 قيمة = -999؛ قاموس البيانات يعرّفها كـ«غير معروف»")
    if st.form_submit_button("سجّل القرار", icon=":material/bookmark:", type="primary") and item and reason:
        log_decision("manual", item, decision, reason)
        st.toast("سُجّل القرار.")
log = st.session_state.get("decision_log", [])
if log:
    d = pd.DataFrame(log)
    st.dataframe(d, hide_index=True)
    st.download_button("حمّل السجل CSV", d.to_csv(index=False).encode("utf-8-sig"), "decision_log.csv", "text/csv",
                       icon=":material/download:")
else:
    st.info("السجل فارغ بعد.")

if at_least("advanced"):
    st.markdown("## متقدم: مستويات الأتمتة")
    comparison_table([
        {"المستوى": "1", "الوصف": "الإنسان يفعل كل شيء"}, {"المستوى": "2", "الوصف": "الآلة تقترح خيارات"},
        {"المستوى": "3", "الوصف": "الآلة تقترح خيارًا واحدًا والإنسان يعتمد"},
        {"المستوى": "4", "الوصف": "الآلة تنفذ ما لم يعترض الإنسان خلال مهلة"},
        {"المستوى": "5", "الوصف": "الآلة تنفذ وتُبلغ"}, {"المستوى": "6", "الوصف": "الآلة تنفذ دون إبلاغ"},
    ], caption="مقتبس بتصرف من سلم Sheridan & Verplank لمستويات الأتمتة. أغلب قرارات تنظيف البيانات الحساسة يجب ألا تتجاوز المستوى 3.")
if at_least("research"):
    researcher_note(["سجل القرارات جزء من الملحق المنهجي؛ يسمح للمراجعين بإعادة بناء التحليل.",
                     "قِس اتفاق المراجعين عند وجود أكثر من مراجع (Inter-rater reliability)."])
real_world(["من يملك صلاحية القرار لكل نوع مشكلة؟", "ما زمن الاستجابة المقبول للمراجعة؟", "هل يُحفظ السجل مع البيانات؟",
            "كيف نمنع المراجعة الشكلية؟"])

page_footer("human_in_loop",
            takeaways=["Detection → Review → Domain check → Decision → Transformation → Validation → Documentation.",
                       "التدخل البشري إلزامي في القرارات غير القابلة للتراجع.", "سجل القرارات يجعل التحليل قابلًا للتدقيق."],
            mistakes=["مراجعة شكلية لكل شيء.", "قرارات بلا توثيق.", "تعديل الأصل مباشرة."])
