import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.page import footer, page_header
from core.state import mark_lab
from core.theme import SEQUENCE
from utils.cleaning import clean_customers_reference
from utils.datasets import customers_raw
from utils.plotting import plot
from utils.profiling import column_summary
from utils.types import semantic_missing_mask

page_header("data_story")
mark_lab("data_story")

raw = customers_raw()
rng = np.random.default_rng(3)
# The institution also sent a column that is filled in only AFTER a customer leaves.
raw["cancellation_ticket"] = np.where(raw["churned"] == 1, rng.choice(["T-" + str(i) for i in range(100, 999)], len(raw)), None)
clean, _ = clean_customers_reference(raw.drop(columns="cancellation_ticket"))

st.markdown("> **وصلتنا Dataset خام من مؤسسة اشتراكات.** الطلب: «نريد نموذجًا يتنبأ بالعملاء الذين سيغادرون، وتقريرًا نقدمه للإدارة "
            "الأسبوع القادم.» الملف: `customers_export_final_v2.csv`. لا يوجد قاموس بيانات. لنبدأ.")

CHAPTERS = ["Schema", "Missingness", "Duplicates", "Outliers", "Categories", "Dates", "Leakage risk", "EDA", "Modeling",
            "Final interpretation"]
st.session_state.setdefault("story_ch", 0)
st.session_state.setdefault("story_answers", {})
ch = st.session_state["story_ch"]
st.progress((ch + 1) / len(CHAPTERS), text=f"المحطة {ch + 1} من {len(CHAPTERS)}: {CHAPTERS[ch]}")


def question(q: str, options: list[str], correct: int, feedback: str) -> None:
    ans = st.radio(q, options, index=None, key=f"story_q_{ch}")
    if ans is not None:
        ok = options.index(ans) == correct
        st.session_state["story_answers"][ch] = ok
        if ok:
            st.success(f"قرار سليم. {feedback}", icon=":material/check_circle:")
        else:
            st.warning(f"فكّر مرة أخرى. {feedback}", icon=":material/lightbulb:")


with st.container(border=True):
    if ch == 0:
        st.markdown("### 1. المخطط: هل الأعمدة كما تبدو؟")
        st.markdown("أول ما فعلناه: `df.dtypes`. أعمدة يُفترض أنها رقمية جاءت نصًا.")
        cs = column_summary(raw)
        st.dataframe(cs[["column", "dtype", "semantic_type", "example"]], hide_index=True)
        question("ما أول خطوة صحيحة؟", ["حساب متوسط العمر مباشرة", "تحويل الأنواع بعد فحص القيم التي تمنع التحويل",
                                        "حذف الأعمدة النصية"], 1,
                 "القيم مثل «unknown» و«$61,200» تمنع التحويل؛ نفحصها ثم نحول بقواعد صريحة.")
    elif ch == 1:
        st.markdown("### 2. الفقد: الظاهر والخفي")
        sem = {c: int(semantic_missing_mask(raw[c]).sum()) for c in raw.columns}
        d = pd.DataFrame({"isna()": raw.isna().sum(), "semantic (-999, unknown, N/A, ?)": pd.Series(sem)})
        st.dataframe(d[(d.iloc[:, 0] > 0) | (d.iloc[:, 1] > 0)])
        st.markdown(f"عمود `cancellation_ticket` مفقود لـ{raw['cancellation_ticket'].isna().mean():.0%} من العملاء — لاحظه، سنعود إليه.")
        question("ماذا نفعل بـ-999 في العمر؟", ["نتركها؛ هي رقم", "نحوّلها إلى NaN ونسأل المؤسسة عن معناها", "نعوضها بالمتوسط فورًا"],
                 1, "Sentinel تعني «غير معروف»؛ التحويل أولًا ثم التحقق من المصدر، والتعويض لاحقًا بعد تشخيص الآلية.")
    elif ch == 2:
        st.markdown("### 3. التكرار")
        dup_ids = raw[raw["customer_id"].duplicated(keep=False)].sort_values("customer_id")
        st.markdown(f"{int(raw.duplicated(subset=[c for c in raw.columns if c != 'cancellation_ticket']).sum())} صفًا مكررًا تمامًا، "
                    f"و{int(raw['customer_id'].duplicated().sum())} تكرارًا للمعرّف.")
        st.dataframe(dup_ids.head(8)[["customer_id", "monthly_spend", "country", "signup_date"]], hide_index=True)
        question("بعض التكرارات لها monthly_spend مختلف. ماذا نفعل؟", ["نحذف الكل", "نبقي الأول دائمًا دون سؤال",
                                                                    "نحذف التام، ونسأل المؤسسة عن السجل الأحدث للمتعارض"], 2,
                 "التكرار التام آمن حذفه؛ المتعارض يحتاج مصدر الحقيقة. إن تعذر، نوثّق قاعدة (الأول) كافتراض.")
    elif ch == 3:
        st.markdown("### 4. القيم الشاذة")
        h = raw["height_cm"]
        plot(px.histogram(h, nbins=80, color_discrete_sequence=SEQUENCE, labels={"value": "height_cm"}), height=280)
        st.markdown(f"{int((h < 3).sum())} قيمة طول بين 1.5 و2! و5 عملاء إنفاقهم أكبر بـ25 مرة من الوسيط.")
        question("كيف نتعامل مع الاثنين؟", ["نحذف الاثنين", "نصحح الطول (×100) ونُبقي كبار المنفقين بعد التحقق",
                                              "نقص الاثنين عند 1.5×IQR"], 1,
                 "الطول خطأ وحدة واضح قابل للتصحيح؛ كبار المنفقين حقيقيون محتملون ومهمون تجاريًا — لا يُحذفون آليًا.")
    elif ch == 4:
        st.markdown("### 5. الفئات غير الصالحة")
        st.dataframe(raw["country"].value_counts().rename("count").to_frame().T)
        question("«USA»، «usa»، «U.S.A»، «US» و«United States»:", ["خمس دول مختلفة", "كيان واحد يحتاج قاموس توحيد موثق",
                                                                     "نحذف الصفوف غير الموحدة"], 1,
                 "strip + lower يحل بعضها؛ الاختصارات تحتاج قاموسًا يراجعه إنسان.")
    elif ch == 5:
        st.markdown("### 6. مشكلات التواريخ")
        st.dataframe(raw["signup_date"].astype(str).str.replace(r"\d", "9", regex=True).value_counts().rename("pattern count"))
        question("«01/05/2023» تعني:", ["1 مايو", "5 يناير", "لا يمكن الحسم دون معرفة صيغة المصدر"], 2,
                 "الإحصاء لا يحسم الصيغة؛ نسأل المصدر أو نستنتج من قيم مثل 25/03 التي لا تكون إلا يوم/شهر.")
    elif ch == 6:
        st.markdown("### 7. خطر التسرب")
        st.markdown("عمود `cancellation_ticket` مملوء فقط لمن غادروا. لنرَ ماذا يحدث لو استخدمناه:")
        d = raw.assign(has_ticket=raw["cancellation_ticket"].notna().astype(int))
        auc = roc_auc_score(d["churned"], d["has_ticket"])
        st.metric("ROC-AUC لخاصية has_ticket وحدها", f"{auc:.3f}")
        question("هل نستخدمها في النموذج؟", ["نعم، أداء ممتاز", "لا: تذكرة الإلغاء تُنشأ عند المغادرة — غير متاحة وقت التنبؤ"], 1,
                 "Target leakage كلاسيكي. النموذج سيكون عديم الفائدة في الإنتاج رغم AUC شبه مثالي.")
    elif ch == 7:
        st.markdown("### 8. الاستكشاف بعد التنظيف")
        r = clean.groupby("satisfaction")["churned"].agg(["mean", "size"]).reset_index()
        plot(px.bar(r, x="satisfaction", y="mean", text="size", color_discrete_sequence=SEQUENCE,
                    labels={"mean": "churn rate"}), height=300)
        question("ماذا نستنتج؟", ["الرضا المنخفض يسبب المغادرة", "الرضا المنخفض يرتبط بمغادرة أعلى؛ السببية تحتاج تصميمًا آخر"], 1,
                 "بيانات رصدية: ارتباط قوي ومفيد للتنبؤ والاستهداف، لكن ليس دليلًا سببيًا.")
    elif ch == 8:
        st.markdown("### 9. النمذجة دون تسرب")
        d = clean.dropna(subset=["satisfaction", "num_orders", "monthly_spend", "age"]).copy()
        X = pd.DataFrame({"satisfaction": d["satisfaction"].astype(float), "num_orders": d["num_orders"].astype(float),
                          "log_spend": np.log1p(d["monthly_spend"]), "age": d["age"].astype(float)})
        Xtr, Xte, ytr, yte = train_test_split(X, d["churned"], test_size=0.3, random_state=0, stratify=d["churned"])
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        st.metric("ROC-AUC على الاختبار (بدون تسرب)", f"{auc:.3f}")
        st.dataframe(pd.DataFrame({"feature": X.columns, "standardized coef": m[-1].coef_[0].round(3)}), hide_index=True)
        question("AUC أقل بكثير من نسخة التسرب. هل هذا سيئ؟", ["نعم، نعود للخاصية المسربة", "لا: هذا أداء حقيقي قابل للتطبيق"], 1,
                 "الأداء الصادق أقل لكنه ما سيحدث فعلًا في الإنتاج.")
    else:
        st.markdown("### 10. التفسير النهائي للإدارة")
        right = sum(st.session_state["story_answers"].values())
        base = raw.drop(columns="cancellation_ticket")
        hidden = sum(int(semantic_missing_mask(base[c]).sum()) for c in base.columns)
        explicit = int(base.isna().sum().sum())
        st.markdown(f"أجبت إجابات سليمة في **{right} من {len(CHAPTERS) - 1}** محطة.")
        st.markdown(
            "**ملخص تنفيذي مقترح:**\n"
            f"1. البيانات احتاجت 8 خطوات تنظيف موثقة؛ وُجدت {explicit} قيمة مفقودة صريحة و{hidden} قيمة مفقودة «مخفية» "
            f"(Sentinels) من أصل {base.size:,} خلية، إضافة إلى قيم مستحيلة وأخطاء وحدات.\n"
            "2. استبعدنا عمود «تذكرة الإلغاء» لأنه يُسجل بعد المغادرة (كان سيعطي دقة وهمية).\n"
            "3. أهم مؤشرات المغادرة: رضا منخفض وعدد طلبات قليل (ارتباط، لا سببية).\n"
            "4. النموذج الصادق يرتب العملاء حسب الخطر بقدرة متوسطة؛ مناسب لتحديد أولويات التواصل لا لقرارات آلية.\n"
            "5. التوصية: تجربة برنامج احتفاظ على شريحة عالية الخطر مع مجموعة ضابطة لقياس الأثر الحقيقي.\n"
            "6. طلبات للمؤسسة: قاموس بيانات، مصدر الحقيقة للتكرارات المتعارضة، وتوحيد صيغ التاريخ عند المصدر."
        )
        if right >= 7:
            st.balloons()

c1, c2, c3 = st.columns(3)
if c1.button("السابق", disabled=ch == 0, icon=":material/skip_next:", key="story_prev"):
    st.session_state["story_ch"] -= 1
    st.rerun()
if c2.button("التالي", disabled=ch == len(CHAPTERS) - 1, type="primary", icon=":material/skip_previous:", key="story_next"):
    st.session_state["story_ch"] += 1
    st.rerun()
if c3.button("ابدأ من جديد", icon=":material/replay:", key="story_reset"):
    st.session_state["story_ch"] = 0
    st.session_state["story_answers"] = {}
    st.rerun()
footer()
