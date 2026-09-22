import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit, daily_sales
from utils.plotting import plot

page_header("data_leakage")

st.markdown("## ما هو التسرب؟")
st.markdown("**Data Leakage**: استخدام معلومات أثناء التدريب **لن تكون متاحة لحظة التنبؤ الحقيقي**. النتيجة: نموذج يبدو "
            "ممتازًا في التقييم ويفشل في الواقع. هو من أخطر الأخطاء لأنه **لا يُنتج رسالة خطأ** بل نتائج «رائعة».")
comparison_table([
    {"النوع": "Target leakage", "الوصف": "خاصية تنتج عن الهدف أو تُسجل بعده", "مثال": "«أُحيل للتحصيل» للتنبؤ بالتعثر"},
    {"النوع": "Train-test contamination", "الوصف": "صفوف الاختبار تتسرب للتدريب", "مثال": "تكرارات في الجانبين"},
    {"النوع": "Preprocessing leakage", "الوصف": "إحصاءات المعالجة من كامل البيانات", "مثال": "Scaling/Imputation/Selection قبل التقسيم"},
    {"النوع": "Temporal leakage", "الوصف": "معلومات مستقبلية في خصائص الماضي", "مثال": "Random split لسلسلة زمنية، Rolling مركزي"},
    {"النوع": "Group leakage", "الوصف": "نفس الكيان في التدريب والاختبار", "مثال": "زيارات نفس المريض"},
    {"النوع": "Feature leakage", "الوصف": "خاصية وكيلة للهدف بطريقة غير سببية", "مثال": "رقم الملف مرتبط بتاريخ التعثر"},
])
mermaid("""
flowchart LR
  subgraph Wrong
    A1[All data] --> B1[Scale / impute / select] --> C1[Split] --> D1[Train] --> E1[Inflated score]
  end
  subgraph Right
    A2[All data] --> C2[Split] --> B2[Fit preprocessing on train only] --> D2[Train] --> E2[Honest score]
  end
""")

d = credit()
FEATS = ["age", "income", "loan_amount", "term_months", "late_payments", "debt_to_income", "credit_history_years"]

st.markdown("## العرض 1: Target leakage — نموذج «مثالي» زائف")
use_leak = st.toggle("أضف الخاصية sent_to_collections", True, key="lk_t")


@st.cache_data(show_spinner="تدريب النموذج…")
def target_leak_demo(with_leak: bool) -> tuple[float, pd.Series]:
    feats = FEATS + (["sent_to_collections"] if with_leak else [])
    Xtr, Xte, ytr, yte = train_test_split(d[feats], d["default"], test_size=0.3, random_state=0, stratify=d["default"])
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    return roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), pd.Series(rf.feature_importances_, index=feats).sort_values()


auc, imp = target_leak_demo(use_leak)
c1, c2 = st.columns([1, 2])
c1.metric("ROC-AUC على الاختبار", f"{auc:.3f}")
with c2:
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color=[PALETTE["softred"] if i == "sent_to_collections" else PALETTE["purple"] for i in imp.index]))
    fig.update_layout(height=300, title="Feature importance")
    plot(fig)
why("عندما تهيمن خاصية واحدة على الأهمية والأداء قريب من المثالي، اسأل: متى تُسجل هذه القيمة؟",
    "الإحالة للتحصيل تحدث **بعد** التعثر؛ لحظة قرار القرض لا يعرفها أحد. التقييم على بيانات تاريخية يخفي ذلك تمامًا.")

st.markdown("## العرض 2: Preprocessing leakage — اختيار الخصائص قبل التقسيم")
st.caption("بيانات عشوائية تمامًا: 1000 خاصية ضوضاء بلا أي علاقة بالهدف. الأداء الحقيقي الممكن = عشوائي (AUC ≈ 0.5).")


@st.cache_data(show_spinner="جارٍ التجربة…")
def selection_leak(n: int, p: int, k: int) -> dict:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)
    sel = SelectKBest(f_classif, k=k).fit(X, y)  # WRONG: uses all rows, including future test folds
    wrong = cross_val_score(LogisticRegression(max_iter=1000), sel.transform(X), y, cv=5, scoring="roc_auc").mean()
    right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000)), X, y, cv=5,
                            scoring="roc_auc").mean()
    return {"wrong": wrong, "right": right}


c1, c2 = st.columns(2)
k = c1.slider("عدد الخصائص المختارة k", 5, 100, 20, key="lk_k")
n = c2.select_slider("عدد الصفوف", [100, 200, 500], value=200, key="lk_n")
r = selection_leak(n, 1000, k)
fig = go.Figure(go.Bar(x=["اختيار على كل البيانات ثم CV ❌", "الاختيار داخل Pipeline ✅"], y=[r["wrong"], r["right"]],
                       marker_color=[PALETTE["softred"], PALETTE["teal"]], text=[f"{r['wrong']:.3f}", f"{r['right']:.3f}"],
                       textposition="outside"))
fig.add_hline(y=0.5, line_dash="dash", annotation_text="random")
fig.update_layout(yaxis=dict(range=[0.3, 1.05], title="CV ROC-AUC"), height=340)
plot(fig)
st.markdown("الطريقة الخاطئة «تكتشف» علاقة في ضوضاء خالصة لأن اختيار الخصائص رأى كل الـfolds. هذا المثال الكلاسيكي "
            "مذكور في *The Elements of Statistical Learning* (§7.10.2).")

st.markdown("## العرض 3: Temporal leakage — Random split لسلسلة زمنية")
ts = daily_sales().set_index("date").asfreq("D")
ts["sales"] = ts["sales"].interpolate()
ts["promo"] = ts["promo"].fillna(0)
feat = pd.DataFrame({"lag1": ts["sales"].shift(1), "lag7": ts["sales"].shift(7), "dow": ts.index.dayofweek,
                     "promo": ts["promo"], "centered_mean": ts["sales"].rolling(7, center=True).mean(),
                     "y": ts["sales"]}).dropna()
use_center = st.toggle("أضف centered rolling mean (يستخدم المستقبل)", False, key="lk_center")
cols = ["lag1", "lag7", "dow", "promo"] + (["centered_mean"] if use_center else [])
rows = []
for name, (tr_idx, te_idx) in {
    "Random split": train_test_split(np.arange(len(feat)), test_size=0.25, random_state=0),
    "Time split (last 25%)": (np.arange(int(len(feat) * 0.75)), np.arange(int(len(feat) * 0.75), len(feat))),
}.items():
    m = GradientBoostingRegressor(random_state=0).fit(feat.iloc[tr_idx][cols], feat.iloc[tr_idx]["y"])
    rows.append({"التقسيم": name, "MAE": mean_absolute_error(feat.iloc[te_idx]["y"], m.predict(feat.iloc[te_idx][cols]))})
st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
st.markdown("Random split يسمح للنموذج بالاستفادة من أيام «محاطة» بأيام مرئية، وcentered mean يحتوي على الإجابة جزئيًا. "
            "Time split يحاكي الواقع: نتنبأ بمستقبل لم نره، ومع structural break في 2024 يظهر الأداء الحقيقي الأصعب.")

st.markdown("## قائمة فحص التسرب")
st.markdown(
    "1. لكل خاصية: **متى** تُعرف قيمتها مقارنة بلحظة التنبؤ؟\n"
    "2. هل أي معالجة (Scaling، Imputation، Encoding، Selection) تُضبط قبل التقسيم؟\n"
    "3. هل الكيان نفسه (عميل، مريض) في التدريب والاختبار؟\n"
    "4. هل التقسيم يحترم الزمن؟\n"
    "5. هل النتيجة «أفضل من المتوقع» بشكل مريب؟ افحص أهم الخصائص.\n"
    "6. هل تكرارات الصفوف موزعة بين الجانبين؟"
)

if at_least("advanced"):
    st.markdown("## متقدم: Point-in-time correctness")
    st.markdown("في مخازن الخصائص Feature stores تُحفظ كل قيمة مع **وقت صلاحيتها**، ويُبنى التدريب بـ«Point-in-time join»: "
                "لكل مثال نأخذ آخر قيمة معروفة **قبل** وقته. هذا يمنع التسرب الزمني في الأنظمة الكبيرة.")
if at_least("research"):
    researcher_note(["Kaufman et al. (2012) صنفوا التسرب كمشكلة منهجية في التنقيب.",
                     "Kapoor & Narayanan (2023) وثّقوا التسرب في مئات الأوراق البحثية التي استخدمت ML.",
                     "أرفق «بطاقة نموذج» توضح توقيت كل خاصية ومخطط التقسيم."])
real_world(["اسأل خبير المجال عن توقيت تسجيل كل حقل.", "هل تتغير قيم الحقول بعد الحدث (تحديث بأثر رجعي)؟",
            "هل يوجد عمود «حالة» يعكس النتيجة؟", "جرّب النموذج على فترة لاحقة لم تُستخدم إطلاقًا."])

page_footer("data_leakage",
            takeaways=["التسرب = معلومات غير متاحة وقت التنبؤ.", "ستة أنواع: Target، Contamination، Preprocessing، Temporal، Group، Feature.",
                       "Pipeline داخل CV يمنع تسرب المعالجة.", "النتائج المثالية المفاجئة علامة إنذار."],
            mistakes=["Scaling قبل التقسيم.", "Feature selection على كل البيانات.", "Random split للسلاسل الزمنية.",
                      "الثقة بخاصية تُسجل بعد الحدث."])
