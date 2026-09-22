import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit
from utils.plotting import plot

page_header("feature_engineering")


@st.cache_data(show_spinner=False)
def build_features() -> pd.DataFrame:
    d = credit().drop(columns=["sent_to_collections"])  # leakage column excluded — see Data Leakage module
    d["debt_income_ratio_check"] = d["loan_amount"] / d["income"]
    d["loan_per_month"] = d["loan_amount"] / d["term_months"]
    d["installment_to_income"] = d["loan_per_month"] * 12 / d["income"]
    d["age_band"] = pd.cut(d["age"], [20, 30, 40, 50, 60, 80], labels=["20s", "30s", "40s", "50s", "60+"])
    d["late_x_dti"] = d["late_payments"] * d["debt_to_income"]
    d["app_month"] = d["application_date"].dt.month
    d["app_dow"] = d["application_date"].dt.dayofweek
    d = d.sort_values(["customer_id", "application_date"])
    d["prior_applications"] = d.groupby("customer_id").cumcount()
    d["income_vs_region_median"] = d["income"] / d.groupby("region")["income"].transform("median")
    return d.sort_index()


fe = build_features()

st.markdown("## إنشاء الخصائص · Feature creation")
comparison_table([
    {"النوع": "Ratios نسب", "مثال": "loan_amount / income", "لماذا": "المعنى في النسبة لا في القيمة المطلقة"},
    {"النوع": "Interactions تفاعلات", "مثال": "late_payments × debt_to_income", "لماذا": "أثر متغير يعتمد على آخر"},
    {"النوع": "Binning فئات", "مثال": "age → 20s/30s/…", "لماذا": "علاقات غير خطية، تفسير أسهل (مع فقد معلومة)"},
    {"النوع": "Polynomial", "مثال": "age², age³", "لماذا": "انحناء في نموذج خطي"},
    {"النوع": "Aggregations تجميع", "مثال": "عدد الطلبات السابقة للعميل", "لماذا": "سياق الكيان من سجلات متعددة"},
    {"النوع": "Date features", "مثال": "شهر ويوم الطلب", "لماذا": "موسمية وسلوك زمني"},
    {"النوع": "Lag / Rolling", "مثال": "متوسط آخر 3 أشهر", "لماذا": "الماضي القريب يتنبأ بالقريب القادم"},
    {"النوع": "Text features", "مثال": "طول النص، TF-IDF", "لماذا": "تحويل غير المنظم إلى أرقام"},
    {"النوع": "Domain-driven", "مثال": "installment_to_income", "لماذا": "معرفة الائتمان: قدرة السداد الشهرية"},
])
st.dataframe(fe[["loan_amount", "income", "term_months", "loan_per_month", "installment_to_income", "age", "age_band",
                 "late_payments", "debt_to_income", "late_x_dti", "prior_applications", "income_vs_region_median"]].head(8)
             .round(3), hide_index=True)
why("ابدأ بالخصائص المبنية على المجال قبل التوليد الآلي الضخم.",
    "خاصية واحدة ذات معنى (قدرة السداد) غالبًا تتفوق على مئات التركيبات العشوائية، وتبقى قابلة للتفسير والدفاع عنها.")
st.warning("كل خاصية تجميعية يجب أن تُحسب **حتى لحظة القرار فقط**: `prior_applications` يعدّ الطلبات **السابقة** "
           "(cumcount بعد الترتيب الزمني)، لا كل طلبات العميل.", icon=":material/warning:")

st.markdown("## أثر الخاصية على الأداء")
BASE = ["age", "income", "loan_amount", "term_months", "late_payments", "credit_history_years"]
EXTRA = {"debt_to_income": "debt_to_income", "installment_to_income": "installment_to_income",
         "late_x_dti": "late_x_dti", "prior_applications": "prior_applications"}
add = st.multiselect("أضف خصائص مشتقة إلى النموذج الأساسي", list(EXTRA), default=["debt_to_income"], key="fe_add")
feats = BASE + [EXTRA[a] for a in add]
X_tr, X_te, y_tr, y_te = train_test_split(fe[feats], fe["default"], test_size=0.3, random_state=0, stratify=fe["default"])
sc = StandardScaler().fit(X_tr)
lr = LogisticRegression(max_iter=1000).fit(sc.transform(X_tr), y_tr)
auc_new = roc_auc_score(y_te, lr.predict_proba(sc.transform(X_te))[:, 1])
sc0 = StandardScaler().fit(X_tr[BASE])
auc_base = roc_auc_score(y_te, LogisticRegression(max_iter=1000).fit(sc0.transform(X_tr[BASE]), y_tr)
                         .predict_proba(sc0.transform(X_te[BASE]))[:, 1])
c1, c2 = st.columns(2)
c1.metric("ROC-AUC الأساسي", f"{auc_base:.3f}")
c2.metric("ROC-AUC مع الخصائص المشتقة", f"{auc_new:.3f}", f"{auc_new - auc_base:+.3f}")

st.markdown("## اختيار الخصائص · Feature selection")
comparison_table([
    {"العائلة": "Filter", "الفكرة": "ترتيب الخصائص بمقياس مستقل عن النموذج", "أمثلة": "Correlation، Mutual information، Chi²", "المزايا/العيوب": "سريع؛ يتجاهل التفاعلات"},
    {"العائلة": "Wrapper", "الفكرة": "تجربة مجموعات بتدريب النموذج", "أمثلة": "RFE، Forward/Backward selection", "المزايا/العيوب": "يراعي النموذج؛ مكلف وخطر Overfitting"},
    {"العائلة": "Embedded", "الفكرة": "الاختيار جزء من التدريب", "أمثلة": "Lasso (L1)، أهمية الأشجار", "المزايا/العيوب": "كفء؛ مرتبط بالنموذج"},
])
ALL = BASE + list(EXTRA.values()) + ["app_month", "app_dow"]
X = fe[ALL]
y = fe["default"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
method = st.segmented_control("الطريقة", ["Mutual information", "RFE", "Lasso (L1)", "Tree importance", "Permutation importance"],
                              default="Permutation importance", key="fe_method")


@st.cache_data(show_spinner="جارٍ الحساب…")
def selection_scores(method: str) -> pd.Series:
    scaler = StandardScaler().fit(Xtr)
    Z = scaler.transform(Xtr)
    if method == "Mutual information":
        return pd.Series(mutual_info_classif(Z, ytr, random_state=0), index=ALL)
    if method == "RFE":
        rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=1).fit(Z, ytr)
        return pd.Series(len(ALL) + 1 - rfe.ranking_, index=ALL).astype(float)
    if method == "Lasso (L1)":
        m = LogisticRegression(penalty="l1", C=0.05, solver="liblinear").fit(Z, ytr)
        return pd.Series(np.abs(m.coef_[0]), index=ALL)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    if method == "Tree importance":
        return pd.Series(rf.feature_importances_, index=ALL)
    pi = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=0, n_jobs=-1)
    return pd.Series(pi.importances_mean, index=ALL)


scores = selection_scores(method or "Permutation importance").sort_values()
fig = go.Figure(go.Bar(x=scores.values, y=scores.index, orientation="h",
                       marker_color=[PALETTE["coral"] if v > 0 else "#CED4DA" for v in scores.values]))
fig.update_layout(title=method, height=440)
plot(fig)
notes = {"Mutual information": "يلتقط علاقات غير خطية مع الهدف، لكن كل خاصية منفردة.",
         "RFE": "يزيل الأضعف تكراريًا ويعيد التدريب؛ القيم هنا ترتيب معكوس (الأعلى = بقي أطول).",
         "Lasso (L1)": "المعاملات الصفرية = خصائص مستبعدة. قوة العقوبة C تحدد عدد الخصائص الباقية.",
         "Tree importance": "انخفاض عدم النقاء؛ منحاز للخصائص كثيرة القيم.",
         "Permutation importance": "انخفاض AUC على **بيانات الاختبار** عند خلط الخاصية: أصدق مقياس للاعتماد الفعلي للنموذج."}
st.caption(notes[method or "Permutation importance"] + " لاحظ app_month وapp_dow: لا علاقة لهما بالتعثر في هذه البيانات، "
           "فأهميتهما قريبة من الصفر في الطرق الجيدة.")


def _kbest(k: int):
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(StandardScaler(), SelectKBest(mutual_info_classif, k=k), LogisticRegression(max_iter=1000))
    s = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    return f"k = {k}: ROC-AUC (5-fold CV) = **{s.mean():.3f}** ± {s.std():.3f}"


code_lab("fe_kbest", "الاختيار داخل Pipeline (بلا تسرّب)",
         lambda p: ("pipe = make_pipeline(StandardScaler(),\n"
                    f"                     SelectKBest(mutual_info_classif, k={p['k']}),\n"
                    "                     LogisticRegression(max_iter=1000))\n"
                    "cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')"),
         _kbest, params=lambda: {"k": st.slider("k", 1, len(ALL), 5, key="fe_k")},
         explanation="الاختيار يُعاد داخل كل fold على بيانات التدريب فقط. اختيار الخصائص على كامل البيانات ثم CV يعطي تقديرًا متفائلًا.")

if at_least("advanced"):
    st.markdown("## متقدم: Regularization")
    st.markdown("- **L1 (Lasso):** Σ|β| ← معاملات صفرية (اختيار).\n- **L2 (Ridge):** Σβ² ← تقليص دون حذف، مستقر مع التعدد الخطي.\n"
                "- **Elastic Net:** مزيج الاثنين. كلها تتطلب Scaling لأن العقوبة تعامل المعاملات بالتساوي.")
    Cs = np.logspace(-3, 1, 25)
    Z = StandardScaler().fit_transform(Xtr)
    path = np.array([LogisticRegression(penalty="l1", C=c, solver="liblinear").fit(Z, ytr).coef_[0] for c in Cs])
    fig = go.Figure([go.Scatter(x=Cs, y=path[:, j], name=ALL[j], mode="lines") for j in range(len(ALL))])
    fig.update_layout(xaxis_type="log", xaxis_title="C (inverse regularization strength)", yaxis_title="coefficient",
                      title="مسار Lasso: الخصائص تدخل النموذج مع تخفيف العقوبة", height=420)
    plot(fig)
if at_least("research"):
    researcher_note(["الأهمية التنبؤية ليست أثرًا سببيًا: خاصية مهمة قد تكون وكيلًا Proxy لسبب آخر.",
                     "الاختيار التلقائي ثم الاستدلال على نفس البيانات يبطل p-values المعتادة (Post-selection inference)."])
real_world(["هل كل خاصية متاحة لحظة التنبؤ؟", "هل يستطيع الخبير فهم الخاصية وشرحها؟", "هل الخاصية مستقرة عبر الزمن؟",
            "هل تسبب الخاصية مشكلات عدالة (Proxy لمتغير محمي)؟"])

page_footer("feature_engineering",
            takeaways=["الخصائص المبنية على المجال قوية وقابلة للتفسير.", "Filter/Wrapper/Embedded عائلات الاختيار.",
                       "Permutation importance على بيانات الاختبار أصدق من أهمية الأشجار.", "الاختيار داخل Pipeline."],
            mistakes=["تجميع يتضمن معلومات مستقبلية.", "الاختيار على كل البيانات قبل CV.", "تفسير الأهمية كسببية."])
