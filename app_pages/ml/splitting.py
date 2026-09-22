import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from components.animation import stepper
from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit
from utils.plotting import plot

page_header("splitting")

st.markdown("## الأدوار الثلاثة")
mermaid("""
flowchart LR
  D[All data] --> TR[Train<br/>fit parameters]
  D --> VA[Validation<br/>choose model & hyperparameters]
  D --> TE[Test<br/>final, one-time estimate]
  TR --> VA --> TE
""")
comparison_table([
    {"الجزء": "Train", "الدور": "تعلّم المعاملات", "تحذير": "الأداء هنا متفائل دائمًا"},
    {"الجزء": "Validation", "الدور": "اختيار النموذج وضبط Hyperparameters", "تحذير": "كثرة المقارنات تجعله متفائلًا أيضًا"},
    {"الجزء": "Test (Holdout)", "الدور": "تقدير نهائي غير متحيز", "تحذير": "يُستخدم مرة واحدة؛ لا ضبط عليه"},
])

st.markdown("## رسم متحرك: كيف تتحرك الـFolds؟")
scheme = st.segmented_control("المخطط", ["K-Fold", "Stratified K-Fold", "Group K-Fold", "Time Series Split"],
                              default="K-Fold", key="sp_scheme")
N = 40
rng = np.random.default_rng(0)
y = (rng.random(N) < 0.25).astype(int)
groups = np.repeat(np.arange(10), 4)
rng.shuffle(groups)
K = 5
splitter = {"K-Fold": KFold(K, shuffle=True, random_state=0), "Stratified K-Fold": StratifiedKFold(K, shuffle=True, random_state=0),
            "Group K-Fold": GroupKFold(K), "Time Series Split": TimeSeriesSplit(K)}[scheme or "K-Fold"]
splits = list(splitter.split(np.zeros((N, 1)), y, groups))


def _render(i: int) -> None:
    fig = go.Figure()
    for f, (tr, te) in enumerate(splits[: i + 1]):
        role = np.full(N, "unused", dtype=object)
        role[tr] = "train"
        role[te] = "test"
        colors = [PALETTE["purple"] if r == "train" else PALETTE["coral"] if r == "test" else "#E9ECEF" for r in role]
        fig.add_trace(go.Scatter(x=np.arange(N), y=[f] * N, mode="markers", showlegend=False,
                                 marker=dict(symbol="square", size=15, color=colors,
                                             line=dict(width=[2 if y[j] else 0 for j in range(N)], color="black"))))
    fig.update_layout(height=120 + 45 * len(splits), xaxis=dict(title="observation index" if scheme != "Group K-Fold" else
                                                             "observation index (groups of 4 share an id)"),
                      yaxis=dict(title="fold", autorange="reversed", dtick=1, range=[-0.5, len(splits) - 0.5]),
                      margin=dict(t=20))
    plot(fig)
    tr, te = splits[i]
    txt = f"Fold {i + 1}: train = {len(tr)}, test = {len(te)}, positive rate in test = {y[te].mean():.0%}"
    if scheme == "Group K-Fold":
        txt += f" · groups in test: {sorted(set(groups[te]))} — لا تظهر في التدريب"
    st.caption(txt + " · (المربعات المحاطة بإطار = الفئة الموجبة)")


stepper(f"split_{scheme}", len(splits), _render, labels=[f"fold {i + 1}" for i in range(len(splits))])

comparison_table([
    {"المخطط": "Holdout", "متى": "بيانات كبيرة جدًا", "الخطر": "تقدير متقلب مع بيانات صغيرة"},
    {"المخطط": "K-Fold", "متى": "مشاهدات مستقلة", "الخطر": "يفترض عدم وجود مجموعات أو زمن"},
    {"المخطط": "Stratified K-Fold", "متى": "تصنيف بفئات غير متوازنة", "الخطر": "لا يحل مشكلة المجموعات"},
    {"المخطط": "Group K-Fold", "متى": "عدة صفوف لنفس الكيان (مريض، عميل، شركة)", "الخطر": "توزيع غير متوازن للمجموعات"},
    {"المخطط": "Time Series Split", "متى": "التنبؤ بالمستقبل", "الخطر": "بيانات تدريب أقل في الـFolds الأولى"},
    {"المخطط": "Nested CV", "متى": "ضبط المعاملات + تقدير غير متحيز معًا", "الخطر": "مكلف حسابيًا"},
])

st.markdown("## لماذا لا يصلح Random Split دائمًا؟ تجربة على بيانات القروض")
st.caption("بعض العملاء لهم عدة طلبات؛ نموذج يحفظ «هذا العميل تعثر سابقًا» سيبدو ممتازًا إن رأى طلبات العميل نفسه في الاختبار.")


@st.cache_data(show_spinner="جارٍ التقييم…")
def compare_schemes() -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    d = credit()
    d["cust_code"] = d["customer_id"].str[2:].astype(int)  # an identifier the model can memorize
    X = d[["age", "income", "loan_amount", "term_months", "late_payments", "debt_to_income", "credit_history_years", "cust_code"]]
    y = d["default"]
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, random_state=0, n_jobs=-1)
    rows = []
    for name, cv, grp in (("Random K-Fold", KFold(5, shuffle=True, random_state=0), None),
                          ("Group K-Fold (by customer)", GroupKFold(5), d["customer_id"]),
                          ("Time Series Split", TimeSeriesSplit(5), None)):
        s = cross_val_score(model, X, y, cv=cv, groups=grp, scoring="roc_auc")
        rows.append({"المخطط": name, "ROC-AUC": s.mean(), "± SD": s.std()})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", icon=":material/play_arrow:", key="sp_run"):
    st.session_state["sp_done"] = True
if st.session_state.get("sp_done"):
    res = compare_schemes()
    st.dataframe(res.round(3), hide_index=True)
    st.markdown("إن كان Random K-Fold أعلى، فجزء من «الأداء» هو حفظ هوية العملاء المتكررين لا تعلّم نمط عام. "
                "Group K-Fold وTime Series Split يقيسان ما يهم فعلًا: الأداء على **عملاء جدد** و**المستقبل**.")
why("طابق مخطط التقسيم مع سيناريو الاستخدام الحقيقي.",
    "اسأل: على ماذا سيُطبق النموذج؟ عملاء جدد ← Group split. الشهر القادم ← Time split. عينة عشوائية من نفس المجتمع ← K-Fold.")

if at_least("advanced"):
    st.markdown("## متقدم: Nested CV")
    st.code("from sklearn.model_selection import GridSearchCV, cross_val_score\n"
            "inner = GridSearchCV(model, param_grid, cv=3)      # chooses hyperparameters\n"
            "outer_scores = cross_val_score(inner, X, y, cv=5)  # estimates performance of the whole procedure",
            language="python")
    st.markdown("الحلقة الداخلية تضبط، والخارجية تقيّم **الإجراء كاملًا** (بما فيه الضبط)، فلا يتسرب اختيار المعاملات إلى التقدير.")
if at_least("research"):
    researcher_note(["في Panel data: Group split بالكيان، أو تقسيم زمني، أو الاثنان معًا حسب سؤال التعميم.",
                     "أبلغ عن مخطط التقسيم بدقة؛ هو جزء من تعريف النتيجة.",
                     "Time series CV: استخدم نافذة متوسعة أو منزلقة مع فجوة Gap لتجنب التسرب القريب."])
real_world(["هل يوجد كيان يتكرر عبر الصفوف؟", "هل الترتيب الزمني مهم؟", "هل الفئات غير متوازنة؟",
            "هل مجموعة الاختبار لم تُستخدم في أي قرار؟"])

page_footer("splitting",
            takeaways=["Train للتعلم، Validation للاختيار، Test للتقدير النهائي مرة واحدة.",
                       "Stratified للفئات غير المتوازنة، Group للكيانات المتكررة، Time للتنبؤ.",
                       "مخطط التقسيم يجب أن يحاكي الاستخدام الحقيقي."],
            mistakes=["Random split لبيانات زمنية أو مجمّعة.", "ضبط المعاملات على مجموعة الاختبار.", "خلط زيارات نفس المريض بين التدريب والاختبار."])
