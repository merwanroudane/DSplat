import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit, students
from utils.plotting import plot

page_header("model_evaluation")

FEATS = ["age", "income", "loan_amount", "term_months", "late_payments", "debt_to_income", "credit_history_years"]


@st.cache_data(show_spinner="تدريب النماذج…")
def fit_models():
    d = credit()
    Xtr, Xte, ytr, yte = train_test_split(d[FEATS], d["default"], test_size=0.3, random_state=0, stratify=d["default"])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    return yte.to_numpy(), lr.predict_proba(Xte)[:, 1], gb.predict_proba(Xte)[:, 1]


y_true, p_lr, p_gb = fit_models()

st.markdown("## مقاييس التصنيف")
model = st.segmented_control("النموذج", ["Logistic Regression", "Gradient Boosting"], default="Gradient Boosting", key="ev_model")
p = p_gb if model != "Logistic Regression" else p_lr
thr = st.slider("العتبة Threshold", 0.01, 0.99, 0.5, 0.01, key="ev_thr")
pred = (p >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
acc = (tp + tn) / len(y_true)
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
with st.container(horizontal=True):
    st.metric("Accuracy", f"{acc:.3f}", border=True)
    st.metric("Precision", f"{prec:.3f}", border=True)
    st.metric("Recall", f"{rec:.3f}", border=True)
    st.metric("F1", f"{f1:.3f}", border=True)
    st.metric("ROC-AUC", f"{roc_auc_score(y_true, p):.3f}", border=True)
    st.metric("PR-AUC (AP)", f"{average_precision_score(y_true, p):.3f}", border=True)
c1, c2, c3 = st.columns(3)
with c1:
    cm = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(go.Heatmap(z=cm, x=["pred 0", "pred 1"], y=["true 0", "true 1"], text=cm, texttemplate="%{text}",
                               colorscale=[[0, "#FFFBEA"], [1, PALETTE["coral"]]], showscale=False))
    fig.update_layout(title="Confusion matrix", height=320, yaxis=dict(autorange="reversed"))
    plot(fig)
with c2:
    fpr, tpr, roc_thr = roc_curve(y_true, p)
    j = np.argmin(np.abs(roc_thr - thr))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC", line=dict(color=PALETTE["purple"], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#AAA"), showlegend=False))
    fig.add_trace(go.Scatter(x=[fpr[j]], y=[tpr[j]], mode="markers", marker=dict(size=13, color=PALETTE["coral"]), name="threshold"))
    fig.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR (Recall)", height=320, showlegend=False)
    plot(fig)
with c3:
    pr, rc, pr_thr = precision_recall_curve(y_true, p)
    k = np.argmin(np.abs(pr_thr - thr))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rc, y=pr, line=dict(color=PALETTE["amber"], width=3)))
    fig.add_hline(y=y_true.mean(), line_dash="dash", annotation_text="baseline = prevalence")
    fig.add_trace(go.Scatter(x=[rc[k]], y=[pr[k]], mode="markers", marker=dict(size=13, color=PALETTE["coral"])))
    fig.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision", height=320, showlegend=False)
    plot(fig)
formula(r"\text{Precision} = \frac{TP}{TP+FP},\quad \text{Recall} = \frac{TP}{TP+FN},\quad F_1 = \frac{2PR}{P+R}",
        title="المقاييس من مصفوفة الالتباس",
        symbols={"TP": "إيجابي صحيح", "FP": "إيجابي كاذب (إنذار خاطئ)", "FN": "سلبي كاذب (حالة فائتة)", "TN": "سلبي صحيح"},
        intuition="Precision: «عندما يقول النموذج نعم، كم مرة يصيب؟». Recall: «من كل الحالات الحقيقية، كم التقط؟».")

st.markdown("## العتبة والتكلفة")
c1, c2 = st.columns(2)
cost_fn = c1.number_input("تكلفة تفويت متعثر (FN)", value=10.0, min_value=0.0, key="ev_cfn")
cost_fp = c2.number_input("تكلفة رفض عميل جيد (FP)", value=1.0, min_value=0.0, key="ev_cfp")
ths = np.linspace(0.01, 0.99, 99)
costs = [((p >= t) & (y_true == 0)).sum() * cost_fp + ((p < t) & (y_true == 1)).sum() * cost_fn for t in ths]
best = ths[int(np.argmin(costs))]
fig = go.Figure(go.Scatter(x=ths, y=costs, line=dict(color=PALETTE["coral"], width=3)))
fig.add_vline(x=best, line_dash="dash", annotation_text=f"best ≈ {best:.2f}")
fig.add_vline(x=thr, line_color=PALETTE["purple"], annotation_text="current")
fig.update_layout(xaxis_title="threshold", yaxis_title="total cost", height=320)
plot(fig)
why(f"استخدم العتبة ≈ {best:.2f} بدل 0.5 الافتراضية لهذه التكاليف.",
    "0.5 لا معنى خاصًا لها؛ العتبة المثلى تعتمد على تكلفة كل نوع خطأ ونسبة الفئة الموجبة. ROC-AUC لا يعتمد على العتبة، أما القرار فيعتمد.")

st.markdown("## لماذا تضلل Accuracy؟")
prev = st.slider("نسبة الفئة الموجبة", 0.01, 0.5, 0.05, 0.01, key="ev_prev")
st.markdown(f"نموذج يقول «سلبي» دائمًا: Accuracy = **{1 - prev:.0%}**، Recall = **0%**. "
            "مع البيانات غير المتوازنة استخدم Precision/Recall/F1 وPR-AUC، وقارن دائمًا بخط الأساس (نسبة الانتشار).")

st.markdown("## مقاييس الانحدار")
s = students()
Xtr, Xte, ytr, yte = train_test_split(s[["study_hours", "pre_score", "attendance_pct"]], s["post_score"], test_size=0.3,
                                      random_state=0)
pred_r = LinearRegression().fit(Xtr, ytr).predict(Xte)
outlier = st.toggle("أضف خطأً كبيرًا واحدًا (قيمة شاذة في الاختبار)", False, key="ev_out")
yte_v = yte.to_numpy().copy()
if outlier:
    yte_v[0] += 60
mae, rmse, r2 = mean_absolute_error(yte_v, pred_r), np.sqrt(mean_squared_error(yte_v, pred_r)), r2_score(yte_v, pred_r)
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.3f}")
formula(r"\text{MAE} = \frac{1}{n}\sum|y_i-\hat y_i|,\quad \text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i-\hat y_i)^2},\quad "
        r"R^2 = 1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}", title="MAE وMSE/RMSE وR²",
        intuition="MAE: متوسط الخطأ بوحدة الهدف. RMSE: يعاقب الأخطاء الكبيرة أكثر (لاحظ قفزته مع القيمة الشاذة). "
                  "R²: نسبة التباين المفسر مقارنة بالتنبؤ بالمتوسط؛ قد يكون سالبًا على بيانات الاختبار.")

comparison_table([
    {"المقياس": "Accuracy", "متى": "فئات متوازنة وتكاليف متساوية", "الحذر": "مضلل مع عدم التوازن"},
    {"المقياس": "Precision", "متى": "الإيجابي الكاذب مكلف (رسائل مزعجة)", "الحذر": "يتجاهل الحالات الفائتة"},
    {"المقياس": "Recall", "متى": "الحالة الفائتة مكلفة (تشخيص طبي)", "الحذر": "يتجاهل الإنذارات الكاذبة"},
    {"المقياس": "F1", "متى": "توازن بين الاثنين", "الحذر": "يفترض أهمية متساوية"},
    {"المقياس": "ROC-AUC", "متى": "جودة الترتيب مستقلة عن العتبة", "الحذر": "متفائل مع عدم التوازن الشديد"},
    {"المقياس": "PR-AUC", "متى": "فئة موجبة نادرة", "الحذر": "يعتمد على الانتشار"},
    {"المقياس": "MAE / RMSE / R²", "متى": "الانحدار", "الحذر": "RMSE حساس للقيم الشاذة؛ R² لا يعني صحة النموذج"},
])
if at_least("advanced"):
    st.markdown("## متقدم: المعايرة Calibration")
    bins = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y_true, "bin": bins}).groupby("bin", observed=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["p"], y=cal["y"], mode="lines+markers", line=dict(color=PALETTE["purple"]), name=model))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#AAA"), name="perfect"))
    fig.update_layout(title="Reliability diagram", xaxis_title="predicted probability", yaxis_title="observed rate", height=340)
    plot(fig)
    st.caption("نموذج معاير: من أُعطوا احتمال 0.3 يتعثر منهم ≈ 30%. مهم عندما تُستخدم الاحتمالات نفسها في القرار.")
if at_least("research"):
    researcher_note(["أبلغ عن فترات ثقة للمقاييس (Bootstrap على مجموعة الاختبار).",
                     "قارن النماذج بنفس التقسيم؛ فروق AUC الصغيرة قد تكون ضوضاء (اختبار DeLong).",
                     "اختر العتبة على Validation لا على Test."])
real_world(["ما تكلفة كل نوع خطأ؟", "ما نسبة الفئة الموجبة في الإنتاج؟", "هل تُستخدم الاحتمالات أم القرار الثنائي فقط؟",
            "هل الأداء متساوٍ عبر المجموعات؟"])

page_footer("model_evaluation",
            takeaways=["المقياس يعكس تكلفة الأخطاء في السياق.", "العتبة قرار أعمال لا ثابت 0.5.",
                       "Accuracy مضللة مع عدم التوازن؛ استخدم PR-AUC وRecall/Precision.", "RMSE يعاقب الأخطاء الكبيرة أكثر من MAE."],
            mistakes=["الاكتفاء بـAccuracy.", "عتبة 0.5 دائمًا.", "اختيار العتبة على مجموعة الاختبار.", "مقارنة نماذج بتقسيمات مختلفة."])
