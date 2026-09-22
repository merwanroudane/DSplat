import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.formulas import formula
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import customers_clean, wine
from utils.plotting import add_animation_controls, plot

page_header("transformations")
cust = customers_clean()

TRANSFORMS = {
    "log1p": lambda x: np.log1p(x),
    "sqrt": lambda x: np.sqrt(x),
    "Box-Cox": lambda x: pd.Series(stats.boxcox(x + (1 if x.min() <= 0 else 0))[0], index=x.index),  # shift if zeros
    "Yeo-Johnson": lambda x: pd.Series(PowerTransformer("yeo-johnson", standardize=False)
                                       .fit_transform(x.to_frame()).ravel(), index=x.index),
    "Winsorize 1%/99%": lambda x: x.clip(*x.quantile([0.01, 0.99])),
    "Standardization (z)": lambda x: (x - x.mean()) / x.std(),
    "Min-Max [0,1]": lambda x: (x - x.min()) / (x.max() - x.min()),
    "Robust (median/IQR)": lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)),
}

st.markdown("## قبل/بعد: شاهد التحويل يتحرك")
c1, c2 = st.columns(2)
col = c1.selectbox("المتغير", ["monthly_spend", "annual_income", "num_orders", "age"], key="tr_col")
name = c2.selectbox("التحويل", list(TRANSFORMS), key="tr_name")
x = cust[col].astype(float)
y = TRANSFORMS[name](x)
# Animate by morphing the standardized "before" into the standardized "after" so both share one axis.
zb = (x - x.mean()) / x.std()
za = (y - y.mean()) / y.std()
steps = np.linspace(0, 1, 9)
frames = []
for t in steps:
    v = (1 - t) * zb + t * za
    frames.append(go.Frame(name=f"{t:.2f}", data=[go.Histogram(x=v, xbins=dict(start=-4, end=8, size=0.2),
                                                               marker_color=PALETTE["coral"] if t < 1 else PALETTE["purple"])],
                           layout=go.Layout(title=f"t = {t:.2f} · skewness = {stats.skew(v):.2f}")))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(xaxis=dict(range=[-4, 8], title="standardized value (for comparison)"), height=420,
                  title=frames[0].layout.title.text, yaxis=dict(range=[0, len(x) / 6]))
add_animation_controls(fig, [f.name for f in frames], duration=500, prefix="before → after: ")
plot(fig)
c1, c2, c3 = st.columns(3)
c1.metric("Skewness قبل", f"{x.skew():.2f}")
c2.metric("Skewness بعد", f"{y.skew():.2f}")
c3.metric("المدى بعد", f"[{y.min():.2f}, {y.max():.2f}]")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Histogram(x=x, nbinsx=50, marker_color=PALETTE["coral"]))
    fig.update_layout(title=f"قبل: {col}", height=280)
    plot(fig)
with c2:
    fig = go.Figure(go.Histogram(x=y, nbinsx=50, marker_color=PALETTE["purple"]))
    fig.update_layout(title=f"بعد: {name}", height=280)
    plot(fig)
st.caption("الرسم المتحرك يعرض النسختين بعد توحيدهما على نفس المقياس لتتضح **تغيرات الشكل**. "
           "لاحظ: Standardization وMin-Max وRobust لا تغيّر الشكل إطلاقًا (Skewness ثابت) — هي تغيّر الموقع والمقياس فقط.")

st.markdown("## التحويلات التي تغيّر الشكل")
formula(r"y = \log(1 + x)", title="Log / log1p", symbols={"x": "قيمة غير سالبة"},
        intuition="يضغط القيم الكبيرة أكثر من الصغيرة، فيقصّر الذيل الأيمن. الفروق النسبية تصبح فروقًا مطلقة.",
        example="x = 9 ← 2.30، x = 99 ← 4.61، x = 999 ← 6.91: كل عشرة أضعاف تضيف ≈ 2.3 فقط.")
formula(r"y^{(\lambda)} = \begin{cases} \dfrac{x^{\lambda} - 1}{\lambda} & \lambda \neq 0 \\ \log x & \lambda = 0 \end{cases}",
        title="Box-Cox (x > 0)", symbols={r"\lambda": "معامل القوة يُقدَّر بالإمكان الأعظم ليجعل التوزيع أقرب للطبيعي"},
        intuition="عائلة تشمل log (λ=0) والجذر (λ=0.5) والتحويل الخطي (λ=1)، ونترك البيانات تختار.",
        example=f"λ المقدرة لـ{col}: {stats.boxcox(x + (1 if x.min() <= 0 else 0))[1]:.3f}"
                + (" (أُضيف 1 لأن العمود يحتوي أصفارًا)" if x.min() <= 0 else ""))
st.markdown("**Yeo-Johnson** يعمم Box-Cox ليقبل الصفر والقيم السالبة. **Winsorization/Clipping** لا يغيّر الشكل الأساسي "
            "بل يقص الأطراف فقط.")

st.markdown("## التحويلات التي تغيّر المقياس فقط (Scaling)")
formula(r"z = \frac{x - \mu}{\sigma}", title="Standardization",
        symbols={r"\mu": "المتوسط (من بيانات التدريب)", r"\sigma": "الانحراف المعياري (من بيانات التدريب)"},
        intuition="كم انحرافًا معياريًا تبعد القيمة عن المتوسط. الناتج بمتوسط 0 وانحراف 1، والشكل كما هو.",
        example="μ = 50، σ = 10: x = 65 ← z = 1.5")
formula(r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}", title="Min-Max normalization",
        symbols={r"x_{\min}, x_{\max}": "أصغر وأكبر قيمة في التدريب"},
        intuition="يضع القيم في [0, 1]. قيمة متطرفة واحدة تحدد x_max فتضغط الباقي قرب الصفر.",
        example="القيم [10, 20, 30, 1000] ← [0, 0.01, 0.02, 1]")
formula(r"x' = \frac{x - \operatorname{median}(x)}{\operatorname{IQR}(x)}", title="Robust scaling",
        symbols={"IQR": "Q3 − Q1"}, intuition="يستخدم مقاييس متينة فلا تشوهه القيم المتطرفة.")

st.markdown("### مقارنة Scalers على بيانات Wine (مقاييس مختلفة جدًا)")
w = wine()
feat = st.multiselect("الخصائص", [c for c in w.columns if c != "cultivar"], default=["proline", "magnesium", "hue"],
                      key="tr_wine")
if feat:
    scalers = {"Original": None, "StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
    fig = go.Figure()
    for sname, sc in scalers.items():
        vals = w[feat].to_numpy() if sc is None else sc.fit_transform(w[feat])
        for j, f in enumerate(feat):
            fig.add_trace(go.Box(y=vals[:, j], name=f, legendgroup=sname, offsetgroup=f, x=[sname] * len(vals),
                                 showlegend=(sname == "Original"), marker_color=[PALETTE["coral"], PALETTE["purple"],
                                                                                   PALETTE["amber"], PALETTE["sky"]][j % 4]))
    fig.update_layout(boxmode="group", height=420, yaxis_type="log" if st.toggle("مقياس لوغاريتمي (يخفي القيم السالبة بعد Scaling)", False, key="tr_log") else "linear")
    plot(fig)
    st.caption("قبل Scaling: proline بالمئات وhue قرب 1 — أي خوارزمية تعتمد على المسافة سترى proline فقط.")

st.markdown("## Standardization مقابل Normalization")
comparison_table([
    {"": "الناتج", "Standardization": "متوسط 0، انحراف 1، غير محدود", "Min-Max": "[0, 1]", "Robust": "وسيط 0، IQR = 1"},
    {"": "متانة أمام القيم المتطرفة", "Standardization": "متوسطة", "Min-Max": "ضعيفة", "Robust": "عالية"},
    {"": "يغيّر الشكل؟", "Standardization": "لا", "Min-Max": "لا", "Robust": "لا"},
    {"": "مناسب لـ", "Standardization": "PCA، الانحدار المنظم، SVM", "Min-Max": "شبكات عصبية، صور، مدخلات محدودة", "Robust": "بيانات ملوثة"},
])

st.markdown("## متى نحتاج Scaling؟")
comparison_table([
    {"الخوارزمية": "KNN، K-Means، SVM", "Scaling؟": "ضروري", "لماذا": "تعتمد على المسافات"},
    {"الخوارزمية": "PCA", "Scaling؟": "ضروري غالبًا", "لماذا": "تعظم التباين الذي يعتمد على الوحدة"},
    {"الخوارزمية": "Ridge / Lasso", "Scaling؟": "ضروري", "لماذا": "العقوبة تعامل كل المعاملات بالتساوي"},
    {"الخوارزمية": "الشبكات العصبية", "Scaling؟": "ضروري عمليًا", "لماذا": "استقرار التدرج"},
    {"الخوارزمية": "OLS (بدون تنظيم)", "Scaling؟": "غير ضروري للتنبؤ", "لماذا": "يغيّر تفسير المعاملات فقط"},
    {"الخوارزمية": "أشجار القرار، Random Forest، Gradient Boosting", "Scaling؟": "غير ضروري", "لماذا": "التقسيم على عتبات لا يتأثر بالمقياس"},
])
why("اضبط (fit) الـScaler على بيانات التدريب فقط ثم طبّقه (transform) على الاختبار.",
    "حساب μ وσ من كامل البيانات يُدخل معلومات من الاختبار إلى التدريب (Preprocessing leakage). الحل: Pipeline.")
st.code("from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neighbors import KNeighborsClassifier\n\n"
        "pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())\n"
        "pipe.fit(X_train, y_train)          # scaler learns mean/std from X_train only\n"
        "pipe.score(X_test, y_test)", language="python")

if at_least("advanced"):
    st.markdown("## متقدم: التفسير بعد التحويل")
    st.markdown("- إن كان y = log(الدخل): زيادة وحدة في x ← تغير بنسبة ≈ 100·β% (بدقة 100·(e^β − 1)%).\n"
                "- إعادة التحويل Back-transform لمتوسط log لا تعطي المتوسط الأصلي (تعطي المتوسط الهندسي تقريبًا) — "
                "انحياز إعادة التحويل Retransformation bias (Duan's smearing).")
if at_least("research"):
    researcher_note(["التحويل يغيّر السؤال: متوسط log(x) ليس log(متوسط x). صرّح بالمقياس الذي تُفسَّر عليه النتائج.",
                     "لا تحوّل المتغيرات لتحقيق الطبيعية فقط إن كانت النماذج لا تتطلبها (مثل GLM بتوزيع مناسب)."])
real_world(["هل يحتاج النموذج Scaling فعلًا؟", "هل التحويل قابل للتفسير لجمهور التقرير؟", "هل يوجد صفر أو قيم سالبة؟",
            "هل الـScaler داخل Pipeline التدريب؟"])

page_footer("transformations",
            takeaways=["log/Box-Cox/Yeo-Johnson تغيّر الشكل؛ Standard/Min-Max/Robust تغيّر المقياس فقط.",
                       "Scaling ضروري للخوارزميات المعتمدة على المسافة والتنظيم، وغير ضروري للأشجار.",
                       "Fit على التدريب فقط."],
            mistakes=["Standardization لتصحيح الالتواء.", "Min-Max مع قيم متطرفة.", "Scaling قبل التقسيم.",
                      "log على قيم صفرية دون +1."])
