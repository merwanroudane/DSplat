import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import credit
from utils.plotting import plot

page_header("encoding")

st.markdown("## لماذا نرمّز؟")
st.markdown("أغلب النماذج تعمل على أرقام. **الترميز Encoding** يحوّل الفئات إلى أرقام، وكل طريقة تضيف افتراضًا ضمنيًا "
            "عن العلاقة بين الفئات. الاختيار الخاطئ يخلق ترتيبًا وهميًا، أو ينفجر بالأبعاد، أو يسرّب الهدف.")
demo = pd.DataFrame({"city": ["Rabat", "Cairo", "Amman", "Cairo", "Rabat", "Tunis"],
                     "size": ["S", "L", "M", "M", "S", "L"], "bought": [1, 0, 1, 1, 0, 0]})
method = st.segmented_control("الطريقة", ["Label", "Ordinal", "One-Hot", "Frequency", "Target", "Hashing"],
                              default="One-Hot", key="enc_method")
out = demo.copy()
code = ""
if method == "Label":
    out["city_label"] = out["city"].astype("category").cat.codes
    code = 'df["city_label"] = df["city"].astype("category").cat.codes'
    note = "أرقام اعتباطية (أبجدية): Amman=0 < Cairo=1 < Rabat=2. النموذج الخطي سيفهم ترتيبًا غير موجود."
elif method == "Ordinal":
    out["size_ord"] = out["size"].map({"S": 1, "M": 2, "L": 3})
    code = 'df["size_ord"] = df["size"].map({"S": 1, "M": 2, "L": 3})   # explicit order'
    note = "مناسب لـsize لأن الترتيب حقيقي. الافتراض الضمني: المسافة S→M تساوي M→L."
elif method == "One-Hot":
    out = pd.concat([out, pd.get_dummies(out["city"], prefix="city", dtype=int)], axis=1)
    code = 'pd.get_dummies(df["city"], prefix="city", dtype=int)   # or OneHotEncoder(handle_unknown="ignore")'
    note = "عمود لكل فئة بلا ترتيب. مع k فئة ينتج k أعمدة (أو k−1 مع drop='first' للنماذج الخطية بتقاطع)."
elif method == "Frequency":
    out["city_freq"] = out["city"].map(out["city"].value_counts(normalize=True))
    code = 'df["city_freq"] = df["city"].map(df["city"].value_counts(normalize=True))'
    note = "عمود واحد مهما كان عدد الفئات. فئتان بنفس التكرار تصبحان متطابقتين."
elif method == "Target":
    out["city_target"] = out.groupby("city")["bought"].transform("mean")
    code = 'df["city_target"] = df.groupby("city")["bought"].transform("mean")   # ⚠ leakage if done on all data'
    note = "يستبدل الفئة بمتوسط الهدف فيها. قوي جدًا لكنه يسرب الهدف إن حُسب على الصف نفسه وبيانات الاختبار."
else:
    k = 4
    out["city_hash"] = out["city"].map(lambda v: zlib.crc32(v.encode()) % k)  # stable hash
    code = ("from sklearn.feature_extraction import FeatureHasher\n"
            'FeatureHasher(n_features=4, input_type="string").transform(df[["city"]].values)')
    note = "يوزع الفئات على عدد ثابت من الأعمدة بدالة Hash؛ لا يحتاج قاموسًا ويتحمل الفئات الجديدة، لكن قد تتصادم فئتان."
st.dataframe(out, hide_index=True)
st.code(code, language="python")
st.info(note, icon=":material/info:")

st.markdown("## المقارنة")
comparison_table([
    {"الطريقة": "Label", "متى": "الأشجار فقط، أو الهدف في التصنيف", "خطر Leakage": "لا", "High cardinality": "جيد", "الأبعاد": "1"},
    {"الطريقة": "Ordinal", "متى": "متغيرات ترتيبية حقيقية", "خطر Leakage": "لا", "High cardinality": "—", "الأبعاد": "1"},
    {"الطريقة": "One-Hot", "متى": "Nominal بفئات قليلة، نماذج خطية", "خطر Leakage": "لا", "High cardinality": "سيئ", "الأبعاد": "k"},
    {"الطريقة": "Frequency", "متى": "Cardinality عالية، الأشجار", "خطر Leakage": "منخفض", "High cardinality": "جيد", "الأبعاد": "1"},
    {"الطريقة": "Target", "متى": "Cardinality عالية مع هدف", "خطر Leakage": "مرتفع جدًا", "High cardinality": "ممتاز", "الأبعاد": "1"},
    {"الطريقة": "Hashing", "متى": "فئات ضخمة ومتجددة", "خطر Leakage": "لا", "High cardinality": "ممتاز", "الأبعاد": "ثابت"},
    {"الطريقة": "Embeddings", "متى": "شبكات عصبية، تشابه دلالي", "خطر Leakage": "حسب التدريب", "High cardinality": "ممتاز", "الأبعاد": "d صغير"},
])

st.markdown("## أثر الأبعاد")
k = st.slider("عدد الفئات في المتغير", 2, 2000, 300, key="enc_k")
n = st.slider("عدد الصفوف", 500, 100000, 5000, step=500, key="enc_n")
st.markdown(f"One-Hot سينتج **{k}** عمودًا؛ بمتوسط **{n / k:.1f}** صف لكل فئة. "
            + ("⚠️ فئات كثيرة بعدد صفوف قليل ← تقديرات غير مستقرة وOverfitting." if n / k < 20 else "مقبول نسبيًا."))

st.markdown("## عرض حي: Target encoding يسرّب")
st.caption("نضيف عمودًا فئويًا **عشوائيًا تمامًا** (لا علاقة له بالهدف) بـ1000 فئة ثم نرمّزه بطريقتين.")
df = credit()
rng = np.random.default_rng(0)
df["random_code"] = rng.integers(0, 1000, len(df)).astype(str)
train = df.sample(frac=0.7, random_state=1)
test = df.drop(train.index)
naive_map = train.groupby("random_code")["default"].mean()
tr_naive = train["random_code"].map(naive_map)
te_naive = test["random_code"].map(naive_map).fillna(train["default"].mean())
oof = pd.Series(np.nan, index=train.index)
for fit_idx, val_idx in KFold(5, shuffle=True, random_state=0).split(train):
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    stats = fit.groupby("random_code")["default"].agg(["mean", "count"])
    prior, m = fit["default"].mean(), 10
    smooth = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
    oof.iloc[val_idx] = val["random_code"].map(smooth).fillna(prior).to_numpy()
stats = train.groupby("random_code")["default"].agg(["mean", "count"])
prior = train["default"].mean()
te_oof = test["random_code"].map((stats["mean"] * stats["count"] + prior * 10) / (stats["count"] + 10)).fillna(prior)
rows = []
for name, tr_x, te_x in (("Naive target encoding", tr_naive, te_naive), ("Out-of-fold + smoothing", oof, te_oof)):
    mdl = LogisticRegression().fit(tr_x.to_frame(), train["default"])
    rows.append({"الطريقة": name, "AUC على التدريب": roc_auc_score(train["default"], mdl.predict_proba(tr_x.to_frame())[:, 1]),
                 "AUC على الاختبار": roc_auc_score(test["default"], mdl.predict_proba(te_x.to_frame())[:, 1])})
res = pd.DataFrame(rows)
fig = go.Figure()
fig.add_trace(go.Bar(x=res["الطريقة"], y=res["AUC على التدريب"], name="train AUC", marker_color=PALETTE["coral"]))
fig.add_trace(go.Bar(x=res["الطريقة"], y=res["AUC على الاختبار"], name="test AUC", marker_color=PALETTE["purple"]))
fig.add_hline(y=0.5, line_dash="dash", annotation_text="random")
fig.update_layout(barmode="group", yaxis=dict(range=[0.4, 1]), height=340)
plot(fig)
st.dataframe(res.round(3), hide_index=True)
why("احسب Target encoding خارج الـFold (Out-of-fold) مع Smoothing، أو استخدم sklearn.preprocessing.TargetEncoder الذي يطبق Cross-fitting.",
    "في الطريقة الساذجة كل صف يرى هدفه ضمن متوسط فئته؛ مع فئات صغيرة يصبح الترميز شبه نسخة من الهدف. "
    "النتيجة: AUC تدريب مرتفع على عمود عشوائي، ثم انهيار على الاختبار.")
st.code("from sklearn.preprocessing import TargetEncoder   # scikit-learn >= 1.3\n"
        'enc = TargetEncoder(target_type="binary", cv=5, random_state=0)\n'
        'X_train_enc = enc.fit_transform(X_train[["city"]], y_train)   # cross-fitting inside\n'
        'X_test_enc = enc.transform(X_test[["city"]])', language="python")

if at_least("advanced"):
    st.markdown("## متقدم: Embeddings للفئات")
    st.markdown("في الشبكات العصبية تُتعلم لكل فئة متجهات بطول صغير (مثل 8) أثناء التدريب، فتقترب الفئات المتشابهة في السلوك. "
                "بديلًا: Embeddings مسبقة التدريب لأسماء الفئات النصية (مثل أسماء المنتجات) عبر نماذج لغوية.")
if at_least("research"):
    researcher_note(["في النماذج الخطية للاستدلال: One-hot مع فئة مرجعية واضحة وتفسير المعاملات كفروق عنها.",
                     "Target encoding يجعل تفسير المعاملات صعبًا؛ مناسب للتنبؤ أكثر من التفسير."])
real_world(["هل تظهر فئات جديدة في الإنتاج؟ (handle_unknown)", "كم صفًا لكل فئة؟", "هل الترميز داخل Pipeline التدريب؟",
            "هل النموذج خطي أم شجري؟ (يغيّر الاختيار)"])

page_footer("encoding",
            takeaways=["كل ترميز يحمل افتراضًا عن العلاقة بين الفئات.", "One-hot للفئات القليلة، Ordinal للترتيب الحقيقي.",
                       "Target encoding قوي لكنه يسرب إن لم يُحسب خارج الـFold."],
            mistakes=["Label encoding لمتغير Nominal في نموذج خطي.", "Target encoding على كامل البيانات.",
                      "One-hot لآلاف الفئات."])
