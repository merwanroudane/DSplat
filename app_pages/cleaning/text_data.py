import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from components.diagrams import flow
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import reviews
from utils.plotting import plot

page_header("text_data")
rv = reviews()

st.markdown("## من النص إلى الأرقام")
flow(["Raw text", "Cleaning", "Normalization", "Tokenization", "Stopwords", "Stemming/Lemmatization",
      "Representation (BoW / TF-IDF / Embeddings)"])
st.caption("وحدة موجّهة لعلم البيانات: كيف نحوّل النص إلى خصائص، دون تحويل المقرر إلى تخصص NLP كامل.")

STOP = {"the", "a", "an", "this", "my", "new", "is", "it", "and", "for", "to", "of", "with", "bought"}
ARABIC_DIACRITICS = re.compile(r"[ً-ْـ]")


def clean(t: str, lower: bool, html: bool, punct: bool, arabic: bool) -> str:
    if html:
        t = re.sub(r"<[^>]+>", " ", t)
    if lower:
        t = t.lower()
    if arabic:
        t = ARABIC_DIACRITICS.sub("", t)
        t = re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    if punct:
        t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stem(w: str) -> str:
    for suf in ("ing", "ly", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w


st.markdown("## خط تنظيف تفاعلي")
c1, c2, c3, c4, c5, c6 = st.columns(6)
o_html = c1.toggle("HTML", True, key="tx_html")
o_lower = c2.toggle("lower", True, key="tx_lower")
o_punct = c3.toggle("الترقيم", True, key="tx_punct")
o_ar = c4.toggle("تطبيع عربي", True, key="tx_ar")
o_stop = c5.toggle("Stopwords", True, key="tx_stop")
o_stem = c6.toggle("Stemming", False, key="tx_stem")
sample_ids = st.multiselect("اختر مراجعات", rv["review_id"].tolist(),
                            default=rv[rv["text"].str.contains("<br>|!!!")]["review_id"].head(2).tolist()
                            + rv[rv["language"] == "ar"]["review_id"].head(1).tolist(), key="tx_ids")
rows = []
for rid in sample_ids:
    t = rv.loc[rv["review_id"] == rid, "text"].iloc[0]
    c = clean(t, o_lower, o_html, o_punct, o_ar)
    tokens = c.split()
    if o_stop:
        tokens = [w for w in tokens if w not in STOP]
    if o_stem:
        tokens = [stem(w) for w in tokens]
    rows.append({"original": t, "cleaned": c, "tokens": " | ".join(tokens)})
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
comparison_table([
    {"الخطوة": "Cleaning", "الغرض": "إزالة HTML والروابط والرموز غير المفيدة", "الحذر": "الرموز التعبيرية قد تحمل مشاعر"},
    {"الخطوة": "Normalization", "الغرض": "توحيد الحالة وأشكال الحروف", "الحذر": "«US» (الدولة) تصبح «us»"},
    {"الخطوة": "Tokenization", "الغرض": "تقسيم النص إلى وحدات", "الحذر": "العربية: الضمائر المتصلة (كتابهم)"},
    {"الخطوة": "Stopwords", "الغرض": "حذف الكلمات الشائعة", "الحذر": "«not» مهمة في المشاعر"},
    {"الخطوة": "Stemming", "الغرض": "قص اللواحق آليًا", "الحذر": "ينتج جذورًا غير قاموسية (studi)"},
    {"الخطوة": "Lemmatization", "الغرض": "إرجاع الصيغة القاموسية", "الحذر": "يحتاج قاموسًا وتحليلًا نحويًا"},
])

st.markdown("## N-grams وBag of Words")
ng = st.slider("N-gram range (1..n)", 1, 3, 2, key="tx_ng")
en = rv[rv["language"] == "en"]["text"].map(lambda t: clean(t, True, True, True, False))
cv = CountVectorizer(ngram_range=(1, ng), stop_words=list(STOP), min_df=3)
X = cv.fit_transform(en)
freq = pd.Series(np.asarray(X.sum(axis=0)).ravel(), index=cv.get_feature_names_out()).sort_values(ascending=False)
fig = go.Figure(go.Bar(x=freq.head(20).values, y=freq.head(20).index, orientation="h", marker_color=PALETTE["purple"]))
fig.update_layout(title=f"أكثر {20} مصطلحًا تكرارًا — المصفوفة {X.shape[0]}×{X.shape[1]} (sparse)", height=460,
                  yaxis=dict(autorange="reversed"))
plot(fig)
st.caption(f"كثافة المصفوفة: {X.nnz / (X.shape[0] * X.shape[1]):.2%} فقط غير صفري — لذلك تُخزن Sparse.")

st.markdown("## TF-IDF")
st.latex(r"\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\frac{1 + N}{1 + \text{df}(t)} + 1")
st.markdown("- **tf**: تكرار المصطلح t في المستند d.\n- **df**: عدد المستندات التي تحوي t.\n- **N**: عدد المستندات.\n"
            "- الكلمة الموجودة في كل مكان (product) وزنها منخفض؛ الكلمة المميزة لمستند وزنها مرتفع. (صيغة scikit-learn بـsmooth_idf.)")
tf = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOP), min_df=3)
T = tf.fit_transform(en)
labels = rv.loc[en.index, "sentiment"]
top = {}
for lab in ["positive", "negative", "neutral"]:
    m = np.asarray(T[(labels == lab).to_numpy()].mean(axis=0)).ravel()
    top[lab] = pd.Series(m, index=tf.get_feature_names_out()).nlargest(6).index.tolist()
st.dataframe(pd.DataFrame(top), hide_index=True)
st.caption("المصطلحات الأعلى وزنًا في المتوسط لكل فئة مشاعر.")


def _classify(features: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    vec = CountVectorizer(ngram_range=(1, 2)) if features == "BoW" else TfidfVectorizer(ngram_range=(1, 2))
    pipe = make_pipeline(vec, LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipe, en, labels, cv=5, scoring="f1_macro")
    return f"Macro-F1 (5-fold CV) = **{scores.mean():.3f}** ± {scores.std():.3f}"


code_lab("tx_clf", "تصنيف المشاعر: BoW مقابل TF-IDF داخل Pipeline",
         lambda p: ("pipe = make_pipeline(" + ("CountVectorizer" if p["features"] == "BoW" else "TfidfVectorizer")
                    + "(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))\n"
                    "cross_val_score(pipe, texts, labels, cv=5, scoring='f1_macro')"),
         _classify, params=lambda: {"features": st.radio("التمثيل", ["BoW", "TF-IDF"], horizontal=True, key="tx_feat")},
         explanation="المتجه Vectorizer داخل Pipeline فيتعلم المفردات وIDF من folds التدريب فقط. البيانات اصطناعية بسيطة "
                     "لذا الأداء مرتفع؛ النصوص الحقيقية أصعب بكثير.")

st.markdown("## Embeddings وتمثيلات Transformers (مفهوميًا)")
st.markdown(
    "- **Word embeddings** (word2vec، GloVe): متجه كثيف لكل كلمة؛ الكلمات المتشابهة معنى متقاربة.\n"
    "- **Sentence embeddings**: متجه للجملة كاملة (مثل نماذج sentence-transformers) — مفيد للتجميع والبحث الدلالي.\n"
    "- **Transformers**: التمثيل يعتمد على السياق؛ «bank» في «river bank» تختلف عن «bank account».\n"
    "- **مقارنة بـTF-IDF**: الـEmbeddings تفهم المرادفات (excellent ≈ great)، وTF-IDF يعاملها ككلمات منفصلة."
)
toy = {"great": (0.9, 0.8), "excellent": (0.85, 0.9), "happy": (0.7, 0.6), "poor": (-0.8, -0.7), "damaged": (-0.9, -0.5),
       "late": (-0.6, -0.9), "kettle": (0.05, -0.1), "blender": (0.1, -0.05)}
fig = go.Figure(go.Scatter(x=[v[0] for v in toy.values()], y=[v[1] for v in toy.values()], text=list(toy), mode="markers+text",
                           textposition="top center", marker=dict(size=12, color=PALETTE["coral"])))
fig.update_layout(title="توضيح مفاهيمي لفضاء Embeddings ثنائي الأبعاد (قيم توضيحية)", height=360)
plot(fig)
why("استخدم Embeddings عندما تهم المرادفات والسياق وتتوفر موارد حسابية؛ وTF-IDF كخط أساس سريع وقابل للتفسير.",
    "TF-IDF + نموذج خطي غالبًا قوي بشكل مفاجئ على النصوص القصيرة، وتفسيره مباشر عبر أوزان الكلمات.")

if at_least("advanced"):
    st.markdown("## متقدم: خصوصية العربية")
    st.markdown("التطبيع العربي: حذف التشكيل والتطويل، توحيد الألف (أ إ آ ← ا)، الياء/الألف المقصورة، التاء المربوطة. "
                "الصرف غني فالـStemming البسيط يضر؛ أدوات مثل CAMeL Tools وFarasa تقدم تحليلًا صرفيًا. "
                "واللهجات والكتابة بالحروف اللاتينية (Arabizi) تحديات إضافية.")
    counts = Counter(" ".join(rv[rv["language"] == "ar"]["text"].map(lambda t: clean(t, True, True, True, True))).split())
    st.dataframe(pd.Series(counts).nlargest(10).rename("count").to_frame())
if at_least("research"):
    researcher_note(["وثّق خطوات المعالجة المسبقة للنص بدقة؛ تغييرات صغيرة تغيّر النتائج.",
                     "لا تستخدم نماذج لغوية خارجية على نصوص حساسة دون موافقة وحوكمة."])
real_world(["ما لغة/لهجة النصوص؟", "هل الرموز التعبيرية أو علامات التعجب تحمل معنى؟", "هل النفي محفوظ بعد المعالجة؟",
            "هل تُحفظ النسخة الأصلية من النص؟"])

page_footer("text_data",
            takeaways=["النص يمر بتنظيف وتطبيع وتجزئة قبل التمثيل الرقمي.", "BoW وTF-IDF خطوط أساس قوية وقابلة للتفسير.",
                       "Embeddings تلتقط المعنى والسياق.", "ضع Vectorizer داخل Pipeline."],
            mistakes=["حذف «not» مع Stopwords.", "تطبيق Stemming إنجليزي على العربية.", "حساب IDF على كل البيانات قبل التقسيم."])
