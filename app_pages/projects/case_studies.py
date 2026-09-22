import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

from components.callouts import why
from components.diagrams import flow
from core.page import footer, page_header
from core.state import mark_lab
from core.theme import PALETTE, SEQUENCE
from utils.cleaning import clean_customers_reference
from utils.datasets import customers_raw, daily_sales, panel, reviews, survey, transactions
from utils.mining import apriori, association_rules
from utils.plotting import plot

page_header("case_studies")
mark_lab("case_studies")
st.markdown("كل دراسة حالة تمر بنفس الـWorkflow الكامل على بيانات مختلفة. افتح الخطوات بالترتيب.")
flow(["Problem", "Data", "Quality", "Cleaning", "EDA", "Analysis / Model", "Interpretation"])

case = st.segmented_control("دراسة الحالة", ["1. العملاء", "2. التجزئة", "3. الاستبيان", "4. السلسلة الزمنية", "5. Panel", "6. النصوص"],
                            default="1. العملاء", key="cs_case")


def step(title: str, expanded: bool = False):
    return st.expander(title, expanded=expanded, icon=":material/chevron_left:")


if case == "1. العملاء":
    raw = customers_raw()
    clean, log = clean_customers_reference(raw)
    with step("1. المشكلة", True):
        st.markdown("شركة اشتراكات تريد فهم **من يغادر ولماذا** قبل تصميم برنامج احتفاظ. السؤال **تشخيصي** أولًا ثم تنبؤي.")
    with step("2. البيانات"):
        st.markdown(f"{len(raw)} صفًا خامًا من نظام CRM قديم؛ الهدف `churned`.")
        st.dataframe(raw.head(6))
    with step("3. الجودة"):
        st.markdown("مشكلات: أنواع خاطئة، Sentinels، تكرار، وحدات، تسميات غير متسقة، تواريخ مختلطة (انظر مختبر التنظيف).")
    with step("4. التنظيف"):
        st.dataframe(pd.DataFrame(log)[["الخطوة", "صفوف/قيم متأثرة"]], hide_index=True)
    with step("5. EDA"):
        c1, c2 = st.columns(2)
        with c1:
            r = clean.groupby("satisfaction")["churned"].mean().reset_index()
            plot(px.bar(r, x="satisfaction", y="churned", color_discrete_sequence=SEQUENCE, labels={"churned": "churn rate"}), height=300)
        with c2:
            r = clean.assign(orders_band=pd.cut(clean["num_orders"].astype(float), [-1, 2, 5, 10, 100])).groupby(
                "orders_band", observed=True)["churned"].mean().reset_index()
            r["orders_band"] = r["orders_band"].astype(str)
            plot(px.bar(r, x="orders_band", y="churned", color_discrete_sequence=[PALETTE["purple"]]), height=300)
    with step("6. التحليل"):
        d = clean.dropna(subset=["satisfaction", "num_orders"])
        X = pd.DataFrame({"low_satisfaction": (d["satisfaction"] <= 2).astype(int), "num_orders": d["num_orders"].astype(float),
                          "days_since_signup": (pd.Timestamp("2024-12-31") - d["signup_date"]).dt.days.fillna(0) / 365})
        m = LogisticRegression().fit(X, d["churned"])
        st.dataframe(pd.DataFrame({"feature": X.columns, "odds ratio": np.exp(m.coef_[0]).round(3)}), hide_index=True)
    with step("7. التفسير"):
        st.markdown("الرضا المنخفض يضاعف احتمال المغادرة تقريبًا، وكل طلب إضافي يخفضه. **ارتباطي لا سببي**: "
                    "لتقييم برنامج احتفاظ نحتاج تجربة. التوصية: استهداف العملاء منخفضي الرضا وقليلي الطلبات مع مجموعة ضابطة.")

elif case == "2. التجزئة":
    tr = transactions()
    with step("1. المشكلة", True):
        st.markdown("سلسلة متاجر تريد: أداء الفروع والفئات، المنتجات التي تُشترى معًا، وشرائح العملاء لحملات موجهة.")
    with step("2. البيانات والجودة"):
        st.markdown(f"{len(tr):,} سطر فاتورة، {tr['invoice_id'].nunique():,} فاتورة، {tr['customer_id'].nunique()} عميل. "
                    "التكرار هنا **مشروع** (سلوك شراء متكرر).")
    with step("3. EDA"):
        m = tr.assign(month=tr["datetime"].dt.to_period("M").dt.to_timestamp()).groupby(["month", "store"])["revenue"].sum().reset_index()
        plot(px.line(m, x="month", y="revenue", color="store", color_discrete_sequence=SEQUENCE), height=320)
        cat = tr.groupby("category")["revenue"].sum().sort_values()
        plot(px.bar(x=cat.values, y=cat.index, orientation="h", color_discrete_sequence=SEQUENCE, labels={"x": "revenue", "y": ""}),
             height=300)
    with step("4. قواعد الارتباط"):
        b = [set(g) for g in tr.groupby("invoice_id")["product"].apply(list)]
        rules = association_rules(apriori(b, 0.03, 2), 0.3).head(6)
        st.dataframe(rules.round(3), hide_index=True)
    with step("5. تجزئة RFM"):
        ref = tr["datetime"].max() + pd.Timedelta(days=1)
        rfm = tr.groupby("customer_id").agg(recency=("datetime", lambda s: (ref - s.max()).days),
                                            frequency=("invoice_id", "nunique"), monetary=("revenue", "sum"))
        for c, asc in (("recency", False), ("frequency", True), ("monetary", True)):
            rfm[c[0].upper()] = pd.qcut(rfm[c].rank(method="first", ascending=asc), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["segment"] = np.select([(rfm["R"] >= 3) & (rfm["F"] >= 3), rfm["R"] <= 1, rfm["M"] >= 4],
                                   ["Loyal", "At risk", "Big spenders"], "Regular")
        st.dataframe(rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1).assign(
            customers=rfm["segment"].value_counts()))
    with step("6. التفسير"):
        st.markdown("قواعد بـLift مرتفع (مثل Pasta ← Tomato Sauce) تقترح عروضًا مجمعة؛ شريحة «At risk» مرشحة لحملة استرجاع. "
                    "قياس أثر أي حملة يتطلب A/B test.")

elif case == "3. الاستبيان":
    sv = survey()
    sv["work_hours"] = sv["work_hours"].replace(-99, np.nan)
    with step("1. المشكلة", True):
        st.markdown("إدارة موارد بشرية تسأل: ما علاقة الضغط وساعات العمل بالرضا الوظيفي؟ وهل يختلف الرضا حسب التعليم؟")
    with step("2. الجودة والفقد"):
        st.dataframe(sv.isna().mean().mul(100).round(1).rename("missing %").to_frame().query("`missing %` > 0"))
        st.markdown("income مفقود بنسبة كبيرة ومرتبط على الأرجح بقيمته (MNAR)؛ stress_score مفقود أكثر لدى الشباب (MAR).")
    with step("3. EDA"):
        edu = ["Primary", "Secondary", "Bachelor", "Master", "PhD"]
        plot(px.box(sv, x="education", y="job_satisfaction", category_orders={"education": edu}, color="education",
                    color_discrete_sequence=SEQUENCE), height=320)
    with step("4. التحليل"):
        d = sv.dropna(subset=["stress_score", "work_hours"])
        r1 = stats.spearmanr(d["stress_score"], d["job_satisfaction"])
        r2 = stats.spearmanr(d["work_hours"], d["stress_score"])
        h = stats.kruskal(*[g["job_satisfaction"] for _, g in sv.groupby("education")])
        st.markdown(f"- Spearman(stress, satisfaction) = **{r1.statistic:.2f}** (p = {r1.pvalue:.2g})\n"
                    f"- Spearman(hours, stress) = **{r2.statistic:.2f}** (p = {r2.pvalue:.2g})\n"
                    f"- Kruskal–Wallis للرضا حسب التعليم: H = {h.statistic:.1f}, p = {h.pvalue:.2g}")
    with step("5. التفسير"):
        st.markdown("الضغط يرتبط سلبيًا بالرضا، وساعات العمل ترتبط بالضغط. تحذيران: (1) التحليل على الحالات المكتملة "
                    "قد ينحاز لأن الشباب أكثر فقدًا لسؤال الضغط؛ (2) البيانات مقطعية فلا نستنتج اتجاه السببية.")

elif case == "4. السلسلة الزمنية":
    ts = daily_sales().set_index("date").asfreq("D")
    with step("1. المشكلة", True):
        st.markdown("التنبؤ بالمبيعات اليومية للأسبوعين القادمين لتخطيط المخزون.")
    with step("2. الجودة"):
        st.markdown(f"أيام غائبة: {int(ts['sales'].isna().sum())} (بعد إعادة الفهرسة)؛ قفزات شاذة؛ structural break في يونيو 2024.")
    ts["sales"] = ts["sales"].interpolate(limit=5)
    ts["promo"] = ts["promo"].fillna(0)
    with step("3. EDA"):
        plot(px.line(ts.reset_index(), x="date", y="sales", color_discrete_sequence=SEQUENCE), height=300)
    with step("4. النموذج (تقسيم زمني)"):
        f = pd.DataFrame({"y": ts["sales"], "lag7": ts["sales"].shift(7), "lag14": ts["sales"].shift(14),
                          "roll7": ts["sales"].shift(1).rolling(7).mean(), "dow": ts.index.dayofweek, "promo": ts["promo"]}).dropna()
        split = f.index >= "2024-10-01"
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(random_state=0).fit(f.loc[~split].drop(columns="y"), f.loc[~split, "y"])
        pred = m.predict(f.loc[split].drop(columns="y"))
        naive = f.loc[split, "lag7"]
        c1, c2 = st.columns(2)
        c1.metric("MAE — النموذج", f"{mean_absolute_error(f.loc[split, 'y'], pred):.1f}")
        c2.metric("MAE — Seasonal naive (قيمة الأسبوع الماضي)", f"{mean_absolute_error(f.loc[split, 'y'], naive):.1f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f.index[split], y=f.loc[split, "y"], name="actual", line=dict(color="#999")))
        fig.add_trace(go.Scatter(x=f.index[split], y=pred, name="model", line=dict(color=PALETTE["coral"])))
        fig.update_layout(height=320)
        plot(fig)
    with step("5. التفسير"):
        st.markdown("قارن دائمًا بـSeasonal naive: إن لم يتفوق النموذج بوضوح فلا يستحق التعقيد. الـLags والـRolling محسوبة "
                    "من الماضي فقط، والتقسيم زمني.")

elif case == "5. Panel":
    p = panel()
    with step("1. المشكلة", True):
        st.markdown("هل ترتبط زيادة الإنفاق على البحث والتطوير بزيادة الإيرادات داخل الشركة نفسها؟")
    with step("2. البنية والجودة"):
        st.markdown(f"{p['firm_id'].nunique()} شركة، {p['year'].min()}–{p['year'].max()}، Panel غير متوازن؛ "
                    f"rnd_spend مفقود {p['rnd_spend'].isna().mean():.1%} ويزداد مع الزمن.")
    with step("3. Pooled مقابل Within"):
        d = p.dropna(subset=["rnd_spend"]).copy()
        d["log_rev"] = np.log(d["revenue"])
        pooled = stats.pearsonr(d["rnd_spend"], d["log_rev"]).statistic
        dm = d[["rnd_spend", "log_rev"]] - d.groupby("firm_id")[["rnd_spend", "log_rev"]].transform("mean")
        within = stats.pearsonr(dm["rnd_spend"], dm["log_rev"]).statistic
        c1, c2 = st.columns(2)
        c1.metric("ارتباط Pooled (بين الشركات)", f"{pooled:.2f}")
        c2.metric("ارتباط Within (داخل الشركة)", f"{within:.2f}")
        plot(px.scatter(dm, x="rnd_spend", y="log_rev", opacity=0.5, color_discrete_sequence=SEQUENCE,
                        labels={"rnd_spend": "R&D (demeaned)", "log_rev": "log revenue (demeaned)"}), height=320)
    with step("4. التفسير"):
        st.markdown("الارتباط Pooled يخلط الفروق بين الشركات الكبيرة والصغيرة؛ التحويل Within (طرح متوسط كل شركة) يقارن الشركة "
                    "بنفسها عبر الزمن — فكرة نماذج Fixed effects. في النمذجة استخدم Group K-Fold حسب الشركة.")

else:
    rv = reviews()
    with step("1. المشكلة", True):
        st.markdown("تصنيف مراجعات المنتجات آليًا (إيجابي/سلبي/محايد) لرصد المشكلات مبكرًا.")
    with step("2. الجودة والتنظيف"):
        st.markdown(f"{len(rv)} مراجعة، {int((rv['language'] == 'ar').sum())} بالعربية؛ HTML وأحرف كبيرة ومسافات.")
    en = rv[rv["language"] == "en"]
    text = en["text"].str.replace(r"<[^>]+>", " ", regex=True).str.lower()
    with step("3. النموذج"):
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000))
        pred = cross_val_predict(pipe, text, en["sentiment"], cv=5)
        rep = pd.DataFrame(classification_report(en["sentiment"], pred, output_dict=True)).T.round(3)
        st.dataframe(rep)
    with step("4. تحليل الأخطاء"):
        errs = en.assign(pred=pred)[pred != en["sentiment"]][["text", "sentiment", "pred"]].head(8)
        st.dataframe(errs, hide_index=True)
    with step("5. التفسير"):
        st.markdown("الفئة المحايدة الأصعب (أقل أمثلة وكلمات أقل تميزًا). المراجعات العربية تحتاج نموذجًا منفصلًا أو "
                    "Embeddings متعددة اللغات. في الإنتاج راقب انجراف المفردات.")
why("طبّق نفس الـWorkflow على بياناتك الخاصة.", "التسلسل الثابت (مشكلة ← جودة ← تنظيف ← استكشاف ← تحليل ← تفسير) يمنع القفز إلى النموذج قبل فهم البيانات.")
footer()
