import streamlit as st

from components.cards import comparison_table
from core.page import footer, page_header, page_link

page_header("cheat_sheets")
t1, t2, t3 = st.tabs(["أدوات قرار سريعة", "جداول مقارنة", "أوراق مراجعة الكود"])

with t1:
    st.markdown("أدوات بقواعد صريحة — ليست Chatbot — تجيب عن الأسئلة المتكررة مع السبب.")
    with st.expander("هل أحتاج Scaling؟", icon=":material/straighten:", expanded=True):
        algo = st.selectbox("الخوارزمية", ["KNN / K-Means / SVM", "PCA", "Ridge / Lasso / Logistic with penalty", "Neural network",
                                           "Linear regression (OLS, no penalty)", "Decision tree / Random Forest / Gradient Boosting"],
                            key="cs_algo")
        ans = {"KNN / K-Means / SVM": ("نعم", "تعتمد على المسافات؛ المتغير الأكبر مقياسًا يهيمن."),
               "PCA": ("نعم غالبًا", "يعظم التباين الذي يعتمد على الوحدات."),
               "Ridge / Lasso / Logistic with penalty": ("نعم", "العقوبة تعامل كل المعاملات بالتساوي."),
               "Neural network": ("نعم عمليًا", "يحسّن استقرار التدرج وسرعة التقارب."),
               "Linear regression (OLS, no penalty)": ("ليس للتنبؤ", "التنبؤ لا يتغير؛ يتغير تفسير المعاملات فقط."),
               "Decision tree / Random Forest / Gradient Boosting": ("لا", "التقسيم على عتبات لا يتأثر بالمقياس.")}[algo]
        st.markdown(f"**{ans[0]}** — لماذا؟ {ans[1]}")
    with st.expander("أي ترميز أستخدم؟", icon=":material/data_object:"):
        c1, c2, c3 = st.columns(3)
        kind = c1.radio("نوع المتغير", ["Nominal", "Ordinal", "Binary"], key="cs_ekind")
        card = c2.radio("عدد الفئات", ["قليل (< 15)", "كبير (15–1000)", "ضخم/متجدد"], key="cs_ecard")
        model = c3.radio("النموذج", ["خطي", "أشجار", "شبكة عصبية"], key="cs_emodel")
        if kind == "Binary":
            rec = ("0/1", "فئتان فقط؛ عمود واحد يكفي.")
        elif kind == "Ordinal":
            rec = ("Ordinal encoding بترتيب صريح", "الترتيب حقيقي؛ صرّح بافتراض المسافات المتساوية.")
        elif card.startswith("قليل"):
            rec = ("One-Hot", "لا يفرض ترتيبًا والأبعاد معقولة.")
        elif card.startswith("ضخم"):
            rec = ("Hashing أو Embeddings" if model == "شبكة عصبية" else "Hashing",
                   "عدد أعمدة ثابت ويتحمل الفئات الجديدة.")
        else:
            rec = ("Target encoding (out-of-fold) أو Frequency" if model != "شبكة عصبية" else "Embeddings",
                   "يتجنب انفجار الأبعاد؛ Target encoding يجب حسابه خارج الـFold لتجنب التسرب.")
        st.markdown(f"**التوصية:** {rec[0]} — لماذا؟ {rec[1]}")
    with st.expander("هل يمكنني استخدام Mean imputation؟", icon=":material/help_center:"):
        c1, c2, c3 = st.columns(3)
        share = c1.radio("نسبة الفقد", ["< 5%", "5–20%", "> 20%"], key="cs_mshare")
        skew = c2.radio("التوزيع", ["متماثل", "ملتوٍ"], key="cs_mskew")
        goal = c3.radio("الهدف", ["وصف سريع", "نموذج تنبؤي", "استدلال إحصائي"], key="cs_mgoal")
        if goal == "استدلال إحصائي":
            st.markdown("**لا يُنصح** — يخفض التباين والأخطاء المعيارية؛ استخدم Multiple Imputation.")
        elif share == "> 20%":
            st.markdown("**لا يُنصح** — كتلة كبيرة عند المتوسط تشوه التوزيع والعلاقات؛ استخدم طريقة نموذجية + مؤشر فقد.")
        elif skew == "ملتوٍ":
            st.markdown("**استخدم الوسيط بدلًا منه** — المتوسط يُسحب نحو الذيل.")
        else:
            st.markdown("**مقبول كخط أساس** — فقد قليل وتوزيع متماثل؛ داخل Pipeline، ومع مؤشر فقد للنماذج.")
    with st.expander("كيف أحقق في القيم المفقودة؟", icon=":material/checklist:"):
        for i, s in enumerate(["حوّل Sentinels (-999, N/A, ?, '') إلى NaN", "احسب العدد والنسبة لكل عمود", "افحص الأنماط والفقد المشترك",
                               "قارن من فُقدت قيمته بمن لم تُفقد (فروق = ليس MCAR)", "افحص الفقد حسب المجموعة والزمن",
                               "اسأل مالك البيانات عن سبب الفقد", "اختر المعالجة حسب الآلية والهدف، ووثّق"]):
            st.checkbox(s, key=f"cs_miss_{i}")
        page_link("missing_values")
    with st.expander("ما الرسم المناسب؟ / ما الاختبار المناسب؟ / هل هذه القيمة الشاذة خطأ؟", icon=":material/insights:"):
        page_link("visualization", label="محدد الرسم Chart Selector")
        page_link("hypothesis_tests", label="نظام اختيار الاختبار")
        page_link("outliers", label="أداة قرار القيم الشاذة")

with t2:
    tables = {
        "Standardization vs Normalization": [
            {"": "الصيغة", "Standardization": "(x − μ)/σ", "Normalization (Min-Max)": "(x − min)/(max − min)"},
            {"": "المدى", "Standardization": "غير محدود", "Normalization (Min-Max)": "[0, 1]"},
            {"": "القيم المتطرفة", "Standardization": "تأثر متوسط", "Normalization (Min-Max)": "تأثر كبير"},
            {"": "الاستخدام", "Standardization": "PCA، الانحدار المنظم، SVM", "Normalization (Min-Max)": "شبكات، صور، مدخلات محدودة"}],
        "Mean vs Median": [
            {"": "التعريف", "Mean": "مجموع/عدد", "Median": "القيمة الوسطى"},
            {"": "القيم المتطرفة", "Mean": "حساس", "Median": "متين"},
            {"": "الاستخدام", "Mean": "توزيعات متماثلة، مجاميع", "Median": "دخل، أسعار، بيانات ملتوية"}],
        "Outlier vs Anomaly": [
            {"": "التعريف", "Outlier": "قيمة بعيدة إحصائيًا", "Anomaly": "مشاهدة من آلية مختلفة"},
            {"": "مثال", "Outlier": "أعلى دخل في المدينة", "Anomaly": "معاملة احتيالية"},
            {"": "الإجراء", "Outlier": "تحقق، ثم احتفظ/صحح/قص", "Anomaly": "تحقيق وتصعيد"}],
        "Deletion vs Imputation": [
            {"": "الافتراض", "Deletion": "MCAR لعدم الانحياز", "Imputation": "MAR للطرق النموذجية"},
            {"": "الخسارة", "Deletion": "صفوف وقوة إحصائية", "Imputation": "خطر تشويه التباين والعلاقات"},
            {"": "متى", "Deletion": "فقد قليل وعشوائي", "Imputation": "فقد متوسط مع متغيرات تفسره"}],
        "Manual EDA vs Automated EDA": [
            {"": "السرعة", "Manual": "بطيء", "Automated": "ثوانٍ"},
            {"": "العمق", "Manual": "مرتبط بالسؤال", "Automated": "شامل وسطحي"},
            {"": "الخطر", "Manual": "الإغفال", "Automated": "إغراق بالمعلومات وثقة زائفة"}],
        "Data Analysis vs Data Analytics": [
            {"": "النطاق", "Analysis": "فحص محدد لسؤال", "Analytics": "ممارسة منهجية مستمرة"},
            {"": "المخرج", "Analysis": "إجابة/تقرير", "Analytics": "مؤشرات، لوحات، أنظمة قرار"}],
        "Data Mining vs Machine Learning": [
            {"": "الهدف", "Data Mining": "اكتشاف أنماط مفهومة", "Machine Learning": "أداء تنبؤي"},
            {"": "التقييم", "Data Mining": "الفائدة والجدّة", "Machine Learning": "الأداء خارج العينة"}],
        "Traditional vs Modern Data Science": [
            {"": "العمل", "Traditional": "يدوي، كود مخصص", "Modern": "أتمتة، AutoML، مساعدة AI"},
            {"": "المبدأ الموصى به", "Traditional": "—", "Modern": "Hybrid: الآلة تقترح والإنسان يقرر"}],
        "Rule-based vs ML-based cleaning": [
            {"": "الشفافية", "Rule-based": "عالية", "ML-based": "منخفضة"},
            {"": "التغطية", "Rule-based": "الحالات المتوقعة", "ML-based": "أنماط معقدة غير متوقعة"},
            {"": "الخطر", "Rule-based": "قواعد قديمة", "ML-based": "تطبيع الحالات النادرة الحقيقية"}],
    }
    pick = st.selectbox("المقارنة", list(tables), key="cs_table")
    comparison_table(tables[pick])

with t3:
    sheet = st.segmented_control("الورقة", ["pandas: الفحص", "pandas: التنظيف", "scikit-learn", "الإحصاء"], default="pandas: الفحص",
                                 key="cs_sheet")
    code = {
        "pandas: الفحص": '''df.shape; df.dtypes; df.head(); df.info(memory_usage="deep")
df.describe(include="all")                  # numeric + categorical summaries
df.isna().sum(); df.isna().mean()           # missing count / share
df.duplicated().sum()                       # exact duplicate rows
df["col"].value_counts(dropna=False)        # frequencies incl. NaN
df["col"].nunique()                         # cardinality
df.select_dtypes("number").corr(method="spearman")''',
        "pandas: التنظيف": '''df = df.replace([-999, "N/A", "?", ""], np.nan)                 # sentinels -> NaN
df["x"] = pd.to_numeric(df["x"].str.replace(r"[$,]", "", regex=True), errors="coerce")
df["d"] = pd.to_datetime(df["d"], format="%Y-%m-%d", errors="coerce")
df["c"] = df["c"].str.strip().str.lower().map(mapping)            # label standardization
df = df.drop_duplicates(); df = df.drop_duplicates(subset="id", keep="first")
df.loc[~df["age"].between(18, 100), "age"] = np.nan               # domain rule
df["x"] = df["x"].clip(*df["x"].quantile([0.01, 0.99]))           # winsorize
df["x_was_missing"] = df["x"].isna().astype(int); df["x"] = df["x"].fillna(df["x"].median())''',
        "scikit-learn": '''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])
pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])
cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")''',
        "الإحصاء": '''from scipy import stats
stats.ttest_ind(a, b, equal_var=False)       # Welch t-test
stats.ttest_rel(before, after)               # paired t-test
stats.mannwhitneyu(a, b)                     # non-parametric, 2 groups
stats.f_oneway(g1, g2, g3); stats.kruskal(g1, g2, g3)
stats.chi2_contingency(pd.crosstab(df.a, df.b))
stats.pearsonr(x, y); stats.spearmanr(x, y)
stats.shapiro(x); stats.levene(g1, g2, center="median")''',
    }[sheet or "pandas: الفحص"]
    st.code(code, language="python")
footer()
