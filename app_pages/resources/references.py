import streamlit as st

from core.page import footer, page_header

page_header("references")
st.markdown("المصادر التي بُني عليها المحتوى والقرارات التقنية. القائمة التفصيلية مع سبب استخدام كل مصدر في "
            "`docs/references.md` و`docs/research_notes.md`.")

REFS = {
    "التوثيق الرسمي للأدوات": [
        ("Streamlit documentation — st.navigation, st.Page, theming, AppTest", "https://docs.streamlit.io/"),
        ("pandas documentation (incl. migration guide to pandas 3 string dtype)", "https://pandas.pydata.org/docs/"),
        ("NumPy documentation", "https://numpy.org/doc/"),
        ("SciPy stats reference", "https://docs.scipy.org/doc/scipy/reference/stats.html"),
        ("statsmodels documentation", "https://www.statsmodels.org/"),
        ("scikit-learn user guide (pipelines, imputation, model selection, TargetEncoder)", "https://scikit-learn.org/stable/user_guide.html"),
        ("Plotly Python documentation (animations, frames)", "https://plotly.com/python/"),
        ("Pandera documentation", "https://pandera.readthedocs.io/"),
        ("Pydantic documentation", "https://docs.pydantic.dev/"),
        ("Great Expectations (GX Core) documentation", "https://docs.greatexpectations.io/"),
        ("fg-data-profiling on PyPI (successor of ydata-profiling)", "https://pypi.org/project/fg-data-profiling/"),
        ("Streamlit Community Cloud — app dependencies and Python versions",
         "https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies"),
    ],
    "كتب ومراجع منهجية": [
        ("James, Witten, Hastie, Tibshirani & Taylor — An Introduction to Statistical Learning (with Python), 2023", "https://www.statlearning.com/"),
        ("Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, 2nd ed., 2009", "https://hastie.su.domains/ElemStatLearn/"),
        ("Wickham, Çetinkaya-Rundel & Grolemund — R for Data Science, 2nd ed. (concepts of tidy data & EDA)", "https://r4ds.hadley.nz/"),
        ("McKinney — Python for Data Analysis, 3rd ed., 2022", "https://wesmckinney.com/book/"),
        ("Little & Rubin — Statistical Analysis with Missing Data, 3rd ed., Wiley, 2019", None),
        ("van Buuren — Flexible Imputation of Missing Data, 2nd ed., 2018", "https://stefvanbuuren.name/fimd/"),
        ("Han, Pei & Tong — Data Mining: Concepts and Techniques, 4th ed., 2022", None),
        ("Tukey — Exploratory Data Analysis, Addison-Wesley, 1977", None),
        ("Barocas, Hardt & Narayanan — Fairness and Machine Learning, 2023", "https://fairmlbook.org/"),
    ],
    "أوراق علمية": [
        ("Rubin (1976) Inference and missing data. Biometrika 63(3)", "https://doi.org/10.1093/biomet/63.3.581"),
        ("Little (1988) A test of missing completely at random… JASA 83(404)", "https://doi.org/10.1080/01621459.1988.10478722"),
        ("Fayyad, Piatetsky-Shapiro & Smyth (1996) From data mining to knowledge discovery in databases. AI Magazine 17(3)", None),
        ("Wirth & Hipp (2000) CRISP-DM: Towards a standard process model for data mining", None),
        ("Wang & Strong (1996) Beyond accuracy: What data quality means to data consumers. JMIS 12(4)", None),
        ("Liu, Ting & Zhou (2008) Isolation Forest. ICDM", "https://doi.org/10.1109/ICDM.2008.17"),
        ("Breunig et al. (2000) LOF: Identifying density-based local outliers. SIGMOD", None),
        ("Iglewicz & Hoaglin (1993) How to Detect and Handle Outliers. ASQC", None),
        ("Leys et al. (2013) Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. JESP 49(4)", None),
        ("Wickham (2014) Tidy data. Journal of Statistical Software 59(10)", "https://doi.org/10.18637/jss.v059.i10"),
        ("Breiman (2001) Statistical modeling: The two cultures. Statistical Science 16(3)", None),
        ("Shmueli (2010) To explain or to predict? Statistical Science 25(3)", None),
        ("Wasserstein & Lazar (2016) The ASA statement on p-values. The American Statistician 70(2)", "https://doi.org/10.1080/00031305.2016.1154108"),
        ("Kaufman, Rosset, Perlich & Stitelman (2012) Leakage in data mining. ACM TKDD 6(4)", None),
        ("Kapoor & Narayanan (2023) Leakage and the reproducibility crisis in ML-based science. Patterns 4(9)", None),
        ("Delacre, Lakens & Leys (2017) Why psychologists should by default use Welch's t-test. IRSP 30(1)", None),
        ("White, Royston & Wood (2011) Multiple imputation using chained equations. Statistics in Medicine 30(4)", None),
        ("Kleinberg, Mullainathan & Raghavan (2016) Inherent trade-offs in the fair determination of risk scores. arXiv:1609.05807", "https://arxiv.org/abs/1609.05807"),
    ],
}
for section, items in REFS.items():
    st.markdown(f"### {section}")
    for title, url in items:
        st.markdown(f"- [{title}]({url})" if url else f"- {title}")
st.caption("تاريخ آخر مراجعة لحالة المكتبات: سبتمبر 2026.")
footer()
