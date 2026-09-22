# Research notes

Access date for all web sources below: **22 September 2026**. Only decisions that changed the code or the curriculum are
recorded.

## Tooling and platform

| Source | Organisation | URL | Why used | Decision affected |
| --- | --- | --- | --- | --- |
| Streamlit developer docs + the version-matched docs bundled in the installed package (1.63) | Snowflake / Streamlit | https://docs.streamlit.io/ | Current multipage API, theming, testing | `st.navigation`/`st.Page` with `app_pages/`; theme in `config.toml`; `AppTest` smoke tests; `st.mermaid_chart` for diagrams; `width="stretch"` instead of deprecated `use_container_width` |
| Community Cloud: app dependencies & Python versions | Streamlit | https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies | Deployment constraints | Supported Python 3.10–3.14, default 3.12; `requirements.txt` only, no system packages → client-side Mermaid/Graphviz, no heavy binaries |
| pandas 3 string migration guide | pandas | https://pandas.pydata.org/docs/user_guide/migration-3-strings.html | Environment had pandas 3.0 | Type helpers in `utils/types.py` use `pd.api.types` instead of `dtype == object`; `select_dtypes("object")` avoided |
| scikit-learn user guide | scikit-learn | https://scikit-learn.org/stable/user_guide.html | Pipelines, imputers, CV splitters, `TargetEncoder` | Leakage-safe generated code; out-of-fold target encoding demo; Group/Time splits |
| Pandera docs (0.29, Jan 2026; Python 3.10–3.14; pandas 2/3) | Union.ai / community | https://pandera.readthedocs.io/ | Modern DataFrame validation | Recommended in the Validation module; `import pandera.pandas as pa` API; optional dependency |
| GX Core changelog (1.16.x, Apr 2026) | Great Expectations | https://docs.greatexpectations.io/docs/core/changelog/ | Is GX maintained? | Presented as the enterprise option; not a dependency |
| fg-data-profiling on PyPI (rename of ydata-profiling, Apr 2026) | Data-Centric AI Community | https://pypi.org/project/fg-data-profiling/ | Old package no longer updated | Automated EDA page teaches the new name/import (`from data_profiling import ProfileReport`); not a dependency |
| missingno GitHub | R. Bilogur | https://github.com/ResidentMario/missingno | Maintenance status | Listed as "maintenance mode"; missingness plots re-implemented in Plotly |
| Sweetviz on PyPI | fbdesignpro | https://pypi.org/project/sweetviz/ | Maintenance status | Listed with limited activity; not a dependency |
| Open-source AutoML projects in 2026 (overview) | MLJAR | https://mljar.com/blog/open-source-automl-projects-in-2026/ | Which AutoML tools are active | FLAML, AutoGluon active; TPOT rebuilt; auto-sklearn last release 2023 → "legacy". Built an in-house mini-AutoML instead of a heavy dependency |
| Evidently (GitHub) | Evidently AI | https://github.com/evidentlyai/evidently | Drift monitoring landscape | Mentioned for observability; PSI/KS implemented in `utils/drift.py` |

## Methodology

| Source | Why used | Decision affected |
| --- | --- | --- |
| Rubin (1976), Biometrika — doi:10.1093/biomet/63.3.581 | Definitions of MCAR/MAR/MNAR | Missingness module and simulator |
| Little (1988), JASA — doi:10.1080/01621459.1988.10478722 | Test of MCAR | EM-based implementation in `utils/missing.py`, with stated limitations |
| Little & Rubin (2019); van Buuren (2018) | Imputation and Rubin's rules | Multiple-imputation demo; "m ≥ % missing" guidance (White, Royston & Wood 2011) |
| Tukey (1977); Iglewicz & Hoaglin (1993); Leys et al. (2013) | Outlier rules | IQR shown as exploration tool, MAD/modified-z as robust alternative, decision tool instead of auto-deletion |
| Liu, Ting & Zhou (2008) — doi:10.1109/ICDM.2008.17 | Isolation Forest | Isolation animation and scoring formula |
| Wang & Strong (1996); ISO/IEC 25012 | Data-quality dimensions | Dimension list; transparent weighted score |
| Fayyad et al. (1996); Wirth & Hipp (2000) | KDD, CRISP-DM | Workflows module; unified course workflow |
| Wickham (2014) — doi:10.18637/jss.v059.i10 | Tidy data | Data Concepts module |
| Wasserstein & Lazar (2016) — doi:10.1080/00031305.2016.1154108 | ASA statement on p-values | Interpretation text never says a test "proves"; effect sizes + CIs always reported |
| Delacre, Lakens & Leys (2017) | Welch's t-test as default | Test selector recommends Welch |
| Hastie, Tibshirani & Friedman (2009), §7.10.2 | Selection-before-CV leakage | Leakage demo 2 |
| Kaufman et al. (2012); Kapoor & Narayanan (2023) | Leakage taxonomy and prevalence | Leakage module structure |
| Breiman (2001); Shmueli (2010) | Explain vs predict | What is Data Science? advanced section |
| Barocas, Hardt & Narayanan (2023); Kleinberg et al. (2016) | Fairness metrics and impossibility | Ethics module |

## Modules added beyond the brief

- **Invalid Values & Consistency** as its own module (units, cross-field and label consistency deserved separation
  from outliers).
- **Observability & Drift** with PSI/KS simulator, schema diff, lineage and a runnable data contract.
- **Interactive Data Story** as a ten-stop guided project with a planted leakage column.
- **Decision log** shared across labs, exported as CSV and appended to the Capstone report.
- **Truth-based scoring** of cleaning: the challenge dataset is generated from a clean ground truth, so learners can
  measure how close their cleaning gets to reality.
