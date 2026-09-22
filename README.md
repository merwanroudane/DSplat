# أكاديمية علم البيانات التفاعلية · Data Science Interactive Academy

**منصة تفاعلية متكاملة لتعلّم علم البيانات من البيانات الخام إلى التحليل والذكاء الاصطناعي**

Developed by **Dr. Marwan Roudane** · تطوير: **الدكتور مروان رودان**

An Arabic, right-to-left, multi-page Streamlit course in Data Science — from raw data, quality and cleaning, through
EDA, statistics, feature engineering, data mining and machine learning, to automated and AI-assisted data science with
a human in the loop.

## Features

- **63 pages** in 13 groups: Start Here, Data Foundations, Data Quality, Data Cleaning, EDA, Statistics, Features & ML,
  Data Mining, Modern Data Science, Labs, Projects, Professional Practice, Resources.
- Every lesson follows *Concept → Intuition → Mathematics → Example → Code → Run → Visualization → Interpretation →
  Mistakes → Exercise → Quiz → Real-world context*, with **Beginner / Advanced / Research** explanation levels.
- **Interactive labs:** Dataset Explorer, Importing Data Lab, Data Quality Dashboard (transparent weighted score),
  Rule Builder, MCAR/MAR/MNAR simulator, Imputation experiment, Missing Data Lab, Outlier Lab + Decision Tool,
  Manual Cleaning Lab (scored against ground truth), EDA Lab, Pipeline Builder (generates pandas and leakage-safe
  scikit-learn code), Manual vs Automated vs Hybrid Lab, Automated Report (Markdown/HTML), Mini-AutoML, Drift simulator.
- **Animations** with Play/Pause/Previous/Next/Reset/Speed: K-Means centroid movement, Isolation Forest splits,
  confidence-interval coverage, missingness mechanisms, mean-imputation effect, before/after transformations,
  PCA rotation, CV folds, workflow steps. A *reduced motion* switch disables autoplay.
- **44 quizzes, 161 questions** (MCQ, true/false, scenario, code and chart interpretation) with explanations, and three-level
  exercises with model answers.
- **Projects:** six end-to-end case studies, a ten-stop interactive data story, and a capstone that assembles a
  downloadable report.
- **11 built-in datasets** (9 synthetic with documented, intentional problems + Iris and Wine). See `docs/datasets.md`.
- Searchable Arabic/English glossary (105 terms), cheat sheets, decision tools, references, progress tracking with
  JSON export/import, bookmarks and a shared decision log.

## Installation

Python **3.10–3.13** (developed and tested on 3.11).

### pip + venv

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### uv

```bash
uv venv
uv pip install -r requirements.txt
```

Optional extras: `pip install pandera` (live Pandera demo), `pip install pytest ruff` (development).

## Running

```bash
streamlit run streamlit_app.py
```

## Project structure

```text
streamlit_app.py        entry point: page config, theme, sidebar, st.navigation
config.py               author, platform name, limits (single place for the identity)
core/                   curriculum registry, navigation, page template, session state, theme
components/             callouts, cards, formulas, code labs, quizzes, animations, diagrams, dataset picker
content/                quiz bank, exercises, glossary
utils/                  datasets, loader, profiling, validation, quality, missing, outliers, cleaning,
                        statistics, pipeline, mining, drift, report, plotting, types
app_pages/<group>/      one script per page
assets/css/theme.css    RTL rules and design-system styles (colours live in .streamlit/config.toml)
data/synthetic/         CSV exports of the built-in datasets
docs/                   architecture, curriculum map, datasets, research notes, references
scripts/build_docs.py   regenerates docs/curriculum.md, docs/datasets.md and data/synthetic/
tests/                  unit tests + headless smoke test of every page
```

## Adding content

- **A module:** add a `Module(...)` entry in `core/curriculum.py` and create its file under `app_pages/`. Start the page
  with `page_header("<id>")` and end with `page_footer("<id>", takeaways=[...], mistakes=[...])`. Navigation, search,
  progress and the curriculum doc pick it up automatically (`python scripts/build_docs.py`).
- **A dataset:** write a generator in `utils/datasets.py` and register it in `DATASET_INFO` (purpose, variables,
  intentional issues, suggested lessons). It then appears in every dataset picker.
- **A quiz:** add a list of `Q(...)` items under the module id in `content/quizzes.py` (types: mcq, tf, scenario,
  code, chart). Exercises go in `content/exercises.py`.
- **A lab:** create a page under `app_pages/labs/`, register it with `kind="lab"`, and use `dataset_picker`,
  `code_lab` and `stepper` from `components/`.

## Tests

```bash
python -m pytest
```

`tests/test_app_smoke.py` renders every page headlessly with Streamlit's `AppTest` (a few minutes).

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. On https://share.streamlit.io create an app pointing to `streamlit_app.py`.
3. Choose Python 3.11 or 3.12 in *Advanced settings*. Dependencies install from `requirements.txt`; no system
   packages or secrets are needed (Mermaid and Graphviz diagrams render in the browser).

## Troubleshooting

- **Arabic text looks garbled in an uploaded CSV:** the importer tries UTF-8, UTF-8-SIG, Windows-1256 and Latin-1;
  re-save the file as UTF-8 if needed.
- **A single column after upload:** the delimiter was not detected; the importer warns about this.
- **Animations are distracting:** enable *تقليل الحركة* in the sidebar.
- **Progress lost after closing the browser:** progress lives in the session; export it from *تقدّمي* and import it later.

## Privacy and security

No user code is executed, uploaded files stay in the session, and nothing is sent to external services. All built-in
data is synthetic or public and non-sensitive.
