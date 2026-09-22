# Architecture decisions

## Platform name

Candidates considered: *Data Science Interactive Academy*, *Interactive Data Science Lab*, *Modern Data Science Academy*,
*Data Science Learning Studio*, *Data Science Research & Learning Lab*. Chosen: **Data Science Interactive Academy —
أكاديمية علم البيانات التفاعلية**. "Academy" signals a structured university-level course (not a single tool), and
"Interactive" names the main teaching method. Name and author are defined once in `config.py`.

## Why this multipage approach?

- `st.navigation` + `st.Page` (Streamlit ≥ 1.36; verified on 1.63) instead of the legacy `pages/` auto-discovery.
  Pages live in `app_pages/` so Streamlit does not auto-discover them.
- **Single source of truth:** `core/curriculum.py` defines every module (id, file, group, titles, icon, objectives,
  prerequisites, difficulty, minutes, keywords). Navigation, page headers, search, the progress tracker,
  "next module" links and `docs/curriculum.md` are all generated from it.
- Page files are plain scripts (no wrapping functions), following Streamlit's recommendation; reusable logic lives in
  `utils/` and UI building blocks in `components/`.

## Why these libraries?

| Need | Choice | Reason |
| --- | --- | --- |
| UI | Streamlit 1.63 | Native theming, `st.mermaid_chart`, `st.fragment(run_every=…)` for animations, `AppTest` for testing |
| Data | pandas (2.2+, verified on 3.0), NumPy | Standard; code avoids `object`-dtype assumptions so it works with pandas 3 string dtype |
| Statistics | SciPy, statsmodels | Tests, KDE, Tukey HSD, OLS trendlines |
| ML | scikit-learn | Pipelines, imputers (incl. `IterativeImputer`), model selection, anomaly detectors |
| Charts | Plotly | Frame-based animations with Play/Pause/slider; interactive hover |
| Diagrams | Mermaid via `st.mermaid_chart`, Graphviz DOT via `st.graphviz_chart` | Rendered client-side; no system binaries needed on Community Cloud |
| Excel | openpyxl | Reading `.xlsx` uploads |

Deliberately **not** required (heavy, or maintenance concerns — see `research_notes.md`): fg-data-profiling /
ydata-profiling, Sweetviz, missingno, Pandera, AutoML frameworks, umap-learn. The platform ships small in-house,
tested equivalents (profiler, validation engine, issue detector, Apriori, mini-AutoML, PSI) and shows the external
tools' code as optional examples. Pandera runs live if installed (`pip install pandera`), otherwise an explanatory
message is shown.

## Why these components?

- `components/callouts.py` — Researcher Note, Real-world checks, Common Mistakes, Why?, Intuition, Domain knowledge.
  Implemented as native `st.container(key="ds-<kind>-N")` so Markdown renders natively; CSS targets the key class.
- `components/formulas.py` — every equation followed by symbols, intuition and a numerical example.
- `components/code_lab.py` — Code → Run → Result → Explanation → Reset with optional parameters. The code shown is
  documentation of a predefined Python function that actually runs; user text is never executed.
- `components/animation.py` — step-based animation with **Play / Pause / Previous / Next / Reset / Speed**, driven by
  `st.fragment(run_every=…)` so only the animated area reruns. Plotly frame animations (with Play/Pause and a slider)
  are used where the motion is continuous (K-Means, transformations, PCA rotation, imputation, outlier pull).
- `components/quiz.py` — MCQ / True-False / Scenario / Code / Chart questions, with an explanation for every answer;
  exercises at three levels with model answers.
- `components/dataset_viewer.py` — one dataset picker used by all labs; includes the user's uploaded file.

## State management strategy

- `core/state.py` initialises all per-user keys once per run: level, reduced-motion flag, completed modules, visited
  pages, quiz scores, exercises, labs visited, bookmarks, uploaded dataset, decision log.
- Persistence without accounts: progress is exported/imported as JSON on the *My Progress* page. No database is needed
  for the core experience (brief §72: "don't start with complex accounts").
- Widget keys are explicit and prefixed per page; callbacks (`on_click`) are used when a button must change other
  widgets' state.

## Performance strategy

- Dataset generators are deterministic and memoised with `functools.lru_cache` (shared, read-only, copies returned).
- Expensive computations (model training, CV comparisons, t-SNE, AutoML, reports) use `st.cache_data`.
- Heavy demos run behind explicit buttons (AutoML, CV scheme comparison, multiple imputation).
- Large scatter plots are sampled (`config.MAX_PLOT_POINTS`).
- Heavy optional libraries are never imported at start-up.

## Security strategy

- No `eval`, `exec`, `subprocess` or shell calls on user input anywhere. Code labs run predefined functions.
- Uploads: extension allow-list, size limit (50 MB, also in `.streamlit/config.toml`), empty-file check, encoding
  detection (UTF-8, UTF-8-SIG, Windows-1256, Latin-1), delimiter sniffing, malformed-row counting (done with the `csv`
  module because pandas can silently shift extra fields), duplicate/missing header warnings, JSON flattening.
  Errors are shown in Arabic; raw tracebacks are hidden (`client.showErrorDetails = "none"`) and logged server-side.
- Uploaded data stays in the session; nothing is sent to external services. There is **no** LLM integration; the
  NL→code demo on the AI-assisted page is a transparent, rule-based simulation and says so.
- Custom regex typed in the Rule Builder is compiled inside a `try/except` with an Arabic error message.

## Optional dependencies and fallbacks

| Optional | Where | Fallback |
| --- | --- | --- |
| pandera | Data Validation page | Built-in rule engine results + install hint |
| fg-data-profiling, sweetviz, skimpy, missingno | Automated EDA page | Built-in profiler; availability shown |
| umap-learn | Dimensionality reduction | t-SNE (scikit-learn) + explanation |

## RTL / LTR

The main area and sidebar are `direction: rtl`. Code blocks, inline code, LaTeX, Plotly, Vega, Graphviz, dataframes,
sliders and JSON are forced LTR in `assets/css/theme.css`. English report previews use `st.container(key="ltr-…")`.

## Design system

Colours are native Streamlit theme tokens in `.streamlit/config.toml`: a light, bright multicolour palette (near-white
background, blue primary, light lavender sidebar, and violet / amber / orange / teal / cyan accents; no pink, no dark
backgrounds, green is not dominant). Each callout type, card column and roadmap stop gets its own light colour. `assets/css/theme.css` adds only what theming cannot:
RTL rules, callouts, cards, formula boxes, roadmap chips, footer, reduced-motion and narrow-screen rules.
`core/theme.py` registers a matching Plotly template.

## Features implemented as alternatives (brief §95)

- **Full Play/Pause animation**: implemented two ways (Plotly frames; fragment-driven stepper). Autoplay is disabled
  when *Reduced motion* is on.
- **AI-assisted analysis**: no external model is called; instead, a rule-based NL→code simulator and an *AI output
  audit* exercise teach reviewing AI outputs without sending data anywhere.
- **SQLite persistence**: replaced by JSON export/import of progress (no server-side personal data).

## Testing

- `tests/test_*.py` — unit tests for loading, validation, type inference, cleaning, imputation, outliers, pipeline code
  generation (generated code must compile), statistics (vs SciPy), Little's test, Apriori, K-Means, PSI.
- `tests/test_app_smoke.py` — every page rendered headlessly with `st.testing.v1.AppTest` at the "research" level.
- During development an additional interaction sweep clicked every button and cycled every keyed selectbox, radio and
  segmented control on every page.
