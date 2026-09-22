# MASTER BUILD PROMPT
## منصة عربية متكاملة لتعليم Data Science باستخدام Python وStreamlit

**Project Owner / Developer:** الدكتور مروان رودان — **Dr. Marwan Roudane**

---

# 0. المهمة الكبرى

أنت لا تبني Dashboard عاديًا، ولا Demo، ولا Prototype، ولا مجرد مجموعة صفحات Streamlit تحتوي نصوصًا وأكوادًا مبعثرة.

أريد منك أن تعمل كفريق متكامل يضم في الوقت نفسه:

- Senior Data Scientist
- Senior Python Engineer
- Senior Streamlit Engineer
- Data Science Curriculum Designer
- University-Level Data Science Instructor
- Statistician
- Machine Learning Engineer
- Data Mining Specialist
- Data Visualization Specialist
- Data Quality Engineer
- Data Governance Specialist
- UX/UI Designer
- Interactive Learning Designer
- Research Assistant
- Technical Writer
- QA Engineer
- Software Architect

وأن تبني **منصة تعليمية عربية كاملة، متعددة الصفحات، تفاعلية، حديثة، قابلة للتوسعة، وقابلة للنشر الفعلي** لتدريس مقرر شامل في **Data Science** من الأساسيات إلى المفاهيم الحديثة.

المنصة يجب أن تجمع بين:

1. الشرح النظري الأكاديمي.
2. الشرح الرياضي والإحصائي.
3. الشرح البصري والـInteractive.
4. أمثلة عملية قابلة للتنفيذ.
5. أكواد Python واضحة.
6. مختبرات Interactive Labs.
7. رسوم وDiagrams وAnimations.
8. اختبارات Quizzes وتمارين.
9. مشاريع واقعية End-to-End Projects.
10. الطرق التقليدية Manual Data Science.
11. الطرق الحديثة Automated Data Science.
12. AI-Assisted Data Science.
13. AutoML وAutomated EDA وAutomated Cleaning.
14. Human-in-the-Loop Data Science.
15. أفضل الممارسات الحديثة في Data Quality وReproducibility وValidation وData Governance.

الهدف النهائي هو بناء **مقرر Data Science متكامل داخل Streamlit** يمكن استخدامه فعليًا في التعليم الجامعي، التدريب المهني، والدراسة الذاتية.

---

# 1. شرط البحث العميق Deep Research Requirement

قبل بناء المنصة، وأثناء بنائها، لا تعتمد فقط على معلوماتك الداخلية.

أريد منك أن تقوم ببحث معمق وحديث في المصادر الموثوقة، وأن تضيف بنفسك أي موضوعات أو أدوات أو ممارسات حديثة تراها ضرورية لكي يكون المنهج قويًا ومواكبًا للعصر.

## يجب أن تبحث في مصادر مثل:

- Official Streamlit Documentation
- Python Documentation
- pandas Documentation
- NumPy Documentation
- SciPy Documentation
- statsmodels Documentation
- scikit-learn Documentation
- Plotly Documentation
- Altair Documentation
- Matplotlib Documentation
- PyData ecosystem documentation
- Data quality frameworks
- Data validation frameworks
- AutoML frameworks
- Automated EDA libraries
- Data profiling tools
- Data observability concepts
- Academic syllabi from strong universities
- University Data Science curricula
- Modern Data Science textbooks
- Statistical learning references
- Machine learning references
- Data mining references
- Data engineering references where relevant
- Reproducible research best practices
- Recent AI-assisted analytics approaches
- Recent LLM/Data Agent approaches

## البحث يجب أن يحقق الآتي:

- اكتشاف الموضوعات التي ينقصها البرومبت وإضافتها.
- مقارنة الأدوات الحالية وعدم استخدام مكتبة قديمة فقط لأنها مشهورة تاريخيًا.
- التحقق من أن المكتبات المقترحة ما زالت maintained قدر الإمكان.
- التحقق من توافق المكتبات مع الإصدارات الحديثة من Python.
- التحقق من أفضل ممارسات Streamlit الحديثة.
- التحقق من أفضل طريقة حاليًا لبناء Multipage Navigation.
- التحقق من الأدوات المناسبة للـInteractive Visualization.
- التحقق من أدوات Automated EDA الحالية.
- التحقق من أدوات Data Validation الحديثة.
- التحقق من الأدوات الممكن استخدامها على Streamlit Community Cloud.

## مهم جدًا

إذا وجدت أثناء البحث موضوعًا مهمًا غير مذكور في هذا البرومبت، **أضفه تلقائيًا** إذا كان يخدم المنهج.

لا تلتزم حرفيًا بالقائمة إذا كانت هناك بنية تعليمية أفضل.

لكن لا تحذف أي محور أساسي مذكور هنا دون سبب تقني أو تعليمي واضح.

---

# 2. Research Notes & Sources

أنشئ داخل المشروع ملفًا مثل:

`docs/research_notes.md`

أو:

`docs/references.md`

وسجل فيه أهم المصادر التي اعتمدت عليها في تحديث المحتوى والبنية التقنية.

لكل مصدر اذكر عند الإمكان:

- Title
- Organization / Author
- URL
- Access date
- لماذا تم استخدام المصدر؟
- أي قرار تقني أو تعليمي تأثر به؟

لا تملأ الملف بروابط عشوائية. استخدم مصادر موثوقة ومباشرة قدر الإمكان.

---

# 3. الهوية الأكاديمية للمنصة

يجب أن تظهر هوية المطور في المنصة بشكل احترافي:

**الدكتور مروان رودان**  
**Dr. Marwan Roudane**

يمكن أن تظهر في:

- Home page
- About page
- Footer
- Course information card

لكن بشكل أكاديمي راقٍ وليس بصورة دعائية مبالغ فيها.

ضع الاسم في ملف إعدادات مركزي مثل:

```python
APP_AUTHOR_AR = "الدكتور مروان رودان"
APP_AUTHOR_EN = "Dr. Marwan Roudane"
```

ولا تكرر الاسم Hard-coded في كل ملف.

---

# 4. اسم المنصة

اقترح عدة أسماء احترافية ثم اختر الأفضل من حيث الهوية الأكاديمية.

أمثلة فقط:

- Data Science Interactive Academy
- Interactive Data Science Lab
- Modern Data Science Academy
- Data Science Learning Studio
- Data Science Research & Learning Lab

يمكن إضافة Subtitle عربي مثل:

**منصة تفاعلية متكاملة لتعلم علم البيانات من البيانات الخام إلى التحليل والذكاء الاصطناعي**

لكن لا تستخدم اسمًا طفوليًا أو تسويقيًا بشكل مبالغ فيه.

---

# 5. اللغة

اللغة الرئيسية للمنصة:

**العربية**

مع الحفاظ على المصطلحات التقنية والعلمية باللغة الإنجليزية.

مثال:

- القيم المفقودة Missing Values
- النقاط الشاذة Outliers
- التحليل الاستكشافي للبيانات Exploratory Data Analysis (EDA)
- هندسة الخصائص Feature Engineering
- البيانات الفئوية Categorical Data
- التحقق من صحة البيانات Data Validation
- جودة البيانات Data Quality
- اكتشاف الأنماط Pattern Discovery

لا تترجم المصطلح الإنجليزي ثم تخفيه.

الهدف هو أن يتعلم الطالب المفهوم بالعربية والمصطلح العلمي الإنجليزي في الوقت نفسه.

---

# 6. RTL / LTR

يجب دعم العربية دعمًا ممتازًا.

- Arabic text → RTL
- Python code → LTR
- Mathematical formulas → natural rendering
- Variable names → LTR
- Tables → readable with Arabic labels
- English technical expressions → LTR where necessary

لا تسمح بأن يؤدي RTL إلى تشويه:

- Code
- Equations
- File paths
- Library names
- Function names

---

# 7. الفلسفة التعليمية

لا أريد موسوعة صامتة.

أريد منصة تتبع التسلسل:

**Concept → Intuition → Mathematics → Example → Code → Run → Visualization → Interpretation → Mistakes → Exercise → Quiz → Real-world Context**

لكل مفهوم مهم، حاول أن تعرض:

1. ما هو؟
2. لماذا نحتاجه؟
3. ما المشكلة التي يحلها؟
4. ما الفكرة البديهية Intuition؟
5. ما الأساس الرياضي؟
6. كيف يُنفذ في Python؟
7. ماذا ينتج؟
8. كيف نفسر النتيجة؟
9. ما الأخطاء الشائعة؟
10. متى لا نستخدمه؟
11. ما البدائل؟
12. ما علاقته بالموضوع التالي؟

---

# 8. مستويات الشرح

أضف على الأقل مستويين:

## Beginner Mode

- شرح أكبر.
- مصطلحات أبسط.
- أكواد قصيرة.
- رياضيات تدريجية.
- أمثلة مباشرة.

## Advanced Mode

- تفاصيل إحصائية ورياضية أعمق.
- Assumptions.
- Edge Cases.
- Diagnostics.
- Trade-offs.
- Advanced methods.
- References.

يمكن أيضًا إضافة:

## Research Mode

إذا رأيت ذلك مناسبًا، وفيه:

- Methodological notes
- Statistical cautions
- Researcher notes
- Reproducibility
- Reporting recommendations

---

# 9. أنماط التعلم Learning Modes

وفّر أو صمم المعمارية بحيث تدعم:

- Learn
- Explore
- Practice
- Project

## Learn
شرح منظم.

## Explore
تغيير Parameters ورؤية النتائج.

## Practice
Exercises وQuizzes.

## Project
End-to-End Workflow.

---

# 10. التصميم المرئي

أريد تصميمًا:

- Modern
- Bright
- Warm
- Clean
- Academic
- Elegant
- Accessible

تجنب:

- Black backgrounds
- Very dark blue
- Dark navy
- Heavy neon colors
- Excessive gradients
- Excessive emojis

استخدم Palette دافئة ومشرقة، مثل:

- White
- Ivory
- Warm Beige
- Cream
- Amber
- Golden Yellow
- Coral
- Peach
- Soft Red
- Soft Purple
- Light Cyan
- Light Sky Blue كـAccent فقط

لا تجعل اللون الأخضر هو اللون المهيمن على الواجهة.

---

# 11. Design System

أنشئ Design System مركزيًا:

- Color tokens
- Typography
- Spacing
- Border radius
- Shadow levels
- Card styles
- Alert styles
- Formula styles
- Code styles
- Quiz styles
- Section headers

لا تكرر CSS بشكل عشوائي في كل صفحة.

---

# 12. الصفحة الرئيسية Home

يجب أن تكون الصفحة الرئيسية Landing Page قوية وتحتوي على:

- اسم المنصة.
- وصف أكاديمي قصير.
- اسم المطور.
- ما الذي سيتعلمه المستخدم؟
- Course Roadmap.
- Progress Overview.
- Learning Paths.
- Continue Learning button.
- Explore Dataset button.
- Start First Module button.

أنشئ Roadmap تفاعلية للرحلة:

Raw Data
→ Data Collection
→ Data Understanding
→ Data Validation
→ Data Quality
→ Data Cleaning
→ EDA
→ Statistics
→ Feature Engineering
→ Data Mining
→ Machine Learning
→ Evaluation
→ Interpretation
→ Communication
→ Automation
→ AI-Assisted Data Science

---

# 13. Information Architecture

قبل كتابة جميع الصفحات، صمم Information Architecture واضحة.

Navigation Groups مقترحة:

## Start Here
- Home
- What is Data Science?
- Data Science Roadmap
- Course Guide

## Data Foundations
- Data Concepts
- Data Structures
- Variable Types
- Data Sources
- Importing Data

## Data Quality
- Data Quality Dimensions
- Data Validation
- Missing Values
- Duplicates
- Outliers
- Invalid Values
- Consistency Problems

## Data Cleaning
- Numerical Data
- Categorical Data
- Date/Time Data
- Text Data
- Transformations
- Scaling
- Encoding

## Exploratory Data Analysis
- EDA Foundations
- Univariate
- Bivariate
- Multivariate
- Visualization

## Statistics
- Descriptive Statistics
- Probability Foundations where needed
- Statistical Inference
- Hypothesis Testing
- Effect Size
- Confidence Intervals

## Data Mining
- Pattern Discovery
- Association Rules
- Clustering
- Anomaly Detection
- Dimensionality Reduction

## Machine Learning Foundations
- Supervised Learning
- Unsupervised Learning
- Train/Validation/Test
- Cross Validation
- Evaluation
- Data Leakage

## Modern Data Science
- Automated EDA
- Automated Cleaning
- AutoML
- Data Validation Frameworks
- Data Observability
- AI-Assisted Analytics
- LLMs for Data Analysis
- Data Agents
- Human-in-the-Loop

## Labs
- Missing Data Lab
- Outlier Lab
- Data Cleaning Lab
- EDA Lab
- Pipeline Builder
- Automated vs Manual Lab

## Projects
- Customer Data Project
- Retail Project
- Survey Project
- Time Series Project
- Panel Data Project
- Text Project
- Capstone Project

## Resources
- Glossary
- Cheat Sheets
- References
- About

حسّن هذا الهيكل إذا توصلت إلى بنية تعليمية أفضل.

---

# 14. الفرق بين المجالات

أنشئ وحدة قوية تشرح الفرق والعلاقة بين:

- Data Science
- Data Analysis
- Data Analytics
- Data Mining
- Statistics
- Machine Learning
- Artificial Intelligence
- Business Intelligence
- Data Engineering
- Big Data Analytics
- Deep Learning
- Generative AI
- AutoML
- Automated Analytics

لا تكتفِ بتعريف قاموسي.

لكل مجال وضح:

- Main goal
- Typical questions
- Typical data
- Common methods
- Common tools
- Typical outputs
- Overlap with other fields
- Example project

وأضف Interactive Comparison.

---

# 15. Workflow Frameworks

اشرح وربط الأطر المهمة مثل:

- CRISP-DM
- OSEMN
- KDD process
- Modern ML lifecycle
- Research-oriented Data Science workflow

لكن لا تجعل المنصة أسيرة إطار واحد.

أنشئ Workflow جامعًا يعكس أفضل العناصر المشتركة.

---

# 16. Data Concepts

اشرح بالتفصيل:

- Dataset
- Observation
- Record
- Row
- Column
- Variable
- Feature
- Target
- Label
- Predictor
- Response
- Identifier
- Index
- Metadata
- Schema
- Data Dictionary
- Primary Key concept
- Entity
- Measurement

---

# 17. أنواع البيانات

اشرح:

## حسب البنية
- Structured
- Semi-Structured
- Unstructured

## حسب نوع المتغير
- Numerical
  - Continuous
  - Discrete
- Categorical
  - Nominal
  - Ordinal
  - Binary
- Boolean
- Date/Time
- Text
- Geospatial
- Image
- Audio
- Video
- Sequence

## حسب التصميم الزمني/البحثي
- Cross-sectional Data
- Time Series
- Panel Data
- Longitudinal Data
- Repeated Measures
- Event Data
- Transaction Data
- Streaming Data

اجعل هناك Interactive Variable Inspector.

---

# 18. Data Sources & Collection

غطِّ:

- CSV
- Excel
- JSON
- Parquet
- SQL
- APIs
- Web Scraping
- Surveys
- Experiments
- Administrative Data
- Sensors / IoT concept
- Public Data
- Cloud Data
- Logs
- Streaming Data

اشرح الفروق من حيث:

- Structure
- Size
- Reliability
- Update frequency
- Typical problems
- Privacy considerations

---

# 19. Importing Data Lab

أنشئ مختبرًا حقيقيًا لقراءة البيانات.

يدعم على الأقل:

- CSV
- Excel
- JSON

ويعرض:

- Preview
- Shape
- Columns
- Dtypes
- Memory usage
- Missingness summary
- Duplicate summary
- Initial warnings

---

# 20. Data Quality

هذه من أهم الوحدات.

اشرح:

- Completeness
- Accuracy
- Consistency
- Validity
- Uniqueness
- Timeliness
- Integrity
- Reliability
- Conformity
- Relevance where appropriate

أنشئ Data Quality Dashboard.

لكن لا تنشئ Score واحدًا غامضًا.

إذا استخدمت Data Quality Score:

- اشرح Formula.
- اشرح weights.
- اسمح بتغيير weights.
- اعرض sub-scores.

---

# 21. Data Validation

أضف وحدة حديثة حول Data Validation.

اشرح:

- Schema validation
- Type validation
- Range validation
- Category validation
- Regex validation
- Cross-field validation
- Referential integrity concept
- Business rules
- Temporal consistency

وابحث في الأدوات الحديثة مثل:

- Pandera
- Pydantic where relevant
- Great Expectations or current maintained alternatives

لكن استخدم فقط ما هو مناسب ومحدث بعد البحث.

---

# 22. Missing Values

أنشئ وحدة ضخمة مستقلة.

اشرح:

- NaN
- None
- Null
- NA
- Empty string
- Placeholder values
- Sentinel values

مثل:

-999
9999
N/A
Unknown
?

وضح أن بعض القيم تبدو غير مفقودة برمجيًا لكنها Missing Semantically.

---

# 23. Missingness Mechanisms

اشرح:

- MCAR
- MAR
- MNAR

مع أمثلة واقعية.

أنشئ Interactive Simulation.

أزرار:

- Simulate MCAR
- Simulate MAR
- Simulate MNAR

واعرض تأثيرها على:

- Mean
- Variance
- Distribution
- Correlation
- Regression coefficient

---

# 24. Missingness Diagnostics

غطِّ:

- Count
- Percentage
- Pattern
- Matrix
- Heatmap
- Missingness by group
- Missingness by time
- Missingness correlation concept
- Joint missingness
- Monotone missingness concept

وبالنسبة لـ:

- Cross-sectional data
- Time series
- Panel data

اشرح الفروق في التشخيص.

ناقش الاختبارات الإحصائية المرتبطة بالـMissingness عندما تكون مناسبة، مع توضيح حدودها.

---

# 25. Missing Data Treatment

اشرح:

- Listwise deletion
- Pairwise deletion
- Mean imputation
- Median imputation
- Mode imputation
- Constant imputation
- Forward fill
- Backward fill
- Interpolation
- KNN imputation
- Regression imputation
- Iterative imputation
- Multiple imputation concept
- Model-based methods

لكل طريقة:

- Assumptions
- Advantages
- Risks
- When appropriate
- When inappropriate
- Effect on distribution
- Effect on variance
- Effect on relationships
- Potential bias

أنشئ Interactive Missing Data Experiment.

---

# 26. Duplicates

اشرح:

- Exact duplicates
- Partial duplicates
- Near duplicates
- Legitimate repeated observations
- Repeated measures
- Transaction repetition
- Duplicate IDs
- Key-level duplicates

وضح أن Duplicate ليست دائمًا Observation يجب حذفها.

اعرض:

- duplicated()
- drop_duplicates()
- subset=
- keep=

مع أمثلة متعددة.

---

# 27. Outliers & Anomalies

اشرح الفرق بين:

- Outlier
- Extreme value
- Anomaly
- Influential observation
- Measurement error
- Data error
- Rare event
- Novelty
- Structural break

ثم قسم الوحدة إلى:

## Statistical Methods
- Boxplot
- IQR
- Z-score
- Modified Z-score
- MAD
- Percentile rules
- Robust statistics

## Model-Based Methods
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Robust covariance
- Density-based methods
- Autoencoder concept

وضح أن 1.5 × IQR ليست قاعدة حذف أوتوماتيكية.

---

# 28. Outlier Decision Tool

أنشئ Interactive Decision Tree يسأل:

- Is the value logically possible?
- Does it violate domain rules?
- Could it be a unit problem?
- Could it be a measurement problem?
- Is it present in the original source?
- Is it rare but valid?
- Does it strongly affect the model?

ثم يقترح:

- Keep
- Investigate
- Correct
- Transform
- Cap / Winsorize
- Robust modeling
- Remove

مع تفسير.

---

# 29. Numerical Data Cleaning

غطِّ:

- Invalid numeric values
- Impossible negatives
- Impossible zeros
- Infinite values
- Decimal errors
- Unit inconsistencies
- Range violations
- Scale problems
- Precision
- Rounding
- Skewness
- Heavy tails

---

# 30. Transformations & Scaling

اشرح:

- Log
- Square root
- Power transforms
- Box-Cox
- Yeo-Johnson
- Winsorization
- Clipping
- Standardization
- Normalization
- Min-Max scaling
- Robust scaling

مع Animations تظهر Before/After Distribution.

---

# 31. Categorical Data

اشرح:

- Nominal
- Ordinal
- Binary
- High cardinality
- Rare categories
- Inconsistent labels
- Unknown categories
- Typos
- Whitespace
- Case sensitivity

مثال:

USA
usa
U.S.A
United States

اشرح Standardization of Labels.

---

# 32. Encoding

اشرح:

- Label Encoding
- Ordinal Encoding
- One-Hot Encoding
- Frequency Encoding
- Target Encoding
- Hashing
- Embeddings concept

مع:

- When to use
- Leakage risk
- High cardinality issues
- Dimensionality impact

---

# 33. Text Data

أنشئ وحدة Data Science-focused Text Processing.

اشرح:

- Cleaning
- Normalization
- Tokenization
- Stopwords
- Stemming
- Lemmatization
- N-grams
- Bag of Words
- TF-IDF
- Embeddings
- Sentence embeddings
- Transformer representations concept

لا تحولها إلى NLP specialization كاملة.

---

# 34. Date & Time Data

اشرح:

- datetime parsing
- Formatting
- Year
- Month
- Day
- Week
- Quarter
- Hour
- Time zones
- Irregular time
- Missing timestamps
- Resampling
- Lag
- Lead
- Rolling windows
- Seasonality
- Trend
- Calendar features

---

# 35. EDA Foundations

اشرح الفرق بين:

- Descriptive Analysis
- Exploratory Analysis
- Confirmatory Analysis
- Diagnostic Analysis
- Predictive Analysis

ثم ابنِ EDA Workflow واضحًا.

---

# 36. Univariate Analysis

## Numerical
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Range
- Quantiles
- IQR
- Skewness
- Kurtosis
- Histogram
- Density plot
- Boxplot
- ECDF

## Categorical
- Frequency
- Percentage
- Mode
- Bar chart
- Pareto concept

---

# 37. Bivariate Analysis

## Numerical vs Numerical
- Scatter plot
- Correlation
- Regression line concept

## Numerical vs Categorical
- Group summaries
- Boxplot
- Violin plot

## Categorical vs Categorical
- Contingency table
- Crosstab
- Stacked bar chart

---

# 38. Multivariate Analysis

اشرح:

- Correlation matrix
- Heatmap
- Pairwise relationships
- Pairplot concept
- Parallel coordinates concept
- PCA visualization concept
- Multivariate anomaly concept
- Multicollinearity concept

---

# 39. Statistical Foundations

أضف ما يحتاجه طالب Data Science من:

- Population vs Sample
- Parameter vs Statistic
- Random variable concept
- Distribution concept
- Sampling variability
- Standard error
- Confidence interval
- Effect size
- Statistical significance

بدون تحويل المنصة إلى مقرر Probability كامل.

---

# 40. Statistical Tests

ابنِ Decision System:

Research Question
→ Variable Types
→ Design
→ Assumptions
→ Candidate Test
→ Interpretation

غطِّ على الأقل:

- One-sample t-test
- Independent t-test
- Paired t-test
- ANOVA
- Chi-square
- Pearson correlation
- Spearman correlation
- Mann-Whitney
- Wilcoxon
- Kruskal-Wallis
- Normality tests
- Variance tests

مع:

- H0
- H1
- Statistic
- p-value
- Confidence interval
- Effect size
- Assumptions
- Interpretation

ولا تختزل التحليل إلى p < 0.05 فقط.

---

# 41. Visualization

اشرح اختيار الرسم الصحيح.

غطِّ:

- Histogram
- Bar plot
- Boxplot
- Violin plot
- Scatter plot
- Line plot
- Heatmap
- Density
- ECDF
- Area chart
- Treemap concept
- Maps concept
- Interactive charts

وأضف وحدة عن الأخطاء:

- Misleading axes
- Bad scales
- Too many colors
- 3D chart misuse
- Overplotting
- Truncated axes
- Chartjunk

---

# 42. Feature Engineering

اشرح:

- Feature creation
- Interactions
- Ratios
- Binning
- Polynomial features
- Aggregations
- Date features
- Lag features
- Rolling features
- Text features
- Domain-driven features

ثم Feature Selection:

- Filter methods
- Wrapper methods
- Embedded methods
- Regularization concept
- Tree importance
- Permutation importance

---

# 43. Data Leakage

أنشئ وحدة قوية جدًا حول:

- Target leakage
- Train-test contamination
- Preprocessing leakage
- Temporal leakage
- Group leakage
- Feature leakage

اعرض Demo يبين نموذجًا يحقق نتائج ممتازة ظاهريًا بسبب Leakage.

---

# 44. Train / Validation / Test

اشرح:

- Holdout
- Train
- Validation
- Test
- K-Fold
- Stratified K-Fold
- Group K-Fold
- Time Series Split
- Nested CV concept if appropriate

وضح لماذا Random Split ليس مناسبًا دائمًا.

---

# 45. Data Mining

اشرح:

- Pattern discovery
- Association rules
- Clustering
- Classification
- Anomaly detection
- Sequential patterns concept
- Dimensionality reduction

واربط Data Mining بـData Science وML وStatistics.

---

# 46. Machine Learning Foundations

اشرح بإيجاز عميق ومنظم:

- Supervised learning
- Unsupervised learning
- Semi-supervised learning
- Self-supervised learning concept
- Regression
- Classification
- Clustering
- Dimensionality reduction
- Anomaly detection

لا تحول المنصة إلى مقرر ML كامل، لكن يجب أن تكون الروابط واضحة.

---

# 47. Model Evaluation

## Regression
- MAE
- MSE
- RMSE
- R²

## Classification
- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- ROC-AUC
- PR curve

اشرح:

- Class imbalance
- Threshold concept
- Why accuracy can mislead

---

# 48. Traditional vs Modern Data Science

أنشئ مقارنة شاملة بين:

## Traditional Manual Workflow

Researcher:
- inspects data
- writes code
- checks distributions
- cleans data
- selects methods
- builds features
- models
- interprets

## Modern Automated Workflow

Tools may provide:
- automated profiling
- automatic type inference
- anomaly detection
- automatic chart suggestions
- automated feature engineering
- AutoML
- AI-assisted summaries
- AI-generated code

## Hybrid Workflow

Automatic detection
→ Human review
→ Domain validation
→ Controlled transformation
→ Re-analysis
→ Documentation

اجعل Hybrid Philosophy محورًا واضحًا.

---

# 49. Automated EDA

ابحث في الأدوات الحالية وأدرج الأنسب، مثل أو بدائل:

- ydata-profiling
- Sweetviz
- skimpy
- missingno
- D-Tale concept if appropriate

لكن:

- لا تعتمد المنصة عليها وحدها.
- اجعلها Optional.
- تحقق من maintenance والحالة الحالية قبل اعتمادها.

---

# 50. Automated Data Cleaning

اشرح الفرق بين:

- Manual cleaning
- Rule-based cleaning
- Statistical cleaning
- ML-based cleaning
- AI-assisted cleaning
- Automated cleaning

أنشئ Pipeline يولد:

- Detected issues
- Severity
- Evidence
- Suggested action
- Why?

ولا ينفذ حذفًا أو تعديلًا غير قابل للرجوع بشكل تلقائي.

---

# 51. Automatic Data Discovery

اشرح:

- Schema inference
- Semantic type detection
- Relationship discovery
- Distribution detection
- Pattern discovery
- Anomaly discovery
- Cardinality analysis
- Feature suggestions
- Metadata discovery

---

# 52. Data Observability & Modern Quality

أضف مقدمة حديثة إلى:

- Data observability
- Data drift concept
- Schema drift
- Data freshness
- Data lineage concept
- Data contracts concept
- Data quality monitoring

بحجم يناسب Data Science course وليس Data Engineering specialization.

---

# 53. AI-Assisted Data Science

اشرح:

- AI Assistant
- AI Copilot
- LLM-assisted analysis
- Natural language to code
- Natural language to chart
- Data Agent
- Tool-using agent
- Automated report generation
- AI-assisted documentation

لكن ناقش المخاطر:

- Hallucination
- Wrong statistics
- Data leakage
- Privacy
- Reproducibility
- Hidden transformations
- Over-automation

---

# 54. Human-in-the-Loop

ضع مبدأ واضحًا:

**Automation should assist judgment, not erase judgment.**

أنشئ Interactive Diagram:

Machine Detection
→ Human Review
→ Domain Check
→ Decision
→ Transformation
→ Validation
→ Documentation

---

# 55. Automated vs Manual Lab

يختار المستخدم Dataset ثم Mode:

- Manual
- Automated
- Hybrid

## Manual
الطالب ينفذ الخطوات.

## Automated
المنصة تبني report آليًا.

## Hybrid
المنصة تكتشف وتقترح، والطالب يقرر.

---

# 56. Interactive Code Labs

كل درس تطبيقي مهم يجب أن يحتوي على:

- Code
- Run
- Result
- Explanation
- Reset
- Optional Parameters

لكن لا تستخدم arbitrary:

- eval()
- exec()
- subprocess execution of arbitrary user input

في النسخة المنشورة.

استخدم predefined safe functions.

---

# 57. Explain Code

عند عرض كود مثل:

```python
df.isna().sum()
```

لا تكتفِ بعرضه.

اشرح:

- df
- isna()
- sum()
- ما نوع الناتج؟
- كيف نفسره؟

---

# 58. Interactive Parameter Playgrounds

استخدم:

- Slider
- Selectbox
- Radio
- Toggle
- Number input

أمثلة:

- IQR multiplier
- z-score threshold
- missingness rate
- imputation method
- contamination parameter
- train-test ratio
- number of clusters

واجعل الرسوم والنتائج تتغير فوريًا عند الإمكان.

---

# 59. Animations

أريد Animations تعليمية فعلية.

استخدم الأدوات المناسبة بعد البحث، مثل:

- Plotly frames
- SVG
- Graphviz
- Streamlit components
- Lottie where appropriate
- HTML/CSS/JavaScript components

أمثلة:

- Mean imputation effect
- Outlier movement
- Distribution before/after log
- Standardization
- Train-test split
- K-Means centroid movement
- Isolation Forest concept
- Data leakage demonstration
- Missingness mechanism

إذا كان Full Play/Pause صعبًا تقنيًا في Streamlit، فاختر UX بديلًا قويًا مثل step-based animation بدل pseudo-animation ضعيف.

---

# 60. Animation Controls

عند الإمكان:

- Play
- Pause
- Previous
- Next
- Reset
- Speed

---

# 61. Diagrams

أنشئ Diagrams حقيقية لـ:

- Data Science Lifecycle
- Data Cleaning Pipeline
- Missing Data Decision Tree
- Outlier Decision Tree
- Statistical Test Selector
- Chart Selector
- Manual vs Automated
- ML Lifecycle
- Variable Type Tree
- Data Quality Workflow
- Human-in-the-Loop

---

# 62. Mathematical Layer

استخدم `st.latex()` للصيغ.

اشرح عند الحاجة:

- Mean
- Variance
- Standard deviation
- Z-score
- IQR
- Correlation
- Standardization
- Min-Max normalization
- Distance
- Loss concepts

بعد كل Equation:

1. شرح الرموز.
2. Intuition.
3. Numerical example.
4. Visualization.

---

# 63. Dataset Explorer

أنشئ صفحة مركزية:

**Dataset Explorer**

المستخدم يستطيع:

- Upload CSV
- Upload Excel
- Choose built-in dataset

ثم يحصل على:

- Shape
- Column types
- Missingness
- Duplicates
- Numeric summary
- Categorical summary
- Cardinality
- Potential outliers
- Correlations
- Memory usage
- Warnings
- Validation results
- Recommended next steps

---

# 64. Built-in Datasets

وفر datasets تعليمية متعددة، مثل:

- Retail
- Banking-like synthetic
- Customer
- Survey
- Time series
- Panel-style
- Text
- Sales
- Data-quality challenge dataset

يجب أن تكون هناك Dataset مصممة عمدًا بمشاكل:

- Missing values
- Duplicates
- Wrong types
- Impossible values
- Outliers
- Rare categories
- Inconsistent labels
- Date problems
- Whitespace
- Unit inconsistencies

استخدم Synthetic أوPublic non-sensitive datasets.

---

# 65. Before vs After Cleaning

أنشئ مقارنة Side-by-Side:

- Before
- After

وتعرض:

- Missingness
- Duplicates
- Outliers
- Distribution
- Summary statistics
- Data types
- Quality metrics

---

# 66. Pipeline Builder

أنشئ Interactive Data Cleaning Pipeline Builder.

يختار الطالب:

1. Missing strategy
2. Duplicate strategy
3. Outlier strategy
4. Transformation
5. Scaling
6. Encoding

ثم:

- Build Pipeline
- Show Diagram
- Run Pipeline
- Show Output
- Show Generated Python Code
- Explain each stage

---

# 67. Quizzes

بعد كل وحدة مهمة:

- MCQ
- True/False
- Scenario-based
- Code interpretation
- Chart interpretation

بعد الإجابة:

لا تعرض فقط Correct/Incorrect.

اعرض Explanation.

---

# 68. Exercises

لكل Module:

- Beginner exercise
- Intermediate exercise
- Research challenge

مثال:

- Identify missing values.
- Diagnose their pattern.
- Select a treatment.
- Explain your reasoning.

---

# 69. Case Studies

أنشئ عدة Case Studies كاملة:

1. Customer Data Cleaning
2. Retail Analytics
3. Survey Data
4. Time Series
5. Panel-style Data
6. Text Data

كل Case Study يمر عبر Workflow كامل.

---

# 70. Capstone Project

يجب أن يوجد مشروع نهائي يبدأ من Raw Data وينتهي بتقرير تحليلي.

المراحل:

- Problem definition
- Data import
- Understanding
- Validation
- Quality
- Cleaning
- EDA
- Statistical analysis
- Feature engineering
- Modeling
- Evaluation
- Interpretation
- Visualization
- Final report

---

# 71. Interactive Data Story

أنشئ مشروعًا قصصيًا:

"وصلتنا Dataset خام من مؤسسة."

ثم تظهر المشاكل تدريجيًا:

1. Schema issues
2. Missingness
3. Duplicates
4. Outliers
5. Invalid categories
6. Date problems
7. Leakage risk
8. EDA
9. Modeling
10. Final interpretation

الهدف أن يشعر الطالب أنه يحل مشروعًا حقيقيًا.

---

# 72. Progress Tracking

استخدم `st.session_state` لتتبع:

- Current module
- Completed lessons
- Quiz scores
- Exercises attempted
- Visited labs
- Progress percentage
- Bookmarks

يمكن إضافة SQLite persistence اختياريًا، لكن لا تبدأ بنظام Accounts معقد إذا لم يكن ضروريًا.

---

# 73. Search

أضف Search على الأقل داخل:

- Lessons
- Glossary
- Topics

مثلاً البحث عن:

- Missing Values
- Outliers
- Scaling
- Encoding
- Data Mining
- Leakage

---

# 74. Glossary

أنشئ Glossary عربي/إنجليزي قابلًا للبحث.

مثال:

- Feature
- Observation
- Target
- Imputation
- Normalization
- Standardization
- Outlier
- Anomaly
- Cardinality
- Data Leakage
- Pipeline
- Cross Validation
- Schema
- Metadata

---

# 75. Common Mistakes

في كل وحدة أضف Common Mistakes.

أمثلة:

- Delete every outlier.
- Use mean imputation blindly.
- Scale before splitting.
- Target encode on full data.
- Treat correlation as causation.
- Trust automated cleaning without validation.
- Use accuracy on imbalanced data.

---

# 76. Researcher Notes

أضف Callouts بعنوان:

**Researcher Note**

مثال:

> A statistically unusual value is not automatically a data error.

وملاحظات عن:

- Assumptions
- Bias
- Validity
- Reproducibility
- Domain knowledge

---

# 77. Real-World Thinking

في كل وحدة أضف مربعًا:

**In real projects, what should I check?**

مثلاً Missing Values:

- Is missingness systematic?
- Is it related to a group?
- Is it caused by collection process?
- Is it time-dependent?
- Does missingness itself carry information?

---

# 78. Domain Awareness

استخدم أمثلة مثل:

- Age = 250 → likely invalid
- Income = 0 → may be valid or invalid
- Temperature = -5 → depends on context

الهدف تعليم:

**Statistics + Domain Knowledge**

---

# 79. Reproducibility

أضف وحدة حول:

- Virtual environments
- requirements.txt
- pyproject.toml
- Random seeds
- Versioning
- Git concept
- Notebook vs Script
- Pipelines
- Experiment tracking concept
- Data versioning concept

---

# 80. Data Ethics & Privacy

أضف مقدمة مناسبة حول:

- Privacy
- Sensitive data
- Data minimization
- Bias
- Fairness concept
- Consent concept
- Responsible use
- Avoiding leakage of personal data

لا تحولها إلى مقرر قانوني، لكنها يجب أن تكون موجودة.

---

# 81. Security

في النسخة المنشورة:

لا تستخدم arbitrary execution:

- eval
- exec
- shell commands from user input
- subprocess on arbitrary input

لا ترسل uploaded data إلى API خارجي تلقائيًا.

أي AI integration خارجية يجب أن تكون Optional وواضحة.

---

# 82. File Upload Safety

تحقق من:

- file extension
- file size
- empty file
- malformed CSV
- delimiter problems
- encoding problems
- duplicate columns
- missing headers

اعرض رسائل خطأ واضحة بالعربية.

---

# 83. Accessibility

اهتم بـ:

- Contrast
- Font sizes
- Readable Arabic typography
- Keyboard usability
- Responsive layout
- Mobile/tablet behavior
- Reduced motion consideration

---

# 84. Performance

استخدم:

- `st.cache_data`
- Lazy loading
- Sampling for huge plots
- Avoid unnecessary recomputation
- Optional heavy dependencies

لا تجعل startup بطيئًا بسبب تحميل كل شيء مرة واحدة.

---

# 85. Automated Report

أضف صفحة تولد Automated Data Analysis Report يحتوي:

- Overview
- Types
- Missingness
- Duplicates
- Numeric summary
- Categorical summary
- Outliers
- Correlations
- Charts
- Warnings
- Suggestions

ويجب أن تفرق بوضوح بين:

- Detected issue
- Potential issue
- Recommendation

---

# 86. Explain Why

كل Recommendation مهمة يجب أن يكون معها:

**Why?**

مثال:

"Median imputation may be considered."

Why?

"Because the variable is strongly skewed and the median is more robust to extreme values than the mean."

---

# 87. Decision Support Tools

أنشئ أدوات تفاعلية للإجابة عن:

- What chart should I use?
- What statistical test may fit?
- Should I scale?
- How should I investigate missing values?
- Is this outlier necessarily wrong?
- Which encoding strategy is suitable?
- Can I use mean imputation?

لا تجعلها Chatbot فقط.

استخدم Decision Trees / Rules / Interactive Form.

---

# 88. Comparison Tables

أنشئ جداول مقارنة قوية مثل:

- Standardization vs Normalization
- Mean vs Median
- Outlier vs Anomaly
- Deletion vs Imputation
- Manual EDA vs Automated EDA
- Data Analysis vs Data Analytics
- Data Mining vs ML
- Traditional vs Modern Data Science
- Rule-based vs ML-based cleaning

---

# 89. Modern Topics

بعد البحث، أضف ما هو مناسب من:

- AutoML
- Automated feature engineering
- Data validation
- Data observability
- Data contracts
- Synthetic data
- Embeddings
- LLM-assisted analytics
- Data agents
- Semantic layers concept
- Human-in-the-loop
- Metadata-driven analysis

لكن لا تضف Buzzwords بلا شرح أو سياق.

---

# 90. Libraries

استخدم المكتبات المناسبة بعد البحث، مثل:

- streamlit
- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- plotly
- matplotlib
- altair
- graphviz

وأدوات أخرى عند الحاجة.

أي dependency غير ضرورية يجب تجنبها.

أي مكتبة قديمة أو غير maintained يجب عدم استخدامها فقط لأنها معروفة.

---

# 91. Architecture

ممنوع وضع كل شيء في `app.py`.

أريد Modular Architecture.

مثال أولي:

```text
data_science_academy/
│
├── app.py
├── config.py
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── assets/
│   ├── css/
│   ├── images/
│   ├── icons/
│   └── animations/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── core/
│   ├── navigation.py
│   ├── state.py
│   ├── theme.py
│   └── settings.py
│
├── components/
│   ├── cards.py
│   ├── callouts.py
│   ├── code_lab.py
│   ├── quiz.py
│   ├── diagrams.py
│   ├── formulas.py
│   ├── dataset_viewer.py
│   └── progress.py
│
├── modules/
│   ├── foundations/
│   ├── data_quality/
│   ├── cleaning/
│   ├── eda/
│   ├── statistics/
│   ├── mining/
│   ├── ml/
│   └── modern_data_science/
│
├── labs/
│   ├── missing_lab.py
│   ├── outlier_lab.py
│   ├── cleaning_lab.py
│   ├── eda_lab.py
│   ├── pipeline_lab.py
│   └── automation_lab.py
│
├── utils/
│   ├── data_loader.py
│   ├── validation.py
│   ├── cleaning.py
│   ├── statistics.py
│   └── plotting.py
│
├── docs/
│   ├── architecture.md
│   ├── curriculum.md
│   ├── research_notes.md
│   └── references.md
│
└── tests/
    ├── test_loader.py
    ├── test_validation.py
    ├── test_cleaning.py
    └── test_statistics.py
```

عدّل هذه البنية إذا كان هناك تصميم أفضل.

---

# 92. Multipage Navigation

استخدم الطريقة الحديثة والمناسبة لـStreamlit بعد التحقق من Documentation الحالية.

لا تستخدم Pattern قديمًا إذا كان هناك API أحدث وأكثر استقرارًا.

أنشئ Navigation منظمة ومقسمة إلى Groups.

---

# 93. Page Template

كل Module يجب أن يبدأ بـ:

- Title
- Short description
- Learning objectives
- Prerequisites
- Difficulty
- Estimated study time

ثم المحتوى.

وينتهي بـ:

- Key takeaways
- Common mistakes
- Quiz
- Practice exercise
- Next module

---

# 94. Content Depth

ممنوع أن تكون صفحة مثل Missing Values عبارة عن 3 فقرات وكود واحد.

الصفحات الأساسية يجب أن تكون تعليمية فعلًا وتغطي:

- Theory
- Intuition
- Mathematics
- Diagnostics
- Methods
- Comparison
- Code
- Interactive example
- Visualization
- Interpretation
- Pitfalls
- Exercise
- Quiz
- Real-world note

---

# 95. No Placeholder Policy

ممنوع:

- TODO
- Coming soon
- Placeholder content
- Lorem ipsum
- Add content here
- Dummy buttons
- Fake results

إذا لم تستطع تنفيذ Feature معينة، نفذ بديلًا فعليًا واذكر القرار في `docs/architecture.md`.

---

# 96. No Empty Prototype

لا تعتبر المشروع مكتملًا لمجرد إنشاء الملفات.

يجب أن تعمل فعليًا:

- Navigation
- Datasets
- Run buttons
- Charts
- Labs
- Quizzes
- Animations الأساسية
- RTL
- Progress tracking
- Dataset Explorer
- Pipeline Builder

---

# 97. Code Quality

استخدم:

- Functions
- Reusable components
- Type hints
- Docstrings
- Error handling
- Clear names
- Separation of concerns
- Minimal duplication

ولا تستخدم giant functions طويلة بلا حاجة.

---

# 98. Testing

أضف tests للوظائف المهمة:

- Data loading
- Missing detection
- Duplicate detection
- Type inference
- Outlier helpers
- Scaling
- Validation rules

وأضف Smoke Test أو checks مناسبة للتطبيق عند الإمكان.

---

# 99. Quality Assurance

قبل اعتبار المشروع منتهيًا:

- Run the app.
- Check imports.
- Check navigation.
- Check all datasets.
- Check buttons.
- Check charts.
- Check animations.
- Check Run actions.
- Check Arabic RTL.
- Check mobile layout where possible.
- Check no empty pages.
- Check no broken links.
- Check optional dependency fallbacks.

أصلح الأخطاء التي تظهر.

---

# 100. Documentation

أنشئ README احترافيًا يشرح:

- Project overview
- Features
- Screenshots section placeholder only if screenshots will actually be generated; otherwise omit it
- Installation
- Virtual environment
- pip setup
- uv setup if supported
- Running Streamlit
- Project structure
- Adding a module
- Adding a dataset
- Adding a quiz
- Adding a lab
- Deployment
- Troubleshooting

---

# 101. Deployment

يجب أن يكون المشروع جاهزًا قدر الإمكان لـ:

- Streamlit Community Cloud

تجنب dependencies التي تحتاج System configuration معقدًا إلا إذا كانت Optional.

وفر:

- requirements.txt أو pyproject.toml
- clear Python version guidance
- secrets example فقط إذا احتاج المشروع secrets

ولا تضع API keys في الكود.

---

# 102. Python Compatibility

تحقق أثناء البحث من إصدارات Python المناسبة للمكتبات المستخدمة.

اختر نطاقًا حديثًا ومستقرًا.

لا تدّعي دعم إصدار غير مجرب أو غير متوافق مع dependencies.

---

# 103. Implementation Strategy

نفذ المشروع على مراحل داخل نفس المهمة:

## Phase 1 — Research
- Search current references.
- Inspect Streamlit current APIs.
- Inspect suitable libraries.
- Create research notes.

## Phase 2 — Curriculum Architecture
- Build module map.
- Define prerequisites.
- Define learning objectives.

## Phase 3 — Technical Architecture
- Define folders.
- Reusable components.
- Theme.
- Navigation.

## Phase 4 — Core Functional App
- Home
- Navigation
- Theme
- Dataset Explorer
- Core components

## Phase 5 — Core Curriculum
ابدأ بالوحدات الأساسية الأكثر أهمية.

## Phase 6 — Interactive Labs
- Missing
- Outliers
- Cleaning
- EDA
- Pipeline

## Phase 7 — Modern Data Science
- Automated EDA
- Automated Cleaning
- AutoML concepts
- AI-assisted analytics

## Phase 8 — Projects
- Case studies
- Capstone

## Phase 9 — QA
- Run
- Test
- Fix

لا تتوقف بعد Phase 1 أو 2.

---

# 104. Prioritization Rule

إذا كانت المهمة ضخمة جدًا بحيث لا يمكن إنتاج كل المحتوى دفعة واحدة، فلا تنشئ صفحات فارغة.

بدلًا من ذلك:

1. ابنِ البنية كاملة.
2. نفذ Core Modules بجودة عالية.
3. نفذ المكونات القابلة لإعادة الاستخدام.
4. أضف بقية الوحدات تدريجيًا بجودة متسقة.
5. استخدم محتوى حقيقي دائمًا.

لكن لا تقل: "سأكمل لاحقًا" وتترك المشروع مجرد Skeleton.

---

# 105. Content Expansion Autonomy

أعطيك صلاحية أن تضيف Modules جديدة إذا وجدت أثناء البحث أنها مهمة لمقرر Data Science حديث.

أمثلة محتملة:

- Data validation
- Data drift
- Schema drift
- Feature stores concept
- Metadata systems concept
- Experiment tracking
- Model monitoring concept
- Responsible AI
- Synthetic data

لكن أضف فقط ما يخدم المنهج ولا يحوله إلى مجموعة Buzzwords.

---

# 106. Source Discipline

عند إدخال معلومة حديثة، مكتبة، Framework أو Best Practice:

- تحقق من المصدر.
- فضل official docs.
- تجنب Blog غير موثوق عندما توجد docs رسمية.
- تجنب مكتبة obsolete إذا كان هناك بديل maintained.
- اذكر المصادر المهمة في References.

---

# 107. Design Consistency

لا تجعل كل صفحة تبدو كأنها تطبيق مختلف.

استخدم:

- Shared components
- Shared cards
- Shared typography
- Shared section patterns
- Shared colors
- Shared layout

---

# 108. Academic Tone

المحتوى يجب أن يكون:

- واضحًا.
- دقيقًا.
- أكاديميًا.
- غير متكلف.
- لا يبدو كأنه نص AI تسويقي.
- لا يكثر من العبارات الإنشائية.

اشرح المصطلح مباشرة.

استخدم أمثلة واقعية.

---

# 109. Mathematical Accuracy

كل Formula يجب أن تكون صحيحة.

كل Statistical statement يجب ألا يكون مبالغًا فيه.

لا تستخدم لغة مثل:

"هذا الاختبار يثبت أن..."

إذا كان الصحيح:

"يوفر دليلًا إحصائيًا تحت مجموعة Assumptions معينة."

---

# 110. Interpretation First

لا أريد المنصة أن تعلم الطالب كيف يستدعي Function فقط.

يجب أن تعلمه:

- لماذا يستخدمها؟
- ماذا تقيس؟
- ماذا تعني النتيجة؟
- ما حدودها؟

---

# 111. Built-in Learning Datasets

لكل Dataset ملف Documentation يشرح:

- Dataset purpose
- Variables
- Intentional issues
- Suggested lessons

ولا تستخدم بيانات حساسة حقيقية.

---

# 112. Generated Code

عندما يبني المستخدم Cleaning Pipeline، اعرض الكود المقابل.

لكن اجعل الكود:

- Readable
- Beginner-friendly
- Reproducible
- Sequential

مع comments مفيدة.

---

# 113. Export Features

إذا كان مناسبًا، أضف إمكانات مثل:

- Download cleaned dataset
- Download summary CSV
- Download generated Python code
- Download report as HTML/Markdown إذا كان ذلك عمليًا

لكن لا تجعل هذه Features تؤخر core learning experience.

---

# 114. Responsive UX

اختبر الواجهة على الأقل من حيث التصميم لـ:

- Desktop
- Tablet
- Narrow screens

ولا تستخدم 5 columns ثابتة إذا كانت ستنهار على الهاتف.

---

# 115. Fallbacks

إذا كانت Library optional غير موجودة:

- لا تجعل التطبيق ينهار.
- اعرض رسالة تعليمية.
- استخدم fallback method.

---

# 116. Logging & Debugging

أضف logging داخليًا عند الحاجة للأخطاء المهمة، لكن لا تعرض Stack Trace خام للمستخدم النهائي.

في Developer mode يمكن توفير تفاصيل أكبر.

---

# 117. Course Cohesion

المحتوى يجب أن يكون مترابطًا.

مثال:

- Data Types قبل Encoding.
- Missing Values قبل Imputation.
- Train/Test Split قبل preprocessing pipelines التي قد تسبب leakage.
- Data Quality قبل Automated Cleaning.

ضع prerequisite map.

---

# 118. Curriculum Map

أنشئ ملف:

`docs/curriculum.md`

ويحتوي على جدول:

- Module
- Prerequisites
- Learning objectives
- Practical lab
- Quiz
- Dataset
- Difficulty

---

# 119. Architecture Decisions

أنشئ ملف:

`docs/architecture.md`

واشرح فيه:

- Why this multipage approach?
- Why these libraries?
- Why these components?
- State management strategy
- Performance strategy
- Security strategy
- Optional dependencies

---

# 120. Final Acceptance Criteria

لا أعتبر المشروع منتهيًا إذا كان:

- صفحة واحدة.
- Dashboard فقط.
- Prototype فقط.
- Skeleton فقط.
- نصوص بلا تفاعل.
- صفحات قصيرة جدًا.
- بدون كود قابل للتشغيل.
- بدون بيانات.
- بدون رسوم.
- بدون Labs.
- بدون Quizzes.
- بدون Mathematics.
- بدون Automation.
- بدون Modern Data Science.
- بدون بحث خارجي حديث.
- بدون References.
- بدون اختبارات تشغيل.

---

# 121. Required Final Deliverable

أريد Repository كاملًا يعمل مباشرة قدر الإمكان.

ويحتوي على:

- Streamlit multipage app
- Arabic RTL interface
- Professional warm bright theme
- Developer identity: Dr. Marwan Roudane / الدكتور مروان رودان
- Complete course architecture
- Core theoretical lectures
- Mathematical explanations
- Interactive visual explanations
- Diagrams
- Animations
- Runnable code examples
- Built-in datasets
- Dataset Explorer
- Data Quality Dashboard
- Manual Cleaning Lab
- Automated Cleaning Lab
- Hybrid Workflow Lab
- EDA Lab
- Pipeline Builder
- Quizzes
- Exercises
- Case studies
- Capstone project
- Modern Data Science modules
- Research notes
- References
- Tests
- Documentation
- Deployment files

---

# 122. المطلوب منك قبل كتابة أول سطر Implementation

قم أولًا بما يلي:

1. ابحث في المصادر الحديثة.
2. راجع أحدث Streamlit patterns.
3. اقترح final curriculum architecture.
4. حدد technical architecture.
5. حدد optional vs required dependencies.
6. أنشئ قائمة بالمصادر الأساسية.
7. لاحظ أي Modules مهمة أضفتها بنفسك.

ثم ابدأ التنفيذ مباشرة.

لا تتوقف عند تقديم الخطة.

---

# 123. Final Instruction to Claude

تصرف كأنك المسؤول التقني والأكاديمي الأول عن هذا المشروع.

لا تنفذ المتطلبات بصورة سطحية.

إذا وجدت أثناء البحث أن هناك طريقة تعليمية أو تقنية أفضل مما طلبته، استخدمها بشرط:

- الحفاظ على أهداف المنصة.
- عدم حذف المحاور الأساسية.
- توثيق القرار.

أريد أن تكون النتيجة أقرب إلى:

**Interactive Arabic Data Science University Course + Data Lab + Modern Automated Data Science Platform**

وليست:

**Streamlit Dashboard with a few pages**.

اجعل كل جزء مفيدًا فعليًا للطالب أو الباحث.

اجعل الطالب يفهم البيانات، مشاكلها، تشخيصها، تنظيفها، تحليلها، تفسيرها، وبناء Workflow حديث عليها، سواء بطريقة Manual أو Automated أو Hybrid.

وفي جميع صفحات المنصة حافظ على الهوية الأكاديمية:

**Developed by Dr. Marwan Roudane**  
**تطوير: الدكتور مروان رودان**

---

# END OF MASTER PROMPT
