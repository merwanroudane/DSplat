# Curriculum map · خريطة المقرر

Generated from `core/curriculum.py` by `scripts/build_docs.py` — do not edit by hand.

## ابدأ هنا · Start Here

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Home**<br>الرئيسية | — | الصفحة الرئيسية: خريطة المقرر، مسارات التعلّم، والتقدّم. | — | — | — | مبتدئ | 30 |
| **What is Data Science?**<br>ما هو علم البيانات؟ | — | التمييز بين Data Science وData Analysis وData Analytics.<br>فهم التداخل بين Statistics وMachine Learning وData Mining.<br>ربط كل مجال بنوع الأسئلة والمخرجات التي ينتجها. | — | 4 questions | — | مبتدئ | 35 |
| **Workflows & Lifecycle**<br>دورة حياة المشروع | What is Data Science? | شرح مراحل CRISP-DM وOSEMN وKDD.<br>مقارنة الأطر وبيان نقاط قوتها.<br>استخدام Workflow جامع يربط الجودة بالتحليل والتوثيق. | — | 4 questions | — | مبتدئ | 30 |
| **Course Guide**<br>دليل المقرر | — | كيف تستخدم المنصة: المستويات، أنماط التعلّم، خريطة المتطلبات السابقة، والمسارات المقترحة. | — | — | — | مبتدئ | 10 |

## أسس البيانات · Data Foundations

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Data Concepts**<br>مفاهيم البيانات | What is Data Science? | تعريف مكونات الـDataset بدقة.<br>التمييز بين Feature وTarget وIdentifier.<br>قراءة Schema وبناء Data Dictionary. | — | 4 questions | customers_clean | مبتدئ | 35 |
| **Variable & Data Types**<br>أنواع البيانات والمتغيرات | Data Concepts | تصنيف المتغيرات إلى Continuous/Discrete/Nominal/Ordinal/Binary.<br>التمييز بين تصاميم البيانات Cross-sectional/Time Series/Panel.<br>استخدام Variable Inspector لاستنتاج النوع الدلالي. | Variable Inspector | 4 questions | all | مبتدئ | 40 |
| **Data Sources & Collection**<br>مصادر البيانات وجمعها | Variable & Data Types | مقارنة مصادر البيانات من حيث البنية والموثوقية والخصوصية.<br>معرفة المشكلات النموذجية لكل مصدر.<br>قراءة البيانات من عدة صيغ باستخدام pandas. | — | 4 questions | — | مبتدئ | 30 |
| **Importing Data Lab**<br>مختبر استيراد البيانات | Data Sources & Collection | استيراد الملفات بأمان والتحقق من سلامتها.<br>قراءة ملخص أولي: Shape وDtypes وMemory.<br>تفسير التحذيرات الأولية قبل أي تحليل. | Importing Data Lab | 4 questions | — | مبتدئ | 25 |

## جودة البيانات · Data Quality

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Data Quality Dimensions**<br>أبعاد جودة البيانات | Data Concepts | تعريف أبعاد جودة البيانات وقياس كل بعد.<br>بناء Quality Score بصيغة وأوزان معلنة.<br>تفسير الـsub-scores بدل رقم واحد غامض. | Data Quality Dashboard | 4 questions | customers_raw | مبتدئ | 40 |
| **Data Validation**<br>التحقق من صحة البيانات | Data Quality Dimensions | كتابة قواعد تحقق قابلة للتنفيذ.<br>التمييز بين أنواع قواعد التحقق.<br>اختيار أداة التحقق المناسبة (Pandera, Pydantic, GX Core). | Rule Builder | 4 questions | customers_raw | متوسط | 45 |
| **Missing Values: Mechanisms & Diagnostics**<br>القيم المفقودة: الفهم والتشخيص | Variable & Data Types, Data Quality Dimensions | اكتشاف القيم المفقودة الصريحة والدلالية (Sentinels).<br>فهم MCAR/MAR/MNAR وتأثيرها على التقديرات.<br>تشخيص نمط الفقد عبر المجموعات والزمن وPanel. | MCAR/MAR/MNAR Simulator | 4 questions | survey | متوسط | 60 |
| **Missing Data Treatment**<br>معالجة القيم المفقودة | Missing Values: Mechanisms & Diagnostics | مقارنة طرق المعالجة من حيث الافتراضات والتحيّز.<br>قياس أثر الـImputation على التوزيع والتباين والعلاقات.<br>فهم Multiple Imputation وقواعد Rubin. | Imputation Experiment | 4 questions | survey | متقدم | 60 |
| **Duplicates**<br>التكرارات | Data Concepts | استخدام duplicated() وdrop_duplicates() مع subset وkeep.<br>التمييز بين التكرار الخاطئ والتكرار المشروع.<br>اكتشاف Near duplicates. | — | 4 questions | customers_raw | مبتدئ | 30 |
| **Outliers & Anomalies**<br>القيم الشاذة والحالات غير الطبيعية | Univariate Analysis | التمييز بين Outlier وAnomaly وInfluential observation.<br>تطبيق الطرق الإحصائية والنموذجية للكشف.<br>اتخاذ قرار مبرر: Keep/Investigate/Correct/Transform/Cap/Remove. | Outlier Decision Tool | 4 questions | customers_raw | متوسط | 55 |
| **Invalid Values & Consistency**<br>القيم غير الصالحة ومشكلات الاتساق | Data Quality Dimensions | اكتشاف القيم المستحيلة منطقيًا.<br>اكتشاف تناقضات Cross-field والوحدات.<br>ربط الإحصاء بمعرفة المجال Domain Knowledge. | — | 4 questions | customers_raw | مبتدئ | 35 |

## تنظيف البيانات · Data Cleaning

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Numerical Data Cleaning**<br>تنظيف البيانات الرقمية | Invalid Values & Consistency | تحويل الأعمدة النصية إلى أرقام بأمان.<br>اكتشاف أخطاء الوحدات والفواصل العشرية.<br>تقييم الالتواء والذيول الثقيلة. | — | 4 questions | customers_raw | مبتدئ | 35 |
| **Categorical Data**<br>البيانات الفئوية | Variable & Data Types | توحيد التسميات Label Standardization.<br>معالجة الفئات النادرة وHigh cardinality.<br>استخدام Categorical dtype المرتب. | — | 4 questions | customers_raw | مبتدئ | 35 |
| **Encoding**<br>ترميز المتغيرات | Categorical Data, Train/Validation/Test & CV | اختيار الترميز المناسب لكل نوع متغير.<br>فهم أثر الترميز على الأبعاد.<br>تطبيق Target Encoding دون Leakage. | — | 4 questions | — | متوسط | 40 |
| **Date & Time Data**<br>بيانات التاريخ والوقت | Variable & Data Types | تحويل النصوص إلى datetime بأمان.<br>بناء Lag وRolling features.<br>اكتشاف الطوابع الزمنية المفقودة وResampling. | — | 4 questions | daily_sales | مبتدئ | 40 |
| **Text Data**<br>البيانات النصية | Categorical Data | تنظيف النص وتطبيعه.<br>تحويل النص إلى Features عبر BoW وTF-IDF.<br>فهم فكرة Embeddings وTransformers دون تخصص NLP كامل. | — | 4 questions | reviews | متوسط | 40 |
| **Transformations & Scaling**<br>التحويلات والقياس | Univariate Analysis | اختيار التحويل المناسب للالتواء.<br>التمييز بين Standardization وNormalization.<br>معرفة متى نحتاج Scaling ومتى لا نحتاجه. | Before/After Animation | 4 questions | — | متوسط | 45 |

## التحليل الاستكشافي · EDA

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **EDA Foundations**<br>أسس التحليل الاستكشافي | Data Quality Dimensions | وضع EDA في سياق دورة التحليل.<br>اتباع Workflow منظم للاستكشاف.<br>تجنب تحويل EDA إلى صيد p-values. | — | 3 questions | — | مبتدئ | 25 |
| **Univariate Analysis**<br>التحليل أحادي المتغير | EDA Foundations | حساب وتفسير مقاييس النزعة المركزية والتشتت.<br>قراءة Histogram وECDF وBoxplot.<br>فهم أثر القيم المتطرفة على المتوسط. | — | 4 questions | all | مبتدئ | 45 |
| **Bivariate Analysis**<br>التحليل ثنائي المتغير | Univariate Analysis | اختيار أداة التحليل حسب نوعي المتغيرين.<br>تفسير الارتباط وحدوده.<br>قراءة Crosstab والنسب الشرطية. | — | 4 questions | all | مبتدئ | 40 |
| **Multivariate Analysis**<br>التحليل متعدد المتغيرات | Bivariate Analysis | قراءة مصفوفة الارتباط والـHeatmap.<br>تشخيص Multicollinearity عبر VIF.<br>استكشاف البنية متعددة الأبعاد بصريًا. | — | 3 questions | all | متوسط | 40 |
| **Data Visualization**<br>التصوير البياني | Univariate Analysis | اختيار نوع الرسم حسب السؤال ونوع البيانات.<br>اكتشاف الرسوم المضللة.<br>تصميم رسم واضح ومتاح للجميع. | Chart Selector | 4 questions | — | مبتدئ | 40 |

## الإحصاء · Statistics

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Statistical Foundations**<br>أسس الاستدلال الإحصائي | Univariate Analysis | التمييز بين Parameter وStatistic.<br>فهم Standard Error وتوزيع المعاينة.<br>تفسير فترة الثقة تفسيرًا صحيحًا. | CI Coverage Animation | 4 questions | — | متوسط | 50 |
| **Statistical Tests**<br>الاختبارات الإحصائية | Statistical Foundations | اختيار الاختبار حسب السؤال والتصميم ونوع المتغير.<br>تفسير p-value وCI وEffect size معًا.<br>فحص الافتراضات قبل التفسير. | Test Selector & Runner | 4 questions | students | متقدم | 60 |

## الخصائص وتعلّم الآلة · Features & ML

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Feature Engineering & Selection**<br>هندسة الخصائص واختيارها | Multivariate Analysis, Encoding | إنشاء خصائص مفيدة مبنية على المجال.<br>تطبيق طرق اختيار الخصائص.<br>استخدام Permutation importance بشكل صحيح. | — | 4 questions | credit | متوسط | 50 |
| **ML Foundations**<br>أسس تعلّم الآلة | Statistical Foundations | تصنيف مسائل التعلم.<br>فهم Bias-Variance وOverfitting بصريًا.<br>ربط ML بالإحصاء والتنقيب. | — | 4 questions | — | متوسط | 40 |
| **Train/Validation/Test & CV**<br>التقسيم والتحقق المتقاطع | ML Foundations | فهم دور Train/Validation/Test.<br>اختيار مخطط التقسيم المناسب للتصميم.<br>معرفة لماذا لا يصلح Random Split دائمًا. | Split Animation | 4 questions | — | متوسط | 40 |
| **Data Leakage**<br>تسرّب البيانات | Train/Validation/Test & CV | اكتشاف أنواع التسرب الستة.<br>إعادة إنتاج نتيجة مضخمة بسبب التسرب ثم إصلاحها.<br>استخدام Pipeline لمنع Preprocessing leakage. | Leakage Demo | 4 questions | credit | متقدم | 45 |
| **Model Evaluation**<br>تقييم النماذج | Train/Validation/Test & CV | اختيار المقياس المناسب للمسألة.<br>فهم أثر العتبة Threshold.<br>معرفة لماذا تضلل Accuracy في البيانات غير المتوازنة. | Threshold Explorer | 4 questions | credit | متوسط | 45 |

## التنقيب في البيانات · Data Mining

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Data Mining & Association Rules**<br>التنقيب وقواعد الارتباط | EDA Foundations | وضع Data Mining في علاقته بالإحصاء وML.<br>حساب Support وConfidence وLift.<br>تفسير القواعد بحذر وتجنب القواعد التافهة. | Apriori Explorer | 3 questions | baskets | متوسط | 45 |
| **Clustering**<br>التجميع | Transformations & Scaling | فهم خوارزمية K-Means خطوة بخطوة.<br>اختيار عدد العناقيد بأدلة متعددة.<br>مقارنة K-Means وDBSCAN وHierarchical. | K-Means Animation | 4 questions | customers_clean | متوسط | 45 |
| **Anomaly Detection**<br>اكتشاف الحالات الشاذة | Outliers & Anomalies | فهم فكرة العزل في Isolation Forest.<br>ضبط contamination وتفسير النتائج.<br>مقارنة الخوارزميات على نفس البيانات. | Isolation Animation | 3 questions | — | متقدم | 40 |
| **Dimensionality Reduction**<br>تقليل الأبعاد | Multivariate Analysis, Transformations & Scaling | فهم PCA هندسيًا ورياضيًا.<br>قراءة Scree plot والـLoadings.<br>معرفة حدود t-SNE/UMAP في التفسير. | — | 3 questions | wine | متقدم | 40 |

## علم البيانات الحديث · Modern Data Science

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Traditional vs Modern DS**<br>التقليدي مقابل الحديث | Data Quality Dimensions, EDA Foundations | مقارنة سير العمل اليدوي والآلي والهجين.<br>معرفة ما يمكن أتمتته وما لا ينبغي.<br>تبني فلسفة Hybrid. | — | 3 questions | — | مبتدئ | 30 |
| **Automated EDA & Discovery**<br>الاستكشاف الآلي | Traditional vs Modern DS, Univariate Analysis | فهم ما تفعله أدوات الـProfiling.<br>تشغيل Profiler مدمج وقراءة نتائجه.<br>تقييم حدود الأدوات الآلية وحالة صيانتها. | — | 3 questions | — | متوسط | 35 |
| **Automated Data Cleaning**<br>التنظيف الآلي | Automated EDA & Discovery, Missing Data Treatment, Outliers & Anomalies | مقارنة أساليب التنظيف الآلي.<br>قراءة تقرير المشكلات: Severity وEvidence وWhy.<br>فهم لماذا لا يُنفذ حذف غير قابل للتراجع آليًا. | — | 3 questions | — | متوسط | 35 |
| **AutoML**<br>التعلّم الآلي المؤتمت | Model Evaluation, Data Leakage | فهم مكونات AutoML: Search space وBudget وValidation.<br>تشغيل Mini-AutoML وقراءة Leaderboard.<br>معرفة مخاطر AutoML: Leakage وOverfitting للـValidation. | Mini-AutoML | 3 questions | credit | متقدم | 40 |
| **Data Observability & Drift**<br>مراقبة البيانات والانجراف | Data Validation | فهم أعمدة Data observability.<br>قياس Data drift بـPSI وKS.<br>كتابة Data contract بسيط. | Drift Simulator | 3 questions | — | متوسط | 35 |
| **AI-Assisted Data Science**<br>علم البيانات بمساعدة الذكاء الاصطناعي | Traditional vs Modern DS | فهم أنماط استخدام LLMs في التحليل.<br>تدقيق مخرجات AI واكتشاف أخطائها.<br>تصميم سير عمل آمن لوكيل بيانات Data Agent. | AI Output Audit | 3 questions | — | متوسط | 40 |
| **Human-in-the-Loop**<br>الإنسان في الحلقة | Automated Data Cleaning | تطبيق دورة Detection→Review→Decision→Validation→Documentation.<br>توثيق القرارات بشكل قابل للتدقيق.<br>تحديد نقاط التدخل البشري الإلزامية. | — | 2 questions | — | مبتدئ | 25 |

## المختبرات · Labs

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Dataset Explorer**<br>مستكشف البيانات | Importing Data Lab | ارفع ملفًا أو اختر Dataset مدمجة واحصل على تشخيص كامل مع توصيات. | Dataset Explorer | — | all | مبتدئ | 20 |
| **Missing Data Lab**<br>مختبر القيم المفقودة | Missing Data Treatment | شخّص الفقد ثم قارن طرق التعويض على بيانات حقيقية مع إخفاء قيم معروفة لقياس الخطأ. | Missing Data Lab | — | all | متوسط | 30 |
| **Outlier Lab**<br>مختبر القيم الشاذة | Outliers & Anomalies | قارن IQR وZ وModified Z وIsolation Forest وLOF على نفس البيانات مع مصفوفة الاتفاق. | Outlier Lab | — | all | متوسط | 30 |
| **Manual Cleaning Lab**<br>مختبر التنظيف اليدوي | Invalid Values & Consistency, Categorical Data | نظّف Dataset التحدي خطوة بخطوة وقارن قبل/بعد في جودة البيانات. | Before vs After | — | customers_raw | متوسط | 40 |
| **EDA Lab**<br>مختبر التحليل الاستكشافي | Bivariate Analysis | استكشف أي Dataset بالرسوم أحادية وثنائية ومتعددة المتغيرات مع تفسير تلقائي. | EDA Lab | — | all | مبتدئ | 30 |
| **Pipeline Builder**<br>باني خط المعالجة | Transformations & Scaling, Encoding, Missing Data Treatment | اختر استراتيجيات المعالجة، اعرض المخطط، شغّل، وحمّل كود Python المكافئ. | Pipeline Builder | — | all | متوسط | 35 |
| **Manual vs Automated vs Hybrid Lab**<br>مختبر اليدوي مقابل الآلي | Automated Data Cleaning | نفس البيانات بثلاثة أنماط: أنت تقرر، أو المنصة، أو المنصة تقترح وأنت تعتمد. | Hybrid Lab | — | all | متوسط | 35 |
| **Automated Analysis Report**<br>التقرير الآلي | Automated EDA & Discovery | تقرير تحليلي آلي يميّز بين Detected issue وPotential issue وRecommendation، قابل للتحميل. | Report Generator | — | all | مبتدئ | 15 |

## المشاريع · Projects

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Case Studies**<br>دراسات الحالة | EDA Lab | ست دراسات حالة كاملة: العملاء، التجزئة، الاستبيان، السلاسل الزمنية، Panel، والنصوص. | — | — | — | متوسط | 90 |
| **Interactive Data Story**<br>قصة البيانات | Data Leakage, Manual Cleaning Lab | «وصلتنا Dataset خام من مؤسسة»: عشر محطات تكشف المشكلات تدريجيًا حتى التفسير النهائي. | — | — | — | متوسط | 45 |
| **Capstone Project**<br>المشروع النهائي | Interactive Data Story | من Raw Data إلى تقرير نهائي: 14 مرحلة مع قائمة تحقق وتقرير قابل للتحميل. | — | — | — | متقدم | 180 |

## الممارسة المهنية · Professional Practice

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Reproducibility**<br>قابلية إعادة الإنتاج | Workflows & Lifecycle | بناء مشروع قابل لإعادة الإنتاج.<br>فهم Seeds وVersioning وData versioning.<br>التمييز بين Notebook وScript. | — | 3 questions | — | مبتدئ | 30 |
| **Data Ethics & Privacy**<br>أخلاقيات البيانات والخصوصية | What is Data Science? | تحديد البيانات الحساسة وتقليلها.<br>قياس فجوة عدالة بسيطة.<br>تطبيق قائمة استخدام مسؤول. | — | 3 questions | — | مبتدئ | 30 |

## الموارد · Resources

| Module | Prerequisites | Learning objectives | Practical lab | Quiz | Dataset | Difficulty | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Search**<br>البحث | — | ابحث في الدروس والمسرد والموضوعات. | — | — | — | مبتدئ | 30 |
| **My Progress**<br>تقدّمي | — | الدروس المكتملة ونتائج الاختبارات والمختبرات والمفضلة، مع تصدير واستيراد التقدم. | — | — | — | مبتدئ | 30 |
| **Glossary**<br>المسرد | — | مسرد عربي/إنجليزي قابل للبحث. | — | — | — | مبتدئ | 30 |
| **Cheat Sheets & Comparisons**<br>أوراق المراجعة | — | أوراق مراجعة pandas/cleaning/stats وجداول مقارنة شاملة وأدوات قرار سريعة. | — | — | — | مبتدئ | 30 |
| **References**<br>المراجع | — | المصادر الرسمية والكتب والمقررات التي بُني عليها المحتوى. | — | — | — | مبتدئ | 30 |
| **About**<br>عن المنصة | — | عن المنصة والمطوّر والمنهجية التقنية. | — | — | — | مبتدئ | 30 |
