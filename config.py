"""Central configuration: identity, naming and global constants.

Every page reads the author name and platform name from here, so they are
never hard-coded anywhere else.
"""

from __future__ import annotations

APP_AUTHOR_AR = "الدكتور مروان رودان"
APP_AUTHOR_EN = "Dr. Marwan Roudane"

APP_NAME_EN = "Data Science Interactive Academy"
APP_NAME_AR = "أكاديمية علم البيانات التفاعلية"
APP_SUBTITLE_AR = (
    "منصة تفاعلية متكاملة لتعلّم علم البيانات من البيانات الخام "
    "إلى التحليل والذكاء الاصطناعي"
)
APP_VERSION = "1.0.0"

# Upload safety limits (see utils/data_loader.py and docs/architecture.md)
MAX_UPLOAD_MB = 50
ALLOWED_UPLOAD_EXTENSIONS = (".csv", ".txt", ".xlsx", ".xls", ".json")

# Performance: maximum points drawn in a single scatter plot before sampling
MAX_PLOT_POINTS = 5_000

# Global random seed used by synthetic datasets and simulations
RANDOM_SEED = 42

LEVELS = {
    "beginner": "مبتدئ",
    "advanced": "متقدم",
    "research": "بحثي",
}
