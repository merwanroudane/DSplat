"""Built-in teaching datasets.

All datasets are synthetic (generated with fixed seeds) or public and
non-sensitive (scikit-learn's bundled Iris and Wine). No real personal data
is used. Each dataset has documentation in DATASET_INFO and docs/datasets.md.

Design note: the customer data is generated as a clean *ground truth* first
and then deliberately corrupted. That lets labs score a learner's cleaning
against the truth instead of against another guess.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from config import RANDOM_SEED

COUNTRIES = ["Morocco", "Egypt", "Saudi Arabia", "United States", "France", "Jordan", "Algeria", "Tunisia"]
COUNTRY_WEIGHTS = [0.22, 0.18, 0.14, 0.14, 0.1, 0.08, 0.08, 0.06]
MEMBERSHIP = ["Bronze", "Silver", "Gold", "Platinum"]


# ------------------------------------------------------------------ customers
@lru_cache(maxsize=1)
def _customers_truth() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    n = 600
    age = np.clip(rng.normal(38, 12, n).round(), 18, 80).astype(int)
    country = rng.choice(COUNTRIES, n, p=COUNTRY_WEIGHTS)
    gender = rng.choice(["Female", "Male"], n)
    base_income = np.exp(rng.normal(10.6, 0.45, n)) * (1 + (age - 38) * 0.008)
    income = np.round(base_income, -2)
    membership = np.select(
        [income > 80000, income > 50000, income > 30000], ["Platinum", "Gold", "Silver"], "Bronze"
    )
    swap = rng.random(n) < 0.15
    membership = np.where(swap, rng.choice(MEMBERSHIP, n), membership)
    signup = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    days_since = rng.integers(1, 400, n)
    last_purchase = signup + pd.to_timedelta(days_since, unit="D")
    last_purchase = pd.Series(last_purchase).where(last_purchase < pd.Timestamp("2024-12-31"),
                                                   pd.Timestamp("2024-12-30")).to_numpy()
    orders = rng.poisson(3 + income / 30000, n)
    spend = np.round(rng.gamma(2.0, 60, n) * (1 + income / 100000), 2)
    height = np.round(np.where(gender == "Male", rng.normal(175, 7, n), rng.normal(162, 6, n)), 1)
    satisfaction = np.clip(np.round(rng.normal(3.6, 1.0, n)), 1, 5).astype(int)
    logit = -1.2 - 0.35 * orders + 0.9 * (satisfaction <= 2) + 0.004 * days_since
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    user = [f"user{i:04d}" for i in range(n)]
    domains = rng.choice(["example.com", "mail.test", "demo.org"], n)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
            "signup_date": pd.to_datetime(signup),
            "age": age,
            "gender": gender,
            "country": country,
            "annual_income": income,
            "membership": membership,
            "monthly_spend": spend,
            "num_orders": orders,
            "height_cm": height,
            "satisfaction": satisfaction,
            "email": [f"{u}@{d}" for u, d in zip(user, domains)],
            "last_purchase_date": pd.to_datetime(last_purchase),
            "churned": churned,
        }
    )
    return df


def customers_clean() -> pd.DataFrame:
    return _customers_truth().copy()


@lru_cache(maxsize=1)
def _customers_raw() -> pd.DataFrame:
    """Corrupt the truth with realistic, documented problems."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    t = _customers_truth()
    n = len(t)
    raw = pd.DataFrame({"customer_id": t["customer_id"].astype(object)})

    # signup_date: mixed formats, impossible dates, blanks
    fmt_choice = rng.choice(4, n, p=[0.7, 0.15, 0.1, 0.05])
    dates = []
    for d, f in zip(t["signup_date"], fmt_choice):
        if f == 0:
            dates.append(d.strftime("%Y-%m-%d"))
        elif f == 1:
            dates.append(d.strftime("%d/%m/%Y"))
        elif f == 2:
            dates.append(d.strftime("%Y/%m/%d"))
        else:
            dates.append("")
    for i in rng.choice(n, 4, replace=False):
        dates[i] = "2023-13-45"
    raw["signup_date"] = pd.Series(dates, dtype=object)

    # age: impossible values, sentinel, text placeholders
    age = t["age"].astype(object).copy()
    age.iloc[rng.choice(n, 3, replace=False)] = 250
    age.iloc[rng.choice(n, 3, replace=False)] = -3
    age.iloc[rng.choice(n, 12, replace=False)] = -999
    age.iloc[rng.choice(n, 8, replace=False)] = "unknown"
    age.iloc[rng.choice(n, 20, replace=False)] = np.nan
    raw["age"] = age.astype(str).replace({"nan": np.nan})

    # gender: inconsistent labels + whitespace + missing
    g_map = {"Female": ["Female", "F", "female", " f ", "FEMALE"], "Male": ["Male", "M", "male", "m ", "MALE"]}
    raw["gender"] = [
        (rng.choice(g_map[g], p=[0.6, 0.15, 0.1, 0.1, 0.05]) if rng.random() > 0.04 else None)
        for g in t["gender"]
    ]

    # country: many spellings of the same entity + one rare real category
    c_map = {
        "United States": ["United States", "USA", "usa", "U.S.A", "US"],
        "Morocco": ["Morocco", "morocco ", "Maroc", "MOROCCO"],
        "Saudi Arabia": ["Saudi Arabia", "KSA", "Saudi arabia"],
        "Egypt": ["Egypt", "egypt", " Egypt"],
    }
    countries = []
    for c in t["country"]:
        opts = c_map.get(c)
        countries.append(rng.choice(opts) if opts and rng.random() < 0.35 else c)
    countries[int(rng.integers(0, n))] = "Iceland"
    raw["country"] = countries

    # annual_income: numbers stored as text, currency symbols, unit errors, zeros, extremes
    inc = []
    for v in t["annual_income"]:
        r = rng.random()
        if r < 0.05:
            inc.append(f"{v / 1000:.0f}")  # recorded in thousands -> unit inconsistency
        elif r < 0.10:
            inc.append(f"${v:,.0f}")
        elif r < 0.14:
            inc.append(f"{v:,.0f}")
        elif r < 0.17:
            inc.append("N/A")
        elif r < 0.19:
            inc.append("0")
        else:
            inc.append(f"{v:.0f}")
    for i in rng.choice(n, 3, replace=False):
        inc[i] = "9999999"
    raw["annual_income"] = inc

    raw["membership"] = [m.lower() if rng.random() < 0.12 else m for m in t["membership"]]
    spend = t["monthly_spend"].copy()
    spend.iloc[rng.choice(n, 5, replace=False)] *= 25  # real but extreme spenders
    spend.iloc[rng.choice(n, 15, replace=False)] = np.nan
    raw["monthly_spend"] = spend
    orders = t["num_orders"].astype(float).copy()
    orders.iloc[rng.choice(n, 4, replace=False)] = -1
    raw["num_orders"] = orders
    height = t["height_cm"].copy()
    idx = rng.choice(n, 10, replace=False)
    height.iloc[idx] = (height.iloc[idx] / 100).round(2)  # recorded in metres
    raw["height_cm"] = height
    sat = t["satisfaction"].astype(object).copy()
    sat.iloc[rng.choice(n, 6, replace=False)] = 9
    sat.iloc[rng.choice(n, 10, replace=False)] = "?"
    raw["satisfaction"] = sat.astype(str)
    email = t["email"].copy()
    for i in rng.choice(n, 8, replace=False):
        email.iloc[i] = email.iloc[i].replace("@", "")
    for i in rng.choice(n, 4, replace=False):
        email.iloc[i] = email.iloc[i].replace(".com", "").replace(".org", "").replace(".test", "")
    raw["email"] = email
    lp = t["last_purchase_date"].copy()
    bad = rng.choice(n, 6, replace=False)
    lp.iloc[bad] = t["signup_date"].iloc[bad] - pd.Timedelta(days=30)  # before signup
    raw["last_purchase_date"] = lp.dt.strftime("%Y-%m-%d")
    raw["churned"] = t["churned"]

    # duplicates: exact copies and key-level duplicates with conflicting values
    exact = raw.iloc[rng.choice(n, 15, replace=False)]
    conflict = raw.iloc[rng.choice(n, 6, replace=False)].copy()
    conflict["monthly_spend"] = conflict["monthly_spend"] * 1.1
    out = pd.concat([raw, exact, conflict], ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return out


def customers_raw() -> pd.DataFrame:
    return _customers_raw().copy()


# --------------------------------------------------------------------- survey
@lru_cache(maxsize=1)
def _survey() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 2)
    n = 800
    age = np.clip(rng.normal(40, 13, n).round(), 18, 75).astype(int)
    region = rng.choice(["North", "South", "East", "West", "Center"], n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    edu_levels = ["Primary", "Secondary", "Bachelor", "Master", "PhD"]
    education = rng.choice(edu_levels, n, p=[0.1, 0.35, 0.33, 0.17, 0.05])
    edu_rank = pd.Series(education).map({e: i for i, e in enumerate(edu_levels)}).to_numpy()
    income = np.round(np.exp(9.6 + 0.18 * edu_rank + 0.01 * (age - 40) + rng.normal(0, 0.35, n)), -1)
    hours = np.clip(rng.normal(40 + 2 * edu_rank, 7, n), 10, 80).round(1)
    stress = np.clip(20 + 0.5 * hours - 0.15 * age + rng.normal(0, 6, n), 0, 60).round(1)
    satisfaction = np.clip(np.round(3.9 - 0.04 * (stress - 30) + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    items = {f"q{i}": np.clip(np.round(satisfaction + rng.normal(0, 0.9, n)), 1, 5).astype(int) for i in range(1, 6)}
    df = pd.DataFrame(
        {
            "respondent_id": [f"R{i:04d}" for i in range(1, n + 1)],
            "age": age,
            "gender": rng.choice(["Female", "Male", "Prefer not to say"], n, p=[0.49, 0.47, 0.04]),
            "region": region,
            "education": education,
            "income": income,
            "work_hours": hours,
            "stress_score": stress,
            "job_satisfaction": satisfaction,
            **items,
        }
    )
    df["income_true"] = df["income"]  # ground truth kept for teaching (hidden in most views)
    # MNAR: high earners skip the income question more often
    p_income = 1 / (1 + np.exp(-(np.log(df["income"]) - np.log(df["income"]).mean()) * 2.5 + 1.6))
    df.loc[rng.random(n) < p_income, "income"] = np.nan
    # MAR: younger respondents skip the stress item more often (depends on observed age)
    p_stress = np.where(df["age"] < 30, 0.30, 0.05)
    df.loc[rng.random(n) < p_stress, "stress_score"] = np.nan
    # MCAR: random item non-response
    for c in ["q2", "q4"]:
        df[c] = df[c].astype(float)
        df.loc[rng.random(n) < 0.06, c] = np.nan
    # Sentinel coding in work_hours
    df.loc[rng.choice(n, 14, replace=False), "work_hours"] = -99
    return df


def survey(include_truth: bool = False) -> pd.DataFrame:
    df = _survey().copy()
    return df if include_truth else df.drop(columns=["income_true"])


# ----------------------------------------------------------------- retail
PRODUCTS = {
    "Bread": ("Bakery", 1.2), "Butter": ("Dairy", 2.5), "Milk": ("Dairy", 1.1), "Cheese": ("Dairy", 4.0),
    "Pasta": ("Grocery", 1.8), "Tomato Sauce": ("Grocery", 2.2), "Olive Oil": ("Grocery", 7.5),
    "Coffee": ("Beverages", 6.0), "Sugar": ("Grocery", 1.5), "Tea": ("Beverages", 3.5),
    "Dates": ("Produce", 5.0), "Apples": ("Produce", 2.8), "Chicken": ("Meat", 8.5), "Rice": ("Grocery", 2.4),
    "Yogurt": ("Dairy", 1.9), "Diapers": ("Baby", 12.0), "Baby Wipes": ("Baby", 4.5), "Water": ("Beverages", 0.8),
}
_RULES = [("Bread", "Butter", 0.55), ("Pasta", "Tomato Sauce", 0.65), ("Coffee", "Sugar", 0.5),
          ("Diapers", "Baby Wipes", 0.6), ("Tea", "Dates", 0.35), ("Chicken", "Rice", 0.45)]


@lru_cache(maxsize=1)
def _transactions() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 3)
    names = list(PRODUCTS)
    popularity = np.array([10, 4, 9, 4, 6, 3, 3, 6, 3, 5, 3, 5, 5, 6, 5, 2, 1, 8], dtype=float)
    popularity /= popularity.sum()
    rows = []
    start = pd.Timestamp("2024-01-01")
    for inv in range(1, 1501):
        size = int(rng.integers(1, 6))
        basket = set(rng.choice(names, size, replace=False, p=popularity))
        for a, b, p in _RULES:
            if a in basket and rng.random() < p:
                basket.add(b)
        date = start + pd.Timedelta(days=int(rng.integers(0, 366)), hours=int(rng.integers(8, 22)))
        store = rng.choice(["Casablanca", "Cairo", "Riyadh", "Amman"], p=[0.35, 0.3, 0.2, 0.15])
        cust = f"K{int(rng.integers(1, 400)):04d}"
        for item in sorted(basket):
            cat, price = PRODUCTS[item]
            rows.append((f"INV{inv:05d}", date, cust, store, item, cat, int(rng.integers(1, 4)),
                         round(price * rng.uniform(0.9, 1.1), 2)))
    df = pd.DataFrame(rows, columns=["invoice_id", "datetime", "customer_id", "store", "product", "category",
                                     "quantity", "unit_price"])
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    return df


def transactions() -> pd.DataFrame:
    return _transactions().copy()


def baskets() -> list[set[str]]:
    return [set(g) for g in _transactions().groupby("invoice_id")["product"].apply(list)]


# -------------------------------------------------------------- time series
@lru_cache(maxsize=1)
def _daily_sales() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 4)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n = len(dates)
    t = np.arange(n)
    weekly = 18 * np.isin(dates.dayofweek, [4, 5])  # Friday/Saturday peaks
    yearly = 25 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
    promo = (rng.random(n) < 0.07).astype(int)
    level = 200 + 0.08 * t + np.where(dates >= "2024-06-01", 35, 0)  # structural break
    sales = level + weekly + yearly + 40 * promo + rng.normal(0, 12, n)
    df = pd.DataFrame({"date": dates, "sales": sales.round(1), "promo": promo,
                       "temperature": (22 + 9 * np.sin(2 * np.pi * (dates.dayofyear - 110) / 365.25)
                                       + rng.normal(0, 2, n)).round(1)})
    spikes = rng.choice(n, 4, replace=False)
    df.loc[spikes, "sales"] = df.loc[spikes, "sales"] * 2.6
    df.loc[rng.choice(n, 12, replace=False), "sales"] = np.nan
    drop = rng.choice(n, 20, replace=False)  # missing timestamps
    return df.drop(index=drop).reset_index(drop=True)


def daily_sales() -> pd.DataFrame:
    return _daily_sales().copy()


# --------------------------------------------------------------------- panel
@lru_cache(maxsize=1)
def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 5)
    rows = []
    sectors = ["Manufacturing", "Services", "Technology", "Agriculture"]
    for f in range(1, 41):
        sector = sectors[f % 4]
        alpha = rng.normal(0, 0.4)
        size = rng.lognormal(4, 0.6)
        exit_year = 2023 if rng.random() > 0.2 else int(rng.integers(2015, 2022))
        entry = 2010 if rng.random() > 0.15 else int(rng.integers(2011, 2016))
        for y in range(entry, exit_year + 1):
            emp = size * np.exp(0.03 * (y - 2010) + rng.normal(0, 0.08))
            rnd = max(0.0, (0.02 + 0.04 * (sector == "Technology")) * emp * 100 + rng.normal(0, 5))
            revenue = np.exp(alpha + 0.9 * np.log(emp) + 0.004 * rnd + rng.normal(0, 0.1)) * 1000
            if y in (2020,):
                revenue *= 0.85  # common shock
            rows.append((f"F{f:02d}", y, sector, round(revenue, 1), round(emp, 1), round(rnd, 2),
                         round(revenue * rng.uniform(0.02, 0.15), 1)))
    df = pd.DataFrame(rows, columns=["firm_id", "year", "sector", "revenue", "employees", "rnd_spend", "profit"])
    # missingness that increases in later years (reporting fatigue)
    p = 0.02 + 0.012 * (df["year"] - 2010)
    df.loc[rng.random(len(df)) < p, "rnd_spend"] = np.nan
    return df


def panel() -> pd.DataFrame:
    return _panel().copy()


# ---------------------------------------------------------------------- text
_POS = ["great quality", "fast delivery", "excellent value", "works perfectly", "very happy", "highly recommend",
        "friendly support", "easy to use", "beautiful design", "arrived early"]
_NEG = ["poor quality", "late delivery", "waste of money", "stopped working", "very disappointed",
        "would not recommend", "rude support", "hard to use", "cheap material", "arrived damaged"]
_NEU = ["as described", "average product", "okay for the price", "nothing special", "does the job"]
_PROD = ["headphones", "kettle", "backpack", "phone case", "desk lamp", "blender", "keyboard", "water bottle"]
_AR = {"pos": ["منتج رائع وجودة ممتازة", "التوصيل سريع جداً", "أنصح به بشدة", "سعر مناسب وأداء ممتاز"],
       "neg": ["جودة سيئة للأسف", "التوصيل متأخر كثيراً", "لا أنصح به", "توقف عن العمل بسرعة"],
       "neu": ["منتج عادي", "كما هو موصوف", "مقبول مقابل السعر"]}


@lru_cache(maxsize=1)
def _reviews() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 6)
    rows = []
    for i in range(1, 421):
        label = rng.choice(["positive", "negative", "neutral"], p=[0.5, 0.33, 0.17])
        prod = rng.choice(_PROD)
        if rng.random() < 0.12:
            key = {"positive": "pos", "negative": "neg", "neutral": "neu"}[label]
            text = " ".join(rng.choice(_AR[key], 2, replace=False)) + "!"
            lang = "ar"
        else:
            pool = {"positive": _POS, "negative": _NEG, "neutral": _NEU}[label]
            phrases = rng.choice(pool, min(2, len(pool)), replace=False)
            opener = rng.choice(["The", "This", "My new", "Bought this"])
            text = f"{opener} {prod}: {phrases[0]}, {phrases[1]}."
            if rng.random() < 0.15:
                text = text.upper() + "!!!"
            if rng.random() < 0.1:
                text = "  " + text + "   <br> "
            lang = "en"
        stars = {"positive": rng.integers(4, 6), "negative": rng.integers(1, 3), "neutral": 3}[label]
        rows.append((f"RV{i:04d}", prod, text, int(stars), label, lang))
    return pd.DataFrame(rows, columns=["review_id", "product", "text", "stars", "sentiment", "language"])


def reviews() -> pd.DataFrame:
    return _reviews().copy()


# -------------------------------------------------------------------- credit
@lru_cache(maxsize=1)
def _credit() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 7)
    n_cust = 2200
    rows = []
    app_id = 0
    for c in range(n_cust):
        n_apps = 1 if rng.random() > 0.25 else int(rng.integers(2, 4))
        age = int(np.clip(rng.normal(41, 11), 21, 75))
        income = float(np.round(np.exp(rng.normal(10.5, 0.5)), -2))
        emp = rng.choice(["Salaried", "Self-employed", "Public sector", "Unemployed"], p=[0.55, 0.2, 0.2, 0.05])
        home = rng.choice(["Rent", "Own", "Mortgage"], p=[0.45, 0.25, 0.3])
        region = rng.choice(["North", "South", "East", "West"])
        hist = int(np.clip(rng.normal(8, 5), 0, 35))
        risk_c = rng.normal(0, 0.8)
        for _ in range(n_apps):
            app_id += 1
            loan = float(np.round(income * rng.uniform(0.1, 0.9), -2))
            term = int(rng.choice([12, 24, 36, 60]))
            late = int(rng.poisson(0.6 + 1.2 * (risk_c > 0.8)))
            dti = float(np.clip(loan / income + rng.normal(0.2, 0.08), 0.02, 1.5))
            date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(rng.integers(0, 1095)))
            logit = (-3.1 + 1.6 * dti + 0.45 * late - 0.04 * hist + 0.9 * (emp == "Unemployed")
                     + 0.3 * (home == "Rent") + risk_c * 0.7 - 0.000004 * income)
            default = int(rng.random() < 1 / (1 + np.exp(-logit)))
            collections = int(default == 1 and rng.random() < 0.93) or int(rng.random() < 0.01)
            rows.append((f"A{app_id:05d}", f"CU{c:05d}", date, age, income, emp, home, region, hist, loan,
                         term, late, round(dti, 3), collections, default))
    df = pd.DataFrame(rows, columns=["application_id", "customer_id", "application_date", "age", "income",
                                     "employment", "home_ownership", "region", "credit_history_years",
                                     "loan_amount", "term_months", "late_payments", "debt_to_income",
                                     "sent_to_collections", "default"])
    return df.sort_values("application_date").reset_index(drop=True)


def credit() -> pd.DataFrame:
    return _credit().copy()


# ------------------------------------------------------------------ students
@lru_cache(maxsize=1)
def _students() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 8)
    n = 240
    method = np.repeat(["Lecture", "Flipped", "Project-based"], n // 3)
    effect = pd.Series(method).map({"Lecture": 0.0, "Flipped": 4.0, "Project-based": 6.5}).to_numpy()
    hours = np.clip(rng.gamma(4, 1.6, n), 0.5, 25).round(1)
    pre = np.clip(rng.normal(58, 11, n), 20, 100).round(1)
    post = np.clip(pre + 3 + effect + 0.8 * hours + rng.normal(0, 7, n), 0, 100).round(1)
    gender = rng.choice(["Female", "Male"], n)
    school = rng.choice(["Public", "Private"], n, p=[0.65, 0.35])
    passed = np.where(post >= 60, "Pass", "Fail")
    attendance = np.clip(rng.beta(8, 2, n) * 100, 30, 100).round(0)
    return pd.DataFrame({"student_id": [f"S{i:03d}" for i in range(1, n + 1)], "teaching_method": method,
                         "gender": gender, "school_type": school, "study_hours": hours, "pre_score": pre,
                         "post_score": post, "attendance_pct": attendance, "result": passed})


def students() -> pd.DataFrame:
    return _students().copy()


# ---------------------------------------------------------------- public sets
@lru_cache(maxsize=1)
def _iris() -> pd.DataFrame:
    from sklearn.datasets import load_iris
    b = load_iris(as_frame=True)
    df = b.frame.copy()
    df["species"] = pd.Categorical.from_codes(b.target, b.target_names)
    return df.drop(columns=["target"])


@lru_cache(maxsize=1)
def _wine() -> pd.DataFrame:
    from sklearn.datasets import load_wine
    b = load_wine(as_frame=True)
    df = b.frame.copy()
    df["cultivar"] = pd.Categorical.from_codes(b.target, ["class_0", "class_1", "class_2"])
    return df.drop(columns=["target"])


def iris() -> pd.DataFrame:
    return _iris().copy()


def wine() -> pd.DataFrame:
    return _wine().copy()


# ------------------------------------------------------------------ registry
DATASET_INFO: dict[str, dict] = {
    "customers_raw": {
        "loader": customers_raw,
        "title": "عملاء — نسخة التحدي (خام) · Customers (raw challenge)",
        "design": "Cross-sectional",
        "purpose": "Dataset مصممة عمدًا بمشاكل جودة واقعية لتدريب التشخيص والتنظيف.",
        "variables": {
            "customer_id": "معرّف العميل (يُفترض أنه فريد)", "signup_date": "تاريخ التسجيل (نص بصيغ مختلطة)",
            "age": "العمر (يحوي قيمًا مستحيلة وSentinels ونصوص)", "gender": "الجنس (تسميات غير متسقة)",
            "country": "الدولة (تهجئات متعددة لنفس الكيان)", "annual_income": "الدخل السنوي (نص، رموز عملة، وحدات)",
            "membership": "فئة العضوية (حالة أحرف غير متسقة)", "monthly_spend": "الإنفاق الشهري (ملتوٍ، قيم متطرفة حقيقية)",
            "num_orders": "عدد الطلبات (قيم سالبة مستحيلة)", "height_cm": "الطول بالسنتيمتر (بعضه بالمتر)",
            "satisfaction": "الرضا 1–5 (قيم خارج المدى و«?»)", "email": "بريد إلكتروني اصطناعي (بعضه غير صالح)",
            "last_purchase_date": "آخر شراء (بعضه قبل التسجيل)", "churned": "الهدف: هل غادر العميل؟ (0/1)",
        },
        "issues": ["Missing values صريحة ودلالية (-999, unknown, N/A, ?)", "Exact duplicates وKey duplicates متعارضة",
                   "Wrong types (أرقام مخزنة نصًا)", "Impossible values (age=250, orders<0)",
                   "Unit inconsistencies (الدخل بالآلاف، الطول بالمتر)", "Inconsistent labels (USA/usa/U.S.A)",
                   "Whitespace وحالة الأحرف", "Date problems (صيغ مختلطة، 2023-13-45، ترتيب زمني مستحيل)",
                   "Outliers حقيقية وأخرى أخطاء", "Rare category (Iceland)"],
        "lessons": ["quality_dimensions", "data_validation", "duplicates", "outliers", "consistency",
                    "numerical_cleaning", "categorical_data", "cleaning_lab"],
    },
    "customers_clean": {
        "loader": customers_clean,
        "title": "عملاء — الحقيقة المرجعية (نظيفة) · Customers (clean truth)",
        "design": "Cross-sectional",
        "purpose": "النسخة الصحيحة التي اشتُقت منها نسخة التحدي؛ تُستخدم للمقارنة وللتجميع والنمذجة.",
        "variables": {"customer_id": "معرّف", "age": "العمر", "annual_income": "الدخل", "monthly_spend": "الإنفاق",
                      "num_orders": "الطلبات", "satisfaction": "الرضا", "churned": "الهدف"},
        "issues": ["لا مشكلات مقصودة؛ الإنفاق والدخل ملتويان طبيعيًا."],
        "lessons": ["data_concepts", "clustering", "cleaning_lab"],
    },
    "survey": {
        "loader": survey,
        "title": "استبيان رضا العاملين · Workplace survey",
        "design": "Cross-sectional survey",
        "purpose": "لتعليم آليات الفقد MCAR/MAR/MNAR والمتغيرات الترتيبية (Likert).",
        "variables": {"respondent_id": "معرّف المستجيب", "age": "العمر", "gender": "الجنس (يتضمن Prefer not to say)",
                      "region": "المنطقة", "education": "التعليم (ترتيبي)", "income": "الدخل (MNAR: ذوو الدخل المرتفع يمتنعون)",
                      "work_hours": "ساعات العمل (Sentinel -99)", "stress_score": "درجة الضغط (MAR: يعتمد على العمر)",
                      "job_satisfaction": "الرضا الوظيفي 1–5", "q1..q5": "بنود Likert (q2 وq4 فيهما فقد MCAR)"},
        "issues": ["MNAR في income", "MAR في stress_score", "MCAR في q2 وq4", "Sentinel -99 في work_hours"],
        "lessons": ["missing_values", "missing_treatment", "missing_lab", "hypothesis_tests"],
    },
    "transactions": {
        "loader": transactions,
        "title": "معاملات متجر تجزئة · Retail transactions",
        "design": "Transaction data",
        "purpose": "لتحليل المبيعات وسلة المشتريات وقواعد الارتباط.",
        "variables": {"invoice_id": "رقم الفاتورة", "datetime": "وقت الشراء", "customer_id": "العميل",
                      "store": "الفرع", "product": "المنتج", "category": "الفئة", "quantity": "الكمية",
                      "unit_price": "سعر الوحدة", "revenue": "الإيراد"},
        "issues": ["التكرار مشروع هنا: نفس العميل والمنتج يظهران مرات عديدة", "قواعد ارتباط مزروعة للاكتشاف"],
        "lessons": ["data_mining", "duplicates", "case_studies"],
    },
    "daily_sales": {
        "loader": daily_sales,
        "title": "مبيعات يومية · Daily sales (time series)",
        "design": "Time series",
        "purpose": "لتعليم التاريخ والوقت، Resampling، Lag/Rolling، والموسمية.",
        "variables": {"date": "التاريخ", "sales": "المبيعات", "promo": "يوم عرض ترويجي", "temperature": "درجة الحرارة"},
        "issues": ["20 يومًا مفقودًا من الفهرس الزمني", "12 قيمة مبيعات مفقودة", "4 قفزات شاذة",
                   "Structural break في يونيو 2024", "موسمية أسبوعية وسنوية"],
        "lessons": ["datetime_data", "splitting", "case_studies"],
    },
    "panel": {
        "loader": panel,
        "title": "بيانات شركات عبر السنوات · Firm panel",
        "design": "Unbalanced panel",
        "purpose": "لتعليم Panel data: التكرار المشروع للمعرّف، الفقد عبر الزمن، Group K-Fold.",
        "variables": {"firm_id": "الشركة", "year": "السنة", "sector": "القطاع", "revenue": "الإيراد",
                      "employees": "الموظفون", "rnd_spend": "الإنفاق على البحث والتطوير", "profit": "الربح"},
        "issues": ["Panel غير متوازن (دخول وخروج الشركات)", "فقد rnd_spend يتزايد مع الزمن", "صدمة 2020"],
        "lessons": ["variable_types", "missing_values", "splitting", "case_studies"],
    },
    "reviews": {
        "loader": reviews,
        "title": "مراجعات منتجات · Product reviews (text)",
        "design": "Unstructured text",
        "purpose": "لتعليم تنظيف النص وTF-IDF وتصنيف المشاعر.",
        "variables": {"review_id": "المعرّف", "product": "المنتج", "text": "نص المراجعة (إنجليزي/عربي)",
                      "stars": "التقييم", "sentiment": "التصنيف", "language": "اللغة"},
        "issues": ["أحرف كبيرة وعلامات تعجب", "HTML tags ومسافات", "نصوص مختلطة اللغة"],
        "lessons": ["text_data", "case_studies"],
    },
    "credit": {
        "loader": credit,
        "title": "طلبات قروض (اصطناعية) · Credit applications (banking-like)",
        "design": "Cross-sectional with repeated customers",
        "purpose": "للنمذجة والتقييم وعدم التوازن وتسرّب البيانات.",
        "variables": {"application_id": "الطلب", "customer_id": "العميل (قد يتكرر)", "application_date": "تاريخ الطلب",
                      "age": "العمر", "income": "الدخل", "employment": "نوع العمل", "home_ownership": "السكن",
                      "region": "المنطقة", "credit_history_years": "طول التاريخ الائتماني",
                      "loan_amount": "مبلغ القرض", "term_months": "المدة", "late_payments": "الدفعات المتأخرة",
                      "debt_to_income": "نسبة الدين إلى الدخل",
                      "sent_to_collections": "أُحيل للتحصيل (يُسجَّل بعد التعثر ← تسرّب!)", "default": "الهدف: التعثر"},
        "issues": ["Class imbalance (~15% تعثر)", "Target leakage في sent_to_collections",
                   "Group leakage: العميل نفسه له عدة طلبات", "بُعد زمني يتطلب Time-based split"],
        "lessons": ["feature_engineering", "data_leakage", "model_evaluation", "automl"],
    },
    "students": {
        "loader": students,
        "title": "أداء الطلاب وطرق التدريس · Student outcomes",
        "design": "Experimental-style with pre/post",
        "purpose": "للاختبارات الإحصائية: مستقلة، مزدوجة، ANOVA، Chi-square، والارتباط.",
        "variables": {"student_id": "الطالب", "teaching_method": "طريقة التدريس (3 مجموعات)", "gender": "الجنس",
                      "school_type": "نوع المدرسة", "study_hours": "ساعات الدراسة", "pre_score": "الاختبار القبلي",
                      "post_score": "الاختبار البعدي", "attendance_pct": "نسبة الحضور", "result": "النتيجة Pass/Fail"},
        "issues": ["study_hours ملتوية", "attendance ملتوية لليسار"],
        "lessons": ["hypothesis_tests", "stat_foundations", "bivariate"],
    },
    "iris": {
        "loader": iris,
        "title": "Iris (بيانات عامة) · Iris",
        "design": "Cross-sectional (public)",
        "purpose": "بيانات عامة كلاسيكية صغيرة للتصنيف والتجميع.",
        "variables": {"sepal/petal length/width (cm)": "قياسات الزهرة", "species": "النوع"},
        "issues": ["لا مشكلات؛ مناسبة للتجارب السريعة."],
        "lessons": ["multivariate", "clustering"],
    },
    "wine": {
        "loader": wine,
        "title": "Wine (بيانات عامة) · Wine",
        "design": "Cross-sectional (public)",
        "purpose": "13 خاصية كيميائية بمقاييس مختلفة جدًا — مثالية لتوضيح Scaling وPCA.",
        "variables": {"13 chemical features": "خصائص كيميائية", "cultivar": "الصنف"},
        "issues": ["مقاييس مختلفة جدًا بين الأعمدة"],
        "lessons": ["dim_reduction", "transformations"],
    },
}


def dataset_names() -> list[str]:
    return list(DATASET_INFO)


def load_dataset(name: str) -> pd.DataFrame:
    return DATASET_INFO[name]["loader"]()
