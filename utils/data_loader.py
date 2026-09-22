"""Safe file loading for uploads (CSV / Excel / JSON).

Checks: extension, size, empty file, encoding (UTF-8, UTF-8-SIG, Windows-1256
for Arabic, Latin-1), delimiter sniffing, malformed rows, duplicate and
missing headers. All user-facing messages are in Arabic. Nothing is ever
executed or sent anywhere.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from config import ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_MB

log = logging.getLogger(__name__)
ENCODINGS = ("utf-8", "utf-8-sig", "cp1256", "latin-1")


@dataclass
class LoadResult:
    df: pd.DataFrame | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.errors


def _decode(raw: bytes) -> tuple[str | None, str | None]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, None


def _header_checks(header: list[str], res: LoadResult) -> None:
    cleaned = [h.strip() for h in header]
    dups = sorted({h for h in cleaned if cleaned.count(h) > 1 and h})
    if dups:
        res.warnings.append(f"أسماء أعمدة مكررة: {dups}. سيضيف pandas لاحقة مثل ‎.1‎ لتمييزها؛ راجع المصدر.")
    blanks = sum(1 for h in cleaned if not h)
    if blanks:
        res.warnings.append(f"{blanks} عمود بلا اسم في الترويسة (Missing header).")
    numeric_like = sum(1 for h in cleaned if h.replace(".", "", 1).replace("-", "", 1).isdigit())
    if header and numeric_like / len(header) > 0.5:
        res.warnings.append("أغلب أسماء الأعمدة أرقام: قد لا يحتوي الملف على صف ترويسة (Header). "
                            "إن كان كذلك فاقرأه بـ header=None.")


def load_bytes(raw: bytes, filename: str) -> LoadResult:
    res = LoadResult(None)
    name = filename.lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        res.errors.append(f"الامتداد {ext or '(بلا امتداد)'} غير مدعوم. الصيغ المسموحة: "
                          + ", ".join(ALLOWED_UPLOAD_EXTENSIONS))
        return res
    size_mb = len(raw) / 1e6
    res.info["size_mb"] = round(size_mb, 3)
    if size_mb > MAX_UPLOAD_MB:
        res.errors.append(f"حجم الملف {size_mb:.1f} MB يتجاوز الحد {MAX_UPLOAD_MB} MB.")
        return res
    if len(raw.strip()) == 0:
        res.errors.append("الملف فارغ.")
        return res
    try:
        if ext in (".csv", ".txt"):
            _load_csv(raw, res)
        elif ext in (".xlsx", ".xls"):
            _load_excel(raw, res)
        elif ext == ".json":
            _load_json(raw, res)
    except Exception as exc:  # never show a raw traceback to the learner
        log.exception("Upload failed for %s", filename)
        res.errors.append(f"تعذّرت قراءة الملف ({type(exc).__name__}). تحقّق من أنه غير تالف وبالصيغة الصحيحة.")
        res.df = None
    if res.df is not None:
        if res.df.empty:
            res.errors.append("قُرئ الملف لكنه لا يحتوي على صفوف بيانات.")
        elif res.df.shape[1] == 1 and ext in (".csv", ".txt"):
            res.warnings.append("قُرئ عمود واحد فقط: قد يكون الفاصل (Delimiter) غير مكتشف بشكل صحيح.")
    return res


def _load_csv(raw: bytes, res: LoadResult) -> None:
    text, enc = _decode(raw)
    if text is None:
        res.errors.append("تعذّر تحديد ترميز الملف (Encoding). احفظه بترميز UTF-8.")
        return
    res.info["encoding"] = enc
    if enc in ("cp1256", "latin-1"):
        res.warnings.append(f"الملف ليس UTF-8؛ قُرئ بترميز {enc}. تحقّق من ظهور النصوص العربية بشكل صحيح.")
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
        res.warnings.append("لم يُكتشف الفاصل تلقائيًا؛ استُخدمت الفاصلة «,».")
    res.info["delimiter"] = {"\t": "TAB", ",": ",", ";": ";", "|": "|"}.get(sep, sep)
    first_line = next(csv.reader(io.StringIO(sample), delimiter=sep), [])
    _header_checks(first_line, res)
    # Count fields ourselves: pandas may silently shift extra fields into an index or drop them.
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    width = len(first_line)
    good = [rows[0]] + [r for r in rows[1:] if len(r) == width]
    n_bad = len(rows) - len(good)
    if n_bad:
        res.warnings.append(f"تُجوهل {n_bad} سطرًا معطوبًا (عدد حقول غير مطابق للترويسة: {width}).")
        res.info["malformed_rows"] = n_bad
    buf = io.StringIO()
    csv.writer(buf, delimiter=sep).writerows(good)
    res.df = pd.read_csv(io.StringIO(buf.getvalue()), sep=sep)


def _load_excel(raw: bytes, res: LoadResult) -> None:
    xls = pd.ExcelFile(io.BytesIO(raw))
    res.info["sheets"] = xls.sheet_names
    if len(xls.sheet_names) > 1:
        res.warnings.append(f"الملف يحتوي {len(xls.sheet_names)} أوراق؛ قُرئت الأولى: «{xls.sheet_names[0]}».")
    df = xls.parse(xls.sheet_names[0])
    _header_checks([str(c) for c in df.columns], res)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        res.warnings.append(f"{len(unnamed)} عمود بلا عنوان (Unnamed) — ربما خلايا مدمجة أو ترويسة غير مكتملة.")
    res.df = df


def _load_json(raw: bytes, res: LoadResult) -> None:
    text, enc = _decode(raw)
    if text is None:
        res.errors.append("تعذّر تحديد ترميز ملف JSON.")
        return
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        res.errors.append(f"JSON غير صالح عند السطر {exc.lineno}، العمود {exc.colno}.")
        return
    if isinstance(data, dict):
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if list_keys:
            res.warnings.append(f"الملف كائن JSON؛ قُرئت القائمة تحت المفتاح «{list_keys[0]}».")
            data = data[list_keys[0]]
        else:
            data = [data]
    if not isinstance(data, list):
        res.errors.append("بنية JSON غير جدولية.")
        return
    df = pd.json_normalize(data)
    nested = [c for c in df.columns if "." in str(c)]
    if nested:
        res.warnings.append(f"فُكّكت {len(nested)} حقول متداخلة (Nested) إلى أعمدة مسطّحة مثل: {nested[:3]}.")
    # Lists/dicts left after flattening are unhashable (break duplicated(), nunique()); keep them as JSON text.
    list_cols = [c for c in df.columns if df[c].map(lambda v: isinstance(v, (list, dict))).any()]
    for c in list_cols:
        df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    if list_cols:
        res.warnings.append(f"أعمدة تحتوي قوائم (Lists) حُفظت كنص JSON: {list_cols}. فكّكها بـexplode() إن احتجت.")
    res.df = df


def initial_warnings(df: pd.DataFrame) -> list[str]:
    """Quick heuristics shown right after import."""
    from utils.types import infer_semantic_type

    out = []
    miss = df.isna().mean()
    for c, v in miss[miss > 0.3].items():
        out.append(f"العمود «{c}» مفقود بنسبة {v:.0%}.")
    dups = int(df.duplicated().sum())
    if dups:
        out.append(f"{dups} صفًا مكررًا تمامًا.")
    for c in df.columns:
        sem, reason = infer_semantic_type(df[c], str(c))
        if sem in ("numeric-as-text", "date-as-text"):
            out.append(f"«{c}»: {reason}")
        if df[c].nunique(dropna=True) == 1:
            out.append(f"«{c}» ثابت (قيمة واحدة).")
    return out
