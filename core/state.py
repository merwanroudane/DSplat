"""Session state: one place that initialises and mutates per-user progress.

Everything is kept in ``st.session_state`` (per browser session). Progress can
be exported/imported as JSON from the "My Progress" page, which gives
persistence without accounts or a database (see docs/architecture.md).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from core.curriculum import tracked_modules

_DEFAULTS: dict[str, Any] = {
    "level": "beginner",
    "reduce_motion": False,
    "completed": set(),
    "visited": set(),
    "quiz_scores": {},  # module_id -> {"score": int, "total": int, "at": iso}
    "exercises": set(),  # "module_id:level"
    "labs_visited": set(),
    "bookmarks": set(),
    "current_module": None,
    "uploaded_df": None,
    "uploaded_name": None,
    "decision_log": [],  # human-in-the-loop decisions across labs
}

_LEVEL_ORDER = {"beginner": 0, "advanced": 1, "research": 2}


def init_state() -> None:
    """Create missing keys with safe defaults. Called once per run."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)
    st.session_state["_ds_counter"] = 0


def next_key(prefix: str) -> str:
    """Unique, stable-per-run widget/container key."""
    st.session_state["_ds_counter"] = st.session_state.get("_ds_counter", 0) + 1
    return f"{prefix}-{st.session_state['_ds_counter']}"


# ----------------------------------------------------------------- levels
def level() -> str:
    return st.session_state.get("level", "beginner") or "beginner"


def at_least(name: str) -> bool:
    """True if the current explanation level is >= ``name``."""
    return _LEVEL_ORDER.get(level(), 0) >= _LEVEL_ORDER[name]


def reduce_motion() -> bool:
    return bool(st.session_state.get("reduce_motion", False))


# --------------------------------------------------------------- progress
def mark_visited(module_id: str) -> None:
    st.session_state["visited"].add(module_id)
    st.session_state["current_module"] = module_id


def mark_lab(module_id: str) -> None:
    st.session_state["labs_visited"].add(module_id)


def set_complete(module_id: str, done: bool = True) -> None:
    if done:
        st.session_state["completed"].add(module_id)
    else:
        st.session_state["completed"].discard(module_id)


def is_complete(module_id: str) -> bool:
    return module_id in st.session_state.get("completed", set())


def record_quiz(module_id: str, score: int, total: int) -> None:
    st.session_state["quiz_scores"][module_id] = {
        "score": int(score),
        "total": int(total),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def record_exercise(module_id: str, level_name: str) -> None:
    st.session_state["exercises"].add(f"{module_id}:{level_name}")


def toggle_bookmark(module_id: str) -> None:
    marks: set = st.session_state["bookmarks"]
    if module_id in marks:
        marks.discard(module_id)
    else:
        marks.add(module_id)


def progress_pct() -> float:
    total = len(tracked_modules())
    done = len([m for m in tracked_modules() if m.id in st.session_state.get("completed", set())])
    return 0.0 if total == 0 else 100.0 * done / total


def log_decision(source: str, item: str, decision: str, reason: str) -> None:
    """Append a human-in-the-loop decision (used by hybrid labs)."""
    st.session_state["decision_log"].append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "item": item,
            "decision": decision,
            "reason": reason,
        }
    )


# ---------------------------------------------------------- export/import
_EXPORT_KEYS = ("completed", "visited", "quiz_scores", "exercises", "labs_visited", "bookmarks", "level")


def export_progress() -> str:
    payload: dict[str, Any] = {"app": "ds-academy", "version": 1}
    for key in _EXPORT_KEYS:
        value = st.session_state.get(key)
        payload[key] = sorted(value) if isinstance(value, set) else value
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_progress(text: str) -> tuple[bool, str]:
    """Load progress JSON; returns (ok, Arabic message)."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False, "الملف ليس JSON صالحًا."
    if not isinstance(payload, dict) or payload.get("app") != "ds-academy":
        return False, "هذا الملف لا يبدو ملف تقدّم صادرًا من هذه المنصة."
    for key in _EXPORT_KEYS:
        if key not in payload:
            continue
        default = _DEFAULTS[key]
        value = payload[key]
        if isinstance(default, set):
            st.session_state[key] = set(value or [])
        elif isinstance(default, dict):
            st.session_state[key] = dict(value or {})
        elif key == "level" and value in _LEVEL_ORDER:
            st.session_state[key] = value
    return True, "تم استيراد التقدّم بنجاح."
