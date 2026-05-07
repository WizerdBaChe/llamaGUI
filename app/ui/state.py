# ui/state.py
# Shared runtime state, version detection, and history helpers.
# No Gradio imports here — keeps this module independently testable.
from __future__ import annotations
import time, uuid
import gradio as gr

# ── Gradio version detection ──────────────────────────────────────────────────
_GR_VER: tuple[int, int] = tuple(int(x) for x in gr.__version__.split(".")[:2])  # type: ignore[assignment]
GR6  = _GR_VER[0] >= 6           # 6.x  — type param removed, css/head in launch()
GR45 = 4 <= _GR_VER[0] <= 5      # 4.x / 5.x — type="messages" must be explicit
GR3  = _GR_VER[0] <= 3           # legacy tuple format

GR_VERSION_STR = gr.__version__

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}

def new_session(name: str | None = None) -> str:
    sid = uuid.uuid4().hex[:8]
    _sessions[sid] = {
        "name":    name or f"Chat {len(_sessions) + 1}",
        "history": [],
        "created": time.time(),
    }
    return sid

def session_choices() -> list[tuple[str, str]]:
    return [
        (_sessions[s]["name"], s)
        for s in sorted(_sessions, key=lambda x: _sessions[x]["created"])
    ]

def get_session(sid: str) -> dict | None:
    return _sessions.get(sid)

def delete_session(sid: str) -> str | None:
    """Delete session and return the next available sid, or None."""
    if sid in _sessions:
        del _sessions[sid]
    return list(_sessions.keys())[-1] if _sessions else None

def rename_session(sid: str, name: str) -> bool:
    if sid in _sessions and name.strip():
        _sessions[sid]["name"] = name.strip()
        return True
    return False

# ── Download task store ───────────────────────────────────────────────────────
_dl_tasks: dict[int, dict] = {}   # id(task_dict) -> task_dict

def register_dl_task(task: dict) -> int:
    tid = id(task)
    _dl_tasks[tid] = task
    return tid

def get_dl_task(tid: int | str) -> dict | None:
    try:
        return _dl_tasks.get(int(tid))
    except (ValueError, TypeError):
        return None

# ── History format helpers ────────────────────────────────────────────────────
def to_gr(history: list[dict]) -> list[dict]:
    """Internal [{role, content}] -> Gradio Chatbot format.
    GR6 and GR45 (type='messages') both use the same dict list directly."""
    return history

def ensure_dicts(history) -> list[dict]:
    """Normalise any incoming history to internal dict-list format.
    Handles legacy GR3 tuple pairs that Gradio <4 may return."""
    if not history:
        return []
    if isinstance(history[0], (list, tuple)):
        out: list[dict] = []
        for pair in history:
            out.append({"role": "user", "content": pair[0] or ""})
            if pair[1] is not None:
                out.append({"role": "assistant", "content": pair[1]})
        return out
    return list(history)

# ── Initialise one default session on module load ─────────────────────────────
INITIAL_SID = new_session("Chat 1")
