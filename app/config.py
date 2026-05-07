# config.py  v2.0
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any


def _resolve_root() -> Path:
    if env := os.environ.get("LLAMAGUI_ROOT"):
        p = Path(env)
        if p.is_dir():
            return p.resolve()
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "app" else here.parent


ROOT    = _resolve_root()
BINDIR  = Path(os.environ.get("LLAMAGUI_BIN", str(ROOT / "bin" / "cuda")))

_PROFILES_PATH = ROOT / "profiles.json"
_SETTINGS_PATH = ROOT / "settings.json"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PROFILE: dict[str, Any] = dict(
    model_path="", mmproj_path="",
    n_gpu_layers=35, n_ctx=4096, n_batch=512,
    temperature=0.7, top_p=0.9, top_k=40,
    repeat_penalty=1.1, max_tokens=2048,
    chat_format="chatml", engine_mode="subprocess", verbose=False,
)

DEFAULT_SETTINGS: dict[str, Any] = dict(
    theme="Soft",
    language="zh-TW",
    auto_open_browser=False,
    stream_delay_ms=0,
    show_system_prompt=True,
    show_token_count=True,
    window_width=1280,
    window_height=820,
    window_min_w=900,
    window_min_h=600,
    default_temperature=0.7,
    default_top_p=0.9,
    default_top_k=40,
    default_repeat_penalty=1.1,
    default_max_tokens=2048,
    default_n_gpu_layers=35,
    default_n_ctx=4096,
    idle_unload_enabled=False,   # ← 新增
    idle_unload_seconds=300,     # ← 新增
    hf_token="",
)

_DEFAULT_DATA: dict[str, Any] = dict(
    active_profile="default",
    models_dir=str(ROOT / "models"),
    profiles=dict(default=DEFAULT_PROFILE.copy()),
)

# ── raw I/O ───────────────────────────────────────────────────────────────────
def _load_raw() -> dict[str, Any]:
    if not _PROFILES_PATH.exists():
        _save_raw(_DEFAULT_DATA)
        return _DEFAULT_DATA.copy()
    try:
        return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))
    except Exception:
        _save_raw(_DEFAULT_DATA)
        return _DEFAULT_DATA.copy()

def _save_raw(data: dict) -> None:
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# ── profiles ──────────────────────────────────────────────────────────────────
def get_profile(name: str | None = None) -> dict[str, Any]:
    data = _load_raw()
    name = name or data.get("active_profile", "default")
    p = data.get("profiles", {}).get(name, DEFAULT_PROFILE.copy())
    for k, v in DEFAULT_PROFILE.items():
        p.setdefault(k, v)
    return p

def save_profile(name: str, profile: dict[str, Any]) -> None:
    data = _load_raw()
    data.setdefault("profiles", {})[name] = {**DEFAULT_PROFILE, **profile}
    _save_raw(data)

def delete_profile(name: str) -> bool:
    if name == "default":
        return False
    data = _load_raw()
    if name not in data.get("profiles", {}):
        return False
    del data["profiles"][name]
    if data.get("active_profile") == name:
        data["active_profile"] = "default"
    _save_raw(data)
    return True

def set_active(name: str) -> bool:
    data = _load_raw()
    if name not in data.get("profiles", {}):
        return False
    data["active_profile"] = name
    _save_raw(data)
    return True

def get_active_name() -> str:
    return _load_raw().get("active_profile", "default")

def list_profiles() -> list[str]:
    return list(_load_raw().get("profiles", {}).keys())

def get_all() -> dict[str, Any]:
    return _load_raw()

# ── models dir ────────────────────────────────────────────────────────────────
def get_models_dir() -> str:
    raw  = _load_raw()
    path = raw.get("models_dir", "")

    # 相對路徑轉絕對（相對於 ROOT）
    if path and not os.path.isabs(path):
        path = str(ROOT / path)

    # 存活檢查：路徑無效時 fallback 到 ROOT/models
    if not path or not os.path.isdir(path):
        fallback = str(ROOT / "models")
        os.makedirs(fallback, exist_ok=True)
        # 修正寫回，避免下次繼續走死路徑
        raw["models_dir"] = _to_rel(fallback)
        _save_raw(raw)
        return fallback

    return path


def set_models_dir(path: str) -> None:
    data = _load_raw()
    data["models_dir"] = _to_rel(path)   # 存相對路徑
    _save_raw(data)


def _to_rel(path: str) -> str:
    """盡量轉成相對於 ROOT 的路徑；跨磁碟時保留絕對路徑。"""
    try:
        rel = os.path.relpath(path, ROOT)
        # relpath 結果若需要往上跳太多層（跨磁碟或完全不相關），保留絕對路徑
        if rel.startswith("..") and len(rel) > 20:
            return path
        return rel
    except ValueError:
        return path   # Windows 跨磁碟 (C: vs D:) 會 raise ValueError


def scan_models() -> list[str]:
    d = get_models_dir()
    if not os.path.isdir(d):
        return []
    found = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".gguf"):
                found.append(os.path.join(root, f))
    found.sort()
    return found

# ── settings ──────────────────────────────────────────────────────────────────
def get_global_settings() -> dict[str, Any]:
    if not _SETTINGS_PATH.exists():
        save_global_settings(DEFAULT_SETTINGS.copy())
        return DEFAULT_SETTINGS.copy()
    try:
        loaded = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_SETTINGS, **loaded}
    except Exception:
        save_global_settings(DEFAULT_SETTINGS.copy())
        return DEFAULT_SETTINGS.copy()

def save_global_settings(settings: dict[str, Any]) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

def get_bin_dir() -> str:
    return str(BINDIR)

def get_variant() -> str:
    return os.environ.get("LLAMAGUI_VARIANT", "cuda")
