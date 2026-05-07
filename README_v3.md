# LlamaGUI — Engineering Reference

> Version 3.0 | Last updated 2026-05-07
> Portable GUI front-end for llama.cpp, targeting Windows + CUDA.

---

## Project Overview

LlamaGUI wraps `llama-server.exe` (subprocess mode) or `llama-cpp-python` (binding mode) with a Gradio 6 web UI and a FastAPI REST layer. The goal is a self-contained, double-click-to-run package similar to ComfyUI portable.

**v3 新增功能：**
- **System Tray 整合** — `pystray` 圖示常駐工作列；按 X 縮小到 Tray 而非關閉，右鍵選單可顯示/隱藏視窗、卸載模型、查看載入狀態
- **Idle 自動卸載** — 閒置超過設定秒數自動釋放 VRAM（可由 Settings 頁啟用）
- **OpenAI-compatible Embeddings** — `POST /v1/embeddings`，subprocess / binding 兩種模式均支援
- **`GET /api/ps`** — Ollama 相容端點，回傳目前載入模型的執行狀態
- **`GET /v1/models`** — OpenAI 相容模型列表端點
- **`engine.ensure_model_loaded()`** — 依名稱按需切換模型，供 `/v1/chat/completions` 使用

---

## Directory Structure

```
llamagui/
├── app/
│   ├── main.py           Entry point — starts FastAPI + Gradio threads, opens pywebview, builds tray icon
│   ├── api.py            FastAPI REST endpoints (OpenAI-compatible + management)
│   ├── engine.py         Engine abstraction (subprocess / binding modes)
│   ├── config.py         Profile CRUD, global settings, models-dir management
│   └── ui/               Gradio UI package
│       ├── __init__.py   build_ui() + launch_server() — only public entry point
│       ├── state.py      Gradio version detection, session store, history helpers
│       ├── style.py      CSS, hotkey JS, theme registry, CHAT_FORMATS constant
│       └── tabs/
│           ├── __init__.py
│           ├── chat.py       Tab 1 — multi-session chat, streaming, image input
│           ├── model.py      Tab 2 — model load/unload, profile management, scan
│           ├── download.py   Tab 3 — HuggingFace search, GGUF download, progress
│           ├── params.py     Tab 4 — live inference parameter tuning
│           ├── monitor.py    Tab 5 — real-time stats, context usage, VRAM
│           └── settings.py   Tab 6 — global settings, theme, window, HF token, idle unload
├── bin/
│   └── cuda/
│       ├── llama-server.exe  (download separately from llama.cpp releases)
│       └── *.dll             CUDA runtime libraries
├── models/               Drop .gguf model files here
├── logs/                 launch.log, setup.log auto-created at runtime
├── cache/                python embed zip + get-pip.py cached here during build
├── icon.png              (optional) Tray icon — 64×64 PNG; auto-generated if absent
├── run_dev.bat           Dev launcher — uses system Python
├── setup_env.bat         First-time env setup for portable Python
└── build_portable.bat    Full portable package builder
```

---

## Module Responsibilities

### `main.py`

- Fixes `sys.path` using `Path(__file__).resolve().parent` (robust across all Python runtimes).
- Starts FastAPI on port `8000` and Gradio on port `7860` as daemon threads.
- Polls `http://127.0.0.1:7860` up to 45 s; exits with error if Gradio fails to start.
- Opens a `pywebview` window if available; falls back to tray-only or headless mode.
- Builds a `pystray` tray icon; window X button hides (not closes) the window; `stop_idle_monitor()` and `engine.unload()` are called on clean exit.
- Ports are overridable via environment variables `LLAMAGUI_GRADIO_PORT` / `LLAMAGUI_API_PORT`.

#### Tray icon behaviour

| Scenario | Behaviour |
|---|---|
| `pywebview` available | Window X → hide to tray; double-click tray → show window |
| `pywebview` not installed | Tray-only mode; "在瀏覽器開啟" opens `GRADIO_URL` |
| Neither `pystray` nor `pywebview` | Headless; process kept alive with `while True: sleep(1)` |

Tray menu items: model status (read-only), show/hide window, open in browser, unload model (free VRAM), quit.

### `api.py`

FastAPI app (`version="2.4.0"`) with the following route groups:

| Group | Routes |
|---|---|
| Health / lifecycle | `GET /health`, `POST /load`, `POST /unload`, `GET /stats` |
| Metadata | `GET /model-metadata`, `GET /vram`, `GET /context-usage` |
| Prompt | `POST /prompt-preview` |
| HuggingFace | `GET /hf-search`, `GET /hf-files`, `POST /hf-download`, `GET/DELETE /hf-download/{task_id}` |
| Profile CRUD | `GET/POST /profiles`, `GET/POST/DELETE /profiles/{name}`, `POST /profiles/{name}/activate` |
| Models dir | `GET/POST /models-dir` |
| OpenAI-compat chat | `POST /v1/chat/completions` (streaming SSE + non-streaming) |
| **OpenAI-compat embed** | **`POST /v1/embeddings`** ← new in v3 |
| **OpenAI-compat models** | **`GET /v1/models`** ← new in v3 |
| **Ollama-compat** | **`GET /api/ps`** ← new in v3 |
| Models list | `GET /models` |

HF downloads run in daemon threads; progress is polled via `GET /hf-download/{task_id}`.

`POST /v1/chat/completions` accepts an optional `model` field; if it differs from the currently loaded model, `engine.ensure_model_loaded()` is called to auto-switch before inference.

`POST /v1/embeddings` accepts `{ "model": "local", "input": "text or list" }` and returns OpenAI-format embedding vectors.

### `engine.py`

Abstracts two backend modes under a single `engine` singleton:

- **subprocess** — spawns `bin/cuda/llama-server.exe`, communicates via HTTP on a local port.
- **binding** — uses `llama_cpp.Llama` Python bindings directly in-process.

Key methods:

| Method | Description |
|---|---|
| `engine.load(profile)` | Load model with given profile dict |
| `engine.unload()` | Unload model and free VRAM |
| `engine.stream(messages, profile, rf)` | Token-by-token generator |
| `engine.generate(messages, profile, rf)` | Full response string |
| `engine.embed(input_texts)` | Returns `list[list[float]]` — both backends |
| `engine.ensure_model_loaded(model_name)` | Auto-switch model by name |
| `engine.get_stats()` | Returns stats dict |
| `engine.get_context_usage()` | Returns `(used, max)` tuple |
| `engine.get_model_metadata()` | Returns GGUF metadata dict |
| `engine.get_ps_info()` | Returns Ollama-compatible process info dict |
| `engine.touch()` | Reset idle timer |
| `engine.start_idle_monitor()` | Start background idle-unload thread |
| `engine.stop_idle_monitor()` | Signal idle thread to stop |
| `format_prompt_preview(messages, fmt)` | Module-level helper, no engine state |

**Idle monitor** (`_idle_monitor`): polls every 30 s; if `idle_unload_enabled` is true in global settings and the engine has been idle longer than `idle_unload_seconds`, `engine.unload()` is called automatically.

**`ensure_model_loaded(model_name)`**: searches `models/` for a file whose stem matches `model_name` (case-insensitive, normalised). If the current model already matches, returns `(True, "already loaded")`. Otherwise calls `load()` with a minimal switch profile.

**Embedding support**:
- `SubprocessEngine.embed()` — POSTs to `/embedding` on the running `llama-server` process.
- `BindingEngine.embed()` — calls `self.llm.embed(text)` directly.

### `config.py`

JSON-backed persistence for two files:

- `profiles.json` — named inference profiles (model path, GPU layers, n_ctx, etc.)
- `settings.json` — global settings (theme, window size, default params, HF token, idle unload)

models_dir is stored as a path relative to ROOT in profiles.json. On first launch after relocating the project folder, the engine detects an invalid path and falls back to <ROOT>/models/, correcting profiles.json automatically.

Key functions: `get_profile(name)`, `save_profile(name, data)`, `delete_profile(name)`, `set_active(name)`, `get_global_settings()`, `save_global_settings(data)`, `scan_models()`, `get_models_dir()`, `set_models_dir(path)`.

**v3 新增 settings 欄位：**

| Key | Default | Description |
|---|---|---|
| `idle_unload_enabled` | `false` | 啟用閒置自動卸載 |
| `idle_unload_seconds` | `300` | 閒置幾秒後卸載（預設 5 分鐘） |

### `ui/state.py`

Pure Python — no Gradio component instantiation, safe for unit testing.

- **Version flags**: `GR6 = _GR_VER[0] >= 6`, `GR45 = 4 <= _GR_VER[0] <= 5`, `GR3 = _GR_VER[0] <= 3`
- **Session store**: `new_session()`, `delete_session()`, `rename_session()`, `get_session()`, `session_choices()`
- **Download task store**: `register_dl_task(task)` → `int`, `get_dl_task(tid)`
- **History helpers**: `to_gr(history)`, `ensure_dicts(history)` — normalises legacy tuple pairs to `[{role, content}]`

### `ui/style.py`

- `CSS` — Gradio custom CSS string
- `HOTKEY_JS` — `<script>` block injected via `launch(head=...)`; binds Ctrl+Enter, Ctrl+K, Ctrl+Shift+S, Ctrl+R
- `THEMES` dict + `get_theme(name)` factory
- `CHAT_FORMATS` list (single source of truth)

### `ui/__init__.py`

`build_ui()` creates `gr.Blocks`, instantiates the global status bar and five `gr.State` objects, then calls each tab builder in order passing a `shared` dict.

`launch_server(server_port, share)` is the **only function `main.py` should call**. It routes `theme/css/head` to `launch()` on GR6 or to `Blocks()` on GR45.

### `ui/tabs/` — Tab builders

Each module exposes a single `build(shared: dict) -> None` function called within an active `gr.Blocks` context.

`shared` keys available to all tabs:

| Key | Type | Description |
|---|---|---|
| `status_md` | `gr.Markdown` | Global status bar updated after each inference action |
| `temp_state` | `gr.State` | Temperature — synced from Params tab |
| `top_p_state` | `gr.State` | Top-P |
| `top_k_state` | `gr.State` | Top-K |
| `rp_state` | `gr.State` | Repeat penalty |
| `mt_state` | `gr.State` | Max tokens |

**`settings.py`** (Tab 6) now surfaces the two idle-unload settings (`idle_unload_enabled`, `idle_unload_seconds`) and saves them via `config.save_global_settings()`.

---

## Gradio Version Compatibility

| Feature | GR 3.x | GR 4.x / 5.x | GR 6.x |
|---|---|---|---|
| `Chatbot(type="messages")` | No | Required | Do not pass — removed |
| `gr.Image(source=...)` | Yes | Yes | Use `sources=[...]` |
| `theme/css/head` location | `Blocks()` | `Blocks()` | `launch()` |
| `gr.Timer` | No | Yes | Yes |

Version detection in `ui/state.py`:

```python
_GR_VER = tuple(int(x) for x in gr.__version__.split(".")[:2])
GR6  = _GR_VER[0] >= 6
GR45 = 4 <= _GR_VER[0] <= 5
GR3  = _GR_VER[0] <= 3
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMAGUI_ROOT` | parent of `app/` | Root path; used for logs, models, bin |
| `LLAMAGUI_VARIANT` | `cuda` | `cuda` or `cpu` |
| `LLAMAGUI_BIN` | `{ROOT}/bin/cuda` | Path to `llama-server.exe` |
| `LLAMAGUI_GRADIO_PORT` | `7860` | Gradio server port |
| `LLAMAGUI_API_PORT` | `8000` | FastAPI server port |

---

## BAT Scripts

### `run_dev.bat`

Uses system Python. Checks for `gradio`, `fastapi`, `uvicorn`, `pywebview`, `pystray`, `pillow`; installs if missing. Prints Python path and Gradio version before launch for quick diagnostic.

### `setup_env.bat [cuda|cpu]`

First-time setup for portable embedded Python. Steps:
1. Patch `python3X._pth` to enable `site-packages`
2. Bootstrap pip via `get-pip.py`
3. Install core packages: `gradio fastapi uvicorn[standard] pywebview pystray pillow`
4. Install `llama-cpp-python` CUDA binding (optional, skipped on failure)

Logs to `logs/setup.log`. All messages are ASCII-safe.

### `build_portable.bat`

Produces a self-contained `dist/LlamaGUI_Portable_CUDA_vX.X/` folder:
1. Create folder structure
2. Copy `app/` source files
3. Copy `bin/cuda/` (must pre-populate `llama-server.exe`)
4. Download Python 3.11.9 embeddable zip (cached in `cache/`)
5. Install pip into embedded Python
6. Install all packages into embedded Python
7. Generate `run_cuda.bat` dynamically, copy `setup_env.bat`, compress to ZIP

Requires `bin/cuda/llama-server.exe` to exist before running.

---

## Adding a New Tab

1. Create `app/ui/tabs/my_tab.py` with:
   ```python
   import gradio as gr

   def build(shared: dict) -> None:
       with gr.Tab("My Tab"):
           ...
   ```
2. Add to `app/ui/tabs/__init__.py`:
   ```python
   from ui.tabs import ..., my_tab
   __all__ = [..., "my_tab"]
   ```
3. Add one line in `app/ui/__init__.py` inside `build_ui()`:
   ```python
   my_tab.build(shared)
   ```

No other files need modification.

---

## Known Issues & Notes

- `llama-server.exe` must be the CUDA build from the official llama.cpp releases page. CPU builds lack the `/v1/chat/completions` endpoint used by subprocess mode.
- `pywebview` is optional. If not installed, the app runs in tray-only mode; set `auto_open_browser: true` in settings to open a browser tab automatically on launch.
- `pystray` + `Pillow` are optional. If absent, the tray icon is silently disabled; the app still runs normally via browser.
- `icon.png` (64×64) in the app root replaces the default generated tray icon. If the file is absent, a blue circle on dark background is used.
- The closing event hook (`win.events.closing += _on_closing`) requires pywebview 4.x. On older versions the window will close normally instead of hiding to tray.
- Path spaces: Windows paths with spaces are supported in Python code but the BAT scripts use quoted paths throughout. Avoid spaces in the root folder name for maximum compatibility.
- The `head=` parameter in `gr.Blocks.launch()` was introduced in Gradio 4.x and may be silently ignored in some 6.x builds; hotkeys are wrapped in `try/except` accordingly.
- Embedding via subprocess mode requires a `llama-server.exe` build that supports the `/embedding` endpoint (available since llama.cpp build b2000+).