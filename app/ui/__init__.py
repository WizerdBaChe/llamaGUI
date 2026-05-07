# Single public entry point: launch_server()
# Assembles gr.Blocks, injects shared state, calls each tab builder.
from __future__ import annotations
import gradio as gr
import config
import engine as eng

from ui.state  import GR6, GR45, INITIAL_SID
from ui.style  import CSS, HOTKEY_JS, get_theme
from ui.tabs   import chat, model, download, params, monitor, settings


def _refresh_status() -> str:
    if eng.engine.is_loaded:
        s    = eng.engine.get_stats()
        tps  = s.get("tokens_per_sec", 0)
        name = s.get("model_name",  "")
        mode = s.get("engine_mode", "")
        tps_str  = f" | **{tps:.1f}** t/s" if tps > 0 else ""
        vram     = eng.get_vram_info()
        vram_str = (f" | VRAM {vram['used_mb']/1024:.1f}/{vram['total_mb']/1024:.1f} GB"
                    if vram["available"] else "")
        return f"**{name}** [{mode}]{tps_str}{vram_str}"
    vram = eng.get_vram_info()
    if vram["available"]:
        return f"no model loaded | VRAM {vram['used_mb']/1024:.1f}/{vram['total_mb']/1024:.1f} GB"
    return "no model loaded"


def build_ui() -> gr.Blocks:
    cfg       = config.get_global_settings()
    theme_name = cfg.get("theme", "Soft")

    # GR6: theme/css/head go into launch(); pass nothing to Blocks
    # GR45: pass them here
    blocks_kw: dict = {}
    if not GR6:
        blocks_kw["theme"] = get_theme(theme_name)
        blocks_kw["css"]   = CSS
        blocks_kw["head"]  = HOTKEY_JS

    with gr.Blocks(title="LlamaGUI", **blocks_kw) as demo:

        # ── Global status bar ─────────────────────────────────────────────────
        with gr.Row():
            status_md = gr.Markdown(value=_refresh_status(), elem_id="status-bar")

        try:
            status_timer = gr.Timer(value=3, active=True)
            status_timer.tick(fn=_refresh_status, outputs=[status_md])
        except Exception:
            pass   # Gradio build without Timer support -- status updates on each action

        # ── Shared state passed to every tab builder ──────────────────────────
        shared: dict = {
            "status_md":   status_md,
            "temp_state":  gr.State(cfg.get("default_temperature",    0.7)),
            "top_p_state": gr.State(cfg.get("default_top_p",          0.9)),
            "top_k_state": gr.State(cfg.get("default_top_k",           40)),
            "rp_state":    gr.State(cfg.get("default_repeat_penalty",  1.1)),
            "mt_state":    gr.State(cfg.get("default_max_tokens",     2048)),
        }

        # ── Build each tab in order ───────────────────────────────────────────
        chat    .build(shared)
        model   .build(shared)
        download.build(shared)
        params  .build(shared)
        monitor .build(shared)
        settings.build(shared)

    return demo


def launch_server(server_port: int = 7860, share: bool = False) -> None:
    """Called exclusively by main.py -- do not call build_ui() directly."""
    cfg        = config.get_global_settings()
    theme_name = cfg.get("theme", "Soft")

    ui = build_ui()
    ui.queue()

    launch_kw: dict = dict(
        server_name="127.0.0.1",
        server_port=server_port,
        share=share,
        inbrowser=False,
        show_error=True,
    )

    if GR6:
        launch_kw["theme"] = get_theme(theme_name)
        launch_kw["css"]   = CSS
        try:
            launch_kw["head"] = HOTKEY_JS
        except Exception:
            pass

    ui.launch(**launch_kw)