# ui/tabs/settings.py
# Tab 6 -- Global settings
from __future__ import annotations
import gradio as gr
import config
from ui.style import THEME_NAMES

def build(shared: dict) -> None:
    s = config.get_global_settings()

    with gr.Tab("Settings"):
        gr.Markdown("### Global Settings")
        gr.Markdown("*Theme and window size take effect after restart.*")

        with gr.Accordion("Interface", open=True):
            s_theme   = gr.Dropdown(label="Theme", choices=THEME_NAMES,
                                    value=s.get("theme", "Soft"))
            s_lang    = gr.Dropdown(label="Language", choices=["zh-TW", "en"],
                                    value=s.get("language", "zh-TW"))
            s_showsys = gr.Checkbox(label="Show System Prompt field",
                                    value=s.get("show_system_prompt", True))
            s_showtok = gr.Checkbox(label="Show token count",
                                    value=s.get("show_token_count", True))

        with gr.Accordion("Window", open=False):
            with gr.Row():
                s_winw = gr.Number(label="Width",  value=s.get("window_width",  1280))
                s_winh = gr.Number(label="Height", value=s.get("window_height",  820))
            s_auto_browser = gr.Checkbox(
                label="Auto-open browser if pywebview unavailable",
                value=s.get("auto_open_browser", False))

        with gr.Accordion("Default Inference Params", open=False):
            s_temp = gr.Slider(0.0, 2.0,   step=0.05, value=s.get("default_temperature",   0.7),  label="Temperature")
            s_tp   = gr.Slider(0.0, 1.0,   step=0.05, value=s.get("default_top_p",          0.9),  label="Top-P")
            s_tk   = gr.Slider(1,   200,   step=1,    value=s.get("default_top_k",           40),   label="Top-K")
            s_rp   = gr.Slider(1.0, 2.0,   step=0.05, value=s.get("default_repeat_penalty",  1.1),  label="Repeat Penalty")
            s_mt   = gr.Slider(64,  8192,  step=64,   value=s.get("default_max_tokens",     2048),  label="Max Tokens")
            s_ngl  = gr.Slider(0,   200,   step=1,    value=s.get("default_n_gpu_layers",     35),  label="n_gpu_layers")
            s_ctx  = gr.Slider(512, 131072, step=512,  value=s.get("default_n_ctx",          4096),  label="n_ctx")

        with gr.Accordion("Idle Auto-Unload（閒置自動釋放 VRAM）", open=False):
            s_idle_en  = gr.Checkbox(
                label="啟用閒置自動卸載模型",
                value=s.get("idle_unload_enabled", False),
                info="無操作超過設定時間後，自動 unload 模型以釋放 VRAM")
            s_idle_sec = gr.Slider(
                30, 3600, step=30,
                value=s.get("idle_unload_seconds", 300),
                label="閒置幾秒後自動卸載（秒）",
                info="預設 300 秒（5 分鐘）。需先勾選上方選項才生效。")

        with gr.Accordion("HuggingFace", open=False):
            s_hf = gr.Textbox(label="HuggingFace Access Token (optional)",
                              value=s.get("hf_token", ""), type="password")

        with gr.Row():
            save_btn  = gr.Button("Save Settings",     variant="primary")
            reset_btn = gr.Button("Reset to Defaults", variant="secondary")
        result_md = gr.Markdown("")

        with gr.Accordion("Keyboard Shortcuts", open=False):
            gr.Markdown(
                "| Shortcut | Action |\n|---|---|\n"
                "| Ctrl+Enter   | Send message |\n"
                "| Ctrl+K       | Clear chat |\n"
                "| Ctrl+Shift+S | Stop generation |\n"
                "| Ctrl+R       | Retry / regenerate |\n"
                "| Shift+Enter  | New line in input |")

        _inputs = [s_theme, s_lang, s_showsys, s_showtok,
                   s_winw, s_winh, s_auto_browser,
                   s_temp, s_tp, s_tk, s_rp, s_mt, s_ngl, s_ctx,
                   s_idle_en, s_idle_sec,
                   s_hf]

        def _save(theme, lang, showsys, showtok, winw, winh, auto_browser,
                  temp, tp, tk, rp, mt, ngl, ctx,
                  idle_en, idle_sec,
                  hft):
            config.save_global_settings(dict(
                theme=theme, language=lang,
                show_system_prompt=showsys, show_token_count=showtok,
                window_width=int(winw), window_height=int(winh),
                auto_open_browser=auto_browser,
                default_temperature=temp, default_top_p=tp,
                default_top_k=int(tk), default_repeat_penalty=rp,
                default_max_tokens=int(mt), default_n_gpu_layers=int(ngl),
                default_n_ctx=int(ctx),
                idle_unload_enabled=idle_en,
                idle_unload_seconds=int(idle_sec),
                hf_token=hft,
            ))
            return "Settings saved. (Theme / window require restart.)"

        def _reset():
            d = config.DEFAULT_SETTINGS.copy()
            config.save_global_settings(d)
            return (d["theme"], d["language"],
                    d["show_system_prompt"], d["show_token_count"],
                    d["window_width"], d["window_height"], d["auto_open_browser"],
                    d["default_temperature"], d["default_top_p"],
                    d["default_top_k"], d["default_repeat_penalty"],
                    d["default_max_tokens"], d["default_n_gpu_layers"],
                    d["default_n_ctx"],
                    d["idle_unload_enabled"], d["idle_unload_seconds"],
                    d["hf_token"],
                    "Reset to defaults.")

        save_btn.click(_save,   _inputs,          [result_md])
        reset_btn.click(_reset, outputs=_inputs + [result_md])