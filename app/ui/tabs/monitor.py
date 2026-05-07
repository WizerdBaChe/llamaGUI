# ui/tabs/monitor.py
# Tab 5 -- Real-time inference stats, context usage, VRAM
from __future__ import annotations
import gradio as gr
import engine as eng

def build(shared: dict) -> None:
    def _make_timer(interval):
        try:
            return gr.Timer(value=interval, active=True)
        except Exception:
            return None

    def _stats():
        s             = eng.engine.get_stats()
        used, max_ctx = eng.engine.get_context_usage()
        pct           = f"{used/max_ctx*100:.1f}%" if max_ctx else "-"
        vram          = eng.get_vram_info()
        na            = "-"
        return (
            s.get("model_name",  na),
            s.get("engine_mode", na),
            f'{s.get("tokens_per_sec",    0):.1f}',
            f'{s.get("elapsed_sec",       0):.2f}s',
            str(s.get("completion_tokens", 0)),
            str(used), str(max_ctx), pct,
            f'{vram["used_mb"]/1024:.1f} GB'  if vram["available"] else na,
            f'{vram["total_mb"]/1024:.1f} GB' if vram["available"] else na,
            f'{vram["utilization_pct"]}%'     if vram["available"] else na,
        )

    with gr.Tab("Monitor"):
        gr.Markdown("### Inference Stats (auto-refresh every 5 s)")

        with gr.Row():
            s_model   = gr.Textbox(label="Model",          interactive=False)
            s_mode    = gr.Textbox(label="Engine",         interactive=False)
            s_tps     = gr.Textbox(label="Tokens/s",       interactive=False)
            s_elapsed = gr.Textbox(label="Elapsed",        interactive=False)
            s_tokens  = gr.Textbox(label="Completion Tok", interactive=False)

        gr.Markdown("**Context Usage**")
        with gr.Row():
            s_ctx_used = gr.Textbox(label="Used",     interactive=False)
            s_ctx_max  = gr.Textbox(label="Max",      interactive=False)
            s_ctx_pct  = gr.Textbox(label="Usage %",  interactive=False)

        gr.Markdown("**GPU VRAM**")
        with gr.Row():
            s_vram_used  = gr.Textbox(label="VRAM Used",  interactive=False)
            s_vram_total = gr.Textbox(label="VRAM Total", interactive=False)
            s_vram_util  = gr.Textbox(label="GPU Util %", interactive=False)

        refresh_btn = gr.Button("Refresh now")

        _outs = [s_model, s_mode, s_tps, s_elapsed, s_tokens,
                 s_ctx_used, s_ctx_max, s_ctx_pct,
                 s_vram_used, s_vram_total, s_vram_util]

        stats_timer = _make_timer(5)
        if stats_timer:
            stats_timer.tick(fn=_stats, outputs=_outs)
        refresh_btn.click(fn=_stats, outputs=_outs)
