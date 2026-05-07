# ui/tabs/params.py
# Tab 4 -- Inference parameter tuning (no model reload required)
from __future__ import annotations
import gradio as gr
import config

def build(shared: dict) -> None:
    """Build the Params tab.

    shared keys consumed:
        temp_state  top_p_state  top_k_state  rp_state  mt_state
    """
    temp_state  = shared["temp_state"]
    top_p_state = shared["top_p_state"]
    top_k_state = shared["top_k_state"]
    rp_state    = shared["rp_state"]
    mt_state    = shared["mt_state"]

    with gr.Tab("Params"):
        gr.Markdown("Adjust inference parameters without reloading the model.")

        temperature    = gr.Slider(0.0, 2.0, step=0.05, value=0.7, label="Temperature")
        top_p          = gr.Slider(0.0, 1.0, step=0.05, value=0.9, label="Top-P")
        top_k          = gr.Slider(1,   200, step=1,    value=40,  label="Top-K")
        repeat_penalty = gr.Slider(1.0, 2.0, step=0.05, value=1.1, label="Repeat Penalty")
        max_tokens     = gr.Slider(64, 8192, step=64,   value=2048, label="Max Tokens")

        with gr.Row():
            apply_btn = gr.Button("Apply to Chat",          variant="primary")
            save_btn  = gr.Button("Save to Active Profile", variant="secondary")
        result_md = gr.Markdown("")

        # Live-sync sliders -> shared States so Chat tab picks up changes
        temperature.change(   lambda v: v, [temperature],    [temp_state])
        top_p.change(         lambda v: v, [top_p],          [top_p_state])
        top_k.change(         lambda v: v, [top_k],          [top_k_state])
        repeat_penalty.change(lambda v: v, [repeat_penalty], [rp_state])
        max_tokens.change(    lambda v: v, [max_tokens],     [mt_state])

        def _apply(temp, tp, tk, rp, mt):
            return temp, tp, tk, rp, mt, "Applied to Chat."

        apply_btn.click(
            _apply,
            [temperature, top_p, top_k, repeat_penalty, max_tokens],
            [temp_state, top_p_state, top_k_state, rp_state, mt_state, result_md])

        def _save(temp, tp, tk, rp, mt):
            name = config.get_active_name()
            p    = config.get_profile(name)
            p.update(temperature=temp, top_p=tp, top_k=int(tk),
                     repeat_penalty=rp, max_tokens=int(mt))
            config.save_profile(name, p)
            return f"Saved to profile **{name}**"

        save_btn.click(
            _save,
            [temperature, top_p, top_k, repeat_penalty, max_tokens],
            [result_md])
