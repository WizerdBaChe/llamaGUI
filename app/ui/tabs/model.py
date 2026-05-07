# ui/tabs/model.py
# Tab 2 -- Model (load / unload / profile / scan)
from __future__ import annotations
import gradio as gr
import engine as eng
import config
from ui.style import CHAT_FORMATS

def build(shared: dict) -> None:
    status_md = shared["status_md"]

    def _make_timer(interval):
        try:
            return gr.Timer(value=interval, active=True)
        except Exception:
            return None

    def _refresh_status() -> str:
        if eng.engine.is_loaded:
            s    = eng.engine.get_stats()
            tps  = s.get("tokens_per_sec", 0)
            name = s.get("model_name", "")
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

    def _fmt_metadata(meta: dict) -> str:
        if not meta:
            return '<p style="color:var(--body-text-color-subdued)">Load a model to see metadata.</p>'
        if "read_error" in meta:
            return f'<p style="color:red">Read error: {meta["read_error"]}</p>'
        def card(val, key):
            return (f'<div class="meta-card">'
                    f'<div class="meta-val">{val}</div>'
                    f'<div class="meta-key">{key}</div></div>')
        arch   = meta.get("arch",    "-")
        params = f'{meta.get("n_params_b", 0):.1f}B' if meta.get("n_params_b") else "-"
        quant  = meta.get("quant_type",       "-")
        ctx    = f'{meta.get("context_length", 0):,}' if meta.get("context_length") else "-"
        layers = str(meta.get("n_layers",     "-"))
        fsize  = f'{meta.get("file_size_gb",  0):.2f} GB'
        emb    = str(meta.get("embedding_length", "-"))
        name   = meta.get("model_name", "") or arch
        return (f'<div class="meta-grid">'
                f'{card(params,"Parameters")}{card(quant,"Quantization")}{card(ctx,"Max Context")}'
                f'{card(layers,"Layers")}{card(fsize,"File Size")}{card(emb,"Embedding")}'
                f'</div>'
                f'<p style="font-size:12px;color:var(--body-text-color-subdued);margin-top:6px">'
                f'Arch: {arch} | Name: {name}</p>')

    with gr.Tab("Model"):

        gr.Markdown("### Model Info")
        metadata_html  = gr.HTML(value=_fmt_metadata({}))
        metadata_timer = _make_timer(5)
        if metadata_timer:
            metadata_timer.tick(
                fn=lambda: _fmt_metadata(eng.engine.get_model_metadata()),
                outputs=[metadata_html])

        gr.Markdown("---\n### Profile")
        with gr.Row():
            _profs  = config.list_profiles()
            _active = config.get_active_name()
            if _active not in _profs:
                _active = _profs[0] if _profs else None
            profile_dd       = gr.Dropdown(label="Profile", choices=_profs,
                                            value=_active, scale=3)
            new_profile_name = gr.Textbox(label="New profile name", scale=2)
            save_profile_btn = gr.Button("Save",   scale=1)
            del_profile_btn  = gr.Button("Delete", variant="stop", scale=1)

        gr.Markdown("---\n### Step 1 — Set Models Folder & Scan")
        with gr.Row():
            models_dir_box = gr.Textbox(
                label="Models folder",
                value=config.get_models_dir(),
                placeholder="D:\\models",
                scale=5)
            set_dir_btn = gr.Button("Set Folder", size="sm", scale=1)
            scan_btn    = gr.Button("Scan", variant="primary", scale=1)
        scan_status = gr.Markdown("")

        gr.Markdown("### Step 2 — Select Model")
        scanned_dd = gr.Dropdown(
            label="Found models (select to fill path below)",
            choices=[], allow_custom_value=False, interactive=True)

        gr.Markdown("### Step 3 — Confirm Path & Options")
        model_path_box = gr.Textbox(
            label="Model path (.gguf)  — auto-filled, or paste manually",
            placeholder="D:\\models\\model.gguf")
        mmproj_box = gr.Textbox(
            label="mmproj path (LLaVA / vision, optional)",
            placeholder="D:\\models\\llava-mmproj.gguf")

        gr.Markdown("---\n### Hardware / Load Settings")
        with gr.Row():
            gpu_layers = gr.Slider(0, 200,      step=1,   value=35,   label="n_gpu_layers")
            n_ctx      = gr.Slider(512, 131072,  step=512, value=4096, label="n_ctx")
        with gr.Row():
            n_batch  = gr.Slider(64, 2048, step=64, value=512, label="n_batch")
            chat_fmt = gr.Dropdown(label="Chat Format",
                                   choices=CHAT_FORMATS, value="chatml")
        engine_mode = gr.Radio(
            choices=["subprocess", "binding"], value="subprocess",
            label="Engine Mode",
            info="subprocess = llama-server.exe (recommended) | binding = llama-cpp-python")
        verbose_chk = gr.Checkbox(label="Verbose (show llama.cpp log)", value=False)
        with gr.Row():
            load_btn   = gr.Button("Load Model", variant="primary",   scale=2)
            unload_btn = gr.Button("Unload",     variant="secondary", scale=1)
        load_result = gr.Markdown("", elem_id="load-result")

        # ── Handlers ──────────────────────────────────────────────────────────
        _form_outputs = [model_path_box, mmproj_box,
                         gpu_layers, n_ctx, n_batch,
                         chat_fmt, engine_mode, verbose_chk]

        def _set_dir_and_scan(path):
            if not path.strip():
                return gr.update(), "Please enter a folder path."
            config.set_models_dir(path.strip())
            files = config.scan_models()
            if not files:
                return gr.update(choices=[], value=None), \
                       f"Folder set: `{path.strip()}` — no .gguf files found"
            return (gr.update(choices=files, value=files[0]),
                    f"Folder set: `{path.strip()}` — **{len(files)}** model(s) found")

        def _scan_only(path):
            if path.strip():
                config.set_models_dir(path.strip())
            files = config.scan_models()
            if not files:
                return gr.update(choices=[], value=None), \
                       "No .gguf files found in models folder"
            return (gr.update(choices=files, value=files[0]),
                    f"**{len(files)}** model(s) found")

        set_dir_btn.click(_set_dir_and_scan, [models_dir_box], [scanned_dd, scan_status])
        scan_btn.click(   _scan_only,        [models_dir_box], [scanned_dd, scan_status])
        scanned_dd.change(lambda v: v or "", [scanned_dd], [model_path_box])

        def _load_profile(name):
            p = config.get_profile(name)
            config.set_active(name)
            return (p.get("model_path", ""),    p.get("mmproj_path", ""),
                    p.get("n_gpu_layers", 35),   p.get("n_ctx", 4096),
                    p.get("n_batch", 512),        p.get("chat_format", "chatml"),
                    p.get("engine_mode", "subprocess"), p.get("verbose", False))
        profile_dd.change(_load_profile, [profile_dd], _form_outputs)

        def _save_profile(pname, new_name, mp, mmproj, ngl, nc, nb, cf, emode, verb):
            name     = new_name.strip() if new_name.strip() else pname
            existing = config.get_profile(name)
            profile  = {"model_path": mp, "mmproj_path": mmproj,
                        "n_gpu_layers": int(ngl), "n_ctx": int(nc),
                        "n_batch": int(nb), "chat_format": cf,
                        "engine_mode": emode, "verbose": verb}
            for k in ["temperature", "top_p", "top_k", "repeat_penalty", "max_tokens"]:
                profile[k] = existing.get(k, config.DEFAULT_PROFILE[k])
            config.save_profile(name, profile)
            config.set_active(name)
            return gr.update(choices=config.list_profiles(), value=name), \
                   f"Saved: **{name}**"
        save_profile_btn.click(
            _save_profile,
            [profile_dd, new_profile_name, model_path_box, mmproj_box,
             gpu_layers, n_ctx, n_batch, chat_fmt, engine_mode, verbose_chk],
            [profile_dd, load_result])

        def _del_profile(name):
            ok     = config.delete_profile(name)
            active = config.get_active_name()
            if ok:
                return gr.update(choices=config.list_profiles(), value=active), \
                       f"Deleted: {name}"
            return gr.update(), "Cannot delete the default profile"
        del_profile_btn.click(_del_profile, [profile_dd], [profile_dd, load_result])

        def _start_load():
            return gr.update(interactive=False, value="Loading..."), \
                   "Loading, please wait..."

        def _do_load(mp, mmproj, ngl, nc, nb, cf, emode, verb):
            if not mp or not mp.strip():
                return gr.update(interactive=True, value="Load Model"), \
                       "ERROR: No model path specified"
            profile = {
                "model_path":   mp.strip(),
                "mmproj_path":  mmproj.strip() if mmproj else "",
                "n_gpu_layers": int(ngl), "n_ctx": int(nc), "n_batch": int(nb),
                "chat_format":  cf, "engine_mode": emode, "verbose": verb,
                **{k: config.get_profile().get(k)
                   for k in ["temperature", "top_p", "top_k",
                              "repeat_penalty", "max_tokens"]},
            }
            ok, msg = eng.engine.load(profile)
            return (gr.update(interactive=True, value="Load Model"),
                    f"{'OK' if ok else 'ERROR'}: {msg}")

        (load_btn.click(_start_load, outputs=[load_btn, load_result])
                 .then(_do_load,
                       [model_path_box, mmproj_box, gpu_layers, n_ctx,
                        n_batch, chat_fmt, engine_mode, verbose_chk],
                       [load_btn, load_result])
                 .then(_refresh_status, outputs=[status_md]))

        (unload_btn.click(lambda: f"OK: {eng.engine.unload()}", outputs=[load_result])
                   .then(_refresh_status, outputs=[status_md]))