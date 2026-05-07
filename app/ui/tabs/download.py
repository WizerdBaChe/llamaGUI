# ui/tabs/download.py
# Tab 3 -- HuggingFace search & download
from __future__ import annotations
import os, threading
import gradio as gr
import engine as eng
import config
from ui.state import register_dl_task, get_dl_task

def build(shared: dict) -> None:
    def _make_timer(interval):
        try:
            return gr.Timer(value=interval, active=True)
        except Exception:
            return None

    with gr.Tab("Download"):
        gr.Markdown("### HuggingFace Model Search & Download")

        with gr.Row(elem_id="search-row", equal_height=False):
            search_box = gr.Textbox(
                label="Search query", placeholder="llama gemma qwen mistral...",
                scale=4, elem_id="search-box")
            with gr.Column(scale=1, min_width=90, elem_id="search-btn-col"):
                gr.HTML('<label class="search-btn-label">&#8203;</label>')
                search_btn = gr.Button("Search", variant="primary", elem_id="search-btn")

        results_dd  = gr.Dropdown(label="Search results", choices=[], interactive=True)
        results_md  = gr.Markdown("")
        files_dd    = gr.Dropdown(label="Select GGUF file", choices=[], interactive=True)
        file_info   = gr.Markdown("")

        with gr.Row():
            dl_btn     = gr.Button("Download", variant="primary", scale=2)
            cancel_btn = gr.Button("Cancel",   variant="stop",    scale=1, interactive=False)
        dl_label = gr.Markdown("")
        dl_bar   = gr.HTML("")

        repo_store = gr.State([])
        file_store = gr.State([])
        tid_state  = gr.State("")

        # ── Search ────────────────────────────────────────────────────────────
        def _search(q):
            if not q.strip():
                return gr.update(), [], "", gr.update(choices=[]), []
            results  = eng.hf_search(q.strip(), 20)
            choices  = [f"{r['id']} dl:{r['downloads']:,} likes:{r['likes']}" for r in results]
            repo_ids = [r["id"] for r in results]
            return (gr.update(choices=choices, value=choices[0] if choices else None),
                    repo_ids, f"{len(results)} results",
                    gr.update(choices=[], value=None), [])

        search_btn.click(_search, [search_box],
                         [results_dd, repo_store, results_md, files_dd, file_store])
        search_box.submit(_search, [search_box],
                          [results_dd, repo_store, results_md, files_dd, file_store])

        # ── Repo -> file list ─────────────────────────────────────────────────
        def _on_repo(choice, repo_ids):
            if not choice or not repo_ids:
                return gr.update(choices=[]), [], ""
            repo_id = next((r for r in repo_ids if choice.startswith(r)), None)
            if not repo_id:
                return gr.update(choices=[]), [], "Cannot identify repo"
            files   = eng.hf_list_gguf_files(repo_id)
            choices = [f"{f['filename']} ({f['size_gb']:.2f} GB)" for f in files]
            return (gr.update(choices=choices, value=choices[0] if choices else None),
                    files, f"**{repo_id}** -- {len(files)} GGUF file(s)")

        results_dd.change(_on_repo, [results_dd, repo_store],
                          [files_dd, file_store, results_md])

        def _on_file(choice, files):
            if not choice or not files:
                return ""
            for f in files:
                if choice.startswith(f["filename"]):
                    return f"Size: {f['size_gb']:.2f} GB"
            return ""

        files_dd.change(_on_file, [files_dd, file_store], [file_info])

        # ── Start download ────────────────────────────────────────────────────
        def _start_dl(choice, files):
            if not choice or not files:
                return "", "Please select a file first", gr.update(), gr.update()
            target = next((f for f in files if choice.startswith(f["filename"])), None)
            if not target:
                return "", "File info not found", gr.update(), gr.update()
            dest      = os.path.join(config.get_models_dir(), target["filename"])
            os.makedirs(config.get_models_dir(), exist_ok=True)
            cancel_ev = threading.Event()
            task      = {"progress": 0, "total": 0, "status": "running",
                         "cancel_event": cancel_ev, "message": ""}
            tid = register_dl_task(task)

            def _run():
                ok, msg = eng.hf_download(
                    target["url"], dest,
                    lambda dl, tot: task.update({"progress": dl, "total": tot}),
                    cancel_ev)
                task["status"]  = "done" if ok else "error"
                task["message"] = msg

            threading.Thread(target=_run, daemon=True).start()
            return (str(tid),
                    f"Downloading: **{target['filename']}** ({target['size_gb']:.2f} GB)...",
                    gr.update(interactive=False), gr.update(interactive=True))

        dl_btn.click(_start_dl, [files_dd, file_store],
                     [tid_state, dl_label, dl_btn, cancel_btn])

        # ── Poll progress ─────────────────────────────────────────────────────
        poll_timer = _make_timer(1)
        if poll_timer:
            poll_timer.active = False

            def _poll(tid_str):
                t = get_dl_task(tid_str)
                if not t:
                    return "", "", gr.update(interactive=True), gr.update(interactive=False)
                prog, total, status = t["progress"], t["total"], t["status"]
                if status in ("done", "error", "cancelled"):
                    icon = "OK" if status == "done" else ("CANCELLED" if status == "cancelled" else "ERROR")
                    return ("", f"{icon}: {t.get('message','')}",
                            gr.update(interactive=True), gr.update(interactive=False))
                pct = prog / total * 100 if total else 0
                bar = (f'<div style="background:var(--background-fill-secondary);'
                       f'border-radius:6px;padding:4px 8px">'
                       f'<div id="dl-bar" style="width:{pct:.1f}%"></div></div>')
                return (tid_str,
                        f"Downloading {prog/1024**2:.1f} / {total/1024**2:.1f} MB ({pct:.1f}%)",
                        gr.update(interactive=False), gr.update(interactive=True))

            poll_timer.tick(_poll, [tid_state],
                            [tid_state, dl_label, dl_btn, cancel_btn])

        # ── Cancel ────────────────────────────────────────────────────────────
        def _cancel(tid_str):
            t = get_dl_task(tid_str)
            if t:
                t["cancel_event"].set()
                t["status"] = "cancelled"
            return "", "Download cancelled", gr.update(interactive=True), gr.update(interactive=False)

        cancel_btn.click(_cancel, [tid_state],
                         [tid_state, dl_label, dl_btn, cancel_btn])
