# ui/tabs/chat.py
# Tab 1 -- Chat
from __future__ import annotations
import gradio as gr
import engine as eng
import config
from ui.state import (
    GR6, GR45,
    new_session, session_choices, get_session, delete_session,
    rename_session, to_gr, ensure_dicts, INITIAL_SID,
)
from ui.style import CHAT_FORMATS

def build(shared: dict) -> None:
    """Build the Chat tab inside an active gr.Blocks context.

    shared keys consumed:
        status_md   : gr.Markdown  (global status bar)
        temp_state  : gr.State
        top_p_state : gr.State
        top_k_state : gr.State
        rp_state    : gr.State
        mt_state    : gr.State
    """
    status_md   = shared["status_md"]
    temp_state  = shared["temp_state"]
    top_p_state = shared["top_p_state"]
    top_k_state = shared["top_k_state"]
    rp_state    = shared["rp_state"]
    mt_state    = shared["mt_state"]

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
            tps_str = f" | **{tps:.1f}** t/s" if tps > 0 else ""
            vram = eng.get_vram_info()
            vram_str = (f" | VRAM {vram['used_mb']/1024:.1f}/{vram['total_mb']/1024:.1f} GB"
                        if vram["available"] else "")
            return f"**{name}** [{mode}]{tps_str}{vram_str}"
        vram = eng.get_vram_info()
        if vram["available"]:
            return f"no model loaded | VRAM {vram['used_mb']/1024:.1f}/{vram['total_mb']/1024:.1f} GB"
        return "no model loaded"

    def _refresh_ctx():
        used, max_ctx = eng.engine.get_context_usage()
        if not max_ctx:
            return ('<div id="ctx-bar-wrap">' +
                    '<span id="ctx-label">Context: --</span>' +
                    '<div id="ctx-bar-track"><div id="ctx-bar-inner" style="width:0"></div></div>' +
                    '</div>'), ""
        pct   = min(used / max_ctx * 100, 100)
        color = "#ef4444" if pct > 85 else "#f59e0b" if pct > 60 else "var(--color-accent)"
        bar   = (f'<div id="ctx-bar-wrap">' +
                 f'<span id="ctx-label">Context: {used:,} / {max_ctx:,} tokens ({pct:.1f}%)</span>' +
                 f'<div id="ctx-bar-track"><div id="ctx-bar-inner" style="width:{pct:.1f}%;background:{color}"></div></div>' +
                 '</div>')
        return bar, ""

    with gr.Tab("Chat"):
        active_sid = gr.State(INITIAL_SID)

        with gr.Row(elem_id="chat-root", equal_height=False):

            # ── Left sidebar ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=180, elem_id="chat-sidebar"):
                gr.HTML('<div class="sidebar-title">Sessions</div>')
                session_dd = gr.Dropdown(
                    label="", choices=session_choices(), value=INITIAL_SID,
                    allow_custom_value=False, show_label=False, container=False,
                    elem_id="session-dropdown")
                with gr.Row(elem_id="sidebar-new-del"):
                    new_btn = gr.Button("+ New", size="sm", scale=3, elem_id="new-session-btn")
                    del_btn = gr.Button("Del",   size="sm", scale=1, variant="stop",
                                        elem_id="del-session-btn")
                gr.HTML('<div class="sidebar-divider"></div>')
                name_box   = gr.Textbox(placeholder="Rename session...",
                                        show_label=False, container=False,
                                        elem_id="rename-box")
                rename_btn = gr.Button("Rename", size="sm", elem_id="rename-btn")

            # ── Right: chat area ──────────────────────────────────────────
            with gr.Column(scale=5, elem_id="chat-main"):

                chatbot_kw: dict = dict(
                    label="Conversation", height=440,
                    placeholder="Load a model first, then start chatting.",
                    elem_id="chatbot")
                if GR45:
                    chatbot_kw["type"] = "messages"
                chatbot = gr.Chatbot(**chatbot_kw)

                # Context bar — label + progress bar in one HTML block
                ctx_bar = gr.HTML(
                    value='<div id="ctx-bar-wrap">' +
                          '<span id="ctx-label">Context: --</span>' +
                          '<div id="ctx-bar-track"><div id="ctx-bar-inner" style="width:0"></div></div>' +
                          '</div>')
                ctx_label = gr.HTML(value="", visible=False)   # dummy output binding
                ctx_timer = _make_timer(4)
                if ctx_timer:
                    ctx_timer.tick(fn=_refresh_ctx, outputs=[ctx_bar, ctx_label])

                # Input area: textarea + image upload side-by-side
                with gr.Row(elem_id="input-row", equal_height=True):
                    user_input = gr.Textbox(
                        placeholder="Type a message... (Enter to send, Shift+Enter for newline)",
                        lines=3, max_lines=8,
                        show_label=False, container=False,
                        elem_id="user-input", scale=5)
                    with gr.Column(scale=1, min_width=130, elem_id="img-col"):
                        img_kw: dict = dict(
                            label="Image", type="filepath",
                            show_label=True, height=108,
                            elem_id="image-input")
                        if GR6:
                            img_kw["sources"] = ["upload", "clipboard"]
                        else:
                            img_kw["source"] = "upload"
                        image_input = gr.Image(**img_kw)

                # Action buttons — all in one row, proportional scales
                with gr.Row(elem_id="action-row"):
                    send_btn  = gr.Button("Send",    variant="primary",
                                          scale=3, elem_id="send-btn")
                    stop_btn  = gr.Button("Stop",    variant="secondary",
                                          interactive=False, scale=1, elem_id="stop-btn")
                    regen_btn = gr.Button("Retry",   variant="secondary",
                                          scale=1, elem_id="retry-btn")
                    clear_btn = gr.Button("Clear",   variant="secondary",
                                          scale=1, elem_id="clear-btn")

                with gr.Accordion("System Prompt", open=False, elem_id="sys-accordion"):
                    system_prompt = gr.Textbox(
                        value="You are a helpful assistant.",
                        lines=3, label="System Prompt")

                with gr.Accordion("Prompt Preview (raw format sent to model)", open=False):
                    prompt_preview = gr.Code(
                        label="", language=None, interactive=False,
                        value="(Send a message to update)")

        # ── Session handlers ──────────────────────────────────────────────────
        def _switch(sid):
            s = get_session(sid)
            return (to_gr(s["history"]) if s else []), sid

        def _new():
            sid = new_session()
            return gr.update(choices=session_choices(), value=sid), sid, []

        def _del(sid):
            if len(session_choices()) <= 1:
                s = get_session(sid)
                return gr.update(), sid, to_gr(s["history"] if s else [])
            next_sid = delete_session(sid)
            s = get_session(next_sid)
            return (gr.update(choices=session_choices(), value=next_sid),
                    next_sid, to_gr(s["history"] if s else []))

        def _rename(sid, new_name):
            rename_session(sid, new_name)
            return gr.update(choices=session_choices(), value=sid), ""

        session_dd.change(_switch, [session_dd], [chatbot, active_sid])
        new_btn.click(_new, outputs=[session_dd, active_sid, chatbot])
        del_btn.click(_del, [active_sid], [session_dd, active_sid, chatbot])
        rename_btn.click(_rename, [active_sid, name_box], [session_dd, name_box])

        # ── Chat handlers ─────────────────────────────────────────────────────
        def _user_submit(message, history, image_path, sid):
            internal = ensure_dicts(history)
            if not message.strip() and not image_path:
                return to_gr(internal), "", None
            if image_path:
                b64     = eng.image_file_to_base64(image_path)
                content = [{"type": "text",      "text": message or "(image)"},
                           {"type": "image_url", "image_url": {"url": b64}}]
                internal.append({"role": "user", "content": content})
            else:
                internal.append({"role": "user", "content": message})
            s = get_session(sid)
            if s is not None:
                s["history"] = internal
            return to_gr(internal), "", None

        def _bot_stream(history, system, temperature, top_p, top_k,
                        repeat_penalty, max_tokens, sid):
            internal = ensure_dicts(history)
            _no_send  = gr.update(interactive=True)
            _no_stop  = gr.update(interactive=False)
            _can_stop = gr.update(interactive=True)
            _no_send2 = gr.update(interactive=False)

            if not internal or internal[-1]["role"] != "user":
                yield to_gr(internal), _no_send, _no_stop, ""
                return

            if not eng.engine.is_loaded:
                internal.append({"role": "assistant",
                                  "content": "No model loaded. Go to the Model tab first."})
                s = get_session(sid)
                if s is not None:
                    s["history"] = internal
                yield to_gr(internal), _no_send, _no_stop, ""
                return

            messages: list[dict] = []
            if system.strip():
                messages.append({"role": "system", "content": system.strip()})
            messages.extend(internal)

            profile_override = {
                **eng.engine.current_profile,
                "temperature": temperature, "top_p": top_p,
                "top_k": top_k, "repeat_penalty": repeat_penalty,
                "max_tokens": max_tokens,
            }

            # Prompt preview (text-only version for display)
            text_msgs = [
                {"role": m["role"],
                 "content": m["content"] if isinstance(m["content"], str)
                            else next((p["text"] for p in m["content"]
                                       if p.get("type") == "text"), "")}
                for m in messages
            ]
            preview = eng.engine.format_prompt_preview(text_msgs)

            internal.append({"role": "assistant", "content": ""})
            yield to_gr(internal), _no_send2, _can_stop, preview

            for token in eng.engine.stream(messages, profile_override):
                internal[-1]["content"] += token
                yield to_gr(internal), _no_send2, _can_stop, preview

            s = get_session(sid)
            if s is not None:
                s["history"] = internal
            yield to_gr(internal), _no_send, _no_stop, preview

        def _regenerate(history, system, temperature, top_p, top_k,
                        repeat_penalty, max_tokens, sid):
            internal = ensure_dicts(history)
            while internal and internal[-1]["role"] == "assistant":
                internal.pop()
            if not internal:
                yield to_gr(internal), gr.update(interactive=True), gr.update(interactive=False), ""
                return
            yield from _bot_stream(
                to_gr(internal), system, temperature,
                top_p, top_k, repeat_penalty, max_tokens, sid)

        def _clear(sid):
            s = get_session(sid)
            if s is not None:
                s["history"] = []
            return []

        _stream_in  = [chatbot, system_prompt,
                       temp_state, top_p_state, top_k_state, rp_state, mt_state,
                       active_sid]
        _stream_out = [chatbot, send_btn, stop_btn, prompt_preview]

        send_ev = (
            send_btn.click(
                _user_submit,
                inputs=[user_input, chatbot, image_input, active_sid],
                outputs=[chatbot, user_input, image_input])
            .then(_bot_stream,   inputs=_stream_in, outputs=_stream_out)
            .then(_refresh_status, outputs=[status_md])
        )
        enter_ev = (
            user_input.submit(
                _user_submit,
                inputs=[user_input, chatbot, image_input, active_sid],
                outputs=[chatbot, user_input, image_input])
            .then(_bot_stream,   inputs=_stream_in, outputs=_stream_out)
            .then(_refresh_status, outputs=[status_md])
        )
        stop_btn.click(fn=None, cancels=[send_ev, enter_ev])
        regen_btn.click(_regenerate, inputs=_stream_in, outputs=_stream_out
                        ).then(_refresh_status, outputs=[status_md])
        clear_btn.click(_clear, [active_sid], [chatbot])
