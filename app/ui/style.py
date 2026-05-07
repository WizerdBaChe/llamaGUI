# ui/style.py
# CSS, hotkey JS injection, and theme registry.
# Pure data — no Gradio component instantiation.
import gradio as gr

# ── Theme registry ────────────────────────────────────────────────────────────
THEMES: dict[str, type] = {
    "Soft":       gr.themes.Soft,
    "Glass":      gr.themes.Glass,
    "Base":       gr.themes.Base,
    "Default":    gr.themes.Default,
    "Monochrome": gr.themes.Monochrome,
}

THEME_NAMES = list(THEMES.keys())

def get_theme(name: str, primary_hue: str = "slate"):
    cls = THEMES.get(name, gr.themes.Soft)
    return cls(primary_hue=primary_hue)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* ── Status bar ─────────────────────────────────────────────────────────── */
#status-bar { font-size:13px; padding:6px 14px;
              background:var(--background-fill-secondary); border-radius:8px; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
#load-result { min-height:28px; }

/* ── Model metadata grid ─────────────────────────────────────────────────── */
.meta-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }
.meta-card { background:var(--background-fill-secondary);
             border-radius:8px; padding:10px 14px; }
.meta-val  { font-size:18px; font-weight:600; }
.meta-key  { font-size:11px; color:var(--body-text-color-subdued); margin-top:2px; }

/* ── Chat layout: sidebar ────────────────────────────────────────────────── */
#chat-root { gap:0 !important; align-items:stretch !important; }

#chat-sidebar {
    min-width:180px; max-width:220px;
    background:var(--background-fill-secondary);
    border-right:1px solid var(--border-color-primary);
    padding:14px 12px !important;
    display:flex; flex-direction:column; gap:8px;
}
/* remove extra gap Gradio adds around Column children */
#chat-sidebar > .gap, #chat-sidebar > div > .gap { gap:6px !important; }

.sidebar-title {
    font-size:13px; font-weight:600; letter-spacing:.04em;
    color:var(--body-text-color-subdued);
    text-transform:uppercase; padding:0 2px 4px;
}
.sidebar-divider {
    height:1px; background:var(--border-color-primary);
    margin:4px 0;
}

/* Session dropdown: compact */
#session-dropdown { font-size:13px !important; }
#session-dropdown .wrap { padding:4px 8px !important; min-height:32px !important; }

/* New / Del buttons */
#sidebar-new-del { gap:6px !important; }
#new-session-btn { font-size:13px !important; }
#del-session-btn { font-size:13px !important; }

/* Rename box */
#rename-box textarea { font-size:13px !important; padding:6px 8px !important; }
#rename-btn          { font-size:13px !important; }

/* ── Chat layout: main area ─────────────────────────────────────────────── */
#chat-main { padding:12px 16px !important; display:flex; flex-direction:column; gap:8px; }
#chat-main > .gap { gap:8px !important; }

/* ── Context bar ─────────────────────────────────────────────────────────── */
#ctx-bar-html { margin:0 !important; padding:0 !important; }
#ctx-bar-html > div { padding:0 !important; margin:0 !important; }

#ctx-bar-wrap {
    background:var(--background-fill-secondary);
    border-radius:6px; padding:5px 10px 4px;
}
#ctx-label {
    display:block; font-size:11.5px;
    color:var(--body-text-color-subdued); margin-bottom:3px;
}
#ctx-bar-track {
    background:var(--border-color-primary);
    border-radius:3px; height:5px; overflow:hidden;
}
#ctx-bar-inner {
    height:5px; border-radius:3px;
    background:var(--color-accent); transition:width .5s ease;
}

/* ── Input row: textarea + image upload ──────────────────────────────────── */
#input-row {
    align-items:stretch !important;
    gap:8px !important;
}
#input-row > .gap { align-items:stretch !important; gap:8px !important; }

/* Textarea — fill row height, no extra container padding */
#user-input { height:100% !important; }
#user-input textarea {
    resize:none; min-height:90px;
    font-size:14px; line-height:1.5;
}

/* Image column: fixed width, no overflow */
#img-col {
    flex:0 0 130px !important; min-width:130px !important; max-width:130px !important;
    padding:0 !important;
}
/* Remove extra wrapper padding Gradio adds around image component */
#image-input {
    height:108px !important;
    overflow:hidden;
}
#image-input .upload-container,
#image-input .upload-btn-container,
#image-input > div > div {
    height:108px !important; min-height:108px !important;
}
/* Label sits above the upload area, never overlaps */
#image-input label {
    display:block; font-size:12px; font-weight:500;
    color:var(--body-text-color-subdued);
    margin-bottom:3px; padding:0 !important;
}
/* The inner drop-zone */
#image-input .wrap.svelte-1aysep3,
#image-input [data-testid="image"] {
    height:86px !important; min-height:86px !important;
}

/* ── Action buttons row ──────────────────────────────────────────────────── */
#action-row { gap:6px !important; margin-top:2px; }
#action-row button { height:38px !important; font-size:14px !important; }
#send-btn { font-weight:600 !important; }

/* ── Download tab: search row alignment ──────────────────────────────────── */
/* Invisible spacer label aligns button bottom edge with the textbox bottom */
#search-btn-col { padding:0 !important; display:flex; flex-direction:column; }
.search-btn-label {
    display:block; font-size:14px; line-height:1.2;
    color:transparent; pointer-events:none;
    margin-bottom:4px;        /* matches Gradio label margin */
}
#search-btn { flex:1 0 auto; }

/* ── Download progress bar ───────────────────────────────────────────────── */
#dl-bar { height:8px; border-radius:4px; background:var(--color-accent); }
"""

# ── Hotkey JS (injected via launch head=) ─────────────────────────────────────
HOTKEY_JS = """<script>
(function () {
  var actions = {
    "ctrlEnter":  function () { var b = document.getElementById("send-btn");          if (b) b.click(); },
    "ctrlk":      function () { var b = document.querySelector("[data-testid=clear-btn]");  if (b) b.click(); },
    "ctrlshiftS": function () { var b = document.querySelector("[data-testid=stop-btn]");   if (b) b.click(); },
    "ctrlr":      function () { var b = document.querySelector("[data-testid=retry-btn]");  if (b) b.click(); }
  };
  document.addEventListener("keydown", function (e) {
    var key = (e.ctrlKey ? "ctrl" : "")
            + (e.shiftKey ? "shift" : "")
            + e.key;
    var fn = actions[key];
    if (fn) { e.preventDefault(); fn(); }
  });
})();
</script>"""

# ── Shared constants ──────────────────────────────────────────────────────────
CHAT_FORMATS = [
    "chatml", "llama-2", "llama-3", "gemma", "phi3",
    "mistral-instruct", "qwen", "deepseek", "alpaca",
]
