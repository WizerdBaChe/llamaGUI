# main.py
from __future__ import annotations
import os, sys, time, threading, logging, signal
from pathlib import Path

THIS   = Path(__file__).resolve()
APPDIR = THIS.parent
if str(APPDIR) not in sys.path:
    sys.path.insert(0, str(APPDIR))

def _setup_logging():
    import config
    logdir = Path(config.ROOT) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(logdir / "launch.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers)

_setup_logging()
log = logging.getLogger("LlamaGUI")

GRADIO_PORT = int(os.environ.get("LLAMAGUI_GRADIO_PORT", 7860))
API_PORT    = int(os.environ.get("LLAMAGUI_API_PORT",    8000))
GRADIO_URL  = f"http://127.0.0.1:{GRADIO_PORT}"
API_URL     = f"http://127.0.0.1:{API_PORT}"

# ── Shutdown ──────────────────────────────────────────────────────────────────
def _shutdown(reason: str = ""):
    log.info(f"Shutting down — {reason}")
    try:
        import engine as eng
        if eng.engine.is_loaded:
            log.info("Unloading model to free VRAM...")
            eng.engine.unload()
        eng.engine.stop_idle_monitor()
    except Exception as e:
        log.warning(f"Unload skipped: {e}")
    log.info("Process exiting.")
    os.kill(os.getpid(), signal.SIGTERM)

# ── Tray icon ─────────────────────────────────────────────────────────────────
def _build_tray_icon(webview_win=None):
    """
    Build and run pystray icon (call in daemon thread via run_detached).
    webview_win: pywebview window ref, or None if not using pywebview.
    """
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        log.warning("pystray / Pillow 未安裝，System Tray 功能停用")
        return None

    # 產生一個簡單的預設圖示（可替換成 icon.png）
    icon_path = APPDIR / "icon.png"
    if icon_path.exists():
        img = Image.open(icon_path).resize((64, 64))
    else:
        img = Image.new("RGBA", (64, 64), (30, 30, 30, 255))
        d = ImageDraw.Draw(img)
        d.ellipse([8, 8, 56, 56], fill=(100, 180, 255, 255))

    # ── 狀態顯示（動態 label）────────────────────────────────────────────────
    def _status_label(item):
        try:
            import engine as eng
            if eng.engine.is_loaded:
                s = eng.engine.get_stats()
                return f"已載入：{s.get('model_name','?')}"
        except Exception:
            pass
        return "未載入模型"

    # ── 視窗顯示 / 隱藏 ──────────────────────────────────────────────────────
    _win_visible = [True]   # mutable container for closure

    def _show_window(icon_obj, item):
        if webview_win is not None:
            try:
                webview_win.show()
                _win_visible[0] = True
            except Exception:
                pass
        else:
            import webbrowser
            webbrowser.open(GRADIO_URL)

    def _hide_window(icon_obj, item):
        if webview_win is not None:
            try:
                webview_win.hide()
                _win_visible[0] = False
            except Exception:
                pass

    def _toggle_window(icon_obj, item):
        if _win_visible[0]:
            _hide_window(icon_obj, item)
        else:
            _show_window(icon_obj, item)

    def _open_browser(icon_obj, item):
        import webbrowser
        webbrowser.open(GRADIO_URL)

    def _unload_model(icon_obj, item):
        try:
            import engine as eng
            if eng.engine.is_loaded:
                eng.engine.unload()
                log.info("Tray: 模型已卸載")
                icon_obj.notify("LlamaGUI", "模型已卸載，VRAM 已釋放")
        except Exception as e:
            log.warning(f"Tray unload: {e}")

    def _quit(icon_obj, item):
        icon_obj.stop()
        _shutdown("Tray → Quit")

    menu = pystray.Menu(
        pystray.MenuItem(
            _status_label,
            None,
            enabled=False),          # 狀態列（灰色不可點）
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "顯示 / 隱藏視窗",
            _toggle_window,
            default=True),           # 雙擊觸發此項
        pystray.MenuItem(
            "在瀏覽器開啟",
            _open_browser),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "卸載模型（釋放 VRAM）",
            _unload_model),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出 LlamaGUI", _quit),
    )

    icon = pystray.Icon(
        name="LlamaGUI",
        icon=img,
        title="LlamaGUI",
        menu=menu)
    return icon

# ── Server threads ────────────────────────────────────────────────────────────
def _start_api():
    try:
        import uvicorn
        from api import app
        log.info(f"FastAPI 啟動中 → {API_URL}")
        uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")
    except Exception as e:
        log.error(f"FastAPI 啟動失敗: {e}")

def _start_gradio():
    try:
        import ui
        log.info(f"Gradio UI 啟動中 → {GRADIO_URL}")
        ui.launch_server(server_port=GRADIO_PORT, share=False)
    except Exception as e:
        log.error(f"Gradio 啟動失敗: {e}", exc_info=True)

def _wait_for_service(url: str, timeout: int = 45) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import config
    log.info("LlamaGUI 啟動")
    log.info(f"  ROOT    : {config.ROOT}")
    log.info(f"  BIN     : {config.BINDIR}")
    log.info(f"  VARIANT : {config.get_variant()}")
    log.info(f"  MODELS  : {config.get_models_dir()}")

    threading.Thread(target=_start_api,    daemon=True).start()
    threading.Thread(target=_start_gradio, daemon=True).start()

    log.info("等待 Gradio 就緒…")
    if not _wait_for_service(GRADIO_URL, timeout=45):
        log.error("Gradio 45 秒內未就緒，請查看 logs/launch.log")
        sys.exit(1)
    log.info("Gradio 已就緒")

    # ── pywebview + pystray ───────────────────────────────────────────────────
    try:
        import webview
        log.info("使用 pywebview 開啟視窗")

        win = webview.create_window(
            title="LlamaGUI",
            url=GRADIO_URL,
            width=1280, height=820,
            min_size=(900, 600),
            resizable=True,
        )

        # 視窗關閉 → 縮到 tray（而非直接 shutdown）
        _tray_icon: list = [None]

        def _on_closing():
            """使用者按 X：隱藏視窗到 tray，不結束程式"""
            try:
                win.hide()
                if _tray_icon[0]:
                    _tray_icon[0].notify("LlamaGUI", "已縮小至系統工作列")
            except Exception:
                pass
            return False    # 回傳 False 阻止 pywebview 真正關閉

        win.events.closing += _on_closing   # pywebview 4.x — closing 可攔截

        def _start_tray():
            icon = _build_tray_icon(webview_win=win)
            if icon:
                _tray_icon[0] = icon
                icon.run_detached()    # ← 非阻塞，跑在自己的 daemon thread
                log.info("System Tray 圖示已啟動")

        threading.Thread(target=_start_tray, daemon=True).start()
        time.sleep(0.5)    # 給 tray 一點時間初始化再 start webview

        webview.start(debug=False)
        # webview.start() 阻塞直到所有視窗被 destroy（不是 hide）
        # 正常情況下按 X 只 hide，所以這行之後的程式碼
        # 只有在 tray → 退出 LlamaGUI 時才會到達

    except ImportError:
        # ── 無 pywebview：純 tray 模式 ───────────────────────────────────────
        log.warning("pywebview 未安裝，使用 Tray-only 模式")
        icon = _build_tray_icon(webview_win=None)
        if icon:
            log.info("System Tray 圖示已啟動（右鍵 → 在瀏覽器開啟）")
            icon.run()      # 阻塞主執行緒，直到選 Quit
        else:
            log.info(f"請手動開啟 {GRADIO_URL}")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                _shutdown("Ctrl+C")

if __name__ == "__main__":
    main()