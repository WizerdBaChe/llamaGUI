# engine.py v2.3 ── LlamaGUI backend engine
from __future__ import annotations
import os, re, json, time, base64, struct, logging, subprocess, threading, urllib.request, urllib.parse
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("llamagui.engine")

try:
    from llama_cpp import Llama as _Llama
    BINDING_AVAILABLE = True
except ImportError:
    _Llama = None
    BINDING_AVAILABLE = False

import config

# ── GGUF metadata reader ─────────────────────────────────────────────────────
GGUF_MAGIC = b"GGUF"

def read_gguf_metadata(path: str) -> dict:
    result: dict[str, Any] = {}
    try:
        fsize = os.path.getsize(path)
        result["file_size_gb"] = round(fsize / 1024**3, 2)
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return result
            version = struct.unpack("<I", f.read(4))[0]
            result["gguf_version"] = version
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count     = struct.unpack("<Q", f.read(8))[0]
            result["tensor_count"] = tensor_count

            def read_str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            def read_val(vtype):
                if vtype == 8:    return read_str()
                elif vtype == 6:  return struct.unpack("<i", f.read(4))[0]
                elif vtype == 7:  return struct.unpack("<Q", f.read(8))[0]
                elif vtype == 4:  return struct.unpack("<I", f.read(4))[0]
                elif vtype == 5:  return struct.unpack("<q", f.read(8))[0]
                elif vtype == 10: return struct.unpack("<d", f.read(8))[0]
                elif vtype == 11: return struct.unpack("<?", f.read(1))[0]
                elif vtype == 12:
                    atype = struct.unpack("<I", f.read(4))[0]
                    alen  = struct.unpack("<Q", f.read(8))[0]
                    return [read_val(atype) for _ in range(min(alen, 64))]
                elif vtype == 1:  return struct.unpack("<B", f.read(1))[0]
                elif vtype == 2:  return struct.unpack("<b", f.read(1))[0]
                elif vtype == 3:  return struct.unpack("<H", f.read(2))[0]
                elif vtype == 9:  return struct.unpack("<h", f.read(2))[0]
                elif vtype == 0:  return struct.unpack("<f", f.read(4))[0]
                else: raise ValueError(f"unknown vtype {vtype}")

            KEEP = {
                "general.name", "general.architecture", "general.quantization_version",
                "tokenizer.chat_template",
                "llama.context_length", "llama.embedding_length",
                "llama.block_count", "llama.attention.head_count",
            }
            for _ in range(kv_count):
                key   = read_str()
                vtype = struct.unpack("<I", f.read(4))[0]
                val   = read_val(vtype)
                short = key.split(".")[-1]
                if key in KEEP or short in ("name", "architecture", "chat_template",
                                            "context_length", "embedding_length",
                                            "block_count", "head_count"):
                    result[short] = val
    except Exception as e:
        result["parse_error"] = str(e)
    return result

# ── Chat format auto-detection ───────────────────────────────────────────────
def detect_chat_format(model_path: str) -> str:
    meta     = read_gguf_metadata(model_path)
    template = meta.get("chat_template", "").lower()
    name     = os.path.basename(model_path).lower()

    if "<start_of_turn>" in template:        fmt = "gemma"
    elif "<|im_start|>" in template:         fmt = "chatml"
    elif "<|start_header_id|>" in template:  fmt = "llama-3"
    elif "[inst]" in template:               fmt = "llama-2"
    elif "<|user|>" in template:             fmt = "phi3"
    elif "[inst]" in template.replace(" ", "").lower(): fmt = "mistral-instruct"
    elif "\u2581<0x0A>" in template or "deepseek" in template: fmt = "deepseek"
    elif "gemma"    in name: fmt = "gemma"
    elif "llama-3"  in name or "llama3"   in name: fmt = "llama-3"
    elif "llama-2"  in name or "llama2"   in name: fmt = "llama-2"
    elif "mistral"  in name: fmt = "mistral-instruct"
    elif "qwen"     in name: fmt = "qwen"
    elif "deepseek" in name: fmt = "deepseek"
    elif "phi"      in name: fmt = "phi3"
    elif "alpaca"   in name: fmt = "alpaca"
    else:                    fmt = "chatml"

    log.info(f"detect_chat_format: '{os.path.basename(model_path)}' → {fmt}")
    return fmt

# ── VRAM info ────────────────────────────────────────────────────────────────
def get_vram_info() -> dict:
    result = dict(used_mb=0, total_mb=0, free_mb=0, utilization_pct=0, available=False)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        ).decode().strip().split("\n")[0]
        parts = [p.strip() for p in out.split(",")]
        if len(parts) == 4:
            result.update(used_mb=int(parts[0]), total_mb=int(parts[1]),
                          free_mb=int(parts[2]), utilization_pct=int(parts[3]),
                          available=True)
    except Exception:
        pass
    return result

# ── Chat templates ───────────────────────────────────────────────────────────
CHAT_TEMPLATES: dict[str, str] = {
    "chatml":           "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n",
    "llama-2":          "{% if messages[0].role=='system' %}<<SYS>>\n{{ messages[0].content }}\n<</SYS>>\n{% endif %}{% for m in messages %}{% if m.role=='user' %}[INST] {{ m.content }} [/INST]{% endif %}{% endfor %}",
    "llama-3":          "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n{{ m.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n",
    "gemma":            "{% for m in messages %}{{ m.role }}\n{{ m.content }}\n{% endfor %}model\n",
    "phi3":             "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}<|end|>\n{% endfor %}<|assistant|>\n",
    "mistral-instruct": "{% for m in messages %}{% if m.role=='user' %}[INST] {{ m.content }} [/INST]{% endif %}{% endfor %}",
    "qwen":             "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n",
    "deepseek":         "{% for m in messages %}{% if m.role=='user' %}User: {{ m.content }}\n{% else %}Assistant: {{ m.content }}\n{% endif %}{% endfor %}",
    "alpaca":           "### Instruction:\n{{ user }}\n### Response:\n",
}

def format_prompt_preview(messages: list[dict], chat_format: str) -> str:
    template = CHAT_TEMPLATES.get(chat_format, CHAT_TEMPLATES["chatml"])
    result = []
    if "{%- for" in template or "{% for" in template:
        inner  = template.split("{% for m in messages %}")[1].split("{% endfor %}")[0]
        suffix = template.split("{% endfor %}")[1] if "{% endfor %}" in template else ""
        for m in messages:
            seg = inner.replace("{{ m.role }}", m.get("role", ""))
            seg = seg.replace("{{ m.content }}", m.get("content", "") if isinstance(m.get("content"), str) else "")
            result.append(seg)
        result.append(suffix)
    else:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
        result.append(template.replace("{{ system }}", system).replace("{{ user }}", user))
    return "".join(result)

# ── Model name fuzzy matching ─────────────────────────────────────────────────
def _normalize(s: str) -> str:
    return re.sub(r"[-_.\s]+", "", s.lower())

def _find_model_by_name(model_name: str) -> str | None:
    models_dir = config.get_models_dir()
    try:
        # 遞迴掃描（與 scan_models() 一致，支援子資料夾）
        candidates = []
        for root, _, files in os.walk(models_dir):
            for f in files:
                if f.lower().endswith(".gguf"):
                    candidates.append(os.path.join(root, f))
    except OSError:
        return None

    norm_input = _normalize(model_name)
    results: list[tuple[int, str]] = []
    for path in candidates:
        stem    = os.path.splitext(os.path.basename(path))[0]
        norm_fn = _normalize(stem)
        if stem == model_name:               results.append((0, path))
        elif norm_fn == norm_input:          results.append((1, path))
        elif norm_input in norm_fn:          results.append((2, path))
        elif norm_fn.startswith(norm_input): results.append((3, path))
    if not results:
        return None
    results.sort(key=lambda x: x[0])
    return results[0][1]

# ── Engine stats ──────────────────────────────────────────────────────────────
class EngineStats:
    def __init__(self): self.reset()
    def reset(self):
        self.prompt_tokens:     int   = 0
        self.completion_tokens: int   = 0
        self.elapsed_sec:       float = 0.0
        self.tokens_per_sec:    float = 0.0
        self.model_name:        str   = ""
        self.engine_mode:       str   = ""
        self.context_used:      int   = 0
        self.context_max:       int   = 0
        self.model_metadata:    dict  = {}
        self.load_time_sec:     float = 0.0   # 項目5：記錄載入耗時
    def to_dict(self) -> dict:
        return dict(
            prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
            elapsed_sec=round(self.elapsed_sec, 2), tokens_per_sec=round(self.tokens_per_sec, 1),
            model_name=self.model_name, engine_mode=self.engine_mode,
            context_used=self.context_used, context_max=self.context_max,
            model_metadata=self.model_metadata, load_time_sec=round(self.load_time_sec, 2),
        )

# ── Subprocess engine ─────────────────────────────────────────────────────────
class SubprocessEngine:
    SERVER_HOST     = "127.0.0.1"
    SERVER_PORT     = 8765
    STARTUP_TIMEOUT = 60

    def __init__(self):
        self.proc:             subprocess.Popen | None = None
        self.lock              = threading.Lock()
        self.stats             = EngineStats()
        self.current_profile:  dict = {}
        self._load_at:         float = 0.0   # 項目5：載入時間戳

    @staticmethod
    def find_server_exe() -> str | None:
        bindir = config.get_bin_dir()
        candidates = [os.path.join(bindir, "llama-server.exe"), os.path.join(bindir, "llama-server")]
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ["bin/cuda/llama-server.exe","bin/llama-server.exe","bin/llama-server",
                    "llama-server.exe","llama-server"]:
            candidates.append(os.path.join(base, rel))
            candidates.append(os.path.join(base, "..", rel))
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c): return c
        return None

    def _server_url(self, path: str) -> str:
        return f"http://{self.SERVER_HOST}:{self.SERVER_PORT}{path}"

    def _wait_ready(self) -> bool:
        deadline = time.time() + self.STARTUP_TIMEOUT
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self._server_url("/health"), timeout=1)
                return True
            except Exception:
                time.sleep(0.4)
        return False

    def load(self, profile: dict) -> tuple[bool, str]:
        exe = self.find_server_exe()
        if not exe: return False, "llama-server.exe not found — see bin/cuda/"
        model_path = profile.get("model_path", "")
        if not model_path or not os.path.isfile(model_path):
            return False, f"model file not found: {model_path}"
        t_start = time.perf_counter()
        with self.lock:
            self._stop_server()
            meta = read_gguf_metadata(model_path)
            self.stats.model_metadata = meta
            cmd = [exe, "--model", model_path, "--host", self.SERVER_HOST,
                   "--port",        str(self.SERVER_PORT),
                   "--n-gpu-layers",str(profile.get("n_gpu_layers", 35)),
                   "--ctx-size",    str(profile.get("n_ctx", 4096)),
                   "--batch-size",  str(profile.get("n_batch", 512)),
                   "--chat-template",str(profile.get("chat_format", "chatml")),
                   "--no-mmap",
                   # --embedding は b9010 以降デフォルト有効のため不要
            ]
            mmproj = profile.get("mmproj_path", "")
            if mmproj and os.path.isfile(mmproj): cmd += ["--mmproj", mmproj]
            if not profile.get("verbose", False): cmd += ["--log-disable"]
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except Exception as e:
                return False, f"Failed to start llama-server.exe: {e}"
            if not self._wait_ready():
                self._stop_server()
                return False, "llama-server did not become ready within 60s"
            self.current_profile       = profile.copy()
            self.stats.model_name      = os.path.basename(model_path)
            self.stats.engine_mode     = "subprocess"
            self.stats.context_max     = profile.get("n_ctx", 4096)
            self.stats.load_time_sec   = time.perf_counter() - t_start
            self._load_at              = time.time()
            return True, f"subprocess: {self.stats.model_name}"

    def unload(self) -> str:
        name = self.stats.model_name
        with self.lock: self._stop_server()
        return f"unloaded: {name}" if name else "unloaded"

    def _stop_server(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.proc.kill()
        self.proc = None; self.stats.reset(); self.current_profile = {}; self._load_at = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stream(self, messages: list[dict], profile: dict,
               response_format: dict | None = None) -> Iterator[str]:
        if not self.is_loaded: yield "[Model not loaded]"; return
        body: dict[str, Any] = {
            "model": "local", "messages": _encode_multimodal_messages(messages),
            "temperature": profile.get("temperature", 0.7),
            "top_p":       profile.get("top_p",       0.9),
            "top_k":       profile.get("top_k",       40),
            "repeat_penalty": profile.get("repeat_penalty", 1.1),
            "max_tokens":  profile.get("max_tokens",  2048),
            "stream":      True,
        }
        # 項目4：透傳 response_format（b9010 原生支援）
        if response_format:
            body["response_format"] = response_format

        payload = json.dumps(body).encode()
        req = urllib.request.Request(self._server_url("/v1/chat/completions"),
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter(); count = 0; prompt_tokens = 0
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"): continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]": break
                    try:
                        chunk = json.loads(data_str)
                        usage = chunk.get("usage")
                        if usage: prompt_tokens = usage.get("prompt_tokens", 0)
                        token = chunk["choices"][0].get("delta", {}).get("content", "")
                        if token: count += 1; yield token
                    except (json.JSONDecodeError, KeyError): continue
        except Exception as e: yield f"[Error: {e}]"
        finally:
            elapsed = time.perf_counter() - t0
            self.stats.completion_tokens = count; self.stats.prompt_tokens = prompt_tokens
            self.stats.elapsed_sec = elapsed
            self.stats.tokens_per_sec = count / elapsed if elapsed > 0 else 0
            self.stats.context_used = prompt_tokens + count

    def embed(self, input_texts: list[str]) -> list[list[float]]:
        if not self.is_loaded: raise RuntimeError("No model loaded")
        results = []
        for text in input_texts:
            payload = json.dumps({"content": text}).encode()
            req = urllib.request.Request(self._server_url("/embedding"),
                data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                results.append(data.get("embedding", []))
            except Exception as e:
                raise RuntimeError(f"Embedding request failed: {e}")
        return results

# ── Binding engine ────────────────────────────────────────────────────────────
class BindingEngine:
    def __init__(self):
        self.llm: Any              = None
        self.lock                  = threading.Lock()
        self.stats                 = EngineStats()
        self.current_profile: dict = {}
        self._load_at:        float = 0.0

    def load(self, profile: dict) -> tuple[bool, str]:
        if not BINDING_AVAILABLE: return False, "llama-cpp-python not installed — use subprocess mode"
        model_path = profile.get("model_path", "")
        if not model_path or not os.path.isfile(model_path):
            return False, f"model file not found: {model_path}"
        t_start = time.perf_counter()
        with self.lock:
            self._unload_internal()
            try:
                meta = read_gguf_metadata(model_path)
                self.stats.model_metadata = meta
                kwargs = dict(model_path=model_path,
                              n_gpu_layers=profile.get("n_gpu_layers", 35),
                              n_ctx=profile.get("n_ctx",   4096),
                              n_batch=profile.get("n_batch", 512),
                              chat_format=profile.get("chat_format", "chatml"),
                              verbose=profile.get("verbose", False),
                              embedding=True)
                mmproj = profile.get("mmproj_path", "")
                if mmproj and os.path.isfile(mmproj):
                    kwargs["chat_handler"] = _build_clip_handler(mmproj)
                self.llm               = _Llama(**kwargs)
                self.current_profile   = profile.copy()
                self.stats.model_name  = os.path.basename(model_path)
                self.stats.engine_mode = "binding"
                self.stats.context_max = profile.get("n_ctx", 4096)
                self.stats.load_time_sec = time.perf_counter() - t_start
                self._load_at = time.time()
                return True, f"binding: {self.stats.model_name}"
            except Exception as e:
                self.llm = None; return False, str(e)

    def unload(self) -> str:
        name = self.stats.model_name
        with self.lock: self._unload_internal()
        return f"unloaded: {name}" if name else "unloaded"

    def _unload_internal(self):
        if self.llm is not None: del self.llm; self.llm = None
        self.stats.reset(); self.current_profile = {}; self._load_at = 0.0

    @property
    def is_loaded(self) -> bool: return self.llm is not None

    def stream(self, messages: list[dict], profile: dict,
               response_format: dict | None = None) -> Iterator[str]:
        if not self.is_loaded: yield "[Model not loaded]"; return
        t0 = time.perf_counter(); count = 0
        kwargs: dict[str, Any] = dict(
            messages=messages,
            temperature=profile.get("temperature", 0.7),
            top_p=profile.get("top_p", 0.9),
            top_k=profile.get("top_k", 40),
            repeat_penalty=profile.get("repeat_penalty", 1.1),
            max_tokens=profile.get("max_tokens", 2048),
            stream=True,
        )
        # 項目4：透傳 response_format
        if response_format:
            kwargs["response_format"] = response_format
        try:
            for chunk in self.llm.create_chat_completion(**kwargs):
                token = chunk["choices"][0]["delta"].get("content", "")
                if token: count += 1; yield token
        except Exception as e: yield f"[Error: {e}]"
        finally:
            elapsed = time.perf_counter() - t0
            self.stats.completion_tokens = count; self.stats.elapsed_sec = elapsed
            self.stats.tokens_per_sec = count / elapsed if elapsed > 0 else 0
            self.stats.context_used = count

    def embed(self, input_texts: list[str]) -> list[list[float]]:
        if not self.is_loaded: raise RuntimeError("No model loaded")
        results = []
        for text in input_texts:
            vec = self.llm.embed(text)
            if vec and isinstance(vec[0], list): results.extend(vec)
            else: results.append(vec)
        return results

# ── Helpers ───────────────────────────────────────────────────────────────────
def _encode_multimodal_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m.get("content", "")} for m in messages]

def _build_clip_handler(mmproj_path: str):
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        return Llava15ChatHandler(clip_model_path=mmproj_path)
    except ImportError: return None

def image_file_to_base64(path: str) -> str:
    ext  = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
            "gif":"image/gif","webp":"image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f: data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def build_image_message(text: str, image_b64: str | None) -> dict:
    if not image_b64: return {"role": "user", "content": text}
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_b64}},
    ]}

# ── HuggingFace helpers ───────────────────────────────────────────────────────
HF_API = "https://huggingface.co/api"
HF_CDN = "https://huggingface.co"

def hf_search(query: str, limit: int = 20) -> list[dict]:
    url = (f"{HF_API}/models?search={urllib.parse.quote(query)}"
           f"&filter=gguf&limit={limit}&sort=downloads&direction=-1")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LlamaGUI/2.3"})
        with urllib.request.urlopen(req, timeout=10) as resp: data = json.loads(resp.read())
        return [{"id": m.get("modelId",""), "downloads": m.get("downloads",0),
                 "likes": m.get("likes",0), "lastModified": m.get("lastModified","")[:10]}
                for m in data]
    except Exception as e:
        return [{"id": f"Error: {e}", "downloads": 0, "likes": 0, "lastModified": ""}]

def hf_list_gguf_files(repo_id: str) -> list[dict]:
    url = f"{HF_API}/models/{repo_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LlamaGUI/2.3"})
        with urllib.request.urlopen(req, timeout=10) as resp: data = json.loads(resp.read())
        results = []
        for s in data.get("siblings", []):
            fn = s.get("rfilename", "")
            if fn.lower().endswith(".gguf"):
                size_bytes = s.get("size", 0) or 0
                results.append({"filename": fn,
                                 "size_gb": round(size_bytes/1024**3, 2) if size_bytes else 0,
                                 "url": f"{HF_CDN}/{repo_id}/resolve/main/{fn}"})
        return results
    except Exception as e:
        return [{"filename": f"Error: {e}", "size_gb": 0, "url": ""}]

def hf_download(url: str, dest_path: str, progress_callback=None,
                cancel_event: threading.Event | None = None) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LlamaGUI/2.3"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0)); downloaded = 0
            with open(dest_path, "wb") as f:
                while True:
                    if cancel_event and cancel_event.is_set(): return False, "cancelled"
                    chunk = resp.read(1024 * 512)
                    if not chunk: break
                    f.write(chunk); downloaded += len(chunk)
                    if progress_callback: progress_callback(downloaded, total)
        return True, dest_path
    except Exception as e:
        if os.path.exists(dest_path):
            try: os.remove(dest_path)
            except Exception: pass
        return False, str(e)

# ── LlamaEngine ───────────────────────────────────────────────────────────────
QUEUE_TIMEOUT_SEC = 30

class LlamaEngine:
    def __init__(self):
        self.sub                           = SubprocessEngine()
        self.bind                          = BindingEngine()
        self.active: SubprocessEngine | BindingEngine = self.sub
        self._current_profile: dict        = {}
        self._last_activity:   float       = 0.0
        self._idle_stop_ev                 = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._req_lock    = threading.Semaphore(1)
        self._switch_lock = threading.Lock()

    def _resolve_chat_format(self, profile: dict) -> dict:
        fmt = profile.get("chat_format", "").strip()
        if not fmt or fmt == "auto":
            model_path = profile.get("model_path", "")
            fmt = detect_chat_format(model_path) if (model_path and os.path.isfile(model_path)) else "chatml"
            return {**profile, "chat_format": fmt}
        return profile

    def _build_switch_profile(self, model_path: str) -> dict:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        try:
            all_profiles = config.get_all().get("profiles", {})
            for pname, pdata in all_profiles.items():
                if _normalize(pname) == _normalize(stem):
                    saved = dict(pdata); saved["model_path"] = model_path; saved["chat_format"] = "auto"
                    return saved
        except Exception: pass
        base    = self._current_profile.copy() if self._current_profile else {}
        hw_keys = ("n_gpu_layers", "n_ctx", "n_batch", "engine_mode", "verbose")
        profile = {k: base[k] for k in hw_keys if k in base}
        profile["model_path"] = model_path; profile["chat_format"] = "auto"
        return profile

    def ensure_model_loaded(self, model_name: str) -> tuple[bool, str]:
        if model_name == "local":
            return (True, "local") if self.is_loaded else (False, "no model loaded")
        with self._switch_lock:
            if self.is_loaded:
                loaded_stem = os.path.splitext(self.stats.model_name)[0]
                if (_normalize(loaded_stem) == _normalize(model_name)
                        or _normalize(model_name) in _normalize(loaded_stem)):
                    return True, self.stats.model_name
            path = _find_model_by_name(model_name)
            if path is None: return False, f"not found: {model_name}"
            if self.is_loaded:
                log.info(f"ensure_model_loaded: unload {self.stats.model_name}")
                self.unload()
            return self.load(self._build_switch_profile(path))

    def touch(self) -> None: self._last_activity = time.time()

    def _idle_monitor(self) -> None:
        import config as _cfg
        while not self._idle_stop_ev.wait(timeout=30):
            try:
                s = _cfg.get_global_settings()
                if not s.get("idle_unload_enabled", False): continue
                if not self.is_loaded: continue
                threshold = float(s.get("idle_unload_seconds", 300))
                if self._last_activity > 0 and (time.time() - self._last_activity) > threshold:
                    log.info(f"Idle >{threshold:.0f}s — auto-unloading to free VRAM")
                    self.unload()
            except Exception: pass

    def start_idle_monitor(self) -> None:
        if self._idle_thread and self._idle_thread.is_alive(): return
        self._idle_stop_ev.clear()
        self._idle_thread = threading.Thread(target=self._idle_monitor, daemon=True, name="IdleMonitor")
        self._idle_thread.start()

    def stop_idle_monitor(self) -> None: self._idle_stop_ev.set()

    def load(self, profile: dict | None = None) -> tuple[bool, str]:
        p       = self._resolve_chat_format(profile or config.get_profile())
        backend = self._pick_backend(p)
        if backend is not self.active and self.active.is_loaded: self.active.unload()
        self.active = backend
        ok, msg = self.active.load(p)
        if ok: self._current_profile = p.copy(); self.touch()
        return ok, msg

    def unload(self) -> str:
        msg = self.active.unload(); self._current_profile = {}; return msg

    def _pick_backend(self, profile: dict):
        if profile.get("engine_mode") == "binding" and BINDING_AVAILABLE: return self.bind
        return self.sub

    @property
    def current_profile(self) -> dict: return self._current_profile.copy()
    @property
    def is_loaded(self) -> bool: return self.active.is_loaded
    @property
    def stats(self): return self.active.stats

    def stream(self, messages: list[dict], profile: dict | None = None,
               response_format: dict | None = None) -> Iterator[str]:
        self.touch()
        p = profile or self.current_profile
        acquired = self._req_lock.acquire(timeout=QUEUE_TIMEOUT_SEC)
        if not acquired:
            yield "[Error: inference queue timeout — server busy]"; return
        try:
            yield from self.active.stream(messages, p, response_format)
        finally:
            self._req_lock.release()

    def generate(self, messages: list[dict], profile: dict | None = None,
                 response_format: dict | None = None) -> str:
        self.touch()
        p = profile or self.current_profile
        acquired = self._req_lock.acquire(timeout=QUEUE_TIMEOUT_SEC)
        if not acquired:
            raise TimeoutError("inference queue timeout — server busy")
        try:
            return "".join(self.active.stream(messages, p, response_format))
        finally:
            self._req_lock.release()

    def embed(self, input_texts: list[str]) -> list[list[float]]:
        if not self.is_loaded: raise RuntimeError("No model loaded")
        return self.active.embed(input_texts)

    # 項目5：running model 詳細狀態（供 GET /api/ps）
    def get_ps_info(self) -> dict | None:
        if not self.is_loaded:
            return None
        s = self.active.stats
        load_at = getattr(self.active, "_load_at", 0.0)
        vram    = get_vram_info()
        return {
            "name":           s.model_name,
            "model":          s.model_name,
            "size":           s.model_metadata.get("file_size_gb", 0),
            "digest":         "",
            "details": {
                "format":          "gguf",
                "family":          s.model_metadata.get("architecture", ""),
                "parameter_size":  f"{s.model_metadata.get('n_params_b', 0)}B",
                "quantization_level": s.model_metadata.get("quant_type", ""),
            },
            "engine_mode":    s.engine_mode,
            "context_max":    s.context_max,
            "context_used":   s.context_used,
            "load_time_sec":  round(s.load_time_sec, 2),
            "loaded_at":      int(load_at),
            "size_vram":      vram.get("used_mb", 0),
            "expires_at":     "",   # idle unload 由 settings 控制，無固定到期時間
        }

    def get_stats(self) -> dict: return self.active.stats.to_dict()
    def get_model_metadata(self) -> dict: return self.active.stats.model_metadata.copy()
    def get_context_usage(self) -> tuple[int, int]:
        s = self.active.stats; return s.context_used, s.context_max
    def format_prompt_preview(self, messages: list[dict]) -> str:
        fmt = self.current_profile.get("chat_format", "chatml")
        return format_prompt_preview(messages, fmt)

# ── Module-level singleton ────────────────────────────────────────────────────
engine = LlamaEngine()
engine.start_idle_monitor()
find_model_by_name = _find_model_by_name   # expose for api.py