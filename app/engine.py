# engine.py v2.5 ── LlamaGUI backend engine
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

# Architecture names that indicate a VLM (vision-language model)
_VLM_ARCHITECTURES = frozenset({
    "llava", "llava_next", "llava_next_video",
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
    "gemma3", "gemma3_text",          # gemma3 multimodal variant
    "internvl", "internvl2",
    "phi3v", "phi4mm",
    "pixtral", "mistral3",
    "idefics3", "smolvlm",
    "moondream",
})

def read_gguf_metadata(path: str) -> dict:
    result: dict[str, Any] = {}
    try:
        fsize = os.path.getsize(path)
        result["file_size_gb"] = round(fsize / 1024**3, 2)
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return result
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count    = struct.unpack("<Q", f.read(8))[0]

            def read_str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            def read_val(vtype):
                if vtype == 4:  return struct.unpack("<I", f.read(4))[0]
                if vtype == 5:  return struct.unpack("<i", f.read(4))[0]
                if vtype == 6:  return struct.unpack("<f", f.read(4))[0]
                if vtype == 7:  return bool(struct.unpack("<?", f.read(1))[0])
                if vtype == 8:  return read_str()
                if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
                if vtype == 11: return struct.unpack("<d", f.read(8))[0]
                if vtype == 12: return struct.unpack("<q", f.read(8))[0]
                if vtype == 0:  return struct.unpack("<B", f.read(1))[0]
                if vtype == 1:  return struct.unpack("<b", f.read(1))[0]
                if vtype == 2:  return struct.unpack("<H", f.read(2))[0]
                if vtype == 3:  return struct.unpack("<h", f.read(2))[0]
                if vtype == 9:
                    item_type = struct.unpack("<I", f.read(4))[0]
                    count = struct.unpack("<Q", f.read(8))[0]
                    return [read_val(item_type) for _ in range(min(count, 256))]
                raise ValueError(f"unknown vtype {vtype}")

            for _ in range(kv_count):
                try:
                    key   = read_str()
                    vtype = struct.unpack("<I", f.read(4))[0]
                    val   = read_val(vtype)
                    result[key] = val
                except Exception:
                    break

        arch = result.get("general.architecture", "")
        result["architecture"]    = arch
        result["model_name"]      = result.get("general.name", "")
        ctx  = result.get(f"{arch}.context_length",  result.get("llama.context_length",  0))
        result["context_length"]  = ctx
        emb  = result.get(f"{arch}.embedding_length", result.get("llama.embedding_length", 0))
        result["embedding_length"] = emb
        layers = result.get(f"{arch}.block_count",   result.get("llama.block_count", 0))
        result["n_layers"] = layers
        ftype  = result.get("general.file_type", -1)
        FTYPE_MAP = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
            6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 14: "Q6_K", 15: "Q8_K",
            11: "Q2_K", 12: "Q3_K_S", 13: "Q3_K_M",
            16: "Q4_K_S", 17: "Q4_K_M", 18: "Q5_K_S", 19: "Q5_K_M",
        }
        result["quant_type"] = FTYPE_MAP.get(ftype, str(ftype))
        n_embd  = emb or 1;  n_layer = layers or 1
        n_vocab = result.get(f"{arch}.vocab_size", result.get("tokenizer.ggml.tokens", 0))
        if isinstance(n_vocab, list): n_vocab = len(n_vocab)
        params = (n_layer * n_embd * n_embd * 4 + n_vocab * n_embd * 2) / 1e9
        result["n_params_b"]    = round(params, 2)
        result["chat_template"] = result.get("tokenizer.chat_template", "")
        # DD-011: 從 architecture 欄位判斷模型是否具備視覺能力
        result["has_vision"] = arch.lower() in _VLM_ARCHITECTURES
    except Exception:
        pass
    return result


def detect_chat_format(model_path: str) -> str:
    meta     = read_gguf_metadata(model_path)
    template = meta.get("chat_template", "").lower()
    name     = os.path.basename(model_path).lower()

    if "<model>" in template:                                   fmt = "gemma"
    elif "<|im_start|>" in template:                           fmt = "chatml"
    elif "<|start_header_id|>" in template:                    fmt = "llama-3"
    elif "<s>[inst]" in template.replace(" ", ""):             fmt = "mistral-instruct"
    elif "[inst]" in template:                                  fmt = "llama-2"
    elif "<|user|>" in template:                               fmt = "phi3"
    elif "\u2581<0x0a>" in template or "deepseek" in template: fmt = "deepseek"
    elif "gemma"   in name:                                    fmt = "gemma"
    elif "llama-3" in name or "llama3" in name:                fmt = "llama-3"
    elif "llama-2" in name or "llama2" in name:                fmt = "llama-2"
    elif "mistral" in name:                                    fmt = "mistral-instruct"
    elif "qwen"    in name:                                    fmt = "qwen"
    elif "deepseek" in name:                                   fmt = "deepseek"
    elif "phi"     in name:                                    fmt = "phi3"
    elif "alpaca"  in name:                                    fmt = "alpaca"
    else:                                                       fmt = "chatml"

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
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if p.get("type") == "text" else "[image]"
                    for p in content
                )
            seg = inner.replace("{{ m.role }}", m.get("role", ""))
            seg = seg.replace("{{ m.content }}", content)
            result.append(seg)
        result.append(suffix)
    else:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user   = next((m["content"] for m in messages if m["role"] == "user"), "")
        result.append(template.replace("{{ system }}", system).replace("{{ user }}", user))
    return "".join(result)


# ── Model name fuzzy matching ─────────────────────────────────────────────────
def _normalize(s: str) -> str:
    return re.sub(r"[-_.\s]+", "", s.lower())

def _find_model_by_name(model_name: str) -> str | None:
    models_dir = config.get_models_dir()
    try:
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


# ── DD-011: mmproj auto-discovery ─────────────────────────────────────────────
def _find_mmproj(model_path: str) -> str | None:
    """
    Search model directory for a matching mmproj file.
    Priority:
      1. Same prefix: {model_stem}-mmproj*.gguf  or  {model_stem}*mmproj*.gguf
      2. Generic:     mmproj*.gguf  (only if exactly one found in directory)
    Returns the absolute path string, or None if not found.
    """
    model_dir  = os.path.dirname(os.path.abspath(model_path))
    model_stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    try:
        all_files = [f for f in os.listdir(model_dir) if f.lower().endswith(".gguf")]
    except OSError:
        return None

    # Priority 1: stem-based match
    stem_matches = [
        f for f in all_files
        if "mmproj" in f.lower() and (
            f.lower().startswith(model_stem[:12])   # prefix match (12 chars tolerance)
            or model_stem[:8] in f.lower()           # looser prefix
        )
    ]
    if stem_matches:
        return os.path.join(model_dir, sorted(stem_matches)[0])

    # Priority 2: generic mmproj (only if unique)
    generic = [f for f in all_files if "mmproj" in f.lower()]
    if len(generic) == 1:
        return os.path.join(model_dir, generic[0])

    return None


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
        self.load_time_sec:     float = 0.0
    def to_dict(self) -> dict:
        return dict(
            prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
            elapsed_sec=round(self.elapsed_sec, 2), tokens_per_sec=round(self.tokens_per_sec, 1),
            model_name=self.model_name, engine_mode=self.engine_mode,
            context_used=self.context_used, context_max=self.context_max,
            model_metadata=self.model_metadata, load_time_sec=round(self.load_time_sec, 2),
        )


# ── OOM keyword detector ──────────────────────────────────────────────────────
_OOM_KEYWORDS = ("out of memory", "oom", "cuda error", "cudaerror",
                 "not enough memory", "allocation failed", "memory allocation")

def _is_oom_error(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in _OOM_KEYWORDS)


# ── Subprocess engine ─────────────────────────────────────────────────────────
class SubprocessEngine:
    SERVER_HOST     = "127.0.0.1"
    SERVER_PORT     = 8765
    STARTUP_TIMEOUT = 60

    def __init__(self):
        self.proc:            subprocess.Popen | None = None
        self.lock             = threading.Lock()
        self.stats            = EngineStats()
        self.current_profile: dict  = {}
        self._load_at:        float = 0.0
        self._has_vision:     bool  = False   # DD-011

    @staticmethod
    def find_server_exe() -> str | None:
        bindir = config.get_bin_dir()
        candidates = [os.path.join(bindir, "llama-server.exe"), os.path.join(bindir, "llama-server")]
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ["bin/cuda/llama-server.exe", "bin/llama-server.exe", "bin/llama-server",
                    "llama-server.exe", "llama-server"]:
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

    # DD-011: query llama-server /props to get vision capability flag
    def _query_vision_capable(self) -> bool:
        try:
            with urllib.request.urlopen(self._server_url("/props"), timeout=2) as resp:
                data = json.loads(resp.read())
                return bool(data.get("vision", False))
        except Exception:
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

            # DD-011: mmproj auto-discovery
            mmproj = profile.get("mmproj_path", "")
            if not mmproj or not os.path.isfile(mmproj):
                discovered = _find_mmproj(model_path)
                if discovered:
                    mmproj = discovered
                    log.info(f"mmproj auto-discovered: {os.path.basename(mmproj)}")

            cmd = [exe, "--model", model_path, "--host", self.SERVER_HOST,
                   "--port",         str(self.SERVER_PORT),
                   "--n-gpu-layers", str(profile.get("n_gpu_layers", 35)),
                   "--ctx-size",     str(profile.get("n_ctx", 4096)),
                   "--batch-size",   str(profile.get("n_batch", 512)),
                   "--chat-template", str(profile.get("chat_format", "chatml")),
                   "--no-mmap",
                   ]
            if mmproj and os.path.isfile(mmproj):
                cmd += ["--mmproj", mmproj]
                log.info(f"Loading with mmproj: {os.path.basename(mmproj)}")

            # DD-001: speculative decoding draft model
            draft = profile.get("draft_model_path", "")
            if draft and os.path.isfile(draft): cmd += ["--draft-model", draft]
            if not profile.get("verbose", False): cmd += ["--log-disable"]
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except Exception as e:
                return False, f"Failed to start llama-server.exe: {e}"
            if not self._wait_ready():
                self._stop_server()
                return False, "llama-server did not become ready within 60s"

            # DD-011: query vision capability after server is ready
            self._has_vision = self._query_vision_capable()
            if not self._has_vision:
                # fallback: check metadata architecture or mmproj presence
                self._has_vision = (
                    meta.get("has_vision", False)
                    or (bool(mmproj) and os.path.isfile(mmproj))
                )

            self.current_profile     = profile.copy()
            self.current_profile["mmproj_path"] = mmproj  # store resolved mmproj path
            self.stats.model_name    = os.path.basename(model_path)
            self.stats.engine_mode   = "subprocess"
            self.stats.context_max   = profile.get("n_ctx", 4096)
            self.stats.load_time_sec = time.perf_counter() - t_start
            self._load_at            = time.time()
            vision_tag = " [vision]" if self._has_vision else ""
            return True, f"subprocess: {self.stats.model_name}{vision_tag}"

    def unload(self) -> str:
        name = self.stats.model_name
        with self.lock:
            self._stop_server()
            self._has_vision = False
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
            "model":          "local",
            "messages":       _encode_multimodal_messages(messages),
            "temperature":    profile.get("temperature", 0.7),
            "top_p":          profile.get("top_p", 0.9),
            "top_k":          profile.get("top_k", 40),
            "repeat_penalty": profile.get("repeat_penalty", 1.1),
            "max_tokens":     profile.get("max_tokens", 2048),
            "stream":         True,
        }
        if response_format:                              body["response_format"]    = response_format
        if profile.get("stop")              is not None: body["stop"]              = profile["stop"]
        if profile.get("seed")              is not None: body["seed"]              = profile["seed"]
        if profile.get("presence_penalty")  is not None: body["presence_penalty"]  = profile["presence_penalty"]
        if profile.get("frequency_penalty") is not None: body["frequency_penalty"] = profile["frequency_penalty"]
        if profile.get("logprobs")          is not None: body["logprobs"]          = profile["logprobs"]
        if profile.get("top_logprobs")      is not None: body["top_logprobs"]      = profile["top_logprobs"]
        if profile.get("tools")       is not None: body["tools"]       = profile["tools"]
        if profile.get("tool_choice") is not None: body["tool_choice"] = profile["tool_choice"]
        if profile.get("id_slot") is not None: body["id_slot"] = profile["id_slot"]

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            self._server_url("/v1/chat/completions"),
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
        except Exception as e:
            import traceback
            log.error(
                f"SubprocessEngine.stream() failed\n"
                f"  Model:  {self.stats.model_name}\n"
                f"  Error:  {type(e).__name__}: {e}\n"
                f"  Trace:\n{traceback.format_exc()}"
            )
            yield f"[Error: {e}]"
        finally:
            elapsed = time.perf_counter() - t0
            self.stats.completion_tokens = count
            self.stats.prompt_tokens     = prompt_tokens
            self.stats.elapsed_sec       = elapsed
            self.stats.tokens_per_sec    = count / elapsed if elapsed > 0 else 0
            self.stats.context_used      = prompt_tokens + count
            log.info(
                f"stream() done | model={self.stats.model_name} | tokens={count}"
                f" | tps={self.stats.tokens_per_sec:.1f} | elapsed={elapsed:.2f}s"
                f" | prompt_tokens={prompt_tokens}"
            )

    def embed(self, input_texts: list[str]) -> list[list[float]]:
        if not self.is_loaded: raise RuntimeError("No model loaded")
        results = []
        for text in input_texts:
            payload = json.dumps({"content": text}).encode()
            req = urllib.request.Request(
                self._server_url("/embedding"),
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
        self.llm:             Any   = None
        self.lock             = threading.Lock()
        self.stats            = EngineStats()
        self.current_profile: dict  = {}
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

                # DD-011: mmproj auto-discovery for binding mode
                mmproj = profile.get("mmproj_path", "")
                if not mmproj or not os.path.isfile(mmproj):
                    discovered = _find_mmproj(model_path)
                    if discovered:
                        mmproj = discovered
                        log.info(f"mmproj auto-discovered (binding): {os.path.basename(mmproj)}")

                kwargs = dict(
                    model_path=model_path,
                    n_gpu_layers=profile.get("n_gpu_layers", 35),
                    n_ctx=profile.get("n_ctx", 4096),
                    n_batch=profile.get("n_batch", 512),
                    chat_format=profile.get("chat_format", "chatml"),
                    verbose=profile.get("verbose", False),
                    embedding=True)
                if mmproj and os.path.isfile(mmproj):
                    handler = _build_clip_handler(mmproj)
                    if handler:
                        kwargs["chat_handler"] = handler
                        log.info(f"LLaVA handler set for: {os.path.basename(mmproj)}")

                self.llm = _Llama(**kwargs)
                self.current_profile     = profile.copy()
                self.current_profile["mmproj_path"] = mmproj
                self.stats.model_name    = os.path.basename(model_path)
                self.stats.engine_mode   = "binding"
                self.stats.context_max   = profile.get("n_ctx", 4096)
                self.stats.load_time_sec = time.perf_counter() - t_start
                self._load_at            = time.time()
                has_vision = self.llm.chat_handler is not None
                vision_tag = " [vision]" if has_vision else ""
                return True, f"binding: {self.stats.model_name}{vision_tag}"
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

    # DD-011: binding mode vision check
    @property
    def has_vision(self) -> bool:
        if not self.is_loaded: return False
        return getattr(self.llm, "chat_handler", None) is not None

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
        if response_format:                                    kwargs["response_format"]    = response_format
        if profile.get("stop")              is not None: kwargs["stop"]              = profile["stop"]
        if profile.get("seed")              is not None: kwargs["seed"]              = profile["seed"]
        if profile.get("presence_penalty")  is not None: kwargs["presence_penalty"]  = profile["presence_penalty"]
        if profile.get("frequency_penalty") is not None: kwargs["frequency_penalty"] = profile["frequency_penalty"]
        if profile.get("logprobs")          is not None: kwargs["logprobs"]          = profile["logprobs"]
        if profile.get("top_logprobs")      is not None: kwargs["top_logprobs"]      = profile["top_logprobs"]
        if profile.get("tools")       is not None: kwargs["tools"]       = profile["tools"]
        if profile.get("tool_choice") is not None: kwargs["tool_choice"] = profile["tool_choice"]

        prompt_tokens = 0
        try:
            for chunk in self.llm.create_chat_completion(**kwargs):
                usage = chunk.get("usage")
                if usage: prompt_tokens = usage.get("prompt_tokens", 0)
                token = chunk["choices"][0]["delta"].get("content", "")
                if token: count += 1; yield token
        except Exception as e:
            import traceback
            log.error(
                f"BindingEngine.stream() failed\n"
                f"  Model:  {self.stats.model_name}\n"
                f"  Error:  {type(e).__name__}: {e}\n"
                f"  Trace:\n{traceback.format_exc()}"
            )
            yield f"[Error: {e}]"
        finally:
            elapsed = time.perf_counter() - t0
            self.stats.completion_tokens = count
            self.stats.elapsed_sec       = elapsed
            self.stats.tokens_per_sec    = count / elapsed if elapsed > 0 else 0
            self.stats.context_used      = prompt_tokens + count
            log.info(
                f"stream() done | model={self.stats.model_name} | tokens={count}"
                f" | tps={self.stats.tokens_per_sec:.1f} | elapsed={elapsed:.2f}s"
                f" | prompt_tokens={prompt_tokens}"
            )

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
    """
    DD-011: Preserve list-type content (ContentParts) for multimodal messages.
    String content is passed through unchanged.
    Pydantic models are serialised to dict before reaching here via api.py.
    """
    result = []
    for m in messages:
        content = m.get("content", "")
        # list content = multimodal parts, pass through as-is for llama-server
        if isinstance(content, list):
            serialised = []
            for part in content:
                if isinstance(part, dict):
                    serialised.append(part)
                else:
                    # Pydantic model → dict (fallback)
                    try:
                        serialised.append(part.model_dump(exclude_none=True))
                    except AttributeError:
                        serialised.append({"type": "text", "text": str(part)})
            result.append({"role": m["role"], "content": serialised})
        else:
            result.append({"role": m["role"], "content": content})
    return result

def _build_clip_handler(mmproj_path: str):
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        return Llava15ChatHandler(clip_model_path=mmproj_path)
    except ImportError: return None

def image_file_to_base64(path: str) -> str:
    ext  = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif",  "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f: data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def build_image_message(text: str, image_b64: str | None) -> dict:
    if not image_b64: return {"role": "user", "content": text}
    return {"role": "user", "content": [
        {"type": "text",      "text": text},
        {"type": "image_url", "image_url": {"url": image_b64, "detail": "auto"}},
    ]}


# ── HuggingFace helpers ───────────────────────────────────────────────────────
HF_API = "https://huggingface.co/api"
HF_CDN = "https://huggingface.co"

def hf_search(query: str, limit: int = 20) -> list[dict]:
    url = (f"{HF_API}/models?search={urllib.parse.quote(query)}"
           f"&filter=gguf&limit={limit}&sort=downloads&direction=-1")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LlamaGUI/2.5"})
        with urllib.request.urlopen(req, timeout=10) as resp: data = json.loads(resp.read())
        return [{"id": m.get("modelId",""), "downloads": m.get("downloads", 0),
                 "likes": m.get("likes", 0), "lastModified": m.get("lastModified","")[:10]}
                for m in data]
    except Exception as e:
        return [{"id": f"Error: {e}", "downloads": 0, "likes": 0, "lastModified": ""}]

def hf_list_gguf_files(repo_id: str) -> list[dict]:
    url = f"{HF_API}/models/{repo_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LlamaGUI/2.5"})
        with urllib.request.urlopen(req, timeout=10) as resp: data = json.loads(resp.read())
        results = []
        for s in data.get("siblings", []):
            fn = s.get("rfilename", "")
            if fn.lower().endswith(".gguf"):
                size_bytes = s.get("size", 0) or 0
                results.append({"filename": fn,
                                 "size_gb": round(size_bytes / 1024**3, 2) if size_bytes else 0,
                                 "url": f"{HF_CDN}/{repo_id}/resolve/main/{fn}"})
        return results
    except Exception as e:
        return [{"filename": f"Error: {e}", "size_gb": 0, "url": ""}]

def hf_download(url: str, dest_path: str, progress_callback=None,
                cancel_event: threading.Event | None = None) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LlamaGUI/2.5"})
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
MAX_WAITING       = 5

class LlamaEngine:
    def __init__(self):
        self.sub           = SubprocessEngine()
        self.bind          = BindingEngine()
        self.active: SubprocessEngine | BindingEngine = self.sub
        self._current_profile: dict  = {}
        self._last_activity:   float = 0.0
        self._idle_stop_ev     = threading.Event()
        self._idle_thread:     threading.Thread | None = None
        self._req_lock         = threading.Semaphore(1)
        self._switch_lock      = threading.Lock()
        self._waiting:     int  = 0
        self._waiting_lock       = threading.Lock()

    # DD-011: unified has_vision property
    @property
    def has_vision(self) -> bool:
        if isinstance(self.active, SubprocessEngine):
            return self.active._has_vision
        if isinstance(self.active, BindingEngine):
            return self.active.has_vision
        return False

    def _resolve_chat_format(self, profile: dict) -> dict:
        fmt = profile.get("chat_format", "").strip()
        if not fmt or fmt == "auto":
            model_path = profile.get("model_path", "")
            fmt = detect_chat_format(model_path) if (model_path and os.path.isfile(model_path)) else "chatml"
        return {**profile, "chat_format": fmt}

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
        profile["model_path"]  = model_path
        profile["chat_format"] = "auto"
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
                # DD-009: subprocess process crash detection
                if isinstance(self.active, SubprocessEngine) and self.active.proc is not None:
                    if self.active.proc.poll() is not None:
                        log.error(
                            f"llama-server process crashed unexpectedly "
                            f"(exit code: {self.active.proc.returncode}) — "
                            f"model: {self.active.stats.model_name}"
                        )
                        with self.active.lock:
                            self.active._stop_server()
                        continue

                s = _cfg.get_global_settings()
                if not s.get("idle_unload_enabled", False): continue
                if not self.is_loaded: continue
                threshold = float(s.get("idle_unload_seconds", 300))
                if self._last_activity > 0 and (time.time() - self._last_activity) > threshold:
                    log.info(f"Idle >{threshold:.0f}s — auto-unloading to free VRAM")
                    self.unload()
            except Exception:
                pass

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

        fallback_layers: list[int] = p.get("fallback_gpu_layers", [35, 20, 10, 0])
        current_layers = p.get("n_gpu_layers", 35)
        layers_to_try  = [current_layers] + [l for l in fallback_layers if l != current_layers]

        last_ok, last_msg = False, ""
        for n_layers in layers_to_try:
            attempt = {**p, "n_gpu_layers": n_layers}
            ok, msg = self.active.load(attempt)
            if ok:
                if n_layers != current_layers:
                    log.warning(f"load() OOM fallback succeeded with n_gpu_layers={n_layers} "
                                f"(requested: {current_layers})")
                self._current_profile = attempt.copy(); self.touch()
                return ok, msg + (f" [fallback: layers={n_layers}]" if n_layers != current_layers else "")
            last_ok, last_msg = ok, msg
            if _is_oom_error(msg):
                log.warning(f"load() OOM with n_gpu_layers={n_layers}: {msg} — trying next fallback")
            else:
                break
        return last_ok, last_msg

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

    def _check_queue(self) -> bool:
        with self._waiting_lock:
            if self._waiting >= MAX_WAITING:
                return False
            self._waiting += 1
            return True

    def _release_queue(self) -> None:
        with self._waiting_lock:
            self._waiting = max(0, self._waiting - 1)

    def stream(self, messages: list[dict], profile: dict | None = None,
               response_format: dict | None = None) -> Iterator[str]:
        self.touch()
        p = profile or self.current_profile
        if not self._check_queue():
            yield f"[Error: server busy — too many pending requests (max {MAX_WAITING})]"
            return
        acquired = self._req_lock.acquire(timeout=QUEUE_TIMEOUT_SEC)
        if not acquired:
            self._release_queue()
            yield "[Error: inference queue timeout — server busy]"; return
        try:
            yield from self.active.stream(messages, p, response_format)
        finally:
            self._req_lock.release()
            self._release_queue()

    def generate(self, messages: list[dict], profile: dict | None = None,
                 response_format: dict | None = None) -> str:
        self.touch()
        p = profile or self.current_profile
        if not self._check_queue():
            raise TimeoutError(f"server busy — too many pending requests (max {MAX_WAITING})")
        acquired = self._req_lock.acquire(timeout=QUEUE_TIMEOUT_SEC)
        if not acquired:
            self._release_queue()
            raise TimeoutError("inference queue timeout — server busy")
        try:
            result = "".join(self.active.stream(messages, p, response_format))
            if result.startswith("[Error:"):
                import traceback
                log.error(
                    f"generate() failed\n"
                    f"  Model:  {self.stats.model_name}\n"
                    f"  Error:  {result}\n"
                    f"  Trace:\n{''.join(traceback.format_stack())}"
                )
                raise RuntimeError(result)
            return result
        except (TimeoutError, RuntimeError):
            raise
        except Exception as e:
            import traceback
            log.error(
                f"generate() failed\n"
                f"  Model:  {self.stats.model_name}\n"
                f"  Error:  {type(e).__name__}: {e}\n"
                f"  Trace:\n{traceback.format_exc()}"
            )
            raise
        finally:
            self._req_lock.release()
            self._release_queue()

    def embed(self, input_texts: list[str]) -> list[list[float]]:
        if not self.is_loaded: raise RuntimeError("No model loaded")
        return self.active.embed(input_texts)

    def get_ps_info(self) -> dict | None:
        if not self.is_loaded: return None
        s       = self.active.stats
        load_at = getattr(self.active, "_load_at", 0.0)
        vram    = get_vram_info()
        return {
            "name":  s.model_name, "model": s.model_name,
            "size":  s.model_metadata.get("file_size_gb", 0),
            "digest": "",
            "details": {
                "format":             "gguf",
                "family":             s.model_metadata.get("architecture", ""),
                "parameter_size":     f"{s.model_metadata.get('n_params_b', 0)}B",
                "quantization_level": s.model_metadata.get("quant_type", ""),
            },
            "engine_mode":   s.engine_mode,
            "context_max":   s.context_max,
            "context_used":  s.context_used,
            "load_time_sec": round(s.load_time_sec, 2),
            "loaded_at":     int(load_at),
            "size_vram":     vram.get("used_mb", 0),
            "expires_at":    "",
            # DD-011: vision capability flag
            "has_vision":    self.has_vision,
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
find_model_by_name = _find_model_by_name  # expose for api.py