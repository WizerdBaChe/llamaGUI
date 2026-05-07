# LlamaGUI

> A portable local LLM GUI and API server built on llama.cpp (b9010+).
> Exposes an OpenAI-compatible REST API on `http://127.0.0.1:8000` and a Gradio web UI on `http://127.0.0.1:7860`.
> One model is active at a time. Sending a request with a specific model name will auto-load the matching model if needed.

---

## Quick Integration Checklist

- Base URL: `http://127.0.0.1:8000`
- No authentication required (local only)
- Use `"model": "local"` to use whatever model is currently loaded
- Use `"model": "<name>"` (e.g. `"gemma-4-E4B"`) to auto-load a matching `.gguf` from the `models/` folder
- Check `GET /health` → `model_loaded: true` before sending prompts if using `"model": "local"`
- `options` dict does NOT exist — all params go at the top level of the request body
- JSON structured output: use `"response_format": {"type": "json_object"}` — no system prompt workaround needed

---

## Authentication

None. All endpoints are unauthenticated and bound to `127.0.0.1` only.

---

## Endpoints

### Health & Lifecycle

#### GET /health
Check server and model status.

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen2.5-7B-Q4_K_M.gguf"
}
```

#### GET /api/ps
Running model details. Ollama-compatible. Returns an empty array if no model is loaded.

Response:
```json
{
  "models": [
    {
      "name":         "Qwen2.5-7B-Q4_K_M.gguf",
      "model":        "Qwen2.5-7B-Q4_K_M.gguf",
      "size":         4.68,
      "details": {
        "format":              "gguf",
        "family":              "llama",
        "parameter_size":      "7.6B",
        "quantization_level":  "Q4_K_M"
      },
      "engine_mode":   "subprocess",
      "context_max":   4096,
      "context_used":  220,
      "load_time_sec": 8.43,
      "loaded_at":     1746230400,
      "size_vram":     5120,
      "expires_at":    ""
    }
  ]
}
```
`expires_at` is empty — idle unload is controlled globally via Settings, not per-request.

#### POST /load
Load a model by profile name or full file path.

Request body:
```json
{
  "profile_name": "default",
  "model_path": null
}
```
- `profile_name`: string | null — load from saved profile
- `model_path`: string | null — **full path** to a `.gguf` file (e.g. `"D:\\models\\Qwen2.5-7B-Q4_K_M.gguf"`).
  This field does **not** support fuzzy name matching. Pass the exact path as returned by `GET /models`.
  Use `/v1/chat/completions` with a `model` name if you want auto-load with fuzzy matching (see Inference section).

> **`/load` vs `model` field — key difference:**
>
> | | `POST /load` (`model_path`) | `/v1/chat/completions` (`model`) |
> |---|---|---|
> | Accepts fuzzy name? | ❌ No — full path required | ✅ Yes — fuzzy match against `models/` |
> | Triggers auto-load? | Manual — you control timing | Automatic — on every request |
> | Can set hardware params? | ✅ Yes (`n_ctx`, `n_gpu_layers`, …) | ❌ No |
>
> If you need **both** fuzzy name matching and hardware parameter overrides, first call `GET /models`
> to resolve the full path, then pass it to `POST /load`.

Response:
```json
{ "status": "loaded", "message": "subprocess: Qwen2.5-7B-Q4_K_M.gguf" }
```

Error (400):
```json
{ "detail": "model path not found" }
```

#### POST /load — Hardware Parameter Override

`POST /load` accepts optional hardware parameters that override the profile's saved values
for that session only. The profile on disk is not modified.

Request body (all fields optional except at least one of `profile_name` / `model_path`):
```json
{
  "profile_name":  "default",
  "model_path":    null,
  "n_ctx":         4096,
  "n_gpu_layers":  35,
  "n_batch":       512,
  "n_threads":     null,
  "temperature":   0.7,
  "top_p":         0.9,
  "top_k":         40,
  "repeat_penalty":1.1,
  "max_tokens":    2048,
  "chat_format":   null,
  "engine_mode":   null
}
```

**Hardware parameter reference:**

| Parameter | Type | Notes |
|---|---|---|
| `n_ctx` | int | Context window size. Set at load time — cannot change without reloading. Larger values consume more VRAM for KV cache. |
| `n_gpu_layers` | int | Number of layers offloaded to GPU. `-1` = all layers. Reduce if VRAM is insufficient. |
| `n_batch` | int | Prompt evaluation batch size. Higher = faster prompt processing but more VRAM. Recommended: 256–512. |
| `n_threads` | int | CPU thread count. Only relevant when `n_gpu_layers` < total layers. |
| `engine_mode` | string | `"subprocess"` (default) or `"binding"`. Binding has lower latency; recommended for high-frequency embedding. |

**Recommended configs by task type:**

| Task | `n_ctx` | `n_gpu_layers` | `n_batch` | Notes |
|---|---|---|---|---|
| Short extraction / classification | 2048 | 35 | 512 | Small ctx → less VRAM → faster |
| Semantic chunking (short) | 4096 | 35 | 512 | Default balance |
| Semantic chunking (long doc) | 8192 | 35 | 256 | Lower batch when ctx is large |
| Embedding (high-frequency) | 2048 | 35 | 512 | Use `engine_mode: binding` |

**Python helper — task-aware reload with deduplication:**
```python
import requests

BASE = "http://127.0.0.1:8000"

TASK_CONFIGS = {
    "extraction": {
        "n_ctx": 2048, "n_gpu_layers": 35, "n_batch": 512,
        "max_tokens": 512, "temperature": 0.0,
    },
    "chunking_fast": {
        "n_ctx": 4096, "n_gpu_layers": 35, "n_batch": 512,
        "max_tokens": 1024, "temperature": 0.2,
    },
    "chunking_smart": {
        "n_ctx": 8192, "n_gpu_layers": 35, "n_batch": 256,
        "max_tokens": 2048, "temperature": 0.3,
    },
}

_active_config: str = ""


def resolve_model_path(model_name: str) -> str:
    """
    Resolve a fuzzy model name to a full path using GET /models.
    GET /models returns full .gguf paths as scanned from the models/ folder.
    Raises RuntimeError if no match is found.
    """
    resp = requests.get(f"{BASE}/models", timeout=5)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    name_lower = model_name.lower()
    for m in models:
        if name_lower in m["id"].lower():
            return m["id"]   # full path, e.g. "D:\\models\\Qwen2.5-7B-Q4_K_M.gguf"
    raise RuntimeError(
        f"Model '{model_name}' not found in models/. "
        f"Available: {[m['id'] for m in models]}"
    )


def ensure_loaded(config_key: str, model_name: str) -> None:
    """
    Switch hardware config only when it changes (deduplication).
    model_name is a fuzzy name (e.g. "qwen2.5-7b"); full path is resolved automatically.

    Note: POST /load requires a full file path — fuzzy matching is NOT supported there.
    This helper resolves the path via GET /models before calling /load.
    """
    global _active_config
    if _active_config == config_key:
        return  # same config, skip reload
    full_path = resolve_model_path(model_name)
    cfg = TASK_CONFIGS[config_key]
    requests.post(f"{BASE}/load",
                  json={"model_path": full_path, **cfg},
                  timeout=120).raise_for_status()
    _active_config = config_key
```

`_active_config` prevents redundant reloads when consecutive requests share the same task type.
Only a true config switch (e.g. extraction → chunking_smart) triggers a model reload (~5–15s).

#### Model Path Resolution in POST /load

`model_path` accepts three input formats. The engine resolves them in order:

1. **Absolute path** — used directly if the file exists
   `"model_path": "D:\\AIWork\\models\\gemma-4-E4B-it-Q4_K_M.gguf"`

2. **Filename stem** — fuzzy-matched against `.gguf` files in the configured `models/` folder
   `"model_path": "gemma-4-E4B-it-Q4_K_M"`

3. **Partial name** — substring / prefix match (case-insensitive, symbols stripped)
   `"model_path": "gemma4e4b"` → matches `gemma-4-E4B-it-Q4_K_M.gguf`

If no match is found, the engine returns HTTP 404 with a message indicating the input value
and instructions to check `GET /v1/models` for valid names.

**Callee contract:** External callers should pass the stem (format 2) and let the engine
resolve the path. Callers must not construct or hardcode absolute paths — the engine's
`models/` directory is configurable and may differ across environments.

Use `GET /models-dir` only if you need to display or log the physical path.
Never pass the result of `GET /v1/models[].id` as a `model_path` directly —
`id` is already a stem and works as format 2 above, but avoid using the API roundtrip
when you already know the model name.

#### POST /unload
Unload current model and free VRAM.

Response:
```json
{ "status": "unloaded", "message": "Qwen2.5-7B-Q4_K_M.gguf" }
```

#### GET /stats
Get inference statistics for the current session.

Response:
```json
{
  "model_name":        "Qwen2.5-7B-Q4_K_M.gguf",
  "engine_mode":       "subprocess",
  "tokens_per_sec":    42.3,
  "elapsed_sec":       3.14,
  "completion_tokens": 133,
  "prompt_tokens":     87,
  "context_used":      220,
  "context_max":       4096,
  "load_time_sec":     8.43
}
```

---

### Inference (OpenAI-Compatible)

#### POST /v1/chat/completions
Send a chat completion request. Compatible with OpenAI SDK.

**Model field behaviour:**

| `model` value | Behaviour |
|---|---|
| `"local"` | Use currently loaded model; returns 503 if none loaded |
| `"<name>"` (e.g. `"gemma-4-E4B"`) | Auto-find and load matching `.gguf` in `models/`; returns 404 if not found |

Model name matching is fuzzy — `"qwen2.5-7b"` will match `Qwen2.5-7B-Instruct-Q4_K_M.gguf`. Matching priority: exact stem → normalised (case-insensitive, symbols stripped) → substring → prefix.

If the requested model differs from the currently loaded one, the old model is unloaded and the new one loaded automatically before inference. `chat_format` is detected from GGUF metadata automatically.

Request body:
```json
{
  "model": "local",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "Explain quantum entanglement." }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "max_tokens": 2048,
  "stream": false,
  "response_format": null
}
```

**Parameter reference:**

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | string | required | `"local"` or a model name for auto-load |
| `messages` | array | required | `[{role, content}]` |
| `temperature` | float | 0.7 | 0.0–2.0 |
| `top_p` | float | 0.9 | 0.0–1.0 |
| `top_k` | int | 40 | 1–200 |
| `repeat_penalty` | float | 1.1 | 1.0–2.0 |
| `max_tokens` | int | 2048 | 1–8192 |
| `stream` | bool | false | SSE if true |
| `response_format` | object | null | See JSON output section below |

**JSON structured output (`response_format`):**

Use `response_format` to force structured JSON output. Requires llama.cpp b9010+. No system prompt workaround needed.

```json
{ "response_format": {"type": "json_object"} }
```

Full JSON schema enforcement:
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {
          "result": {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["result", "confidence"]
      }
    }
  }
}
```

| `type` value | Behaviour |
|---|---|
| `"text"` | Default free-text output |
| `"json_object"` | Forces valid JSON output (any schema) |
| `"json_schema"` | Forces output matching the provided JSON schema |

Non-streaming response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1746230400,
  "model": "local",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "Quantum entanglement is..." },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 87,
    "completion_tokens": 133,
    "total_tokens": 220
  }
}
```

**Extract content (non-streaming):**
```python
content = resp.json()["choices"][0]["message"]["content"]
```

Streaming response (SSE):
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Quantum"},"finish_reason":null}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":" entanglement"},"finish_reason":null}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

**Parse streaming:**
```python
import json, requests

resp = requests.post("http://127.0.0.1:8000/v1/chat/completions",
    json={..., "stream": True}, stream=True)

for line in resp.iter_lines():
    line = line.decode("utf-8").strip()
    if not line.startswith("data:"): continue
    data = line[5:].strip()
    if data == "[DONE]": break
    token = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
    print(token, end="", flush=True)
```

**Error responses:**

| Status | Cause |
|---|---|
| 404 | `model` name specified but no matching `.gguf` found in `models/` |
| 503 | `model: "local"` with nothing loaded, or inference queue timeout (>30s wait) |

---

### Embeddings (OpenAI-Compatible)

#### POST /v1/embeddings
Generate embeddings for one or more texts.

- Binding mode: calls `llm.embed()` directly in-process (faster, no HTTP overhead)
- Subprocess mode: proxies to `llama-server`'s `/embedding` endpoint

For high-frequency or batch embedding workloads, set `engine_mode: binding` in the model's profile.

Request body:
```json
{
  "model": "local",
  "input": "The quick brown fox"
}
```

`input` accepts a single string or an array of strings:
```json
{ "model": "local", "input": ["text one", "text two", "text three"] }
```

Response:
```json
{
  "object": "list",
  "model": "local",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.021, -0.134, 0.007, ...] }
  ],
  "usage": { "prompt_tokens": 5, "total_tokens": 5 }
}
```

Compatible with:
- `openai.embeddings.create()`
- LangChain `OpenAIEmbeddings(base_url="http://127.0.0.1:8000/v1", api_key="none")`

---

### Model Listing (OpenAI-Compatible)

#### GET /v1/models
List available models. Compatible with OpenAI SDK `client.models.list()`.

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id":       "Qwen2.5-7B-Q4_K_M",
      "object":   "model",
      "created":  1746230400,
      "owned_by": "local"
    }
  ]
}
```

Note: `id` is the filename without `.gguf` extension. Use this value in `"model"` field to trigger auto-load.

#### GET /models
Original endpoint. Returns full filenames including `.gguf` extension (full path as stored on disk). Use this to resolve a fuzzy model name to a full path before calling `POST /load`. For GUI use and programmatic path resolution.

---

### Metadata & Monitoring

#### GET /model-metadata
GGUF file metadata for the loaded model.

Response:
```json
{
  "arch": "llama",
  "model_name": "Qwen2.5-7B",
  "n_params_b": 7.6,
  "quant_type": "Q4_K_M",
  "context_length": 32768,
  "embedding_length": 3584,
  "n_layers": 28,
  "file_size_gb": 4.68
}
```
Returns `{"error": "no model loaded"}` if no model is active.

#### GET /vram
GPU VRAM usage via nvidia-smi.

Response:
```json
{
  "used_mb": 5120,
  "total_mb": 8192,
  "free_mb": 3072,
  "utilization_pct": 62,
  "available": true
}
```
`available: false` if nvidia-smi is not found.

#### GET /context-usage
Current context token usage.

Response:
```json
{ "used": 220, "max": 4096, "pct": 5.4 }
```

#### POST /prompt-preview
Preview the raw formatted prompt string before sending to model.

Request:
```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user",   "content": "Hello" }
  ],
  "chat_format": "chatml"
}
```

Response:
```json
{ "prompt": "<|im_start|>system\nYou are helpful.<|im_end|>\n...", "length": 42 }
```

Supported `chat_format` values:
`chatml`, `llama-2`, `llama-3`, `gemma`, `phi3`, `mistral-instruct`, `qwen`, `deepseek`, `alpaca`

---

### Profile Management

#### GET /profiles
List all saved profiles.

Response:
```json
{ "active_profile": "default", "profiles": ["default", "fast", "quality"] }
```

#### GET /profiles/{name}
Get a specific profile's settings.

Response:
```json
{
  "model_path":    "D:\\models\\Qwen2.5-7B-Q4_K_M.gguf",
  "mmproj_path":   "",
  "n_gpu_layers":  35,
  "n_ctx":         4096,
  "n_batch":       512,
  "chat_format":   "chatml",
  "engine_mode":   "subprocess",
  "temperature":   0.7,
  "top_p":         0.9,
  "top_k":         40,
  "repeat_penalty":1.1,
  "max_tokens":    2048,
  "verbose":       false
}
```

`engine_mode` options:

| Value | Description |
|---|---|
| `"subprocess"` | Default. Spawns `bin/cuda/llama-server.exe`. More stable; model crash won't affect main process. |
| `"binding"` | Uses `llama-cpp-python` directly in-process. Lower latency; recommended for high-frequency embedding. |

#### POST /profiles
Create or update a profile.

#### DELETE /profiles/{name}
Delete a profile. Cannot delete `"default"`.

#### POST /profiles/{name}/activate
Set a profile as active (used by GUI and next `/load` call).

---

### HuggingFace Download

#### GET /hf-search?q={query}&limit={n}
Search HuggingFace for GGUF models.

#### GET /hf-files?repo_id={repo_id}
List GGUF files in a repo.

#### POST /hf-download
Start a background download. Returns a task ID to poll.

Request:
```json
{
  "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
  "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
  "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF"
}
```

#### GET /hf-download/{task_id}
Poll download progress. `status`: `running` | `done` | `error` | `cancelled`

#### DELETE /hf-download/{task_id}
Cancel a running download.

---

## Code Examples

### Minimal non-streaming call
```python
import requests

def llm_chat(prompt: str, system: str = "You are a helpful assistant.",
             model: str = "local",
             temperature: float = 0.7, max_tokens: int = 2048) -> str:
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream": False,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

Pass `model="local"` (default) to use whatever is loaded, or `model="qwen2.5-7b"` to trigger auto-load.

### Force JSON output
```python
import json, requests

def llm_json(prompt: str, model: str = "local") -> dict:
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "json_object"},
    }, timeout=300)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])
```

For schema-enforced output:
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "result",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score":  {"type": "number"}
            },
            "required": ["answer", "score"]
        }
    }
}
```

### Embeddings
```python
import requests

def embed(texts: list[str], model: str = "local") -> list[list[float]]:
    resp = requests.post("http://127.0.0.1:8000/v1/embeddings", json={
        "model": model,
        "input": texts,
    }, timeout=60)
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in data]
```

### LangChain integration
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none",
    model="local",
)

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none",
    model="local",
)
```

### Batch processing (sequential, safe)
```python
import requests

BASE = "http://127.0.0.1:8000"


def check_server_alive() -> bool:
    """
    Check that the LlamaGUI server is running.
    Does NOT require a model to already be loaded — use model="<name>" in
    the request body to trigger auto-load, or call POST /load explicitly first.
    Checking model_loaded here is only necessary when using model="local".
    """
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def process_chunks(chunks: list[str], system_prompt: str,
                   model: str = "local") -> list[str]:
    assert check_server_alive(), "LlamaGUI server not running — start via run_dev.bat"
    # If model="local", a model must already be loaded.
    # If model="<name>", auto-load will be triggered on the first request.
    results = []
    for chunk in chunks:
        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model":       model,
            "messages":    [{"role": "system", "content": system_prompt},
                            {"role": "user",   "content": chunk}],
            "temperature": 0.0,
            "max_tokens":  1500,
            "stream":      False,
        }, timeout=300)
        resp.raise_for_status()
        results.append(resp.json()["choices"][0]["message"]["content"])
    return results
```

### Using OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")

# Chat
response = client.chat.completions.create(
    model="local",          # or a model name to trigger auto-load
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)

# List available models
for m in client.models.list():
    print(m.id)

# Embeddings
vectors = client.embeddings.create(model="local", input=["hello world"])
print(vectors.data[0].embedding[:5])
```

### Check running model details
```python
import requests

ps = requests.get("http://127.0.0.1:8000/api/ps").json()
if ps["models"]:
    m = ps["models"][0]
    print(f"Model:   {m['name']}")
    print(f"VRAM:    {m['size_vram']} MB")
    print(f"Context: {m['context_used']} / {m['context_max']}")
    print(f"Engine:  {m['engine_mode']}")
else:
    print("No model loaded")
```

---

## Ollama Migration Reference

| Ollama | LlamaGUI |
|---|---|
| `http://localhost:11434/api/chat` | `http://127.0.0.1:8000/v1/chat/completions` |
| `"model": "llama3"` | `"model": "llama3"` — auto-loads matching `.gguf` |
| `resp["message"]["content"]` | `resp["choices"][0]["message"]["content"]` |
| `"options": {"num_predict": 1000}` | `"max_tokens": 1000` (top-level) |
| `"options": {"temperature": 0.5}` | `"temperature": 0.5` (top-level) |
| `"format": "json"` | `"response_format": {"type": "json_object"}` |
| `"format": {schema}` | `"response_format": {"type": "json_schema", "json_schema": {...}}` |
| `"keep_alive": "5m"` | Configure `idle_unload_seconds` in Settings |
| `GET /api/ps` | `GET /api/ps` — same path, compatible response shape |
| `GET /api/tags` | `GET /v1/models` (OpenAI format) or `GET /models` |
| Multi-model routing | Supported via `"model"` field — auto-loads on demand, one model at a time |
| `/api/embeddings` | `POST /v1/embeddings` (OpenAI-compatible) |

---

## Error Reference

| HTTP Status | Meaning | Fix |
|---|---|---|
| 400 | Bad request (missing path, invalid profile) | Check request body |
| 404 | Model name not found in `models/` folder | Check filename or add `.gguf` to `models/` |
| 409 | File already exists (download) | Delete existing file first |
| 503 | No model loaded (`"model": "local"`) | Call `POST /load` first |
| 503 | Inference queue timeout | Server busy >30s; retry or reduce concurrency |
| Connection refused | Server not running | Start via `run_dev.bat` |

---

## Constraints & Behaviour Notes

- Only one model can be loaded at a time. Loading a new model auto-unloads the previous one.
- `engine_mode: subprocess` spawns `bin/cuda/llama-server.exe` (llama.cpp b9010, CUDA 12). Model crash does not affect the main LlamaGUI process.
- `engine_mode: binding` uses `llama-cpp-python` in-process. Lower latency; recommended for high-frequency embedding. Requires `llama-cpp-python` to be installed.
- Concurrent `/v1/chat/completions` requests are serialised (single queue). Requests waiting more than 30 seconds return HTTP 503.
- `n_ctx` (context window size) is set at load time and cannot be changed without reloading.
- `response_format` is natively supported by llama.cpp b9010+ via grammar sampling. No system prompt engineering needed.
- `model` field: `"local"` uses the currently loaded model; any other value triggers fuzzy filename matching against `models/` and auto-loads if found.
- `chat_format` is detected automatically from GGUF metadata when a model is auto-loaded. Manually loaded models respect the profile's `chat_format` setting.
- Auto-load via `model` field inherits hardware parameters (`n_gpu_layers`, `n_ctx`, `n_batch`) from the active profile, unless a saved profile matching the model name exists. **If you need specific hardware parameters (e.g. a larger `n_ctx`), you cannot set them through `/v1/chat/completions` — you must call `POST /load` explicitly with the full model path and the desired parameters.**
- Idle auto-unload is configurable in the Settings tab. If enabled, the model unloads after N seconds of inactivity.
- For RAG pipelines requiring frequent embedding calls, set `engine_mode: binding` on the embedding model's profile to avoid localhost HTTP overhead.
- Hardware parameters (`n_ctx`, `n_gpu_layers`, `n_batch`) are fixed at load time and cannot
  be changed without reloading the model. Pass them directly in `POST /load` to override
  the profile's saved values for that session without modifying the profile on disk.
- `POST /load` resolves `model_path` in three stages: absolute path → stem fuzzy-match →
  partial substring match. Callee projects should pass a stem or partial name and rely on
  the engine for path resolution. The engine is the authoritative caller layer; external
  projects are callees and must conform to the engine's API contract, not the reverse.
- `GET /v1/models` returns `id` as a filename stem (no `.gguf`, no directory path).
  This value can be passed directly as `model_path` to `POST /load` (stem format),
  but should not be used to construct filesystem paths in callee code.
- `models_dir` is stored as a path relative to `ROOT` in `profiles.json`.
  On first launch after moving the project folder, the engine auto-detects an invalid path
  and falls back to `<ROOT>/models/`, correcting `profiles.json` automatically.
  No manual intervention required when relocating the project directory.