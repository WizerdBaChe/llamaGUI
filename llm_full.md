# LlamaGUI API 調用完整指南

> **版本：** engine v4.0 / api v2.6.2  
> **Base URL：** `http://127.0.0.1:8000`  
> **Gradio UI：** `http://127.0.0.1:7860`  
> 本地部署，無需驗證，單模型常駐，OpenAI 相容。

---

## 快速檢查清單

- Base URL：`http://127.0.0.1:8000`
- 無 API Key，無認證
- `"model": "local"` — 使用目前載入的模型
- `"model": "<name>"` — 模糊匹配 `models/` 內的 `.gguf`，自動載入
- 使用 `"model": "local"` 前先 `GET /health` 確認 `model_loaded: true`
- 所有推論參數放在請求頂層，**不存在** `options` 嵌套物件
- 結構化 JSON 輸出：`"response_format": {"type": "json_object"}`
- Vision（圖片）輸入：`content` 使用 content parts 格式，需載入 VLM 模型

---

## 一、狀態與生命週期

### GET /health

檢查伺服器與模型狀態。

```json
{
  "status":        "ok",
  "model_loaded":  true,
  "model_loading": false,
  "model_name":    "Qwen2.5-7B-Q4_K_M.gguf",
  "has_vision":    false
}
```

| 欄位 | 說明 |
|---|---|
| `model_loaded` | `true` 表示模型就緒，可推論 |
| `model_loading` | `true` 表示正在載入中（最長約 60s），此時 `POST /load` 會回 409 |
| `model_name` | 目前載入的 `.gguf` 檔名；未載入時為空字串 |
| `has_vision` | `true` 表示已載入 VLM 模型（含 mmproj），可傳入圖片 |

### POST /load

載入模型。接受 profile 名稱或模型路徑，支援硬體參數覆蓋。

```json
{
  "profile_name":   "default",
  "model_path":     null,
  "n_ctx":          4096,
  "n_gpu_layers":   35,
  "n_batch":        512,
  "n_threads":      null,
  "temperature":    0.7,
  "top_p":          0.9,
  "top_k":          40,
  "repeat_penalty": 1.1,
  "max_tokens":     2048,
  "chat_format":    null,
  "engine_mode":    null,
  "draft_model_path": null
}
```

`model_path` 支援三種格式（依序解析）：

1. 絕對路徑：`"D:\\models\\Qwen2.5-7B-Q4_K_M.gguf"`
2. 檔名 stem：`"Qwen2.5-7B-Q4_K_M"`
3. 部分名稱：`"qwen2.5-7b"` → 模糊匹配

所有硬體參數僅覆蓋本次 session，不修改 profiles.json。

成功回應：
```json
{
  "status":     "loaded",
  "message":    "subprocess: Qwen2.5-7B-Q4_K_M.gguf",
  "has_vision": false
}
```

錯誤回應：

| 狀態碼 | 原因 |
|---|---|
| 400 | 模型路徑無效或模型啟動失敗 |
| 404 | `model_path` 名稱在 `models/` 中找不到 |
| 409 | 另一個 `POST /load` 正在進行中，稍後重試 |

> **POST /load vs `model` 欄位對比**
>
> | | `POST /load` | `/v1/chat/completions` `model` |
> |---|---|---|
> | 接受模糊名稱 | ✅ stem / 部分名稱 | ✅ 模糊匹配 |
> | 可設硬體參數 | ✅ 可 | ❌ 不可 |
> | 自動觸發 | 手動控制 | 每次請求自動檢查 |
>
> 需要**同時**模糊名稱 + 自訂硬體參數時：先 `GET /v1/models` 取 id，再傳給 `POST /load`。

### POST /unload

卸載模型，釋放 VRAM。

```json
{ "status": "unloaded", "message": "Qwen2.5-7B-Q4_K_M.gguf" }
```

### GET /stats

取得本次 session 的推論統計。

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

### GET /api/ps

Ollama 相容的運行狀態端點。

```json
{
  "models": [
    {
      "name":       "Qwen2.5-7B-Q4_K_M.gguf",
      "model":      "Qwen2.5-7B-Q4_K_M.gguf",
      "size":       4.68,
      "details": {
        "format":             "gguf",
        "family":             "qwen2",
        "parameter_size":     "7.6B",
        "quantization_level": "Q4_K_M"
      },
      "engine_mode":   "subprocess",
      "context_max":   4096,
      "context_used":  220,
      "load_time_sec": 8.43,
      "loaded_at":     1746230400,
      "size_vram":     5120,
      "has_vision":    false,
      "expires_at":    ""
    }
  ]
}
```

無模型載入時回傳 `{"models": []}`。`expires_at` 固定為空字串（idle 卸載由 Settings 頁全域控制）。

---

## 二、推論（OpenAI 相容）

### POST /v1/chat/completions

核心推論端點，相容 OpenAI SDK。

**`model` 欄位行為：**

| 值 | 行為 |
|---|---|
| `"local"` | 使用目前載入的模型；未載入時回 503 |
| `"<name>"` | 模糊匹配 `models/` 中的 `.gguf`；匹配則自動載入，不匹配回 404 |

模型名稱匹配優先順序：完整 stem → 正規化（忽略大小寫、符號）→ 子字串 → 前綴。

自動載入的模型繼承 active profile 的硬體參數（`n_gpu_layers`、`n_ctx`、`n_batch`），
除非 `models/` 中有同名的已儲存 profile。**若需要特定硬體參數，請改用 `POST /load`。**

#### 請求體完整說明

```json
{
  "model":           "local",
  "messages":        [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "解釋量子糾纏。"}
  ],
  "temperature":     0.7,
  "top_p":           0.9,
  "top_k":           40,
  "repeat_penalty":  1.1,
  "max_tokens":      2048,
  "stream":          false,
  "response_format": null,
  "stop":            null,
  "seed":            null,
  "presence_penalty":  null,
  "frequency_penalty": null,
  "logprobs":        null,
  "top_logprobs":    null,
  "tools":           null,
  "tool_choice":     null,
  "id_slot":         null
}
```

**參數完整參考：**

| 參數 | 型別 | 預設 | 說明 |
|---|---|---|---|
| `model` | string | required | `"local"` 或模型名稱（自動載入） |
| `messages` | array | required | `[{role, content}]`，見下方 content 格式說明 |
| `temperature` | float | 0.7 | 0.0–2.0，0.0 為確定性輸出 |
| `top_p` | float | 0.9 | 0.0–1.0，nucleus sampling |
| `top_k` | int | 40 | ≥1，top-k sampling |
| `repeat_penalty` | float | 1.1 | ≥1.0 抑制重複；1.0 = 無懲罰 |
| `max_tokens` | int | 2048 | 最大輸出 token 數 |
| `stream` | bool | false | `true` 啟用 SSE 串流 |
| `response_format` | object | null | 結構化輸出，見下方 |
| `stop` | str \| list[str] | null | 停止序列，生成到此字串停止 |
| `seed` | int | null | 隨機種子，確保可重現輸出 |
| `presence_penalty` | float | null | 懲罰已出現話題的 token，提升主題多樣性 |
| `frequency_penalty` | float | null | 按 token 出現頻率懲罰重複 |
| `logprobs` | bool | null | 回傳 token 對數機率 |
| `top_logprobs` | int | null | 回傳 top-N 個 token 機率 |
| `tools` | list[dict] | null | Tool Calling 工具定義（llama.cpp b9010+） |
| `tool_choice` | str \| dict | null | 工具選擇策略（`"auto"` / `"none"` / 指定工具） |
| `id_slot` | int | null | KV Cache slot 編號，重用 prefix cache（subprocess 模式，需 `--slot-save-path`） |

#### 非串流回應

```json
{
  "id":      "chatcmpl-abc123",
  "object":  "chat.completion",
  "created": 1746230400,
  "model":   "local",
  "choices": [{
    "index":   0,
    "message": {"role": "assistant", "content": "量子糾纏是..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens":     87,
    "completion_tokens": 133,
    "total_tokens":      220
  }
}
```

取得文字：
```python
content = resp.json()["choices"][0]["message"]["content"]
```

#### SSE 串流回應

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"量子"},"finish_reason":null}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"糾纏"},"finish_reason":null}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

串流錯誤格式（OpenAI 標準，不污染 delta.content）：
```json
{
  "id": "chatcmpl-abc123",
  "choices": [],
  "error": {"message": "[Error: ...]", "type": "engine_error", "code": 500}
}
```

解析串流（Python）：
```python
import json, requests

resp = requests.post("http://127.0.0.1:8000/v1/chat/completions",
    json={..., "stream": True}, stream=True)

for line in resp.iter_lines():
    line = line.decode("utf-8").strip()
    if not line.startswith("data:"): continue
    data = line[5:].strip()
    if data == "[DONE]": break
    chunk = json.loads(data)
    if "error" in chunk:
        raise RuntimeError(chunk["error"]["message"])
    token = chunk["choices"][0].get("delta", {}).get("content", "")
    print(token, end="", flush=True)
```

#### 結構化 JSON 輸出（response_format）

不需 system prompt 工程，llama.cpp b9010+ 原生 grammar sampling 支援。

```json
{ "response_format": {"type": "json_object"} }
```

Schema 強制輸出：
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {
          "result":     {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["result", "confidence"]
      }
    }
  }
}
```

| `type` | 行為 |
|---|---|
| `"text"` | 預設自由文字 |
| `"json_object"` | 強制合法 JSON（任意 schema） |
| `"json_schema"` | 強制符合提供的 JSON Schema |

---

## 三、Vision（圖片輸入）

VLM 功能需載入支援視覺的模型（含 mmproj 投影矩陣）。

**支援的模型架構：** LLaVA、LLaVA-Next、Qwen2-VL / Qwen2.5-VL / Qwen3-VL、Gemma3、InternVL / InternVL2、Phi-3-Vision / Phi-4-MM、Pixtral、Mistral3、Idefics3、SmolVLM、Moondream

**mmproj 自動配對：** 若 `mmproj_path` 為空，engine 會在主模型資料夾中自動尋找（前綴匹配 → 唯一 mmproj 檔案），無需手動設定路徑。

`GET /health` 的 `has_vision: true` 表示 VLM 就緒。`GET /api/ps` 的 `has_vision` 同步反映狀態。

### 圖片訊息格式（content parts）

```json
{
  "model": "local",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "這張圖片裡有什麼？"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<BASE64_DATA>",
            "detail": "auto"
          }
        }
      ]
    }
  ]
}
```

`detail` 可選值：`"auto"`（預設）、`"low"`、`"high"`。

`url` 欄位支援：
- base64 data URI：`data:image/jpeg;base64,...`
- 遠端 URL：`https://example.com/photo.jpg`（需 llama-server 網路存取）

**Vision 錯誤回應：**

```json
{
  "detail": "model does not support vision — load a multimodal model with mmproj (e.g. Gemma3-VL, Qwen2-VL, LLaVA). Current model: 'Qwen2.5-7B-Q4_K_M.gguf'"
}
```

> ⚠️ **已知限制：** 載入 mmproj 時，KV cache slot 持久化（`id_slot`）和 prefix cache 會被 llama-server 停用。VLM 模式下 `id_slot` 實際無效。

**Python 工具函數：**
```python
import base64

def image_to_data_uri(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def ask_about_image(image_path: str, question: str) -> str:
    import requests
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model": "local",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path), "detail": "auto"}}
            ]
        }],
        "max_tokens": 1024,
        "temperature": 0.2,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

---

## 四、Tool Calling（工具調用）

llama.cpp b9010+ 原生支援，透傳至 llama-server（subprocess 模式）。工具執行邏輯由呼叫端負責。

```json
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "台北現在氣溫是多少？"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

模型回應含工具調用時，`finish_reason` 為 `"tool_calls"`：
```json
{
  "choices": [{
    "message": {
      "role":       "assistant",
      "content":    null,
      "tool_calls": [{
        "id":       "call_abc123",
        "type":     "function",
        "function": {
          "name":      "get_weather",
          "arguments": "{\"city\": \"Taipei\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

多輪 tool calling 流程（Python）：
```python
import json, requests

BASE = "http://127.0.0.1:8000"

def run_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return json.dumps({"city": args["city"], "temperature": 28, "unit": "celsius"})
    raise ValueError(f"Unknown tool: {name}")

def chat_with_tools(user_message: str, tools: list) -> str:
    messages = [{"role": "user", "content": user_message}]
    while True:
        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "local",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }, timeout=120).json()
        choice = resp["choices"][0]
        if choice["finish_reason"] == "stop":
            return choice["message"]["content"]
        if choice["finish_reason"] == "tool_calls":
            messages.append(choice["message"])
            for tc in choice["message"]["tool_calls"]:
                result = run_tool(tc["function"]["name"],
                                  json.loads(tc["function"]["arguments"]))
                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc["id"],
                    "content":     result,
                })
```

> ⚠️ binding 模式（`llama-cpp-python`）的 `tools` 支援程度依版本而異。
> `id_slot` 在 VLM 模式下無效（兩者互斥）。

---

## 五、文字補全（Legacy Completions）

### POST /v1/completions

裸 prompt 補全，相容舊版 LangChain 等工具鏈。

```json
{
  "model":       "local",
  "prompt":      "The capital of France is",
  "max_tokens":  64,
  "temperature": 0.7,
  "top_p":       0.9,
  "top_k":       40,
  "stop":        ["\n"],
  "seed":        null,
  "stream":      false,
  "echo":        false
}
```

`prompt` 接受單一字串或字串陣列。`echo: true` 在回應中包含原始 prompt。

回應：
```json
{
  "id":      "cmpl-abc123",
  "object":  "text_completion",
  "created": 1746230400,
  "model":   "local",
  "choices": [{"index": 0, "text": " Paris.", "finish_reason": "stop"}]
}
```

---

## 六、Embeddings（向量嵌入）

### POST /v1/embeddings

OpenAI 相容，支援單文字或批次。

```json
{
  "model": "local",
  "input": "The quick brown fox"
}
```

批次：
```json
{ "model": "local", "input": ["文字一", "文字二", "文字三"] }
```

回應：
```json
{
  "object": "list",
  "model":  "local",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.021, -0.134, 0.007, ...]}
  ],
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

- **binding 模式**：直接呼叫 `llm.embed()`，無 HTTP overhead，推薦高頻 embedding 使用
- **subprocess 模式**：代理至 llama-server `/embedding`（需 b2000+ build）

高頻 embedding 推薦配置：`POST /load` 時設 `"engine_mode": "binding"`。

Python 工具函數：
```python
def embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post("http://127.0.0.1:8000/v1/embeddings",
        json={"model": "local", "input": texts}, timeout=60)
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in data]
```

---

## 七、模型列表

### GET /v1/models

OpenAI 相容，`client.models.list()` 可直接使用。

```json
{
  "object": "list",
  "data": [
    {"id": "Qwen2.5-7B-Q4_K_M", "object": "model", "created": 1746230400, "owned_by": "local"}
  ]
}
```

`id` 為不含 `.gguf` 的 filename stem，可直接用於 `/v1/chat/completions` 的 `model` 欄位。

### GET /models

原始端點，回傳完整磁碟路徑（含 `.gguf` 副檔名）。用於需要完整路徑的 `POST /load` 前置解析。

---

## 八、監控與元資料

### GET /model-metadata

目前載入模型的 GGUF 元資料。

```json
{
  "architecture":    "qwen2",
  "model_name":      "Qwen2.5-7B",
  "n_params_b":      7.62,
  "quant_type":      "Q4_K_M",
  "context_length":  32768,
  "embedding_length": 3584,
  "n_layers":        28,
  "file_size_gb":    4.68,
  "has_vision":      false,
  "chat_template":   "..."
}
```

未載入時回 `{"error": "no model loaded"}`。

### GET /vram

GPU VRAM 使用狀況（需 nvidia-smi）。

```json
{
  "used_mb":          5120,
  "total_mb":         8192,
  "free_mb":          3072,
  "utilization_pct":  62,
  "available":        true
}
```

`available: false` 表示 nvidia-smi 不可用。

### GET /context-usage

目前 context window 使用量。

```json
{ "used": 220, "max": 4096, "pct": 5.4 }
```

### POST /prompt-preview

預覽模型收到的原始 prompt 字串。

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "Hello"}
  ],
  "chat_format": "chatml"
}
```

回應：
```json
{ "prompt": "<|im_start|>system\nYou are helpful.<|im_end|>\n...", "length": 42 }
```

支援的 `chat_format`：`chatml`、`llama-2`、`llama-3`、`gemma`、`phi3`、`mistral-instruct`、`qwen`、`deepseek`、`alpaca`

---

## 九、Profile 管理

Profile 是預設推論設定組合，儲存在 `profiles.json`。

### GET /profiles

```json
{ "active": "default", "profiles": ["default", "fast", "quality"] }
```

### GET /profiles/{name}

取得指定 profile 完整設定：
```json
{
  "model_path":     "D:\\models\\Qwen2.5-7B-Q4_K_M.gguf",
  "mmproj_path":    "",
  "n_gpu_layers":   35,
  "n_ctx":          4096,
  "n_batch":        512,
  "chat_format":    "chatml",
  "engine_mode":    "subprocess",
  "temperature":    0.7,
  "top_p":          0.9,
  "top_k":          40,
  "repeat_penalty": 1.1,
  "max_tokens":     2048,
  "verbose":        false,
  "draft_model_path": ""
}
```

`engine_mode` 可選值：

| 值 | 說明 |
|---|---|
| `"subprocess"` | 預設。啟動 `bin/cuda/llama-server.exe`，模型 crash 不影響主程序 |
| `"binding"` | 使用 `llama-cpp-python` in-process。延遲較低，推薦高頻 embedding |

### POST /profiles

新增或更新 profile。Pydantic 型別驗證（DD-012 #4），已知欄位型別錯誤（如 `n_ctx: "abc"`）會被 422 攔截。自訂欄位允許透傳。

```json
{
  "name":    "my-profile",
  "profile": {
    "model_path":   "D:\\models\\Llama3-8B-Q4.gguf",
    "n_gpu_layers": 35,
    "n_ctx":        8192,
    "engine_mode":  "subprocess"
  }
}
```

### DELETE /profiles/{name}

刪除 profile（不可刪除 `"default"`）。

### POST /profiles/{name}/activate

設為目前 active profile。

---

## 十、HuggingFace 下載

### GET /hf-search?q={query}&limit={n}

搜尋 HuggingFace GGUF 模型。

```json
{
  "results": [
    {"id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "downloads": 12345, "likes": 89, "lastModified": "2025-01-01"}
  ]
}
```

### GET /hf-files?repo_id={repo_id}

列出 repo 中的 GGUF 檔案。

```json
{
  "files": [
    {"filename": "qwen2.5-7b-instruct-q4_k_m.gguf", "size_gb": 4.68, "url": "https://..."}
  ]
}
```

### POST /hf-download

在背景啟動下載，回傳 task_id 供輪詢。

```json
{
  "url":      "https://huggingface.co/.../qwen2.5-7b-instruct-q4_k_m.gguf",
  "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
  "repo_id":  "Qwen/Qwen2.5-7B-Instruct-GGUF"
}
```

回應：`{"task_id": "a1b2c3d4", "filename": "qwen2.5-7b-instruct-q4_k_m.gguf"}`

### GET /hf-download/{task_id}

輪詢下載進度。`status` 值：`running` / `done` / `error` / `cancelled`

```json
{
  "task_id":  "a1b2c3d4",
  "status":   "running",
  "progress": 1234567890,
  "total":    4891234567,
  "pct":      25.2,
  "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
  "message":  ""
}
```

### DELETE /hf-download/{task_id}

取消下載，刪除未完成的部分檔案。

---

## 十一、Models Dir 管理

### GET /models-dir

```json
{ "models_dir": "D:\\AIWork\\models" }
```

### POST /models-dir

```json
{ "path": "D:\\AIWork\\models" }
```

---

## 十二、程式碼範例

### 最小非串流調用

```python
import requests

def llm_chat(prompt: str, system: str = "You are a helpful assistant.",
             model: str = "local", temperature: float = 0.7,
             max_tokens: int = 2048) -> str:
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

`model="local"` 使用目前載入的模型；`model="qwen2.5-7b"` 觸發自動載入。

### 強制 JSON 輸出

```python
import json, requests

def llm_json(prompt: str, model: str = "local") -> dict:
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model":           model,
        "messages":        [{"role": "user", "content": prompt}],
        "temperature":     0.0,
        "max_tokens":      1024,
        "stream":          False,
        "response_format": {"type": "json_object"},
    }, timeout=300)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])
```

### 串流輸出

```python
import json, requests

def llm_stream(prompt: str, model: str = "local"):
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   True,
    }, stream=True, timeout=300)
    resp.raise_for_status()
    for line in resp.iter_lines():
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        chunk = json.loads(data)
        if "error" in chunk:
            raise RuntimeError(chunk["error"]["message"])
        token = chunk["choices"][0].get("delta", {}).get("content", "")
        if token:
            yield token

# 使用方式
for token in llm_stream("用繁體中文說明深度學習"):
    print(token, end="", flush=True)
```

### Embeddings

```python
import requests

def embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post("http://127.0.0.1:8000/v1/embeddings",
        json={"model": "local", "input": texts}, timeout=60)
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in data]
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")

# Chat
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)

# 模型列表
for m in client.models.list():
    print(m.id)

# Embeddings
vectors = client.embeddings.create(model="local", input=["hello world"])
print(vectors.data[0].embedding[:5])
```

### LangChain 整合

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

### 批次處理（順序，安全）

```python
import requests

BASE = "http://127.0.0.1:8000"

def check_server() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=3).status_code == 200
    except Exception:
        return False

def process_chunks(chunks: list[str], system_prompt: str,
                   model: str = "local") -> list[str]:
    assert check_server(), "LlamaGUI server not running — start via run_dev.bat"
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

### 任務感知模型載入（硬體參數切換）

```python
import requests

BASE = "http://127.0.0.1:8000"

TASK_CONFIGS = {
    "extraction":     {"n_ctx": 2048, "n_gpu_layers": 35, "n_batch": 512,
                       "max_tokens": 512,  "temperature": 0.0},
    "chunking_fast":  {"n_ctx": 4096, "n_gpu_layers": 35, "n_batch": 512,
                       "max_tokens": 1024, "temperature": 0.2},
    "chunking_smart": {"n_ctx": 8192, "n_gpu_layers": 35, "n_batch": 256,
                       "max_tokens": 2048, "temperature": 0.3},
}

_active_config = ""

def ensure_loaded(config_key: str, model_stem: str) -> None:
    """切換硬體設定；相同 config 不重複 reload（去重複）。"""
    global _active_config
    if _active_config == config_key:
        return
    cfg = TASK_CONFIGS[config_key]
    requests.post(f"{BASE}/load",
                  json={"model_path": model_stem, **cfg},
                  timeout=120).raise_for_status()
    _active_config = config_key
```

### 查詢運行中模型狀態

```python
ps = requests.get("http://127.0.0.1:8000/api/ps").json()
if ps["models"]:
    m = ps["models"][0]
    print(f"Model:   {m['name']}")
    print(f"Vision:  {m['has_vision']}")
    print(f"VRAM:    {m['size_vram']} MB")
    print(f"Context: {m['context_used']} / {m['context_max']}")
    print(f"Engine:  {m['engine_mode']}")
else:
    print("No model loaded")
```

---

## 十三、Ollama 遷移對照

| Ollama | LlamaGUI |
|---|---|
| `http://localhost:11434/api/chat` | `http://127.0.0.1:8000/v1/chat/completions` |
| `"model": "llama3"` | `"model": "llama3"` — 自動匹配 `.gguf` |
| `resp["message"]["content"]` | `resp["choices"][0]["message"]["content"]` |
| `"options": {"num_predict": 1000}` | `"max_tokens": 1000`（頂層） |
| `"options": {"temperature": 0.5}` | `"temperature": 0.5`（頂層） |
| `"format": "json"` | `"response_format": {"type": "json_object"}` |
| `"format": {schema}` | `"response_format": {"type": "json_schema", "json_schema": {...}}` |
| `"keep_alive": "5m"` | Settings 頁設定 `idle_unload_seconds` |
| `GET /api/ps` | `GET /api/ps` — 相同路徑，相容回應格式 |
| `GET /api/tags` | `GET /v1/models`（OpenAI 格式）或 `GET /models` |
| `/api/embeddings` | `POST /v1/embeddings`（OpenAI 相容） |
| 多模型路由 | `"model"` 欄位自動按需載入，每次僅一個模型 |

---

## 十四、錯誤碼速查

| HTTP 狀態碼 | 原因 | 處理方式 |
|---|---|---|
| 400 | 請求格式錯誤、模型路徑無效、VLM guard（傳圖但非 VLM 模型） | 檢查請求 body |
| 404 | `model` 名稱在 `models/` 中找不到 | 確認 `.gguf` 在 models 資料夾中 |
| 409 | `POST /load` 正在進行中 | 等待 `model_loading: false` 後重試 |
| 422 | Pydantic schema 驗證失敗（如 profile 欄位型別錯誤） | 檢查請求欄位型別 |
| 503 | 未載入模型（`model: "local"`）或推論佇列逾時（>30s） | 呼叫 `POST /load` 或降低並發 |
| 500 | 推論引擎錯誤（串流中 error event） | 查看 `logs/api.log` |
| Connection refused | 伺服器未啟動 | 執行 `run_dev.bat` |

---

## 十五、設計約束與行為說明

**單模型限制：** 一次只能載入一個模型。自動載入新模型前會先卸載舊模型。

**並發推論：** `POST /v1/chat/completions` 請求序列化（單一 semaphore）。同時最多 5 個等待請求，超出立即回 503；等待超過 30 秒回 503。

**context window：** `n_ctx` 在載入時固定，無法在推論中途更改，需重新載入才能生效。

**engine_mode 差異：**

| 特性 | subprocess | binding |
|---|---|---|
| 隔離性 | 模型 crash 不影響主程序 | 同程序，crash 影響主程序 |
| 延遲 | 略高（本地 HTTP） | 較低（直接呼叫） |
| embedding 效能 | 較低 | 較高（推薦高頻使用） |
| tool calling | ✅ 完整支援（b9010+） | ✅ 依版本 |
| id_slot | ✅ 支援 | ❌ 不支援 |

**OOM 自動降層：** `POST /load` 偵測到 VRAM OOM 時，自動以 `[35, 20, 10, 0]` 的 `n_gpu_layers` 序列重試。成功時回應訊息會標注 `[fallback: layers=N]`。

**idle 自動卸載：** 可在 Settings 頁啟用。閒置超過 `idle_unload_seconds`（預設 300s）自動卸載以釋放 VRAM。

**TOCTOU 競態：** 若 `idle_unload_seconds < 60`，`ensure_model_loaded()` 完成後到 `stream()` 開始之間存在微小視窗，idle auto-unload 可能在此期間觸發，導致單次 503 或 SSE error event。重試一次即可解決。正式環境建議設定 `idle_unload_seconds ≥ 300`。

**subprocess crash 偵測：** `_idle_monitor` 每 30 秒輪詢一次 `llama-server.exe` 進程存活狀態。偵測到 crash 後主動清理狀態，`GET /health` 即時反映 `model_loaded: false`。

**chat_format 自動偵測：** 自動載入（`model` 欄位觸發）的模型從 GGUF metadata 中的 `tokenizer.chat_template` 欄位偵測正確的 chat format；手動 `POST /load` 的模型使用 profile 中的 `chat_format` 設定（`"auto"` 同樣觸發自動偵測）。

**mmproj 自動配對：** VLM 模型若 `mmproj_path` 為空，engine 依序嘗試：同前綴匹配 → 資料夾內唯一 mmproj 檔。找到自動使用，找不到以純文字模式啟動。

**models_dir 路徑容錯：** profiles.json 儲存相對路徑。專案資料夾移動後，engine 自動偵測失效路徑並 fallback 到 `<ROOT>/models/`，無需手動修正。