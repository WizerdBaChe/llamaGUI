# llamaGUI 更新決策日誌 (Design Decision Log / ADR)

檔案版本：v1.1 (2026-05-11)  
狀態：Active  
最後更新：2026-05-11

## 決策模板說明

每個決策用此格式，聚焦「為何選擇此方案」。  
優先級標示：🔴 最高 / 🟡 高 / 🟢 中 / ⚪ 低

---
決策ID：DD-011  
標題：VLM 多模態視覺支援：ContentPart 型別宣告 + 模型能力偵測 + mmproj 自動配對  
狀態：Accepted  
日期：2026-05-12  
優先級：🟡 高  
來源分析：業界高星專案（llama-server、Ollama、OpenAI SDK）介面分析 + 系統工程三維度評估  

### 情境

llamaGUI 後端已有 mmproj 載入骨架（SubprocessEngine `--mmproj` 參數、BindingEngine `_build_clip_handler()`），但存在三個結構性缺口：

1. **靜默失敗**：`_encode_multimodal_messages()` 對 list 型 content 直接取字串，圖片資訊被丟棄，純文字模型收到帶圖 message 時不會 error，而是無聲地忽略圖片。
2. **無能力自我宣告**：engine 無任何欄位說明當前載入的模型是否支援視覺，外部呼叫端（UI / API 客戶端）無從判斷是否可送圖。
3. **content 型別未宣告**：`Message.content: Any` 在 Pydantic model 層完全不設防，格式錯誤的 image_url（如裸字串而非物件）會傳到 llama-server 才報錯，錯誤邊界過晚。

業界標準（OpenAI API、llama-server `/v1/chat/completions`、Ollama OpenAI 相容端點）一致使用 **content parts** 格式（`list[{"type":"text","text":...} | {"type":"image_url","image_url":{"url":...,"detail":"auto"}}]`）作為多模態 message 結構。模型視覺能力則由 `GET /props`（llama-server）或 model metadata 宣告。

已知限制（上游 llama-server bug）：載入 `--mmproj` 時，KV cache slot 持久化與 prefix cache 會停用，因此 DD-002（`id_slot`）在 VLM 模式下實際無效，需在 API 文件中標注互斥關係。

### 決策

採**方案 C（完整實作）**，三層同步落地：

**Layer 1 — Pydantic 型別宣告（api.py）**  
新增 `ContentPartText`、`ContentPartImage`、`ImageURL` 三個 Pydantic model，`Message.content` 改為 `str | list[ContentPartText | ContentPartImage]`。這是純型別宣告，不改變執行期行為，但讓 FastAPI 的自動 422 驗證替換掉「傳到 llama-server 才報錯」的問題。

```python
class ImageURL(BaseModel):
    url:    str
    detail: Literal["auto", "low", "high"] = "auto"

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class ContentPartImage(BaseModel):
    type:      Literal["image_url"]
    image_url: ImageURL

class Message(BaseModel):
    role:    str
    content: str | list[ContentPartText | ContentPartImage]
```

**Layer 2 — 能力偵測（engine.py）**  
`LlamaEngine.get_ps_info()` 新增 `has_vision: bool`，判斷依據：
- SubprocessEngine：呼叫 `GET /props`，取 `vision` 欄位（llama-server 原生旗標）
- BindingEngine：檢查 `self.llm.chat_handler is not None`  
- 若查詢失敗，fallback 為 `mmproj_path != ""` 的判斷  

新增 `/v1/chat/completions` 前置檢查：若 message 含 `image_url` part 且 `has_vision == False`，回傳 `400 {"detail": "model does not support vision — load a multimodal model with mmproj"}`，而非靜默忽略。

**Layer 3 — mmproj 自動配對（engine.py）**  
`SubprocessEngine.load()` 中，若 `mmproj_path` 為空，自動掃描主模型所在資料夾，依以下優先順序尋找 mmproj：
1. 同前綴：`{model_stem}-mmproj*.gguf`
2. 通用名：`mmproj*.gguf`（資料夾內唯一 mmproj 時）
3. 找不到：不報錯，只記 `log.info()`，以純文字模式啟動

**_encode_multimodal_messages() 修正**：保留 list 型 content 的完整結構（`content` 為 list 時直接透傳，不做字串轉換），讓 llama-server 原生處理多模態 message 組裝。

### 後果

- ✅ 錯誤邊界前移至 API 入口（Pydantic 422），不再依賴 llama-server 回傳不友善的錯誤
- ✅ `has_vision` 旗標讓 UI 可根據模型能力動態顯示/隱藏圖片上傳按鈕
- ✅ mmproj 自動配對大幅降低使用者設定門檻（Gemma 3 / Qwen2-VL / LLaVA 下載後直接可用）
- ✅ ContentPart 結構預留 `detail` 欄位，支援未來 `"low"/"high"` 解析度控制擴充
- ⚠️ DD-002（`id_slot`）在 VLM 模式（mmproj 載入時）實際無效，兩者互斥需在 API 文件標注
- ⚠️ `GET /props` 查詢為同步 HTTP call，`get_ps_info()` 會增加約 1ms 額外延遲（可接受）
- ⚠️ llama-cpp-python binding mode 的視覺能力 API（`chat_handler`）依賴 llama-cpp-python 版本，需在 CHANGELOG 標注最低支援版本
- ❌ UI 層的圖片上傳元件（拖放上傳 → base64 → ContentPartImage 組裝）不在本決策範疇，列為後續 UI-001

### 變更歷史

決策ID：DD-010  
標題：批次實作 DD-001 至 DD-009 對應程式改動  
狀態：Accepted  
日期：2026-05-12  
優先級：N/A（實作紀錄）  
來源分析：系統工程缺口分析批次執行  

### 情境

承接 DD-001 至 DD-009 的九條 Proposed 決策，統一執行為 engine.py v2.4 與 api.py v2.5 的完整覆蓋更新。  
改動涉及推論層、API 介面層、資源管理層、穩健性層共四個維度，且各條決策在程式碼層互不衝突，故採批次合併輸出而非逐一 PR。

### 決策

一次性輸出兩個完整覆蓋檔案（非差分 patch），原因如下：
- 改動跨越同一個 class 內的多個方法（`stream()`、`generate()`、`load()`、`_idle_monitor()`、`__init__()`），差分替換易因縮排錯位或上下文錯位造成靜默破壞
- Pydantic model 的欄位新增需要與 `profile_override` 建構段同步，兩者分屬 api.py 不同位置，整檔覆蓋可確保一致性
- 語法驗證（`ast.parse()`）在輸出前已通過，風險可控

**各 ADR 落地摘要：**

| ADR | 落地檔案 | 落地位置 |
|-----|----------|----------|
| DD-001 | engine.py | `SubprocessEngine.load()` cmd 建構；`LoadRequest` 新增 `draft_model_path` |
| DD-002 | engine.py + api.py | `ChatRequest.id_slot`；`SubprocessEngine.stream()` body 條件透傳 |
| DD-003 | api.py | 新增 `CompletionRequest` + `POST /v1/completions`（subprocess proxy + binding `create_completion`） |
| DD-004 | engine.py | `LlamaEngine.load()` OOM 關鍵字偵測 + `fallback_gpu_layers` 降層重試序列 |
| DD-005 | engine.py | `LlamaEngine._waiting` 計數器 + `_waiting_lock` + `MAX_WAITING=5`；`stream()`/`generate()` 前置 guard |
| DD-006 | engine.py | `SubprocessEngine.stream()` / `BindingEngine.stream()` finally 各加一行 `log.info()` 摘要 |
| DD-007 | engine.py + api.py | `ChatRequest` 新增 `stop`/`seed`/`presence_penalty`/`frequency_penalty`/`logprobs`/`top_logprobs`；兩個 backend stream() 條件透傳 |
| DD-008 | engine.py + api.py | `ChatRequest` 新增 `tools`/`tool_choice`；兩個 backend stream() 條件透傳；binding mode 的 `id_slot` 明確略過並附說明 |
| DD-009 | engine.py | `_idle_monitor()` 加入 `proc.poll() is not None` crash 偵測 + `log.error()` + 主動呼叫 `_stop_server()` 清理 |

### 後果

- ✅ DD-001 至 DD-009 全數狀態更新為 Accepted
- ✅ 兩檔語法驗證通過（`ast.parse()` 無錯誤）
- ✅ api.py 版本號升至 v2.5.0
- ⚠️ DD-003 的 `/v1/completions` binding mode streaming 路徑尚未實作（stream=False 強制），已知限制
- ⚠️ DD-007 的 `logprobs`/`top_logprobs` 在 binding mode 需視 llama-cpp-python 版本，無法靜態保證支援
- ❌ DD-002 binding mode 的 `id_slot` 透傳仍為已知缺口，兩種 backend 行為不一致，待後續版本補齊


- v1（2026-05-12）：初稿，Accepted

決策ID：DD-009  
標題：新增 Subprocess 進程存活主動監控（Heartbeat）  
狀態：Proposed  
日期：2026-05-11  
優先級：🔴 最高  
來源分析：Task-2 Bug 審查 / 系統工程缺口分析  

### 情境

`SubprocessEngine.is_loaded` 目前僅依賴 `self.proc.poll() is None` 判斷進程是否存活。若 `llama-server.exe` 因 VRAM OOM、CUDA 錯誤等原因 crash，`is_loaded` 仍會在 `poll()` 返回前短暫維持 `True`。下一次推論請求到達時才會發現服務已死，此為靜默故障（silent failure），使用者無法即時得知。`IdleMonitor` 的 30 秒輪詢目前只檢查 idle 時間，不做進程健康確認。

替代考量：
- 方案 A：只在推論前額外調用一次 `poll()` — 無法覆蓋 crash 到請求之間的空窗期
- 方案 B：在 `_idle_monitor` 輪詢中加入 `GET /health` heartbeat — 可主動偵測，成本低
- 方案 C：額外啟動獨立 watchdog thread — 過度設計，與現有 idle monitor 職責重疊

### 決策

在現有 `_idle_monitor()` 的 30 秒輪詢內，對 subprocess mode 加入 `GET /health` 存活檢查。偵測到 `proc.poll() is not None`（進程已結束）時，主動呼叫 `_stop_server()` 清理狀態、寫入 `log.error()`，並將 `is_loaded` 還原為 `False`，使 UI 能即時顯示模型已卸載。

### 後果

- ✅ 靜默故障轉為可觀測事件，log 有明確紀錄
- ✅ 使用者下次嘗試推論前即可得知需重新載入
- ✅ 與現有 idle monitor 共用同一執行緒，零額外線程開銷
- ⚠️ heartbeat 每 30 秒一次，仍有最長 30 秒的感知延遲（可接受）
- ❌ 未解決：crash root cause 分析（需額外 stderr 捕捉）

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-008  
標題：Tool Calling / Function Calling 欄位透傳支援  
狀態：Proposed  
日期：2026-05-11  
優先級：🔴 最高  
來源分析：系統工程缺口分析 — API 介面層  

### 情境

OpenAI `/v1/chat/completions` 標準包含 `tools`（工具定義陣列）與 `tool_choice`（工具選擇策略）兩個欄位，為 Agent 架構的核心依賴。llama-server b9010+ 原生支援此規格。目前 `ChatRequest` Pydantic model 未定義這兩欄位，外部呼叫端（如 LangChain、AutoGen）若帶入 `tools` 參數，FastAPI 會靜默丟棄，底層 llama-server 永遠收不到工具定義，`finish_reason: "tool_calls"` 路徑也無法觸發。

替代考量：
- 方案 A：完整實作工具執行邏輯（在 engine.py 內處理 tool 呼叫循環）— 範疇過大，屬應用層非引擎層職責
- 方案 B：僅做透傳（pass-through），不處理工具執行邏輯 — 符合「引擎只做推論」的職責邊界，最小可行實作
- 方案 C：不實作，依賴外部框架自行處理 — 打破 OpenAI 相容性承諾

### 決策

採方案 B：在 `ChatRequest` 加入 `tools: list[dict] | None = None` 與 `tool_choice: str | dict | None = None`，並在 `SubprocessEngine.stream()` 的 `body` 建構段透傳給 llama-server。response 結構不做額外解析，維持原始 JSON 透傳。工具執行邏輯（tool → result → re-inference 循環）由呼叫端負責。

### 後果

- ✅ 與 OpenAI SDK、LangChain、AutoGen 達成結構相容
- ✅ 實作成本極低（Pydantic 2 行 + body 透傳 2 行）
- ✅ `finish_reason: "tool_calls"` 可正確傳回給呼叫端
- ⚠️ Binding mode 需確認 `llama-cpp-python` 對應版本的 `create_chat_completion()` 是否支援 `tools` 參數
- ❌ 未解決：GUI Chat tab 的工具呼叫視覺化（屬 UI 層待辦）

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-007  
標題：補齊 `/v1/chat/completions` 缺失的 OpenAI 標準推論參數  
狀態：Proposed  
日期：2026-05-11  
優先級：🔴 最高  
來源分析：系統工程缺口分析 — API 介面層  

### 情境

`ChatRequest` 目前只定義 `temperature`、`top_p`、`top_k`、`repeat_penalty`、`max_tokens`。OpenAI spec 及 llama-server 均支援以下欄位，外部呼叫端若帶入這些參數，當前實作會靜默丟棄：

| 欄位 | 用途 |
|------|------|
| `stop` | 停止序列（list[str]），控制生成終止條件 |
| `seed` | 亂數種子，確保可重現輸出 |
| `presence_penalty` | 抑制已出現話題的重複傾向 |
| `frequency_penalty` | 按 token 出現頻率懲罰重複 |
| `logprobs` | 回傳 token 機率（布林） |
| `top_logprobs` | 回傳 top-N token 機率（int） |

### 決策

在 `ChatRequest` 逐一加入上述欄位（均設 `None` 為預設值），並在 `SubprocessEngine.stream()` body 建構段以 `if field is not None: body[field] = value` 條件透傳，避免影響未傳入此參數的現有請求行為。`seed`、`stop` 在 BindingEngine 的 `create_chat_completion()` 中同步支援。

### 後果

- ✅ 提升 OpenAI SDK 相容性，現有整合工具不需特殊處理
- ✅ `stop` 對 RAG/Agent 場景（精確控制生成邊界）有實用價值
- ✅ `seed` 對測試與可重現 benchmark 有直接幫助
- ⚠️ `logprobs` 在 binding mode 需額外測試（llama-cpp-python 支援程度待確認）
- ❌ `n`（多候選回答）暫不支援，llama-server subprocess 回應格式需額外處理

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-006  
標題：加入推論完成後的統計摘要 log  
狀態：Proposed  
日期：2026-05-11  
優先級：🟡 高  
來源分析：系統工程缺口分析 — 可觀測性層  

### 情境

`SubprocessEngine.stream()` 與 `BindingEngine.stream()` 在 `finally` 區塊更新 `self.stats`，但不主動輸出任何 log。若要了解推論效能（tps、耗時、token 數），需在推論完成後主動呼叫 `GET /stats`。在高頻呼叫場景（批次處理、自動化測試）中，log 檔案完全無法用於事後效能分析或異常追蹤。

替代考量：
- 方案 A：只在 `generate()` 完成後 log — 無法覆蓋 streaming 場景
- 方案 B：在兩個 backend 的 `finally` 區塊各加一行 `log.info()` — 完整覆蓋，成本極低
- 方案 C：接入外部 metrics 系統（Prometheus）— 過度設計，本決策範疇不含

### 決策

在 `SubprocessEngine.stream()` 與 `BindingEngine.stream()` 的 `finally` 區塊，於 stats 更新後加入一行結構化 `log.info()`，格式為：

```
stream() done | model={model_name} | tokens={count} | tps={tps:.1f} | elapsed={elapsed:.2f}s | prompt_tokens={prompt_tokens}
```

### 後果

- ✅ 每次推論留下可搜尋的效能紀錄，log grep 即可分析
- ✅ 與已有的 `log.error()` 風格一致，無額外依賴
- ✅ 實作成本：2 行程式碼
- ⚠️ 高頻呼叫時 log 量會增加，log rotation 策略需確認（已有 `api.log`，建議共用）

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-005  
標題：請求佇列深度上限保護（Queue Depth Guard）  
狀態：Proposed  
日期：2026-05-11  
優先級：🟡 高  
來源分析：系統工程缺口分析 — 穩健性層  

### 情境

當 `Semaphore(1)` 被占用時，後續請求會在 `_req_lock.acquire(timeout=30)` 處阻塞，消耗 FastAPI/uvicorn 的 async worker。N 個並發請求同時阻塞等待，若 30 秒後全部超時，會同時釋放、造成突發負載。目前沒有任何機制限制等待佇列的深度，理論上可以讓數十個請求同時阻塞。

替代考量：
- 方案 A：換成 `asyncio.Semaphore` + async/await — 需重構 engine.py 為 async，成本高
- 方案 B：加入計數器，超過上限直接回傳 503 — 最小侵入性，不改變現有同步架構
- 方案 C：不處理，依賴 uvicorn worker 自然限制 — 不可控，依賴外部行為

### 決策

在 `LlamaEngine` 加入 `_waiting: int = 0` 計數器與 `MAX_WAITING: int = 5` 上限常數。在 `stream()` 與 `generate()` 取得 semaphore 前，先判斷 `_waiting >= MAX_WAITING` 即直接回 `TimeoutError`（stream 則 yield error string），否則 `_waiting += 1`，finally 時 `_waiting -= 1`。

### 後果

- ✅ 防止過多請求堆積、worker 耗盡
- ✅ 呼叫端能立即收到 503，不必等待 30 秒 timeout
- ✅ `MAX_WAITING` 可設為 settings.json 的可設定項（未來擴充）
- ⚠️ 計數器操作在多執行緒環境需用 `threading.Lock` 保護（非原子操作）
- ❌ 未解決：不公平排隊（先到先服務無優先級，長請求可能餓死短請求）

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-004  
標題：VRAM OOM 自動降層重試（Auto-Fallback n_gpu_layers）  
狀態：Proposed  
日期：2026-05-11  
優先級：🟡 高  
來源分析：系統工程缺口分析 — 資源管理層  

### 情境

`BindingEngine.load()` 在 OOM 時 catch exception 後直接回傳 `(False, str(e))`，使用者只看到錯誤訊息。普通使用者不知道應調低 `n_gpu_layers`，通常需要反覆手動嘗試。SubprocessEngine 目前不直接面對 OOM（由 llama-server 處理），但 llama-server 若因 VRAM 不足啟動失敗，現有邏輯同樣無降級行為。

替代考量：
- 方案 A：只在 UI 層提示建議值 — 不解決根本問題，使用者仍需手動操作
- 方案 B：在 `BindingEngine.load()` 的 except 區塊中，自動以遞減 `n_gpu_layers` 重試 — 對 binding mode 有效
- 方案 C：方案 B 加上 SubprocessEngine 的 startup 失敗偵測，同步支援 — 完整覆蓋

### 決策

採方案 C：在 `LlamaEngine.load()` 加入降層重試邏輯。偵測到 load 失敗且錯誤訊息含 OOM / CUDA / memory 關鍵字時，自動以 `[35, 20, 10, 0]` 的 `n_gpu_layers` 序列重試（0 = 純 CPU）。每次重試前 log `WARNING` 說明降層原因與嘗試值。若所有層數均失敗，回傳最後一個錯誤。成功載入時在回傳訊息中標注實際使用的層數。

### 後果

- ✅ 大幅降低普通使用者的設定門檻，模型載入成功率提升
- ✅ 降層序列可設定為 profile 中的 `fallback_gpu_layers: list[int]`
- ⚠️ 重試會增加 load 總時間（最壞情況：4 次嘗試 × ~15s = ~60s）
- ⚠️ 需精確判斷 OOM 錯誤關鍵字，避免非 OOM 錯誤也觸發降層（浪費時間）
- ❌ 未解決：降層後的效能預期管理（CPU 推論速度落差大，需告知使用者）

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-003  
標題：新增 `/v1/completions` Text Completion 端點  
狀態：Proposed  
日期：2026-05-11  
優先級：🟢 中  
來源分析：系統工程缺口分析 — API 介面層  

### 情境

OpenAI API 規格中，`/v1/completions`（非 chat，裸 prompt 補全）與 `/v1/chat/completions` 是分開的端點。目前 llamaGUI 只實作了 chat 端點。部分工具鏈（舊版 LangChain、某些 fine-tune 測試腳本、PromptLayer）仍使用 completion 端點，無法直接對接。llama-server 原生支援此端點。

替代考量：
- 方案 A：在 api.py 加入 `/v1/completions` → 透傳給 llama-server 對應端點 — 適用 subprocess mode
- 方案 B：在 BindingEngine 實作裸 prompt 補全（`llm.create_completion()`）— 適用 binding mode
- 方案 C：用 system prompt 包裝後轉 chat endpoint — 不符合 spec，行為語意不同

### 決策

採方案 A + B：在 api.py 新增 `POST /v1/completions` 端點，定義 `CompletionRequest` Pydantic model（含 `prompt: str | list[str]`、`max_tokens`、`temperature`、`stop`、`stream`、`echo`、`suffix`）。SubprocessEngine 透傳給 llama-server `/v1/completions`；BindingEngine 呼叫 `llm.create_completion()`。

### 後果

- ✅ 提升舊版工具鏈相容性
- ✅ 裸 prompt 對 fine-tune 模型的行為測試更直觀
- ⚠️ streaming response 格式與 chat 略有差異（`choices[].text` 非 `choices[].delta.content`），需獨立實作 SSE 序列化
- ❌ 未解決：`echo`、`suffix`、`logprobs` 等進階欄位在 binding mode 的支援程度需逐一驗證

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-002  
標題：KV Cache Prefix Slot 管理透傳  
狀態：Proposed  
日期：2026-05-11  
優先級：🟢 中  
來源分析：系統工程缺口分析 — 推論功能層  

### 情境

llama-server 支援 `id_slot` 欄位（在請求 body 中指定 slot 編號），可讓相同 system prompt 的連續對話重用同一 KV cache slot，避免每次推論重新計算 prompt 的注意力矩陣。對 RAG pipeline、多輪長對話、批次任務有顯著加速效果。`--slot-save-path` 參數可將 slot 狀態持久化至磁碟，跨重啟復用。目前 engine.py 完全不管理 slot，每次推論都是全新 context。

替代考量：
- 方案 A：只透傳 `id_slot` 欄位，slot 生命週期由呼叫端管理 — 最小侵入性
- 方案 B：在 LlamaEngine 實作 session → slot 的映射表，自動分配 slot — 完整封裝，對呼叫端透明
- 方案 C：暫不實作，此為 subprocess mode 專屬功能，binding mode 無對應 API — 推遲至需求明確

### 決策

採方案 A 優先：在 `ChatRequest` 加入可選欄位 `id_slot: int | None = None`，SubprocessEngine 若收到此欄位則透傳至 llama-server body。方案 B 的 session-slot 映射表作為後續迭代目標，在 README 中標注為 roadmap 項目。Binding mode 暫時跳過（記錄為已知限制）。

### 後果

- ✅ 進階使用者（RAG 開發者）可立即利用 KV cache 重用能力
- ✅ 最小侵入性，不改變現有 session 模型
- ⚠️ 呼叫端需自行管理 slot 生命週期，有 slot 洩漏風險（未明確釋放）
- ❌ Binding mode 無對應支援，兩種 mode 行為不一致，需在 API 文件中標注

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

決策ID：DD-001  
標題：Speculative Decoding 參數暴露至 Load Profile  
狀態：Proposed  
日期：2026-05-11  
優先級：⚪ 低  
來源分析：系統工程缺口分析 — 推論功能層  

### 情境

llama-server 支援 `--draft-model` 參數啟動 speculative decoding（draft 模型預測多個 token，main 模型批次驗證），可在高 batch size 場景下顯著降低生成 latency（理論加速 2~4×）。目前 `LoadRequest` 和 profile schema 均不包含此參數，GUI 也無對應設定項。此功能需要使用者同時持有 main model 與 draft model（通常是相同架構的小型量化版），使用門檻相對高，普通使用者需求低。

替代考量：
- 方案 A：在 profile 加入 `draft_model_path: str | None` 與 `draft_gpu_layers: int | None`，傳入 `--draft-model` 啟動參數 — 完整支援，但需 UI 對應設定項
- 方案 B：只在 subprocess 的 cmd 建構段預留透傳位置，不暴露給 GUI — 開發者可透過直接編輯 profile JSON 使用
- 方案 C：推遲至 v4，待 llama-server 的 speculative decoding API 穩定後再實作 — 保守策略

### 決策

採方案 B：在 `LoadRequest` 與 profile schema 加入 `draft_model_path: str | None = None`，SubprocessEngine 若偵測到有效路徑則加入 `--draft-model` 啟動參數。GUI Settings 頁暫不新增對應 UI 元件（方案 B），僅供進階使用者透過 API 或直接編輯 profile.json 使用。

### 後果

- ✅ 進階使用者可透過 API 啟用 speculative decoding，無需修改程式碼
- ✅ 實作成本低（profile + LoadRequest 各加 1 欄位，cmd 建構加 3 行）
- ⚠️ Binding mode 的 speculative decoding 支援程度待確認（llama-cpp-python 接口不穩定）
- ❌ GUI 無對應入口，普通使用者無法發現此功能（可接受，屬進階功能）
- ❌ Draft model 版本與 main model 的相容性驗證邏輯暫缺

### 變更歷史

- v1（2026-05-11）：初稿，Proposed

---

## 歷史決策（Archive，按時間升序）

> 本文件 v1.0 建立時無歷史決策。前序工程決策（engine.py v2.3 架構、subprocess/binding 雙 backend 選擇等）待後續補錄。

