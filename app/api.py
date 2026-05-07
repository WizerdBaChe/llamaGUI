# api.py v2.4 ── FastAPI REST 端點
from __future__ import annotations
import os, time, json, uuid, threading
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
import engine as eng

app = FastAPI(title="LlamaGUI API", version="2.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role:    str
    content: Any

class ResponseFormat(BaseModel):
    type:        str = "text"
    json_schema: dict | None = None

class ChatRequest(BaseModel):
    model:           str   = "local"
    messages:        list[Message]
    temperature:     float = Field(default=0.7,  ge=0.0, le=2.0)
    top_p:           float = Field(default=0.9,  ge=0.0, le=1.0)
    top_k:           int   = Field(default=40,   ge=1)
    repeat_penalty:  float = Field(default=1.1,  ge=0.0)
    max_tokens:      int   = Field(default=2048, ge=1)
    stream:          bool  = False
    response_format: ResponseFormat | None = None

class EmbeddingRequest(BaseModel):
    model: str = "local"
    input: str | list[str]

class LoadRequest(BaseModel):
    profile_name:   str   | None = None
    model_path:     str   | None = None
    # 硬體參數：有填才覆蓋 profile 預設值
    n_ctx:          int   | None = None
    n_gpu_layers:   int   | None = None
    n_batch:        int   | None = None
    n_threads:      int   | None = None
    # 推論參數（存進 profile 作為 session 預設）
    temperature:    float | None = None
    top_p:          float | None = None
    top_k:          int   | None = None
    repeat_penalty: float | None = None
    max_tokens:     int   | None = None
    chat_format:    str   | None = None
    engine_mode:    str   | None = None   # "subprocess" | "binding"

class ProfileSaveRequest(BaseModel):
    name:    str
    profile: dict[str, Any]

class PromptPreviewRequest(BaseModel):
    messages:    list[dict]
    chat_format: str = "chatml"

class HFDownloadRequest(BaseModel):
    url:      str
    filename: str
    repo_id:  str = ""

_download_tasks: dict[str, dict] = {}

# ═════════════════════════════════════════════════════════════════════════════
# 基本端點
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": eng.engine.is_loaded,
            "model_name": eng.engine.stats.model_name}

@app.get("/models")
def list_models():
    files = config.scan_models()
    return {"object": "list",
            "data": [{"id": f, "object": "model", "owned_by": "local"} for f in files]}

@app.get("/v1/models")
def list_models_openai():
    files = config.scan_models()
    now   = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id":       os.path.splitext(os.path.basename(f))[0],
             "object":   "model",
             "created":  now,
             "owned_by": "local"}
            for f in files
        ],
    }

@app.post("/load")
def load_model(req: LoadRequest):
    profile = config.get_profile(req.profile_name)

    if req.model_path:
        path = req.model_path
        # 若傳入的不是有效絕對路徑，嘗試當作 model name 做模糊解析
        if not os.path.isfile(path):
            resolved = eng._find_model_by_name(path)
            if resolved:
                path = resolved
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: '{req.model_path}'. "
                           f"Pass an absolute path or a name matching a .gguf in models/. "
                           f"Use GET /v1/models to list available models."
                )
        profile["model_path"] = path
        
    # 有填才覆蓋，None 表示沿用 profile 原始值
    overrides = {
        "n_ctx":          req.n_ctx,
        "n_gpu_layers":   req.n_gpu_layers,
        "n_batch":        req.n_batch,
        "n_threads":      req.n_threads,
        "temperature":    req.temperature,
        "top_p":          req.top_p,
        "top_k":          req.top_k,
        "repeat_penalty": req.repeat_penalty,
        "max_tokens":     req.max_tokens,
        "chat_format":    req.chat_format,
        "engine_mode":    req.engine_mode,
    }
    for k, v in overrides.items():
        if v is not None:
            profile[k] = v

    ok, msg = eng.engine.load(profile)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "loaded", "message": msg}

@app.post("/unload")
def unload_model():
    return {"status": "unloaded", "message": eng.engine.unload()}

@app.get("/stats")
def get_stats():
    return eng.engine.get_stats()

@app.get("/api/ps")
def api_ps():
    info = eng.engine.get_ps_info()
    return {"models": [info] if info else []}

@app.get("/model-metadata")
def get_model_metadata():
    if not eng.engine.is_loaded: return {"error": "no model loaded"}
    return eng.engine.get_model_metadata()

@app.get("/vram")
def get_vram():
    return eng.get_vram_info()

@app.get("/context-usage")
def context_usage():
    used, max_ctx = eng.engine.get_context_usage()
    pct = round(used / max_ctx * 100, 1) if max_ctx else 0
    return {"used": used, "max": max_ctx, "pct": pct}

@app.post("/prompt-preview")
def prompt_preview(req: PromptPreviewRequest):
    formatted = eng.format_prompt_preview(req.messages, req.chat_format)
    return {"prompt": formatted, "length": len(formatted)}

# ═════════════════════════════════════════════════════════════════════════════
# HuggingFace
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/hf-search")
def hf_search(q: str = Query(..., min_length=1), limit: int = 20):
    return {"results": eng.hf_search(q, limit)}

@app.get("/hf-files")
def hf_files(repo_id: str = Query(...)):
    return {"files": eng.hf_list_gguf_files(repo_id)}

@app.post("/hf-download")
def hf_download_start(req: HFDownloadRequest):
    models_dir = config.get_models_dir()
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, req.filename)
    if os.path.exists(dest):
        raise HTTPException(status_code=409, detail=f"File already exists: {req.filename}")
    task_id   = uuid.uuid4().hex[:8]
    cancel_ev = threading.Event()
    _download_tasks[task_id] = {"progress": 0, "total": 0, "status": "running",
                                 "filename": req.filename, "cancel_event": cancel_ev, "message": ""}
    def _run():
        def _cb(dl, tot):
            _download_tasks[task_id]["progress"] = dl
            _download_tasks[task_id]["total"]    = tot
        ok, msg = eng.hf_download(req.url, dest, _cb, cancel_ev)
        _download_tasks[task_id]["status"]  = "done" if ok else "error"
        _download_tasks[task_id]["message"] = msg
    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "filename": req.filename}

@app.get("/hf-download/{task_id}")
def hf_download_progress(task_id: str):
    t = _download_tasks.get(task_id)
    if not t: raise HTTPException(status_code=404, detail="Task not found")
    total = t["total"]; prog = t["progress"]
    return {"task_id": task_id, "status": t["status"], "progress": prog, "total": total,
            "pct": round(prog / total * 100, 1) if total else 0,
            "filename": t.get("filename", ""), "message": t.get("message", "")}

@app.delete("/hf-download/{task_id}")
def hf_download_cancel(task_id: str):
    t = _download_tasks.get(task_id)
    if not t: raise HTTPException(status_code=404, detail="Task not found")
    t["cancel_event"].set(); t["status"] = "cancelled"
    return {"task_id": task_id, "status": "cancelled"}

# ═════════════════════════════════════════════════════════════════════════════
# Profile CRUD
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/profiles")
def list_profiles_api():
    data = config.get_all()
    return {"active": data.get("active_profile"),
            "profiles": list(data.get("profiles", {}).keys())}

@app.get("/profiles/{name}")
def get_profile_api(name: str): return config.get_profile(name)

@app.post("/profiles")
def save_profile_api(req: ProfileSaveRequest):
    config.save_profile(req.name, req.profile)
    return {"status": "saved", "name": req.name}

@app.delete("/profiles/{name}")
def delete_profile_api(name: str):
    ok = config.delete_profile(name)
    if not ok: raise HTTPException(status_code=400, detail="Cannot delete default profile")
    return {"status": "deleted", "name": name}

@app.post("/profiles/{name}/activate")
def activate_profile_api(name: str):
    ok = config.set_active(name)
    if not ok: raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    return {"status": "activated", "name": name}

# ═════════════════════════════════════════════════════════════════════════════
# Models dir
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/models-dir")
def get_models_dir_api(): return {"models_dir": config.get_models_dir()}

@app.post("/models-dir")
def set_models_dir_api(body: dict):
    path = body.get("path", "")
    if not path: raise HTTPException(status_code=400, detail="path is required")
    config.set_models_dir(path)
    return {"status": "updated", "models_dir": path}

# ═════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible chat completions
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.model and req.model != "local":
        ok, msg = eng.engine.ensure_model_loaded(req.model)
        if not ok:
            raise HTTPException(status_code=404 if "not found" in msg else 503, detail=msg)
    elif not eng.engine.is_loaded:
        raise HTTPException(status_code=503, detail="No model loaded. POST /load first.")

    profile_override = {
        **eng.engine.current_profile,
        "temperature":    req.temperature,
        "top_p":          req.top_p,
        "top_k":          req.top_k,
        "repeat_penalty": req.repeat_penalty,
        "max_tokens":     req.max_tokens,
    }
    messages      = [{"role": m.role, "content": m.content} for m in req.messages]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    rf_dict: dict | None = None
    if req.response_format:
        rf_dict = req.response_format.model_dump(exclude_none=True)

    if req.stream:
        async def event_stream() -> AsyncIterator[str]:
            for token in eng.engine.stream(messages, profile_override, rf_dict):
                is_err = token.startswith("[Error:")
                chunk  = {"id": completion_id, "object": "chat.completion.chunk",
                          "created": int(time.time()), "model": req.model,
                          "choices": [{"index": 0, "delta": {"content": token},
                                       "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                if is_err: yield "data: [DONE]\n\n"; return
            end = {"id": completion_id, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": req.model,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(end)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        full = eng.engine.generate(messages, profile_override, rf_dict)
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e))
    stats = eng.engine.get_stats()
    return {"id": completion_id, "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": full},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens":     stats.get("prompt_tokens", 0),
                      "completion_tokens": stats.get("completion_tokens", 0),
                      "total_tokens":      stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)}}

# ═════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible embeddings
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    if req.model and req.model != "local":
        ok, msg = eng.engine.ensure_model_loaded(req.model)
        if not ok:
            raise HTTPException(status_code=404 if "not found" in msg else 503, detail=msg)
    elif not eng.engine.is_loaded:
        raise HTTPException(status_code=503, detail="No model loaded. POST /load first.")

    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    try:
        vectors = eng.engine.embed(texts)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    return {
        "object": "list", "model": req.model,
        "data": [{"object": "embedding", "index": i, "embedding": vec}
                 for i, vec in enumerate(vectors)],
        "usage": {"prompt_tokens":  sum(len(t.split()) for t in texts),
                  "total_tokens":   sum(len(t.split()) for t in texts)},
    }
