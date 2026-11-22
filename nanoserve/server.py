"""OpenAI-compatible HTTP server.

    python -m nanoserve.server --port 8000 [--speculative-k 4]

**Threading model.** The engine is synchronous and owns the GPU. It runs on one dedicated
thread that loops: take newly arrived requests and aborts -> `engine.step()` -> hand each
request's new tokens back to the asyncio event loop. HTTP handlers never touch the engine
directly; they put a request on a queue and await their own per-request asyncio queue.
So a slow client cannot stall the batch, and the event loop never blocks on CUDA.

A client that disconnects mid-stream is aborted, which frees its KV blocks immediately
rather than letting it generate to max_tokens for nobody.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from .engine import EngineConfig, LLMEngine
from .model import resolve_model_dir
from .sequence import SamplingParams


# ---------------------------------------------------------------- engine thread
@dataclass
class _Pending:
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    out: asyncio.Queue
    loop: asyncio.AbstractEventLoop


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._aborts: queue.Queue = queue.Queue()
        self._streams: dict[str, _Pending] = {}
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="engine")
        self._thread.start()

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> tuple[str, asyncio.Queue]:
        rid = f"cmpl-{uuid.uuid4().hex[:16]}"
        q: asyncio.Queue = asyncio.Queue()
        self._inbox.put(_Pending(rid, prompt_ids, params, q, asyncio.get_running_loop()))
        self._wake.set()
        return rid, q

    def abort(self, request_id: str) -> None:
        self._aborts.put(request_id)
        self._wake.set()

    def _run(self) -> None:
        eng = self.engine
        while True:
            if not eng.has_unfinished():
                self._wake.wait()
            self._wake.clear()
            while not self._inbox.empty():
                p: _Pending = self._inbox.get()
                try:
                    eng.add_request(p.prompt_ids, p.params, p.request_id)
                    self._streams[p.request_id] = p
                except ValueError as e:
                    p.loop.call_soon_threadsafe(p.out.put_nowait, e)
            while not self._aborts.empty():
                rid = self._aborts.get()
                eng.abort(rid)
                self._streams.pop(rid, None)
            if not eng.has_unfinished():
                continue
            try:
                outputs = eng.step()
            except Exception as e:  # surface to every waiting client rather than hang them
                for p in self._streams.values():
                    p.loop.call_soon_threadsafe(p.out.put_nowait, e)
                raise
            for o in outputs:
                p = self._streams.get(o.request_id)
                if p is None:
                    continue
                p.loop.call_soon_threadsafe(p.out.put_nowait, o)
                if o.finished:
                    del self._streams[o.request_id]
                    eng.seqs.pop(o.request_id, None)


# ---------------------------------------------------------------- text helpers
class Detokenizer:
    """Incremental detokenisation. Byte-level BPE can split one UTF-8 character across
    tokens, so decoding a prefix may end in U+FFFD. Hold that tail back until the rest of
    the character arrives, otherwise clients see mojibake mid-stream."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.ids: list[int] = []
        self.sent = 0

    def push(self, new_ids: list[int], final: bool = False) -> str:
        self.ids.extend(new_ids)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�") and not final:
            return ""
        delta = text[self.sent:]
        self.sent = len(text)
        return delta


def qwen_chat_prompt(messages: list[dict]) -> str:
    """Qwen2.5's ChatML template, written out so the server needs no Jinja dependency."""
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + messages
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


# ---------------------------------------------------------------- API schema
class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=0)
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=0)
    stream: bool = False


def build_app(engine: LLMEngine, tokenizer: Tokenizer, model_name: str) -> FastAPI:
    app = FastAPI(title="nanoserve")
    aeng = AsyncEngine(engine)
    app.state.aengine = aeng

    async def run(prompt_ids, params, request: Request, stream: bool, chat: bool):
        rid, q = aeng.submit(prompt_ids, params)
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        detok = Detokenizer(tokenizer)

        def chunk(text: str, finish: str | None) -> dict:
            if chat:
                choice = {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}
                return {"id": rid, "object": obj + ".chunk", "created": created, "model": model_name, "choices": [choice]}
            return {"id": rid, "object": obj, "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": finish}]}

        if stream:
            async def gen():
                finished = False
                try:
                    while True:
                        o = await q.get()
                        if isinstance(o, Exception):
                            finished = True
                            yield f"data: {json.dumps({'error': str(o)})}\n\n"
                            return
                        text = detok.push(o.new_token_ids, final=o.finished)
                        if text or o.finished:
                            yield f"data: {json.dumps(chunk(text, o.finish_reason))}\n\n"
                        if o.finished:
                            finished = True
                            yield "data: [DONE]\n\n"
                            return
                finally:
                    # Runs on client disconnect too (the generator is cancelled).
                    if not finished:
                        aeng.abort(rid)
            return StreamingResponse(gen(), media_type="text/event-stream")

        ids: list[int] = []
        while True:
            if await request.is_disconnected():
                aeng.abort(rid)
                raise HTTPException(499, "client disconnected")
            try:
                o = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(o, Exception):
                raise HTTPException(400, str(o))
            ids += o.new_token_ids
            if o.finished:
                break
        text = tokenizer.decode(ids, skip_special_tokens=True)
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                 "total_tokens": len(prompt_ids) + len(ids)}
        if chat:
            body = {"id": rid, "object": obj, "created": created, "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": o.finish_reason}], "usage": usage}
        else:
            body = {"id": rid, "object": obj, "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": o.finish_reason}], "usage": usage}
        return JSONResponse(body)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        ids = req.prompt if isinstance(req.prompt, list) else tokenizer.encode(req.prompt).ids
        params = SamplingParams(req.max_tokens, req.temperature, req.top_p, req.top_k, req.ignore_eos)
        return await run(ids, params, request, req.stream, chat=False)

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request):
        prompt = qwen_chat_prompt([m.model_dump() for m in req.messages])
        ids = tokenizer.encode(prompt).ids
        params = SamplingParams(req.max_tokens, req.temperature, req.top_p, req.top_k)
        return await run(ids, params, request, req.stream, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nanoserve"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        e, s = engine, engine.stats
        return {
            "running": len(e.scheduler.running),
            "waiting": len(e.scheduler.waiting),
            "kv_blocks_total": e.allocator.num_blocks,
            "kv_blocks_free": e.allocator.num_free,
            "kv_capacity_tokens": e.kv_capacity_tokens,
            "steps": s.steps,
            "prefill_tokens": s.prefill_tokens,
            "generated_tokens": s.decode_tokens,
            "preemptions": e.scheduler.num_preemptions,
            "spec_acceptance_rate": round(s.acceptance_rate, 4),
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--speculative-k", type=int, default=0)
    ap.add_argument("--no-skinny-gemm", action="store_true")
    a = ap.parse_args()

    import uvicorn

    mdir = resolve_model_dir(a.model)
    engine = LLMEngine(EngineConfig(
        model=str(mdir), max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
        speculative_k=a.speculative_k, use_skinny_gemm=not a.no_skinny_gemm,
    ))
    engine.warmup()
    tok = Tokenizer.from_file(str(Path(mdir) / "tokenizer.json"))
    print(f"KV pool: {engine.allocator.num_blocks} blocks = {engine.kv_capacity_tokens} tokens "
          f"({engine.kv.nbytes / 2**20:.0f} MiB); GPU {torch.cuda.get_device_name()}", flush=True)
    uvicorn.run(build_app(engine, tok, a.model), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
