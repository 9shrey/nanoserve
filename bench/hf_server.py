"""Baseline: the "FastAPI + model.generate()" server most tutorials build.

One request at a time behind a lock (generate() has no notion of joining a running batch),
streaming via TextIteratorStreamer. Same OpenAI-style /v1/completions interface as
nanoserve, so the load generator drives both identically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from nanoserve.model import resolve_model_dir


class Req(BaseModel):
    prompt: str | list[int]
    max_tokens: int = 128
    stream: bool = True
    ignore_eos: bool = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    d = resolve_model_dir()
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16).cuda().eval()
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(r: Req):
        ids = r.prompt if isinstance(r.prompt, list) else tok(r.prompt).input_ids
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(max_new_tokens=r.max_tokens, do_sample=False, streamer=streamer, pad_token_id=0,
                  repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
        if r.ignore_eos:
            kw["min_new_tokens"] = r.max_tokens

        def work():
            with lock, torch.no_grad():
                model.generate(torch.tensor([ids], device="cuda"), **kw)

        threading.Thread(target=work, daemon=True).start()

        # One pump thread per request moves streamer output onto an asyncio queue. A first
        # version used loop.run_in_executor(next, ...) on the shared default pool (16 threads);
        # with 40 queued requests each parking a pool thread, the request actually generating
        # could not get a thread to deliver its tokens, which inflated this baseline's TPOT p99
        # to ~1.9 s. The fix keeps the baseline honest: sequential, but not artificially slow.
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def pump():
            for text in streamer:
                loop.call_soon_threadsafe(q.put_nowait, text)
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=pump, daemon=True).start()

        async def gen():
            while True:
                text = await q.get()
                if text is None:
                    break
                if text:
                    yield f"data: {json.dumps({'choices': [{'text': text}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
