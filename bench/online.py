"""Online serving benchmark: Poisson arrivals against a live HTTP server, streaming.

    python bench/online.py --arm nanoserve --rates 0.5 1 2 4

Per request it records time-to-first-token (TTFT), time-per-output-token after the first
(TPOT) and end-to-end latency, and reports p50/p99 of each plus achieved output tokens/s.
Arrivals are Poisson (exponential gaps) with a fixed seed, so every arm and every rate gets
the same request list arriving at the same offsets.

Arms (each launches its own server process, so memory is clean):
  nanoserve       continuous batching, max_num_seqs=32
  nanoserve-seq   the same engine forced to one request at a time (max_num_seqs=1):
                  isolates what continuous batching buys, holding the kernels fixed
  hf-server       FastAPI + model.generate(), one request at a time (bench/hf_server.py)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import random_workload  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

ARMS = {
    "nanoserve": [PY, "-m", "nanoserve.server", "--port", "{port}", "--max-num-seqs", "32"],
    "nanoserve-seq": [PY, "-m", "nanoserve.server", "--port", "{port}", "--max-num-seqs", "1"],
    "hf-server": [PY, str(ROOT / "bench" / "hf_server.py"), "--port", "{port}"],
}


async def one(client: httpx.AsyncClient, url: str, req, delay: float, t0: float) -> dict:
    await asyncio.sleep(max(0.0, t0 + delay - time.perf_counter()))
    sent = time.perf_counter()
    first = None
    body = {"prompt": req.prompt_ids, "max_tokens": req.max_tokens, "stream": True, "ignore_eos": True}
    async with client.stream("POST", url + "/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if first is None and json.loads(line[6:])["choices"][0].get("text"):
                first = time.perf_counter()
    end = time.perf_counter()
    first = first or end
    return {"ttft": first - sent, "e2e": end - sent, "n_out": req.max_tokens,
            "tpot": (end - first) / max(1, req.max_tokens - 1), "end": end}


async def run_rate(url: str, reqs, rate: float, seed: int) -> dict:
    rng = random.Random(seed)
    offsets, t = [], 0.0
    for _ in reqs:
        t += rng.expovariate(rate)
        offsets.append(t)
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(client, url, r, o, t0) for r, o in zip(reqs, offsets)))
    span = max(x["end"] for x in res) - t0

    def pct(key, q):
        return float(np.percentile([x[key] for x in res], q))

    return {
        "rate": rate, "n": len(res), "duration_s": span,
        "output_tok_per_s": sum(x["n_out"] for x in res) / span,
        **{f"{k}_p{q}_ms": pct(k, q) * 1000 for k in ("ttft", "tpot", "e2e") for q in (50, 99)},
    }


def wait_ready(url: str, proc, timeout: float = 600):
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during start-up")
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 1, 2, 4])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--out", default=str(ROOT / "results" / "online.jsonl"))
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}"
    cmd = [c.format(port=a.port) for c in ARMS[a.arm]]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(url, proc)
        # warm-up request outside the measurement
        asyncio.run(run_rate(url, random_workload(2, seed=99, output_range=(8, 8)), 10.0, 0))
        reqs = random_workload(a.n, a.seed)
        for rate in a.rates:
            r = asyncio.run(run_rate(url, reqs, rate, a.seed))
            r["arm"] = a.arm
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    finally:
        proc.terminate()
        proc.wait(30)


if __name__ == "__main__":
    main()
