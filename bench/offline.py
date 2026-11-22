"""Offline throughput: the whole request list is available up front; measure wall time to finish it.

    python bench/offline.py --arm nanoserve --n 64
    python bench/offline.py --all            # every arm, each in a fresh process

Each arm runs in its own process so peak-memory numbers are not polluted by another arm's
allocations, and so a crash in one arm (OOM) cannot take down the others.

Arms:
  hf-seq        HF generate, one request at a time. What a naive server does.
  hf-static16   HF generate, static batches of 16, left-padded. Each batch runs until its
                longest request finishes; tokens past a request's own max_tokens are waste.
  nanoserve     continuous batching + paged KV + chunked prefill + skinny-GEMM kernel
  nanoserve-nokernel   same engine, cuBLAS for every matmul (isolates the scheduler's gain)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import random_workload  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["hf-seq", "hf-static16", "nanoserve-nokernel", "nanoserve"]


def gpu_info() -> dict:
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def run_hf(reqs, batch: int) -> dict:
    from transformers import AutoModelForCausalLM

    from nanoserve.model import resolve_model_dir

    d = resolve_model_dir()
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    base_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    gen_kw = dict(do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None, pad_token_id=0)
    # warm-up
    model.generate(torch.tensor([reqs[0].prompt_ids[:16]], device="cuda"), max_new_tokens=4, **gen_kw)
    torch.cuda.synchronize()

    useful = 0
    computed = 0
    t0 = time.perf_counter()
    for i in range(0, len(reqs), batch):
        chunk = reqs[i:i + batch]
        L = max(len(r.prompt_ids) for r in chunk)
        ids = torch.zeros((len(chunk), L), dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, r in enumerate(chunk):  # left padding
            ids[j, L - len(r.prompt_ids):] = torch.tensor(r.prompt_ids)
            mask[j, L - len(r.prompt_ids):] = 1
        new = max(r.max_tokens for r in chunk)
        with torch.no_grad():
            model.generate(ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=new, min_new_tokens=new, **gen_kw)
        useful += sum(r.max_tokens for r in chunk)
        computed += new * len(chunk)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "output_tokens": useful, "computed_output_tokens": computed,
            "peak_mem_mib": (torch.cuda.max_memory_allocated() - base_mem) / 2**20}


def run_nano(reqs, kernel: bool, max_num_seqs: int, spec_k: int = 0) -> dict:
    from nanoserve.engine import EngineConfig, LLMEngine
    from nanoserve.sequence import SamplingParams

    eng = LLMEngine(EngineConfig(use_skinny_gemm=kernel, max_num_seqs=max_num_seqs, speculative_k=spec_k))
    eng.warmup()
    for r in reqs:
        eng.add_request(r.prompt_ids, SamplingParams(max_tokens=r.max_tokens, ignore_eos=True))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while eng.has_unfinished():
        eng.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    s = eng.stats
    return {"wall_s": wall, "output_tokens": s.decode_tokens, "computed_output_tokens": s.decode_tokens,
            "steps": s.steps, "preemptions": eng.scheduler.num_preemptions,
            "mean_batch": sum(s.batch_sizes) / len(s.batch_sizes),
            "kv_pool_mib": eng.kv.nbytes / 2**20, "kv_capacity_tokens": eng.kv_capacity_tokens,
            "peak_kv_blocks_used": s.peak_blocks_used,
            "peak_kv_mib_used": s.peak_blocks_used * eng.cfg.block_size * eng.model.kv_bytes_per_token() / 2**20,
            "kv_slot_utilisation": s.kv_utilisation}


def run_arm(arm: str, n: int, seed: int, max_num_seqs: int) -> dict:
    reqs = random_workload(n, seed)
    if arm == "hf-seq":
        r = run_hf(reqs, 1)
    elif arm == "hf-static16":
        r = run_hf(reqs, 16)
    elif arm == "nanoserve":
        r = run_nano(reqs, True, max_num_seqs)
    elif arm == "nanoserve-nokernel":
        r = run_nano(reqs, False, max_num_seqs)
    else:
        raise SystemExit(f"unknown arm {arm}")
    r.update(arm=arm, n_requests=n, seed=seed, max_num_seqs=max_num_seqs,
             prompt_tokens=sum(len(x.prompt_ids) for x in reqs), **gpu_info())
    r["output_tok_per_s"] = r["output_tokens"] / r["wall_s"]
    r["req_per_s"] = n / r["wall_s"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "results" / "offline.jsonl"))
    a = ap.parse_args()
    if a.all:
        for arm in ARMS:
            subprocess.run([sys.executable, __file__, "--arm", arm, "--n", str(a.n), "--seed", str(a.seed),
                            "--max-num-seqs", str(a.max_num_seqs), "--out", a.out], check=True)
        return
    r = run_arm(a.arm, a.n, a.seed, a.max_num_seqs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
