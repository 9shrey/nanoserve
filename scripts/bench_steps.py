"""Step-level numbers for the README's roofline discussion: prefill throughput and MFU,
and decode step time vs batch size, with the GEMM dispatcher on and off.

MFU here = achieved model FLOPs / the card's fp32 FMA peak at the clock measured during the
run. fp32 is the right denominator because large-M matmuls run as fp32 SGEMM (see README).
Model FLOPs per token = 2 x non-embedding parameters (lm_head runs only on sampled rows);
attention FLOPs are added explicitly.
"""

import json
import subprocess
import sys
import time

import torch

from nanoserve.attention import AttentionMetadata
from nanoserve.engine import EngineConfig, LLMEngine

CUDA_CORES = 1024  # GTX 1650 (TU117): 16 SMs x 64 FP32 cores


def sm_clock_mhz() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def timed(f, n=5):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main(out_path):
    res = {}
    for kernel in (True, False):
        eng = LLMEngine(EngineConfig(use_skinny_gemm=kernel, num_kv_blocks=4096))
        eng.warmup()
        m, c = eng.model, eng.model.cfg
        layer_params = sum(t.numel() for t in vars(m.layers[0]).values())
        flops_tok_base = 2 * layer_params * c.num_layers
        tag = "dispatch" if kernel else "stock_cublas"
        r = {"prefill": [], "decode": []}
        for S, L in ((1, 1024), (4, 256)):
            tables = [list(range(i * 64, i * 64 + (L + 15) // 16)) for i in range(S)]
            meta, pos = AttentionMetadata.build(tables, [L] * S, [L] * S, 16, m.device, gqa_rep=7)
            ids = torch.randint(1000, 50000, (S * L,), device="cuda")
            li = torch.arange(S, device="cuda") * L + L - 1
            sec = timed(lambda: m.forward(ids, pos, eng.kv.layers, meta, li))
            clock = sm_clock_mhz()
            attn = 4 * c.num_layers * c.hidden_size * (L + 1) / 2  # QK^T + PV, causal average
            tflops = S * L * (flops_tok_base + attn) / sec / 1e12
            peak = CUDA_CORES * 2 * clock * 1e6 / 1e12
            r["prefill"].append({"seqs": S, "len": L, "ms": round(sec * 1e3, 1),
                                 "tok_per_s": round(S * L / sec), "tflops": round(tflops, 3),
                                 "sm_clock_mhz": clock, "fp32_peak_tflops": round(peak, 2),
                                 "mfu_vs_fp32_peak": round(tflops / peak, 3)})
        for B in (1, 2, 8, 16, 32):
            L = 300
            tables = [list(range(i * 32, i * 32 + 20)) for i in range(B)]
            meta, pos = AttentionMetadata.build(tables, [L] * B, [1] * B, 16, m.device, gqa_rep=7)
            ids = torch.randint(1000, 50000, (B,), device="cuda")
            li = torch.arange(B, device="cuda")
            sec = timed(lambda: m.forward(ids, pos, eng.kv.layers, meta, li), n=20)
            r["decode"].append({"batch": B, "context": L, "ms_per_step": round(sec * 1e3, 2),
                                "tok_per_s": round(B / sec, 1)})
        res[tag] = r
        print(tag, json.dumps(r, indent=1), flush=True)
        del eng, m
        torch.cuda.empty_cache()
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/steps.json")
