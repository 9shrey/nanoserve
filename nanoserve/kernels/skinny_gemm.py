"""Measured GEMM dispatch: per weight shape and row count, use whichever of cuBLAS or the
row-chunked Triton kernel is actually faster on this GPU.

**Background.** My earlier project inferbench (github.com/9shrey/inferbench) found that on
the GTX 1650 (TU117, no tensor cores) cuBLAS sends fp16 matmuls with M >= 2 rows to a GEMM
kernel built for tensor cores, and wrote a split-K Triton kernel ("v4") that fixes it for
M = 2..16. It used one fixed rule: kernel for 2 <= M <= 16, cuBLAS otherwise.

**What changed here.** A serving engine sees decode batches above 16. Measuring Qwen2.5-0.5B's
four projection shapes separately (`scripts/bench_gemm_dispatch.py`, results/gemm_dispatch.json)
showed the fixed rule is too conservative, and by a different amount per shape:

  shape (N x K)        cuBLAS, M=2..64      v4 on 16-row chunks
  qkv     1152 x 896   ~0.39 ms             2x faster to M=16, slower beyond
  o        896 x 896   ~0.29 ms             faster to M=16, slower beyond
  gate_up 9728 x 896   ~2.8 ms (cliff)      3.6-11x faster to M=32, still ~1.8x at M=256
  down     896 x 4864  ~1.5 ms (cliff)      3.8-6x faster to M=32, still ~2x at M=256

The two wide/deep projections, which are most of the FLOPs, stay on the kernel far past
16 rows if a large M is split into 16-row chunks (one kernel call per chunk). Rather than
hard-code another threshold, `GemmDispatcher.calibrate()` times both options for each
(shape, M bucket) at start-up and records the winner. On a GPU with tensor cores, cuBLAS
should win everywhere and the table will simply say so.
"""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

CHUNK = 16
#: M values that get measured. A call with M rows uses the first bucket >= M; anything
#: past the last bucket uses the last bucket's choice.
BUCKETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048]
#: The chunked kernel is one launch per 16 rows; past this it is never competitive and
#: timing it would only slow calibration down.
KERNEL_MAX_M = 256
#: Upcasting a weight to fp32 per call needs a temporary 2x its fp16 size. Fine for the
#: per-layer projections (<= 35 MB), not for the 545 MB lm_head.
FP32_MAX_WEIGHT_NUMEL = 64 * 2**20
OPTIONS = ("cublas", "kernel", "fp32")


def chunked_v4(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from .skinny_gemm_v4 import skinny_linear_v4

    M = x.shape[0]
    if M <= CHUNK:
        return skinny_linear_v4(x, w) if M > 1 else F.linear(x, w)
    out = torch.empty((M, w.shape[0]), device=x.device, dtype=x.dtype)
    for i in range(0, M, CHUNK):
        xs = x[i:i + CHUNK]
        out[i:i + CHUNK] = skinny_linear_v4(xs, w) if xs.shape[0] > 1 else F.linear(xs, w)
    return out


def _per_call_ms(f, n: int = 20, reps: int = 7) -> float:
    """Throughput timing: n back-to-back calls, wall clock, divided by n; median of reps.

    In a forward pass the matmuls are issued back to back, so what matters is
    max(CPU launch cost, GPU time) per call, not one call's GPU time. A first version timed
    single calls between CUDA events; for small ops that mixed CPU gaps into the result and
    mis-ranked the Triton kernel (0.45 ms measured vs 0.14 ms real for the qkv shape).
    """
    import time

    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    f()
    torch.cuda.synchronize()
    # Keep each rep around ~20 ms so 40 ms calls do not make calibration take minutes.
    n = max(2, min(n, int(20 / max((time.perf_counter() - t) * 1e3, 1e-3))))
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) / n * 1e3)
    return sorted(out)[reps // 2]


def fp32_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Upcast both operands and use cuBLAS SGEMM. On this GPU fp16 cuBLAS GEMM at prefill
    sizes runs at ~0.4 TFLOP/s; SGEMM manages ~1.4, so paying to convert the weight on
    every call still comes out 2-4.5x ahead (gate_up at M=1024: 42.6 -> 13.0 ms)."""
    return F.linear(x.float(), w.float()).to(x.dtype)


_IMPL = {"cublas": F.linear, "kernel": chunked_v4, "fp32": fp32_linear}


class GemmDispatcher:
    """Per (weight shape, M bucket): which of cuBLAS fp16, the chunked Triton kernel, or an
    fp32-upcast SGEMM is fastest on this GPU, measured rather than assumed."""

    def __init__(self):
        #: (N, K) -> list aligned with BUCKETS of an OPTIONS name
        self.table: dict[tuple[int, int], list[str]] = {}
        self.timings: dict[str, dict] = {}

    def calibrate(self, weights: dict[str, torch.Tensor], rounds: int = 3) -> None:
        # The GPU idles at a few hundred MHz and ramps over ~100 ms. Measuring before the
        # ramp would bias whichever option is timed first, so spin it up first.
        w0 = next(iter(weights.values()))
        x0 = torch.randn(256, w0.shape[1], device=w0.device, dtype=w0.dtype)
        for _ in range(200):
            F.linear(x0, w0)
        torch.cuda.synchronize()
        for name, w in weights.items():
            N, K = w.shape
            if (N, K) in self.table:
                continue
            row: list[str] = []
            t: dict[int, dict[str, float]] = {}
            for M in BUCKETS:
                opts = ["cublas"]
                if 2 <= M <= KERNEL_MAX_M:
                    opts.append("kernel")
                if M > CHUNK and w.numel() <= FP32_MAX_WEIGHT_NUMEL:
                    opts.append("fp32")
                if len(opts) == 1:
                    row.append("cublas")
                    continue
                x = torch.randn(M, K, device=w.device, dtype=w.dtype)
                # Interleave the options over several rounds and keep each one's best
                # median, so clock or thermal drift hits all of them alike.
                times = {o: [] for o in opts}
                for _ in range(rounds):
                    for o in opts:
                        times[o].append(_per_call_ms(lambda: _IMPL[o](x, w)))
                best = {o: min(v) for o, v in times.items()}
                winner = min(best, key=best.get)
                # cuBLAS fp16 is the default; leaving it needs a clear 10% win, since
                # near-ties are noise.
                if best[winner] > 0.9 * best["cublas"]:
                    winner = "cublas"
                row.append(winner)
                t[M] = {o: round(v, 4) for o, v in best.items()}
            self.table[(N, K)] = row
            self.timings[name] = {"shape": [N, K], "ms": t}

    # ------------------------------------------------------------- disk cache
    @staticmethod
    def cache_path() -> Path:
        import triton

        gpu = torch.cuda.get_device_name().replace(" ", "_")
        root = Path(os.environ.get("NANOSERVE_CACHE", Path.home() / ".cache" / "nanoserve"))
        return root / f"gemm_{gpu}_torch{torch.__version__}_triton{triton.__version__}.json"

    def load(self, shapes: list[tuple[int, int]]) -> bool:
        p = self.cache_path()
        if not p.exists():
            return False
        d = json.loads(p.read_text())
        if d.get("buckets") != BUCKETS or any(f"{n}x{k}" not in d["table"] for n, k in shapes):
            return False
        self.table = {tuple(map(int, key.split("x"))): v for key, v in d["table"].items()}
        self.timings = d["timings"]
        return True

    def save(self) -> None:
        p = self.cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"buckets": BUCKETS, "timings": self.timings,
                                 "table": {f"{n}x{k}": v for (n, k), v in self.table.items()}}))

    def choice(self, M: int, w: torch.Tensor) -> str:
        row = self.table.get(tuple(w.shape))
        if row is None:
            return "cublas"
        return row[min(bisect.bisect_left(BUCKETS, M), len(BUCKETS) - 1)]

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if x.dtype is not torch.float16 or not x.is_cuda:
            return F.linear(x, w)
        return _IMPL[self.choice(x.shape[0], w)](x, w)

    def summary(self) -> dict[str, str]:
        """Human-readable: for each shape, which option serves which M buckets."""
        out = {}
        for name, d in self.timings.items():
            row = self.table[tuple(d["shape"])]
            parts = []
            for o in OPTIONS:
                ms = [m for m, c in zip(BUCKETS, row) if c == o]
                if ms:
                    parts.append(f"{o}: M in {ms}")
            out[name] = "; ".join(parts)
        return out
