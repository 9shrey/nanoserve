"""fp16 Triton path: accuracy against cuBLAS at every row count the server can produce,
including ragged tails (M % 16 != 0) and the 151,936-wide lm_head."""

import pytest
import torch
import torch.nn.functional as F

from nanoserve.kernels.skinny_gemm import BUCKETS, GemmDispatcher, chunked_v4, fp32_linear

from .conftest import gpu

pytestmark = gpu

SHAPES = [(1152, 896), (896, 4864), (151936, 896)]


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [1, 2, 3, 7, 16, 17, 33, 100])
def test_chunked_matches_cublas(N, K, M):
    g = torch.Generator(device="cuda").manual_seed(M)
    w = torch.randn(N, K, device="cuda", dtype=torch.half, generator=g) * 0.02
    x = torch.randn(M, K, device="cuda", dtype=torch.half, generator=g)
    ref = F.linear(x.float(), w.float())
    out = chunked_v4(x, w).float()
    if N * K <= 64 * 2**20:
        assert ((fp32_linear(x, w).float() - ref).abs().max() / ref.abs().max()).item() < 2e-3
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    # fp32 accumulation in both; the bound is fp16 output rounding, with ~100x headroom
    # over the 3e-4 measured. The bug class this guards (an unzeroed split-K accumulator)
    # doubles values, i.e. rel error ~1.
    assert rel < 2e-3


def test_dispatch_table_lookup_rounds_up_to_bucket():
    d = GemmDispatcher()
    w = torch.empty(8, 4)
    d.table[(8, 4)] = ["kernel" if m in (2, 4) else "fp32" if m >= 512 else "cublas" for m in BUCKETS]
    got = [d.choice(m, w) for m in (1, 2, 3, 4, 5, 300, 512, 5000)]
    assert got == ["cublas", "kernel", "kernel", "kernel", "cublas", "fp32", "fp32", "fp32"]
    assert d.choice(4, torch.empty(3, 3)) == "cublas"  # unknown shape -> default


def test_first_call_with_cold_autotune_cache_is_correct(tmp_path):
    """The kernel accumulates with atomic_add, so the autotuner must zero the output between
    the configs it benchmarks (reset_to_zero). With cache_results=True a warm on-disk cache
    skips autotuning entirely, which hid the removal of that guard from every other test.
    A subprocess with an empty TRITON_CACHE_DIR forces a real autotune on the first call."""
    import os
    import subprocess
    import sys

    code = (
        "import torch, torch.nn.functional as F\n"
        "from nanoserve.kernels.skinny_gemm_v4 import skinny_linear_v4\n"
        "w = torch.randn(1152, 896, device='cuda', dtype=torch.half) * 0.02\n"
        "x = torch.randn(3, 896, device='cuda', dtype=torch.half)\n"
        "ref = F.linear(x.float(), w.float())\n"
        "out = skinny_linear_v4(x, w).float()\n"
        "print(((out - ref).abs().max() / ref.abs().max()).item())\n"
    )
    # Earlier tests leave the fp32 model and cached allocator blocks in this process; the
    # subprocess needs ~600 MB of its own (CUDA context + Triton's 256 MB L2-flush buffer).
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < 700 * 2**20:
        pytest.skip(f"only {free >> 20} MiB free for the subprocess; run this test on its own")
    env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert float(r.stdout.strip().splitlines()[-1]) < 2e-3
