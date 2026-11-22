"""v4 — split-K, with a 2D intermediate instead of a 3D one.

**What still limited v3.** v3 hit 6.6x over cuBLAS at M=2 but decayed to 2.6x by M=8, and
the cause is unchanged from v2: every program materialises a `(BLOCK_M, BLOCK_N, BLOCK_K)`
intermediate, so register demand grows *linearly in M* and occupancy falls away exactly as M
rises. Split-K raised the parallelism floor without touching that.

**v4.** Never form the 3D product. Loop over the M rows explicitly in a `static_range`, and
for each row reduce against the already-loaded weight tile:

    partial[n] = sum_k w[n, k] * x[m, k]          # (BLOCK_N, BLOCK_K) intermediate, 2D

Register demand becomes `BLOCK_N*BLOCK_K` (fixed) plus `BLOCK_M*BLOCK_N` (small), rather
than the product of all three. The arithmetic is identical — M passes over a weight tile that
is already resident — so this trades no extra memory traffic for flat occupancy in M.

Writing back to a single row is the awkward part: Triton has no dynamic row indexing, so the
accumulator is updated with a `tl.where` against the row index. That costs a
`(BLOCK_M, BLOCK_N)` select per row, which is cheap precisely because M is small — the same
property the whole kernel family exists to exploit.

**This kernel is not bit-reproducible, and that is inherent to split-K.** `SPLIT_K` programs
each contribute a partial sum to the same output element via `tl.atomic_add`; the hardware
serialises those atomics in an unspecified order, and fp32 addition is not associative.
Measured over 400 repeated calls on identical inputs: 2 differed, by ~1 fp16 ULP on
small-magnitude elements, with relative error against cuBLAS unchanged at 3.3e-04. So the
output is *reproducibly accurate* but not *reproducibly identical*.

In practice greedy decoding is unaffected — the gap between the top two logits over a 152k
vocabulary dwarfs one ULP — and `tests/test_serving.py` measures that rather than assuming it.
But anything that needs bit-exact replay (caching keyed on activations, differential debugging,
a regression suite comparing raw tensors) should use v2, which walks K in a single program and
is deterministic by construction at roughly 60% of v4's speed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs() -> list[triton.Config]:
    out = []
    for block_n in (64, 128):
        for block_k in (32, 64):
            for split_k in (4, 8, 16):
                out.append(
                    triton.Config(
                        {"BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": split_k},
                        num_warps=4,
                        num_stages=2,
                    )
                )
    return out


# Accumulating kernel: the autotuner re-runs it per config against the same buffer, so
# without reset_to_zero it is correct once and wrong whenever benchmarked. See v3.
#
# nanoserve change: keyed on BLOCK_M (the power of two the kernel is compiled for) rather
# than on M itself. Keyed on M, a server hitting row counts 3, 5, 6, 7, ... re-ran the full
# autotune sweep (~1.5 s each) in the middle of serving, which made the kernel arm of the
# first throughput benchmark *slower* than cuBLAS. cache_results persists the choice on disk.
@triton.autotune(configs=_configs(), key=["N", "K", "BLOCK_M"], reset_to_zero=["y_ptr"], cache_results=True)
@triton.jit
def _v4_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_n = offs_n < N
    k_tiles = tl.cdiv(K, BLOCK_K)

    w_base = w_ptr + offs_n[:, None] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Strided tile assignment, so concurrently-running splits touch neighbouring addresses
    # rather than starting a fraction of the weight matrix apart.
    for t in range(pid_k, k_tiles, SPLIT_K):
        k_off = t * BLOCK_K + offs_k
        mask_k = k_off < K

        # The weight tile is loaded ONCE and reused across all M rows below. That reuse is
        # what keeps this at one streaming pass over the weights despite the inner loop.
        w = tl.load(
            w_base + k_off[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        for m in tl.static_range(BLOCK_M):
            x_m = tl.load(
                x_ptr + m * stride_xm + k_off * stride_xk,
                mask=mask_k & (m < M),
                other=0.0,
            ).to(tl.float32)
            # (BLOCK_N, BLOCK_K) -> (BLOCK_N,). Two-dimensional, independent of BLOCK_M.
            partial = tl.sum(w * x_m[None, :], axis=1)
            acc += tl.where(offs_m[:, None] == m, partial[None, :], 0.0)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.atomic_add(y_ptrs, acc, mask=(offs_m[:, None] < M) & mask_n[None, :])


def skinny_linear_v4(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`F.linear(x, w)` for fp16 with small M: split-K, flat register use in M."""
    if x.dtype is not torch.float16 or w.dtype is not torch.float16:
        raise TypeError(f"fp16 path; got x={x.dtype}, w={w.dtype}")

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    m_rows, k = x2.shape
    n, k_w = w.shape
    if k != k_w:
        raise ValueError(f"shape mismatch: x K={k}, w K={k_w}")

    w = w.contiguous()
    y32 = torch.zeros((m_rows, n), device=x.device, dtype=torch.float32)
    block_m = max(1, triton.next_power_of_2(m_rows))

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])
    _v4_kernel[grid](
        x2, w, y32,
        m_rows, n, k,
        x2.stride(0), x2.stride(1),
        w.stride(0), w.stride(1),
        y32.stride(0), y32.stride(1),
        BLOCK_M=block_m,
    )
    return y32.to(torch.float16).reshape(*orig_shape[:-1], n)
