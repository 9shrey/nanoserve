"""SDPA vs the grouped-GQA matmul (as implemented in nanoserve.attention) across decode,
verify and prefill shapes. Writes results/attention.json."""

import json
import sys
import time

import torch
import torch.nn.functional as F

from nanoserve.attention import AttentionMetadata, _attend

H, HKV, D, B = 14, 2, 64, 16


def per_call(f, n=20, reps=7):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) / n * 1e3)
    return sorted(out)[reps // 2]


def case(S, Q, L):
    nb = (L + B - 1) // B
    k_cache = torch.randn(S * nb * B, HKV, D, device="cuda", dtype=torch.half)
    v_cache = torch.randn_like(k_cache)
    tables = [list(range(s * nb, (s + 1) * nb)) for s in range(S)]
    q = torch.randn(S * Q, H, D, device="cuda", dtype=torch.half)
    meta, _ = AttentionMetadata.build(tables, [L] * S, [Q] * S, B, torch.device("cuda"), gqa_rep=H // HKV)
    g = meta.groups[0]
    # Force each path on the same inputs.
    mask = (torch.arange(nb * B, device="cuda")[None, :] <= (L - Q + torch.arange(Q, device="cuda"))[:, None])
    mask = mask[None, None].expand(S, 1, Q, nb * B)
    bias = torch.zeros(mask.shape, device="cuda").masked_fill(~mask, float("-inf")).repeat(1, 1, H // HKV, 1)
    grouped = type(g)(None, g.block_tables, g.q_seq_idx, g.q_row_idx, None, bias, Q)
    sdpa = type(g)(None, g.block_tables, g.q_seq_idx, g.q_row_idx, mask, None, Q)
    a = _attend(q, k_cache, v_cache, grouped, B)
    b = _attend(q, k_cache, v_cache, sdpa, B)
    err = ((a.float() - b.float()).abs().max()).item()
    return {
        "shape": f"{S} seqs x {Q} query x {L} context",
        "sdpa_ms": round(per_call(lambda: _attend(q, k_cache, v_cache, sdpa, B)), 3),
        "grouped_ms": round(per_call(lambda: _attend(q, k_cache, v_cache, grouped, B)), 3),
        "max_abs_diff": err,
    }


if __name__ == "__main__":
    rows = [case(*c) for c in [(8, 1, 512), (32, 1, 512), (16, 1, 1024), (8, 5, 512), (1, 256, 256), (4, 512, 512)]]
    for r in rows:
        print(r)
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "results/attention.json", "w"), indent=1)
