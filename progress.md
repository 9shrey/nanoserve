# nanoserve — progress log

## 2026-11-22 — Session 1
- Project chosen: #2 LLM inference server (fits the 4 GB GTX 1650; follows on from inferbench).
- Qwen2 forward from scratch: fp32 logits within 2.1e-5 of HF; fp16 all argmax agree.
- Attention: grouped-GQA matmul is 3-6x faster than SDPA for decode/verify shapes, SDPA 2.5x faster
  for prefill -> dispatch on max query length (threshold 16).
- Fixed a wrong claim in a comment (enable_gqa does work with masks on sm_75); measured before keeping.
- HF reference mismatch was HF's repetition_penalty=1.1 from generation_config, not a bug here.
- Engine: paged KV, unified scheduler (chunked prefill, LIFO preemption by recompute), n-gram spec decoding.
- Tests: 12 GPU (fp32, token-identical to HF across batching/chunking/preemption/spec) + 17 CPU = 29 green.
- Mutation check: 5/5 mutants killed (spec residual sampling, preemption reset, RoPE positions,
  greedy correction token, spec num_computed).
- GEMM dispatch: per-shape measured table. gate_up/down keep the Triton kernel up to M=256 (chunked
  in 16 rows); qkv/o only to M=16. First standalone measurement was biased by a cold GPU (qkv measured
  first) — calibration now spins the GPU up and interleaves arms; two runs agree.
- fp16 gate: 7/8 prompts identical to HF over 64 tokens; the 8th diverges at token 54 identically with
  and without the kernel.
- First offline run: nanoserve 56.8 tok/s with kernel vs 68.9 without — kernel arm *slower*. Profiled:
  1. lm_head (151936x896) took 41.5 of 110 ms per B=32 decode step in cuBLAS; never calibrated. Kernel: 11 ms.
  2. v4 autotune keyed on exact M re-tuned (~1.5 s) for every new tail size mid-run. Re-keyed on BLOCK_M + cache_results.
  3. Attention P@V bmm 10x slower than QK^T at same FLOPs (cuBLAS cliff); (V^T P^T)^T is 2.5-7x faster.
  4. Calibration timed single calls between CUDA events — mis-ranked small ops; switched to throughput timing.
  B=32 decode step 110 -> 72 ms (cuBLAS-only: 179 ms). 35 tests green.
- 3-way dispatch (cuBLAS / chunked Triton / fp32-upcast SGEMM): fp16 cuBLAS only wins at M=1 on this card.
- Attention split into short/long query groups: mixed-step padding had cut prefill to 393 tok/s; now ~1,030.
- HF baseline server fixed (shared thread pool starved streaming; TPOT p99 1.85 s -> 35 ms). Rerun.
- FINAL offline: HF seq 24.5, HF static16 28.9, nanoserve stock-cuBLAS 90.2, nanoserve 231.7 tok/s (9.5x).
- Steps: decode B=2 166->46 ms, B=32 176->62 ms (514 tok/s); prefill ~1,250-1,435 tok/s, 25-28% fp32 peak.
- Online @1 req/s: TTFT p50 213 ms vs 71 s (HF). Spec: 3.2x copy-heavy bs1; 0.94x open-ended conc6 k8.
- fp16 gate now 6/8 identical (divergence at 33, 54; same with dispatch on/off).
- Resume: ~/Documents/main-mlinfra.tex + main-swe.tex (new), compiled 1 page each. main.tex untouched.
- Interview guide: resume_fixes/08-nanoserve-interview-guide.md.
- vLLM 0.34 runs on sm_75 (TRITON_ATTN; FA2 needs sm_80): 90.4 tok/s offline = nanoserve on stock cuBLAS (90.2).
  nanoserve 2.6x vLLM is the GEMM dispatch on a tensor-core-less GPU; documented as such.
- Mutation re-check on final code: 7/7 caught. The autotune-reset mutant survived at first (warm
  on-disk autotune cache) -> cold-cache subprocess test added.
- Published github.com/9shrey/nanoserve; CI green (20 CPU tests pass, 42 GPU tests skip on the runner).
