# nanoserve — task plan

## Goal
An LLM inference engine written from scratch in PyTorch (+ one Triton kernel) for Qwen2.5-0.5B on a
4 GB GTX 1650: paged KV cache, continuous batching, speculative decoding, an OpenAI-compatible
streaming server — benchmarked against HF `generate` (and vLLM if it runs on sm_75/4 GB).
Follows inferbench, whose stated limitation was "real servers handle ragged batches; this declines to".

## Next Step
Phase 7 — run benchmarks (offline running), then README/guide/publish.

## Phases
### Phase 1 — Qwen2 forward from scratch, logits parity with HF
**Status:** complete
### Phase 2 — Paged KV allocator + paged attention (prefill + ragged decode)
**Status:** complete
### Phase 3 — Scheduler: continuous batching, admission by KV budget, preemption
**Status:** complete
### Phase 4 — Speculative decoding (n-gram prompt-lookup drafter), output == greedy
**Status:** complete
### Phase 5 — Skinny-GEMM kernel from inferbench wired in for 2..16-row decode
**Status:** complete
### Phase 6 — HTTP server (OpenAI /v1/completions, SSE streaming)
**Status:** complete
### Phase 7 — Benchmarks: offline throughput, online p50/p99 TTFT/TPOT, memory; vs HF (+vLLM attempt)
**Status:** pending
### Phase 8 — README, interview guide, CI, publish
**Status:** pending

## Hard rules
- Correctness gate before every timing: greedy output must match HF token-for-token.
- No `Co-Authored-By` trailer on public commits (user rule from earlier sessions).
- Publish as 9shrey; user authorised GitHub push in this request.

## Decisions Made
| Decision | Rationale |
|---|---|
| Project #2 (inference server) | Fits 4 GB GPU; extends inferbench; AI-infra is the stated hiring target |
| Per-shape measured GEMM dispatch, not inferbench's fixed 2..16 rule | gate_up/down stay pathological in cuBLAS up to M=256 |
| n-gram drafter, not a draft model | No Qwen2.5 smaller than 0.5B; 1.5B target + 0.5B draft does not fit 4 GB |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| HF greedy reference diverged at token 3 | 1 | generation_config repetition_penalty=1.1; override to 1.0 |
| Mutation run interrupted, left scheduler mutated | 1 | restored from backup; mutations now restore via git checkout |
| Cold-GPU bias in first GEMM measurement | 1 | warm-up spin + interleaved rounds in calibrate() |
