"""Speculative decoding benchmark: n-gram drafting on copy-heavy vs open-ended prompts.

    python bench/spec.py

For each workload and draft length k, run the same chat prompts greedily and report emitted
tokens/s, the draft acceptance rate, and tokens emitted per forward pass. k=0 is plain
decoding. Two batch settings: one request at a time (latency-bound, where speculation is
meant to help) and all six concurrently (where the GPU is already busier).

Output equality is checked too: greedy speculation is lossless by construction (and the
fp32 tests assert it), but in fp16 the verify pass runs a different matmul shape (k+1 rows)
than plain decode, so ULP-level differences can flip a near-tie. The fraction of outputs
identical to k=0 is reported rather than assumed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import COPY_TASKS, OPEN_TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(out: str):
    from nanoserve.engine import EngineConfig, LLMEngine
    from nanoserve.model import Qwen2Model, resolve_model_dir
    from nanoserve.sequence import SamplingParams
    from nanoserve.server import qwen_chat_prompt

    d = resolve_model_dir()
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    model = Qwen2Model(d, max_position=8192)
    model.calibrate_gemm()
    sp = SamplingParams(max_tokens=384)
    results = []
    for wname, tasks in (("copy-heavy", COPY_TASKS), ("open-ended", OPEN_TASKS)):
        prompts = [tok.encode(qwen_chat_prompt([{"role": "user", "content": t}])).ids for t in tasks]
        baseline = None
        for conc in (1, len(prompts)):
            for k in (0, 2, 4, 8):
                eng = LLMEngine(EngineConfig(speculative_k=k, max_num_seqs=conc, num_kv_blocks=1024), model=model)
                eng.generate([prompts[0]], SamplingParams(max_tokens=8))  # warm
                eng.stats = type(eng.stats)()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outs = eng.generate(prompts, sp)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                if k == 0 and conc == 1:
                    baseline = outs
                s = eng.stats
                n_tok = sum(len(o) for o in outs)
                r = {
                    "workload": wname, "concurrency": conc, "k": k, "tokens": n_tok,
                    "wall_s": round(wall, 3), "tok_per_s": round(n_tok / wall, 1),
                    "acceptance": round(s.acceptance_rate, 3),
                    "tokens_per_step": round(n_tok / s.steps, 2),
                    "identical_to_plain": sum(a == b for a, b in zip(outs, baseline)),
                    "of": len(outs),
                }
                results.append(r)
                print(json.dumps(r), flush=True)
                del eng
                torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "spec.json"))
