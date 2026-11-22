"""vLLM on the same offline workload as bench/offline.py (run from a separate venv with vllm
installed; vLLM pins its own torch).

    VLLM_PY=/path/to/vllm-venv/bin/python
    $VLLM_PY bench/vllm_offline.py --out results/offline.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import random_workload  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "offline.jsonl"))
    a = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", dtype="half", gpu_memory_utilization=a.gpu_mem,
              max_model_len=1024, max_num_seqs=32, enforce_eager=a.eager, seed=0)
    reqs = random_workload(a.n, a.seed)
    prompts = [{"prompt_token_ids": r.prompt_ids} for r in reqs]
    params = [SamplingParams(max_tokens=r.max_tokens, temperature=0.0, ignore_eos=True) for r in reqs]
    llm.generate(prompts[:2], [SamplingParams(max_tokens=4)] * 2)  # warm-up
    t = time.perf_counter()
    outs = llm.generate(prompts, params)
    wall = time.perf_counter() - t
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    r = {"arm": "vllm" + ("-eager" if a.eager else ""), "wall_s": wall, "output_tokens": n_out,
         "computed_output_tokens": n_out, "output_tok_per_s": n_out / wall, "req_per_s": a.n / wall,
         "n_requests": a.n, "seed": a.seed, "max_num_seqs": 32, "vllm": vllm.__version__,
         "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
         "prompt_tokens": sum(len(x.prompt_ids) for x in reqs)}
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
