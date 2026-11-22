"""fp16 correctness gate, run before any timing: nanoserve greedy vs HF greedy, same prompts.

In fp16 two correct implementations can differ by an ULP (different summation order,
split-K atomics), and a near-tied argmax then sends the continuation down a different but
equally valid path. So this reports how often and how late outputs diverge, and shows the
diverging text, rather than asserting identity. Identity is asserted in fp32 by the test suite.
"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import CORRECTNESS_PROMPTS

N_NEW = 64

def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nanoserve.engine import EngineConfig, LLMEngine
    from nanoserve.model import resolve_model_dir
    from nanoserve.sequence import SamplingParams
    d = resolve_model_dir(); tok = AutoTokenizer.from_pretrained(d)
    prompts = [tok(p).input_ids for p in CORRECTNESS_PROMPTS]
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16).cuda().eval()
    ref = []
    with torch.no_grad():
        for p in prompts:
            g = hf.generate(torch.tensor([p], device="cuda"), max_new_tokens=N_NEW, min_new_tokens=N_NEW, do_sample=False,
                            repetition_penalty=1.0, top_k=None, top_p=None, temperature=None, pad_token_id=0)
            ref.append(g[0, len(p):].tolist())
    del hf; torch.cuda.empty_cache()
    res = {}
    for kernel in (False, True):
        eng = LLMEngine(EngineConfig(use_skinny_gemm=kernel))
        summary = eng.warmup()
        outs = eng.generate(prompts, SamplingParams(max_tokens=N_NEW, ignore_eos=True))
        prefix = []
        for o, r in zip(outs, ref):
            k = next((i for i, (a, b) in enumerate(zip(o, r)) if a != b), N_NEW)
            prefix.append(k)
        res["kernel" if kernel else "cublas"] = {
            "identical": sum(p == N_NEW for p in prefix), "of": len(prompts), "first_divergence": prefix,
            "dispatch": summary,
            "examples": [{"ours": tok.decode(o), "hf": tok.decode(r)} for o, r, k in zip(outs, ref, prefix) if k < N_NEW][:2],
        }
        del eng; torch.cuda.empty_cache()
    print(json.dumps(res, indent=1))
    Path(out).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/correctness_fp16.json")
