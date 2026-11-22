"""Where does offline wall time go? Time every step of the offline workload, split into
pure-decode steps and steps that carry prefill tokens."""
import json, sys, time, torch
sys.path.insert(0, "bench")
from workloads import random_workload
from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.sequence import SamplingParams

eng = LLMEngine(EngineConfig(max_num_seqs=32)); eng.warmup()
for r in random_workload(64, 0):
    eng.add_request(r.prompt_ids, SamplingParams(max_tokens=r.max_tokens, ignore_eos=True))
rows = []
while eng.has_unfinished():
    pre = eng.stats.prefill_tokens; dec = eng.stats.decode_tokens
    torch.cuda.synchronize(); t = time.perf_counter()
    eng.step()
    torch.cuda.synchronize()
    rows.append((time.perf_counter() - t, eng.stats.prefill_tokens - pre, eng.stats.decode_tokens - dec, eng.stats.batch_sizes[-1]))
dec_only = [r for r in rows if r[1] == 0]; mixed = [r for r in rows if r[1] > 0]
tot = sum(r[0] for r in rows)
out = {
    "total_s": round(tot, 2), "steps": len(rows),
    "decode_only": {"steps": len(dec_only), "s": round(sum(r[0] for r in dec_only), 2),
                    "ms_per_step": round(1e3 * sum(r[0] for r in dec_only) / len(dec_only), 1),
                    "mean_batch": round(sum(r[3] for r in dec_only) / len(dec_only), 1)},
    "with_prefill": {"steps": len(mixed), "s": round(sum(r[0] for r in mixed), 2),
                     "prefill_tokens": sum(r[1] for r in mixed),
                     "prefill_tok_per_s": round(sum(r[1] for r in mixed) / sum(r[0] for r in mixed), 1)},
}
print(json.dumps(out, indent=1)); json.dump(out, open("results/step_breakdown.json", "w"), indent=1)
