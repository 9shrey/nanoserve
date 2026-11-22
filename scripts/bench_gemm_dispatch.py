"""Re-measure cuBLAS vs row-chunked v4 for every Qwen2.5-0.5B projection shape and M bucket
(fresh, ignoring the cache) and write the table to results/gemm_dispatch.json."""
import json, sys, torch
from nanoserve.model import Qwen2Model, resolve_model_dir
from nanoserve.kernels.skinny_gemm import BUCKETS

m = Qwen2Model(resolve_model_dir())
summary = m.calibrate_gemm(use_cache=False)
out = {"gpu": torch.cuda.get_device_name(), "buckets": BUCKETS, "dispatch": summary, "timings_ms": m.gemm.timings}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/gemm_dispatch.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
