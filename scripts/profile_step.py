"""Break one decode step at batch B into its parts, and profile the forward."""
import sys, time, torch
sys.path.insert(0, "bench")
from workloads import random_workload
from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.attention import AttentionMetadata
from nanoserve.sequence import SamplingParams

B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
kernel = (sys.argv[2] == "1") if len(sys.argv) > 2 else True
eng = LLMEngine(EngineConfig(use_skinny_gemm=kernel, max_num_seqs=B, num_kv_blocks=4096))
eng.warmup()
for r in random_workload(B, 0, prompt_range=(300, 300)):
    eng.add_request(r.prompt_ids, SamplingParams(max_tokens=200, ignore_eos=True))
while any(s.num_computed < len(s.prompt_ids) for s in eng.scheduler.running) or eng.scheduler.waiting:
    eng.step()
for _ in range(3): eng.step()
torch.cuda.synchronize()
N = 10
t = time.perf_counter()
for _ in range(N): eng.step()
torch.cuda.synchronize()
print(f"B={B} kernel={kernel}: full step {(time.perf_counter()-t)/N*1e3:.1f} ms")

# parts
sch = eng.scheduler
t = time.perf_counter(); batch = sch.schedule(); t_sched = time.perf_counter() - t
tables = [it.seq.blocks for it in batch]; ctx = [it.seq.num_computed + it.num_scheduled for it in batch]; ql = [it.num_scheduled for it in batch]
t = time.perf_counter()
for _ in range(N): meta, pos = AttentionMetadata.build(tables, ctx, ql, 16, eng.model.device, gqa_rep=7)
torch.cuda.synchronize(); t_meta = (time.perf_counter() - t) / N
ids = torch.tensor([it.seq.token_ids[-1] for it in batch], device="cuda"); li = torch.arange(len(batch), device="cuda")
for _ in range(2): eng.model.forward(ids, pos, eng.kv.layers, meta, li)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(N): eng.model.forward(ids, pos, eng.kv.layers, meta, li)
torch.cuda.synchronize(); t_fwd = (time.perf_counter() - t) / N
print(f"schedule {t_sched*1e3:.2f} ms | metadata {t_meta*1e3:.2f} ms | forward {t_fwd*1e3:.1f} ms")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    eng.model.forward(ids, pos, eng.kv.layers, meta, li); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=50))
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
    eng.model.forward(ids, pos, eng.kv.layers, meta, li); torch.cuda.synchronize()
rows = [e for e in prof.key_averages(group_by_input_shape=True) if e.key in ("aten::mm", "aten::linear", "aten::bmm", "aten::addmm")]
for e in sorted(rows, key=lambda e: -e.device_time_total)[:10]:
    print(f"{e.key:12s} calls={e.count:4d} cuda={e.device_time_total/1e3:7.2f} ms  shapes={str(e.input_shapes)[:90]}")
