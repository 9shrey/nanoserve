import sys, time, torch
from torch.profiler import profile, ProfilerActivity
from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.attention import AttentionMetadata
S, L = int(sys.argv[1]), int(sys.argv[2])
eng = LLMEngine(EngineConfig(num_kv_blocks=2048)); eng.warmup()
tables = [list(range(i * 64, i * 64 + (L + 15) // 16)) for i in range(S)]
t = time.perf_counter(); meta, pos = AttentionMetadata.build(tables, [L] * S, [L] * S, 16, eng.model.device, gqa_rep=7); torch.cuda.synchronize()
print(f"metadata build {1e3*(time.perf_counter()-t):.1f} ms")
ids = torch.randint(1000, 50000, (S * L,), device="cuda"); li = torch.arange(S, device="cuda") * L + L - 1
f = lambda: eng.model.forward(ids, pos, eng.kv.layers, meta, li)
for _ in range(2): f()
torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize()
print(f"S={S} L={L}: forward {1e3*(time.perf_counter()-t):.1f} ms = {S*L/(time.perf_counter()-t):.0f} tok/s")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    f(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=45))
