"""Quick parity: full-sequence logits from nanoserve vs HF, fp32 and fp16."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanoserve.model import Qwen2Model, resolve_model_dir
from nanoserve.attention import AttentionMetadata
from nanoserve.kv_cache import KVPool, blocks_needed

dtype = {"fp32": torch.float32, "fp16": torch.float16}[sys.argv[1]]
d = resolve_model_dir()
tok = AutoTokenizer.from_pretrained(d)
ids = tok("The capital of France is Paris. The capital of Germany is", return_tensors="pt").input_ids[0]
m = Qwen2Model(d, dtype=dtype)
c = m.cfg
pool = KVPool(c.num_layers, 64, 16, c.num_kv_heads, c.head_dim, "cuda", dtype)
T = len(ids)
table = list(range(blocks_needed(T, 16)))[::-1]  # deliberately non-contiguous order
meta, pos = AttentionMetadata.build([table], [T], [T], 16, m.device, gqa_rep=7)
ours = m.forward(ids.cuda(), pos, pool.layers, meta, torch.arange(T, device="cuda"))
del m; torch.cuda.empty_cache()
hf = AutoModelForCausalLM.from_pretrained(d, dtype=dtype).cuda().eval()
with torch.no_grad():
    ref = hf(ids[None].cuda()).logits[0].float()
print("max abs diff", (ours - ref).abs().max().item(), "ref scale", ref.abs().max().item())
print("argmax agree", (ours.argmax(-1) == ref.argmax(-1)).float().mean().item())
