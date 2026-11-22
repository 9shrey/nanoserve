"""Qwen2 forward pass, written from scratch against the raw safetensors weights.

No `transformers` modeling code is used at runtime. The HF implementation is only the
reference the tests compare against.

**Packed layout.** Every step the engine flattens all scheduled tokens from all sequences
into one 1-D batch of T tokens. A decode step for 8 sequences is T=8. A step mixing one
300-token prefill chunk with 7 decodes is T=307. Linear layers see one (T, hidden) matrix
regardless of how the tokens are split between sequences. Only attention needs to know
which token belongs to which sequence, and it gets that from `AttentionMetadata`. That is
what lets one forward function serve prefill, decode, chunked prefill and speculative
verification without special cases.

**Fusions.** Q/K/V are one matmul and gate/up are one matmul: 4 GEMMs per layer instead
of 7. On this GPU a decode-sized GEMM is latency-bound, so fewer launches matter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .attention import AttentionMetadata, paged_attention
from .kernels.skinny_gemm import GemmDispatcher

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dir(cls, path: Path) -> "ModelConfig":
        c = json.loads((path / "config.json").read_text())
        eos: list[int] = []
        gen = path / "generation_config.json"
        if gen.exists():
            e = json.loads(gen.read_text()).get("eos_token_id", [])
            eos = e if isinstance(e, list) else [e]
        if not eos:
            eos = [c["eos_token_id"]]
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            max_position=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            eos_token_ids=tuple(eos),
        )


def resolve_model_dir(model_id: str = MODEL_ID) -> Path:
    """Local snapshot directory for a HF model id (downloads on first use)."""
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Normalise in fp32. Squaring fp16 activations overflows for large outliers,
    # and HF does the same upcast, so parity depends on it.
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


class RotaryEmbedding:
    """Non-interleaved ("rotate_half") RoPE, as Qwen2/Llama use. Cos/sin are precomputed once
    and indexed by absolute position, so a token's rotation depends only on its position and
    not on how the step was batched."""

    def __init__(self, head_dim: int, max_pos: int, theta: float, device, dtype):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(max_pos, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv)  # (P, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (P, D)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (T, heads, D); positions: (T,)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        half = x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rot * sin


@dataclass
class Layer:
    ln1: torch.Tensor
    qkv_w: torch.Tensor
    qkv_b: torch.Tensor
    o_w: torch.Tensor
    ln2: torch.Tensor
    gate_up_w: torch.Tensor
    down_w: torch.Tensor


class Qwen2Model:
    def __init__(
        self,
        model_dir: Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        use_skinny_gemm: bool = True,
        max_position: int = 8192,
    ):
        self.cfg = ModelConfig.from_dir(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        # The Triton skinny-GEMM only exists for fp16 on CUDA. Until `calibrate_gemm()` has
        # run, the dispatcher's table is empty and every matmul goes to cuBLAS.
        self.use_skinny_gemm = use_skinny_gemm and dtype is torch.float16 and self.device.type == "cuda"
        self.gemm = GemmDispatcher()
        cfg = self.cfg

        sd: dict[str, torch.Tensor] = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            sd.update(load_file(str(f), device="cpu"))

        def w(name: str) -> torch.Tensor:
            return sd[name].to(device=self.device, dtype=dtype)

        self.embed = w("model.embed_tokens.weight")
        self.final_norm = w("model.norm.weight")
        self.lm_head = self.embed if cfg.tie_word_embeddings else w("lm_head.weight")
        self.layers: list[Layer] = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            m = p + "mlp."
            self.layers.append(
                Layer(
                    ln1=w(p + "input_layernorm.weight"),
                    qkv_w=torch.cat([w(a + "q_proj.weight"), w(a + "k_proj.weight"), w(a + "v_proj.weight")]),
                    qkv_b=torch.cat([w(a + "q_proj.bias"), w(a + "k_proj.bias"), w(a + "v_proj.bias")]),
                    o_w=w(a + "o_proj.weight"),
                    ln2=w(p + "post_attention_layernorm.weight"),
                    gate_up_w=torch.cat([w(m + "gate_proj.weight"), w(m + "up_proj.weight")]),
                    down_w=w(m + "down_proj.weight"),
                )
            )
        del sd
        self.rope = RotaryEmbedding(cfg.head_dim, max_position, cfg.rope_theta, self.device, dtype)
        self.max_position = max_position

    # ------------------------------------------------------------------ helpers
    def _linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.use_skinny_gemm:
            return self.gemm.linear(x, w)
        return F.linear(x, w)

    def calibrate_gemm(self, use_cache: bool = True) -> dict[str, str]:
        """Time cuBLAS vs the Triton kernel for each projection shape (all layers share
        shapes, so layer 0 stands in for every layer). ~40 s on a GTX 1650, so the table is
        cached per GPU + torch + triton version."""
        if not self.use_skinny_gemm:
            return {}
        L = self.layers[0]
        # lm_head matters as much as all 24 layers together: at batch 32 cuBLAS took 41 ms
        # for it alone, out of a 110 ms step.
        ws = {"qkv": L.qkv_w, "o": L.o_w, "gate_up": L.gate_up_w, "down": L.down_w, "lm_head": self.lm_head}
        if not (use_cache and self.gemm.load([tuple(w.shape) for w in ws.values()])):
            self.gemm.calibrate(ws)
            self.gemm.save()
        return self.gemm.summary()

    def kv_bytes_per_token(self) -> int:
        c = self.cfg
        return 2 * c.num_layers * c.num_kv_heads * c.head_dim * torch.finfo(self.dtype).bits // 8

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # (T,) packed tokens of every scheduled sequence
        positions: torch.Tensor,  # (T,) absolute position of each token in its sequence
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],  # per layer (k, v), flat slots
        meta: AttentionMetadata,
        logits_indices: torch.Tensor,  # rows of the packed batch whose logits are needed
    ) -> torch.Tensor:
        cfg = self.cfg
        H, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        q_size, kv_size = H * D, Hkv * D

        h = self.embed[input_ids]  # (T, hidden)
        for li, L in enumerate(self.layers):
            x = rms_norm(h, L.ln1, cfg.rms_norm_eps)
            qkv = self._linear(x, L.qkv_w) + L.qkv_b
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q = self.rope.apply(q.view(-1, H, D), positions)
            k = self.rope.apply(k.view(-1, Hkv, D), positions)
            v = v.view(-1, Hkv, D)
            attn = paged_attention(q, k, v, kv_cache[li], meta)  # (T, H, D)
            h = h + self._linear(attn.reshape(-1, q_size), L.o_w)

            x = rms_norm(h, L.ln2, cfg.rms_norm_eps)
            gate, up = self._linear(x, L.gate_up_w).chunk(2, dim=-1)
            h = h + self._linear(F.silu(gate) * up, L.down_w)

        # Only project the rows we will sample from. For a 500-token prefill that is 1 row
        # instead of 500; at vocab 151,936 the full projection would be ~150 MB of fp16.
        h = rms_norm(h[logits_indices], self.final_norm, cfg.rms_norm_eps)
        return self._linear(h, self.lm_head).float()
