"""The engine loop: schedule -> one packed forward -> sample/verify -> update sequences."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .attention import AttentionMetadata
from .kv_cache import BlockAllocator, KVPool
from .model import MODEL_ID, Qwen2Model, resolve_model_dir
from .sampling import probabilities, sample
from .scheduler import ScheduledSeq, Scheduler, SchedulerConfig
from .sequence import SamplingParams, Sequence, Status
from .spec import NgramDrafter, verify_greedy, verify_sampling


@dataclass
class EngineConfig:
    model: str = MODEL_ID
    dtype: torch.dtype = torch.float16
    block_size: int = 16
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 1024
    #: KV pool size. None = size it from free GPU memory after loading weights.
    num_kv_blocks: int | None = None
    kv_cache_memory_fraction: float = 0.80  # of the memory free after weights + headroom
    activation_headroom_mb: int = 600
    max_model_len: int = 4096
    speculative_k: int = 0  # 0 disables speculative decoding
    ngram_max: int = 3
    use_skinny_gemm: bool = True
    seed: int = 0


@dataclass
class StepOutput:
    request_id: str
    new_token_ids: list[int]
    finished: bool
    finish_reason: str | None = None


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0  # tokens emitted
    draft_proposed: int = 0
    draft_accepted: int = 0
    forward_time: float = 0.0
    step_time: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)
    peak_blocks_used: int = 0
    #: Over all steps: tokens actually held in the cache / slots in allocated blocks.
    #: The gap is internal fragmentation (unfilled tails of each sequence's last block).
    _kv_tokens: int = 0
    _kv_slots: int = 0

    @property
    def kv_utilisation(self) -> float:
        return self._kv_tokens / self._kv_slots if self._kv_slots else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.draft_accepted / self.draft_proposed if self.draft_proposed else 0.0


class LLMEngine:
    def __init__(self, cfg: EngineConfig | None = None, model: Qwen2Model | None = None):
        self.cfg = cfg = cfg or EngineConfig()
        self.model = model or Qwen2Model(
            resolve_model_dir(cfg.model), dtype=cfg.dtype, use_skinny_gemm=cfg.use_skinny_gemm,
            max_position=cfg.max_model_len + cfg.speculative_k + 1,
        )
        mc = self.model.cfg
        self.eos = set(mc.eos_token_ids)
        self.gqa_rep = mc.num_heads // mc.num_kv_heads

        num_blocks = cfg.num_kv_blocks or self._auto_num_blocks()
        self.kv = KVPool(mc.num_layers, num_blocks, cfg.block_size, mc.num_kv_heads, mc.head_dim,
                         self.model.device, cfg.dtype)
        self.allocator = BlockAllocator(num_blocks)
        drafter = NgramDrafter(k=cfg.speculative_k, n_max=cfg.ngram_max) if cfg.speculative_k > 0 else None
        self.scheduler = Scheduler(
            SchedulerConfig(cfg.max_num_seqs, cfg.max_num_batched_tokens, cfg.block_size),
            self.allocator, drafter,
        )
        self.seqs: dict[str, Sequence] = {}
        self.stats = EngineStats()
        self.generator = torch.Generator(device=self.model.device).manual_seed(cfg.seed)

    def warmup(self) -> dict[str, str]:
        """Calibrate GEMM dispatch, then run a throwaway batch so Triton autotuning and
        CUDA context set-up do not land inside the first real request."""
        summary = self.model.calibrate_gemm()
        from .sequence import SamplingParams
        for n in (1, 2, 5, 16, 32):
            ids = [self.add_request([100 + i] * 8, SamplingParams(max_tokens=3, ignore_eos=True)) for i in range(n)]
            while self.has_unfinished():
                self.step()
            for i in ids:
                self.seqs.pop(i, None)
        self.stats = EngineStats()
        self.scheduler.num_preemptions = 0
        return summary

    def _auto_num_blocks(self) -> int:
        free, _ = torch.cuda.mem_get_info(self.model.device)
        budget = (free - self.cfg.activation_headroom_mb * 2**20) * self.cfg.kv_cache_memory_fraction
        per_block = self.model.kv_bytes_per_token() * self.cfg.block_size
        n = int(budget // per_block)
        if n < 16:
            raise RuntimeError(f"not enough GPU memory for a KV cache ({free / 2**20:.0f} MiB free)")
        return n

    @property
    def kv_capacity_tokens(self) -> int:
        return self.allocator.num_blocks * self.cfg.block_size

    # ------------------------------------------------------------------ API
    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None,
                    request_id: str | None = None) -> str:
        params = params or SamplingParams()
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) + params.max_tokens > self.cfg.max_model_len:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({params.max_tokens}) "
                             f"exceeds max_model_len {self.cfg.max_model_len}")
        seq = Sequence(list(prompt_ids), params)
        if request_id is not None:
            seq.request_id = request_id
        if seq.request_id in self.seqs:
            raise ValueError(f"duplicate request id {seq.request_id}")
        self.scheduler.add(seq)
        self.seqs[seq.request_id] = seq
        return seq.request_id

    def abort(self, request_id: str) -> None:
        seq = self.seqs.pop(request_id, None)
        if seq is None or seq.status is Status.FINISHED:
            return
        if seq in self.scheduler.running:
            self.scheduler.finish(seq)
        else:
            self.scheduler.waiting.remove(seq)
            self.scheduler.free(seq)
        seq.status = Status.FINISHED
        seq.finish_reason = "abort"

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        plist = params if isinstance(params, list) else [params] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_unfinished():
            self.step()
        return [self.seqs.pop(i).output_ids for i in ids]

    # ------------------------------------------------------------------ step
    @torch.inference_mode()
    def step(self) -> list[StepOutput]:
        t0 = time.perf_counter()
        batch = self.scheduler.schedule()
        if not batch:
            return []

        input_ids: list[int] = []
        tables, ctx_lens, q_lens = [], [], []
        logit_rows: list[tuple[ScheduledSeq, int, int]] = []  # (item, first row, n rows)
        row = 0
        for item in batch:
            s = item.seq
            toks = s.token_slice(s.num_computed, s.num_computed + item.num_new) + item.drafts
            input_ids += toks
            tables.append(s.blocks)
            ctx_lens.append(s.num_computed + item.num_scheduled)
            q_lens.append(item.num_scheduled)
            if item.samples:
                n = 1 + len(item.drafts)
                logit_rows.append((item, row + item.num_new - 1, n))
            row += item.num_scheduled

        dev = self.model.device
        meta, positions = AttentionMetadata.build(tables, ctx_lens, q_lens, self.cfg.block_size, dev,
                                                  gqa_rep=self.gqa_rep)
        idx = [r for _, first, n in logit_rows for r in range(first, first + n)]
        tf = time.perf_counter()
        logits = self.model.forward(
            torch.tensor(input_ids, dtype=torch.long, device=dev), positions,
            self.kv.layers, meta, torch.tensor(idx, dtype=torch.long, device=dev),
        )
        new_tokens = self._sample(logits, logit_rows)
        self.stats.forward_time += time.perf_counter() - tf

        outputs: list[StepOutput] = []
        now = time.perf_counter()
        for item in batch:
            s = item.seq
            if not item.samples:  # mid-prefill chunk
                s.num_computed += item.num_new
                self.stats.prefill_tokens += item.num_new
                continue
            emitted = new_tokens[s.request_id]
            if item.drafts:
                accepted = len(emitted) - 1
                self.stats.draft_proposed += len(item.drafts)
                self.stats.draft_accepted += accepted
                s.num_computed += item.num_new + accepted
            else:
                if not s.in_decode:
                    self.stats.prefill_tokens += item.num_new
                s.num_computed += item.num_new
            outputs.append(self._append(s, emitted, now))
        self.stats.steps += 1
        self.stats.batch_sizes.append(len(batch))
        used = self.allocator.num_blocks - self.allocator.num_free
        self.stats.peak_blocks_used = max(self.stats.peak_blocks_used, used)
        self.stats._kv_tokens += sum(s.num_computed for s in self.scheduler.running)
        self.stats._kv_slots += used * self.cfg.block_size
        self.stats.step_time += time.perf_counter() - t0
        return outputs

    def _sample(self, logits: torch.Tensor, rows) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        if not rows:
            return out
        dev = logits.device
        reps = torch.tensor([n for _, _, n in rows], device=dev)
        p = [it.seq.params for it, _, _ in rows]
        temps = torch.tensor([x.temperature for x in p], device=dev).repeat_interleave(reps)
        top_ks = torch.tensor([x.top_k for x in p], device=dev).repeat_interleave(reps)
        top_ps = torch.tensor([x.top_p for x in p], device=dev).repeat_interleave(reps)

        any_spec = any(it.drafts for it, _, _ in rows)
        if not any_spec:
            toks = sample(logits, temps, top_ks, top_ps, self.generator).tolist()
            for (it, _, _), t in zip(rows, toks):
                out[it.seq.request_id] = [t]
            return out

        probs = None
        if bool((temps > 0).any()):
            probs = probabilities(logits, temps, top_ks, top_ps)
        r = 0
        for it, _, n in rows:
            sl = slice(r, r + n)
            if it.seq.params.temperature == 0:
                out[it.seq.request_id] = verify_greedy(logits[sl], it.drafts)
            else:
                out[it.seq.request_id] = verify_sampling(probs[sl], it.drafts, self.generator)
            r += n
        return out

    def _append(self, s: Sequence, emitted: list[int], now: float) -> StepOutput:
        if s.first_token_time is None:
            s.first_token_time = now
        kept: list[int] = []
        reason = None
        for t in emitted:
            kept.append(t)
            if (not s.params.ignore_eos and t in self.eos) or t in s.params.stop_token_ids:
                reason = "stop"
                break
            if len(s.output_ids) + len(kept) >= s.params.max_tokens:
                reason = "length"
                break
        s.output_ids.extend(kept)
        self.stats.decode_tokens += len(kept)
        if reason is None and s.num_tokens >= self.cfg.max_model_len:
            reason = "length"
        if reason is not None:
            s.finish_reason = reason
            s.finish_time = now
            self.scheduler.finish(s)
        else:
            self.scheduler.trim(s)
        return StepOutput(s.request_id, kept, reason is not None, reason)
