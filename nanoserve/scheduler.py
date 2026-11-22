"""Continuous batching with chunked prefill and preemption.

The scheduler runs once per model step and decides, for every sequence, how many tokens to
push through the model in that step. There is no separate "prefill phase" and "decode
phase". Each sequence just has `num_tokens - num_computed` tokens that are known but not
yet cached:

    new request            -> the whole prompt (cut into chunks by the token budget)
    decoding request       -> exactly 1 (the token sampled last step)
    preempted + resumed    -> prompt + everything generated so far (recomputed)

Policy, in priority order:

1. **Running sequences first, oldest first.** A running sequence that cannot get a block
   for its next token evicts the *newest* running sequence (LIFO preemption). Newest goes
   first because it has the least computed work to throw away, and evicting the oldest
   would starve it. If nothing newer is left to evict, the sequence preempts itself.
2. **Then admit waiting sequences**, FIFO, while the token budget, the `max_num_seqs` cap
   and the free blocks allow. Nothing new is admitted in a step that preempted, since
   admitting would just cause the next eviction.

**Preemption by recompute, not swap.** A preempted sequence's blocks are freed outright,
and on resumption its prompt plus its generated tokens are prefilled again. Swapping KV to
host memory is the alternative, but at 12 KB/token over PCIe, recomputing a few hundred
tokens of prefill on a 0.5B model is cheaper than the round trip.

**The token budget** (`max_num_batched_tokens`) bounds each step's work. Without it, one
4,000-token prompt arriving would stall every in-flight decode for the length of its
prefill. With it, the prompt is fed in chunks alongside the decodes, which trades a little
time-to-first-token for the new request against tail latency for everyone else.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .kv_cache import BlockAllocator, blocks_needed
from .sequence import Sequence, Status
from .spec import NgramDrafter


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new: int  # known tokens to run this step (prompt chunk, or the 1 sampled token)
    drafts: list[int] = field(default_factory=list)

    @property
    def num_scheduled(self) -> int:
        return self.num_new + len(self.drafts)

    @property
    def samples(self) -> bool:
        """After this step the sequence has no uncached known tokens, so we sample."""
        return self.seq.num_computed + self.num_new == self.seq.num_tokens


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 1024
    block_size: int = 16


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, allocator: BlockAllocator,
                 drafter: NgramDrafter | None = None):
        self.cfg = cfg
        self.alloc = allocator
        self.drafter = drafter
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        need = blocks_needed(seq.num_tokens + 1, self.cfg.block_size)
        if need > self.alloc.num_blocks:
            raise ValueError(
                f"request needs {need} KV blocks but the whole pool is {self.alloc.num_blocks}"
            )
        seq.status = Status.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ---------------------------------------------------------------- internals
    def _ensure_blocks(self, seq: Sequence, num_tokens: int) -> bool:
        need = blocks_needed(num_tokens, self.cfg.block_size) - len(seq.blocks)
        if need <= 0:
            return True
        if not self.alloc.can_allocate(need):
            return False
        seq.blocks.extend(self.alloc.allocate(need))
        return True

    def _preempt(self, seq: Sequence) -> None:
        self.alloc.free(seq.blocks)
        seq.blocks = []
        seq.num_computed = 0
        seq.status = Status.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)  # resumes ahead of never-started requests

    def free(self, seq: Sequence) -> None:
        if seq.blocks:
            self.alloc.free(seq.blocks)
            seq.blocks = []

    def trim(self, seq: Sequence) -> None:
        """Return blocks allocated for rejected draft tokens."""
        keep = blocks_needed(seq.num_tokens, self.cfg.block_size)
        if len(seq.blocks) > keep:
            self.alloc.free(seq.blocks[keep:])
            del seq.blocks[keep:]

    # ---------------------------------------------------------------- policy
    def schedule(self) -> list[ScheduledSeq]:
        budget = self.cfg.max_num_batched_tokens
        out: list[ScheduledSeq] = []
        preempted = False

        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_tokens - seq.num_computed, budget)
            drafts: list[int] = []
            if self.drafter is not None and seq.in_decode:
                # Never draft past max_tokens: the bonus token takes one of the slots.
                k = min(self.drafter.k, budget - n, seq.remaining_tokens - 1)
                drafts = self.drafter.propose(seq.token_ids, k) if k > 0 else []
            while not self._ensure_blocks(seq, seq.num_computed + n + len(drafts)):
                if drafts:  # shed speculation before evicting anyone
                    drafts = []
                    continue
                victim = self.running.pop()
                self._preempt(victim)
                preempted = True
                if victim is seq:
                    break
            if seq.status is Status.WAITING:  # preempted itself
                break
            out.append(ScheduledSeq(seq, n, drafts))
            budget -= n + len(drafts)
            i += 1

        if not preempted:
            while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
                seq = self.waiting[0]
                n = min(seq.num_tokens - seq.num_computed, budget)
                if not self._ensure_blocks(seq, seq.num_computed + n):
                    break
                self.waiting.popleft()
                seq.status = Status.RUNNING
                self.running.append(seq)
                out.append(ScheduledSeq(seq, n))
                budget -= n
        return out

    def finish(self, seq: Sequence) -> None:
        self.free(seq)
        seq.status = Status.FINISHED
        self.running.remove(seq)
