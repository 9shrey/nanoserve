"""CPU-only unit tests: allocator, scheduler policy, drafter, sampling, spec verification."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from nanoserve.kv_cache import BlockAllocator, OutOfBlocks, blocks_needed
from nanoserve.sampling import probabilities, sample
from nanoserve.scheduler import Scheduler, SchedulerConfig
from nanoserve.sequence import SamplingParams, Sequence, Status
from nanoserve.spec import NgramDrafter, verify_greedy, verify_sampling


# ------------------------------------------------------------------ allocator
@settings(max_examples=200, deadline=None)
@given(st.integers(1, 64), st.lists(st.tuples(st.booleans(), st.integers(0, 10)), max_size=60))
def test_allocator_never_double_allocates(n_blocks, ops):
    a = BlockAllocator(n_blocks)
    held: list[list[int]] = []
    for is_alloc, n in ops:
        if is_alloc:
            if a.can_allocate(n):
                held.append(a.allocate(n))
            else:
                with pytest.raises(OutOfBlocks):
                    a.allocate(n)
        elif held:
            a.free(held.pop(n % len(held)))
        flat = [b for h in held for b in h]
        assert len(flat) == len(set(flat))
        assert all(0 <= b < n_blocks for b in flat)
        assert a.num_free + len(flat) == n_blocks


def test_allocator_rejects_double_free():
    a = BlockAllocator(4)
    b = a.allocate(2)
    a.free(b)
    with pytest.raises(ValueError):
        a.free(b)


def test_failed_allocation_is_all_or_nothing():
    a = BlockAllocator(3)
    with pytest.raises(OutOfBlocks):
        a.allocate(4)
    assert a.num_free == 3


def test_blocks_needed():
    assert [blocks_needed(n, 16) for n in (0, 1, 16, 17, 32)] == [0, 1, 1, 2, 2]


# ------------------------------------------------------------------ scheduler
def _seq(n_prompt, max_tokens=8):
    return Sequence(list(range(1, n_prompt + 1)), SamplingParams(max_tokens=max_tokens))


def _fake_step(sched, batch):
    """Mimic the engine: advance num_computed, append one token to sampling sequences."""
    for it in batch:
        it.seq.num_computed += it.num_new
        if it.seq.num_computed == it.seq.num_tokens:
            it.seq.output_ids.append(7)


def test_token_budget_chunks_prefill():
    s = Scheduler(SchedulerConfig(max_num_batched_tokens=10, block_size=4), BlockAllocator(100))
    seq = _seq(25)
    s.add(seq)
    chunks = []
    while not seq.output_ids:
        b = s.schedule()
        chunks.append(b[0].num_new)
        _fake_step(s, b)
    assert chunks == [10, 10, 5]


def test_decodes_are_scheduled_before_new_prefills():
    s = Scheduler(SchedulerConfig(max_num_batched_tokens=8, block_size=4), BlockAllocator(100))
    a, b = _seq(8), _seq(20)
    s.add(a)
    _fake_step(s, s.schedule())  # a prefilled, now decoding
    s.add(b)
    batch = s.schedule()
    assert batch[0].seq is a and batch[0].num_new == 1
    assert batch[1].seq is b and batch[1].num_new == 7  # gets only what is left of the budget


def test_max_num_seqs_caps_admission():
    s = Scheduler(SchedulerConfig(max_num_seqs=2, block_size=4), BlockAllocator(100))
    for _ in range(5):
        s.add(_seq(3))
    assert len(s.schedule()) == 2
    assert len(s.waiting) == 3


def test_lifo_preemption_evicts_newest_and_requeues_it_first():
    alloc = BlockAllocator(4)  # 16 slots of block_size 4
    s = Scheduler(SchedulerConfig(block_size=4), alloc)
    old, new = _seq(7, max_tokens=50), _seq(7, max_tokens=50)
    s.add(old)
    s.add(new)
    _fake_step(s, s.schedule())  # both prefilled: 2 blocks each, pool full
    _fake_step(s, s.schedule())  # 8 tokens each, still fits
    batch = s.schedule()  # old needs a 3rd block -> evict new
    assert [it.seq for it in batch] == [old]
    assert new.status is Status.WAITING and new.num_computed == 0 and new.blocks == []
    assert s.waiting[0] is new and s.num_preemptions == 1


def test_oversized_request_is_rejected_up_front():
    s = Scheduler(SchedulerConfig(block_size=4), BlockAllocator(2))
    with pytest.raises(ValueError):
        s.add(_seq(20))


# ------------------------------------------------------------------ drafter
def test_ngram_finds_most_recent_continuation():
    d = NgramDrafter(k=3, n_max=2)
    toks = [1, 2, 3, 9, 1, 2, 4, 5, 6, 1, 2]
    assert d.propose(toks) == [4, 5, 6]


def test_ngram_backs_off_to_shorter_match():
    d = NgramDrafter(k=2, n_max=3)
    assert d.propose([5, 8, 1, 2, 3, 4, 8]) == [1, 2]


def test_ngram_no_match():
    assert NgramDrafter(k=3).propose([1, 2, 3, 4]) == []


def test_ngram_truncates_at_end_of_history():
    assert NgramDrafter(k=5, n_max=1).propose([3, 1, 2, 3]) == [1, 2, 3]


# ------------------------------------------------------------------ sampling
def test_top_k_and_top_p_masks():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    one = torch.ones(1)
    p = probabilities(logits, one, torch.tensor([2]), one)
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0, 0]]))
    p = probabilities(logits, one, torch.tensor([0]), torch.tensor([0.7]))
    assert (p[0, 2:] == 0).all() and p[0, :2].sum() == pytest.approx(1)
    p = probabilities(logits, one, torch.tensor([0]), torch.tensor([0.1]))
    assert p[0, 0] == pytest.approx(1)  # top token always survives


def test_greedy_rows_ignore_randomness():
    logits = torch.randn(4, 50)
    t = sample(logits, torch.zeros(4), torch.zeros(4, dtype=torch.long), torch.ones(4))
    assert torch.equal(t, logits.argmax(-1))


# ------------------------------------------------------------------ spec verification
def test_verify_greedy_accepts_prefix_then_corrects():
    V = 10
    logits = torch.full((4, V), -1.0)
    for row, tok in enumerate([3, 4, 9, 2]):
        logits[row, tok] = 5.0
    assert verify_greedy(logits, [3, 4, 5]) == [3, 4, 9]
    assert verify_greedy(logits, [3, 4, 9]) == [3, 4, 9, 2]  # all accepted + bonus
    assert verify_greedy(logits[:1], []) == [3]


def test_verify_sampling_preserves_target_distribution():
    """Speculative sampling must emit tokens distributed exactly as the target p,
    whatever the draft. Chi-square over 40k trials, deliberately bad and good drafts."""
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.1, 0.4, 0.2, 0.3])
    probs = p.expand(2, 4)
    N = 40_000
    for draft in (0, 1, 3):
        counts = torch.zeros(4)
        for _ in range(N):
            counts[verify_sampling(probs, [draft], g)[0]] += 1
        expected = p * N
        chi2 = (((counts - expected) ** 2) / expected).sum().item()
        assert chi2 < 16.27, (draft, counts / N)  # p=0.001 critical value, 3 dof


# ------------------------------------------------------------------ attention metadata
def test_mixed_step_splits_short_and_long_queries():
    """3 decodes + one 40-token prefill chunk: two groups, no decode padded to 40 rows."""
    from nanoserve.attention import AttentionMetadata

    tables = [[0, 1], [2], [3, 4, 5], [6, 7, 8]]
    ctx = [20, 5, 33, 40]
    ql = [1, 1, 1, 40]
    meta, pos = AttentionMetadata.build(tables, ctx, ql, 16, torch.device("cpu"), gqa_rep=7)
    short, long_ = meta.groups
    assert short.max_query_len == 1 and long_.max_query_len == 40
    assert short.token_idx.tolist() == [0, 1, 2]
    assert long_.token_idx.tolist() == list(range(3, 43))
    assert short.bias.shape == (3, 1, 7, 48) and short.mask is None
    assert long_.mask.shape == (1, 1, 40, 48)
    assert pos.tolist()[:3] == [19, 4, 32] and pos.tolist()[3:] == list(range(40))
    # position 32 of seq 2 lives in its 3rd block (id 5), offset 0
    assert meta.slot_mapping.tolist()[2] == 5 * 16 + 0
