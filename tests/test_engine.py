"""End-to-end correctness: every scheduling feature must leave greedy output unchanged.

Each test changes *how* the work is batched (all at once, chunked, under memory pressure,
with speculation) and asserts the tokens are identical to HF's one-prompt-at-a-time greedy
decode. Batching is an optimisation; if it changes the answer, it is a bug.
"""

import pytest

from nanoserve.sequence import SamplingParams

from .conftest import MAX_NEW, gpu

pytestmark = gpu
GREEDY = SamplingParams(max_tokens=MAX_NEW, ignore_eos=True)


def test_single_requests_match_hf(make_engine, prompt_ids, hf_greedy):
    eng = make_engine()
    for ids, ref in zip(prompt_ids, hf_greedy):
        assert eng.generate([ids], GREEDY)[0] == ref


def test_continuous_batch_matches_hf(make_engine, prompt_ids, hf_greedy):
    eng = make_engine()
    assert eng.generate(prompt_ids, GREEDY) == hf_greedy
    assert max(eng.stats.batch_sizes) == len(prompt_ids)


def test_staggered_arrivals_match_hf(make_engine, prompt_ids, hf_greedy):
    """Requests join mid-flight, so decodes and prefills share steps."""
    eng = make_engine()
    ids = {}
    pending = list(enumerate(prompt_ids))
    while pending or eng.has_unfinished():
        if pending:
            i, p = pending.pop(0)
            ids[i] = eng.add_request(p, GREEDY)
        eng.step()
    for i, rid in ids.items():
        assert eng.seqs[rid].output_ids == hf_greedy[i]


@pytest.mark.parametrize("budget", [7, 32])
def test_chunked_prefill_matches_hf(make_engine, prompt_ids, hf_greedy, budget):
    eng = make_engine(max_num_batched_tokens=budget)
    assert eng.generate(prompt_ids, GREEDY) == hf_greedy


def test_preemption_matches_hf(make_engine, prompt_ids, hf_greedy):
    """A pool too small for all six sequences forces evict-and-recompute."""
    longest = max(len(p) for p in prompt_ids) + MAX_NEW
    eng = make_engine(num_kv_blocks=(longest + 15) // 16 + 3, block_size=16)
    assert eng.generate(prompt_ids, GREEDY) == hf_greedy
    assert eng.scheduler.num_preemptions > 0
    assert eng.allocator.num_free == eng.allocator.num_blocks  # nothing leaked


@pytest.mark.parametrize("k", [1, 4])
def test_speculative_greedy_is_lossless(make_engine, prompt_ids, hf_greedy, k):
    eng = make_engine(speculative_k=k)
    assert eng.generate(prompt_ids, GREEDY) == hf_greedy
    assert eng.stats.draft_proposed > 0
    assert eng.stats.draft_accepted > 0  # the "repeat after me" prompt guarantees some


def test_speculative_under_memory_pressure(make_engine, prompt_ids, hf_greedy):
    longest = max(len(p) for p in prompt_ids) + MAX_NEW
    eng = make_engine(num_kv_blocks=(longest + 15) // 16 + 3, speculative_k=4)
    assert eng.generate(prompt_ids, GREEDY) == hf_greedy
    assert eng.allocator.num_free == eng.allocator.num_blocks


def test_eos_and_max_tokens(make_engine, prompt_ids):
    eng = make_engine()
    out = eng.generate([prompt_ids[0]], SamplingParams(max_tokens=3, ignore_eos=True))[0]
    assert len(out) == 3
    rid = eng.add_request(prompt_ids[0], SamplingParams(max_tokens=200))
    while eng.has_unfinished():
        eng.step()
    s = eng.seqs[rid]
    assert s.finish_reason in {"stop", "length"}
    if s.finish_reason == "stop":
        assert s.output_ids[-1] in eng.eos


def test_sampling_is_seeded_and_varies(make_engine, prompt_ids):
    sp = SamplingParams(max_tokens=16, temperature=1.0, top_p=0.95, ignore_eos=True)
    a = make_engine(seed=1).generate([prompt_ids[2]] * 3, sp)
    b = make_engine(seed=1).generate([prompt_ids[2]] * 3, sp)
    assert a == b
    assert len({tuple(x) for x in a}) > 1


def test_abort_frees_blocks(make_engine, prompt_ids):
    eng = make_engine()
    rid = eng.add_request(prompt_ids[5], SamplingParams(max_tokens=50))
    eng.step()
    assert eng.allocator.num_free < eng.allocator.num_blocks
    eng.abort(rid)
    assert eng.allocator.num_free == eng.allocator.num_blocks
    assert not eng.has_unfinished()
