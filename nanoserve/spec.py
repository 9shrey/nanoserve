"""Speculative decoding with an n-gram ("prompt lookup") drafter.

**Why n-gram and not a draft model.** The usual draft model needs the same tokenizer and to
be several times smaller than the target. Qwen2.5 has nothing smaller than 0.5B. Using
1.5B as the target with 0.5B as the draft does not fit in 4 GB with any KV cache left. Prompt
lookup needs no second model: it searches the sequence's own history for the most recent
earlier occurrence of its last n tokens and proposes the tokens that followed. That is
cheap and it is exactly right for workloads that copy from their input (summarisation,
extraction, code editing, RAG answers that quote the context).

**Why it is lossless.** Verification runs the target model over [last token, d1..dk] in one
forward pass, producing the target's distribution at every draft position.
* Greedy: accept d_i while it equals the target's argmax. The first mismatch is replaced by
  the target's own argmax, so the output is token-for-token what plain greedy decoding
  would produce. The tests assert this.
* Sampling: the drafter is deterministic (q puts all its mass on d_i), so standard
  speculative rejection sampling reduces to: accept d_i with probability p(d_i); on
  rejection, sample from p with d_i removed and renormalised. The emitted tokens are
  distributed exactly as p. (Leviathan et al. 2023, with q one-hot.)

Every verify step emits at least one token (the correction or the bonus token), so the
worst case is plain decoding plus the cost of verifying k extra positions.
"""

from __future__ import annotations

import numpy as np
import torch


class NgramDrafter:
    def __init__(self, k: int = 4, n_max: int = 3, n_min: int = 1, max_lookback: int = 2048):
        self.k = k
        self.n_max = n_max
        self.n_min = n_min
        self.max_lookback = max_lookback

    def propose(self, tokens: list[int], k: int | None = None) -> list[int]:
        k = self.k if k is None else k
        if k <= 0 or len(tokens) < self.n_min + 1:
            return []
        arr = np.asarray(tokens[-self.max_lookback:], dtype=np.int64)
        L = len(arr)
        for n in range(min(self.n_max, L - 1), self.n_min - 1, -1):
            pattern = arr[L - n:]
            # Candidate starts j with j + n < L, i.e. the match is followed by at least one
            # token and is not the suffix itself.
            windows = np.lib.stride_tricks.sliding_window_view(arr[:L - 1], n)
            hits = np.flatnonzero((windows == pattern).all(axis=1))
            if hits.size:
                j = int(hits[-1])  # most recent occurrence
                return arr[j + n:j + n + k].tolist()
        return []


def verify_greedy(logits: torch.Tensor, drafts: list[int]) -> list[int]:
    """logits (1+k, V): row i is the target's prediction after consuming draft i-1.
    Returns emitted tokens: accepted drafts plus one correction/bonus token."""
    target = logits.argmax(-1).tolist()
    out: list[int] = []
    for i, d in enumerate(drafts):
        if d != target[i]:
            out.append(target[i])
            return out
        out.append(d)
    out.append(target[len(drafts)])
    return out


def verify_sampling(probs: torch.Tensor, drafts: list[int], generator: torch.Generator | None = None) -> list[int]:
    """probs (1+k, V): target distribution at each position (after temperature/top-p)."""
    out: list[int] = []
    u = torch.rand(len(drafts), generator=generator, device=probs.device) if drafts else None
    for i, d in enumerate(drafts):
        p = probs[i]
        if u[i] < p[d]:
            out.append(d)
            continue
        residual = p.clone()
        residual[d] = 0
        s = residual.sum()
        if s <= 0:  # p was a point mass on d, so rejection had probability 0; unreachable
            out.append(d)
            continue
        out.append(int(torch.multinomial(residual / s, 1, generator=generator)))
        return out
    out.append(int(torch.multinomial(probs[len(drafts)], 1, generator=generator)))
    return out
