"""Batched sampling. Every row carries its own temperature / top-k / top-p, so requests with
different settings share one decode step."""

from __future__ import annotations

import torch


def _filter(logits: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Mask logits outside each row's top-k and nucleus (top-p) set to -inf."""
    sorted_logits, idx = logits.sort(dim=-1, descending=True)
    V = logits.shape[-1]
    rank = torch.arange(V, device=logits.device).expand_as(sorted_logits)
    k = torch.where(top_ks > 0, top_ks, torch.full_like(top_ks, V))
    drop = rank >= k[:, None]
    probs = sorted_logits.softmax(-1)
    # Keep the smallest prefix whose mass reaches top_p: drop a token if the mass *before*
    # it already reaches top_p. The top token is therefore always kept.
    cum_before = probs.cumsum(-1) - probs
    drop |= cum_before >= top_ps[:, None]
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, idx, sorted_logits)


def probabilities(logits: torch.Tensor, temps: torch.Tensor, top_ks: torch.Tensor,
                  top_ps: torch.Tensor) -> torch.Tensor:
    """Target distribution for rows with temperature > 0 (callers handle greedy rows)."""
    scaled = logits / temps.clamp_min(1e-5)[:, None]
    return _filter(scaled, top_ks, top_ps).softmax(-1)


def sample(logits: torch.Tensor, temps: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (N, V) fp32 -> token ids (N,). Rows with temperature 0 are argmax."""
    greedy = logits.argmax(-1)
    if bool((temps == 0).all()):
        return greedy
    probs = probabilities(logits, temps, top_ks, top_ps)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(temps == 0, greedy, sampled)
