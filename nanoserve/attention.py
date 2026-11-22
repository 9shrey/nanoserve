"""Paged attention over ragged batches.

The KV cache is a pool of fixed-size blocks (see `kv_cache.py`). A sequence owns a list of
block ids, its *block table*. Logical token position p of that sequence lives in physical
slot `block_table[p // B] * B + p % B`. Blocks need not be contiguous, so a sequence can
grow one block at a time with no reallocation and no fragmentation beyond the last,
partly-filled block.

One function handles every kind of step. Each scheduled sequence s contributes
`query_len[s]` new tokens whose positions are `[context_len[s] - query_len[s], context_len[s])`:

    decode               query_len = 1
    prefill / chunk      query_len = chunk size, context_len = tokens cached so far + chunk
    speculative verify   query_len = 1 + number of draft tokens

A token at position p may attend to every cached position <= p of its own sequence. That
is the causal mask, expressed in absolute positions, so it is the same rule for all three.

**Implementation.** The new K/V rows are scattered into their slots, then each sequence's
blocks are gathered into a padded (S, L_max) tensor and attended with a boolean mask.

Two compute paths, chosen per step by the longest query chunk (measured on the GTX 1650,
`scripts/bench_attention.py`):

* **Short queries (decode, speculative verify): grouped-GQA matmul.** Qwen2.5-0.5B has 14
  query heads sharing 2 KV heads. Viewing the 7 query heads of a group as 7 extra *rows* of
  one matmul against their shared K turns 14 tiny (1 x L) products into 2 (7 x L) ones and
  never materialises an expanded K/V. 3-6x faster than SDPA at these shapes.
* **Long queries (prefill chunks): SDPA memory-efficient kernel**, which is 2.5x faster than
  the grouped matmul once Q is in the hundreds.

**A step is split into the two groups, not padded to one shape.** The first version padded
every sequence in a step to the step's longest query. In a mixed step (24 decodes + one
450-token prefill chunk) that meant 24 x 450 padded query rows for 24 real ones, and
prefill throughput in the offline benchmark fell to ~390 tok/s against ~1,400 measured for
prefill alone. Now short-query and long-query sequences are attended separately, each with
its own block tables and mask, and the outputs are scattered back into packed order. The gather copies the active KV
once per layer per step. A fused paged-attention kernel would read the blocks in place; at
this model size the copy is a few percent of step time (measured in the README), so it is
listed under "what I'd do next" rather than done.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: Up to this many query tokens per sequence the grouped-GQA matmul wins; above it, SDPA.
GROUPED_MAX_QUERY = 16


@dataclass
class AttentionGroup:
    """Sequences attended together in one padded batch."""

    token_idx: torch.Tensor | None  # (T_g,) rows of the packed batch; None = every token, in order
    block_tables: torch.Tensor  # (S_g, max_blocks) physical block ids, 0-padded
    q_seq_idx: torch.Tensor  # (T_g,) sequence (row of block_tables) of each token
    q_row_idx: torch.Tensor  # (T_g,) index of the token within its sequence's query chunk
    mask: torch.Tensor | None  # (S_g, 1, Q_max, L) bool, SDPA path
    bias: torch.Tensor | None  # (S_g, 1, rep*Q_max, L) fp32 additive mask, grouped path
    max_query_len: int


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # (T,) physical slot to write each new token's K/V into
    groups: list[AttentionGroup]
    block_size: int

    @classmethod
    def build(
        cls,
        block_tables: list[list[int]],
        context_lens: list[int],
        query_lens: list[int],
        block_size: int,
        device: torch.device,
        gqa_rep: int = 1,
    ) -> tuple["AttentionMetadata", torch.Tensor]:
        """Build on the CPU with tensor ops, move once. Returns (meta, positions)."""
        S = len(block_tables)
        to = dict(device=device, non_blocking=True)
        ql = torch.tensor(query_lens, dtype=torch.long)
        ctx = torch.tensor(context_lens, dtype=torch.long)
        n_blocks = [(c + block_size - 1) // block_size for c in context_lens]
        bt_all = torch.zeros((S, max(n_blocks)), dtype=torch.long)
        for s, (table, nb) in enumerate(zip(block_tables, n_blocks)):
            bt_all[s, :nb] = torch.tensor(table[:nb], dtype=torch.long)

        # Per token, in packed order: its sequence, its row within the chunk, its position.
        T = int(ql.sum())
        tok_seq = torch.repeat_interleave(torch.arange(S), ql)
        offsets = torch.cumsum(ql, 0) - ql
        tok_row = torch.arange(T) - offsets[tok_seq]
        positions = (ctx - ql)[tok_seq] + tok_row
        slots = bt_all[tok_seq, positions // block_size] * block_size + positions % block_size

        short = [s for s in range(S) if query_lens[s] <= GROUPED_MAX_QUERY]
        long_ = [s for s in range(S) if query_lens[s] > GROUPED_MAX_QUERY]
        groups = []
        for members, grouped in ((short, True), (long_, False)):
            if not members:
                continue
            idx = torch.tensor(members, dtype=torch.long)
            g_ql, g_ctx = ql[idx], ctx[idx]
            q_max, nb = int(g_ql.max()), max(n_blocks[s] for s in members)
            g_seq = torch.repeat_interleave(torch.arange(len(members)), g_ql)
            g_row = torch.arange(int(g_ql.sum())) - (torch.cumsum(g_ql, 0) - g_ql)[g_seq]
            # Padded query rows keep qpos=0, so they attend to key 0 only. That keeps the
            # softmax finite; their outputs are never read back.
            qpos = torch.zeros((len(members), q_max), dtype=torch.long)
            qpos[g_seq, g_row] = (g_ctx - g_ql)[g_seq] + g_row
            mask = (torch.arange(nb * block_size)[None, None, :] <= qpos[:, :, None]).unsqueeze(1)
            bias = None
            if grouped:
                # Rows of the grouped score matrix are ordered (head-in-group j, query i), so
                # the per-query mask is tiled rep times. Built once per step, used by all layers.
                bias = torch.zeros(mask.shape, dtype=torch.float32).masked_fill_(~mask, float("-inf"))
                bias = bias.repeat(1, 1, gqa_rep, 1).to(**to)
                mask = None
            else:
                mask = mask.to(**to)
            token_idx = None
            if len(members) != S:
                token_idx = torch.cat([torch.arange(offsets[s], offsets[s] + ql[s]) for s in members]).to(**to)
            groups.append(AttentionGroup(
                token_idx=token_idx, block_tables=bt_all[idx, :nb].to(**to),
                q_seq_idx=g_seq.to(**to), q_row_idx=g_row.to(**to),
                mask=mask, bias=bias, max_query_len=q_max,
            ))
        meta = cls(slot_mapping=slots.to(**to), groups=groups, block_size=block_size)
        return meta, positions.to(**to)


def _attend(q, k_cache, v_cache, g: AttentionGroup, B: int) -> torch.Tensor:
    # Gather each sequence's blocks into a padded, contiguous view.
    S, nb = g.block_tables.shape
    Hkv, D = k_cache.shape[1], k_cache.shape[2]
    K = k_cache.view(-1, B, Hkv, D)[g.block_tables].view(S, nb * B, Hkv, D).transpose(1, 2)
    V = v_cache.view(-1, B, Hkv, D)[g.block_tables].view(S, nb * B, Hkv, D).transpose(1, 2)

    # Scatter queries into (S, Q_max) rows.
    H = q.shape[1]
    Qm = g.max_query_len
    Q = q.new_zeros((S, Qm, H, D))
    Q[g.q_seq_idx, g.q_row_idx] = q
    Q = Q.transpose(1, 2)  # (S, H, Qm, D)
    if g.bias is not None:
        rep = H // Hkv
        qg = Q.reshape(S, Hkv, rep * Qm, D)  # head h = g*rep + j -> group g, row j*Qm + i
        scores = (qg @ K.transpose(-1, -2)).float() * (D ** -0.5) + g.bias
        probs = scores.softmax(-1).to(V.dtype)
        # P @ V computed as (V^T @ P^T)^T. Same maths, but with only rep*Qm (7) rows the
        # direct form gets cuBLAS's slow tensor-core-emulating kernel on this GPU; with the
        # operands swapped the "rows" are head_dim (64) and it picks a fast one. Measured
        # 2.5-7x faster across decode shapes (scripts/profile_step.py found it).
        out = (V.transpose(-1, -2) @ probs.transpose(-1, -2)).transpose(-1, -2).reshape(S, H, Qm, D)
    else:
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=g.mask, enable_gqa=True)
    return out.transpose(1, 2)[g.q_seq_idx, g.q_row_idx]  # (T_g, H, D)


def paged_attention(
    q: torch.Tensor,  # (T, H, D)
    k: torch.Tensor,  # (T, Hkv, D) new keys, already rotated
    v: torch.Tensor,  # (T, Hkv, D)
    kv_layer: tuple[torch.Tensor, torch.Tensor],  # each (num_slots, Hkv, D)
    meta: AttentionMetadata,
) -> torch.Tensor:
    k_cache, v_cache = kv_layer
    # Write this step's K/V into their slots first. After this the cache holds every
    # position the step's queries may see, including the new tokens themselves.
    k_cache.index_copy_(0, meta.slot_mapping, k)
    v_cache.index_copy_(0, meta.slot_mapping, v)
    if len(meta.groups) == 1 and meta.groups[0].token_idx is None:
        return _attend(q, k_cache, v_cache, meta.groups[0], meta.block_size)
    out = torch.empty_like(q)
    for g in meta.groups:
        out[g.token_idx] = _attend(q[g.token_idx], k_cache, v_cache, g, meta.block_size)
    return out
