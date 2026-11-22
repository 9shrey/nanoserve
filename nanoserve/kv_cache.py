"""Block allocator and the physical KV pool.

The pool is allocated once at start-up and never resized. Its size is the engine's real
capacity limit: the scheduler admits a request only if the blocks it needs are free, and
preempts running requests when decode growth would overrun the pool.

Why blocks instead of one contiguous buffer per request: a contiguous buffer has to be
sized for the request's *maximum* length up front, and that reservation is mostly empty
for most of the request's life. Blocks are handed out as the sequence actually grows, so
the only waste is the unfilled tail of each sequence's last block (< block_size tokens).
"""

from __future__ import annotations

import torch


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    """Free-list allocator over block ids [0, num_blocks).

    LIFO free list: a just-freed block is the next one handed out, which keeps the working
    set of blocks small and cache-warm. Allocation is all-or-nothing, so a failed request
    never leaves a partial allocation behind for the caller to clean up.
    """

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > len(self._free):
            raise OutOfBlocks(f"need {n}, have {len(self._free)}")
        out = [self._free.pop() for _ in range(n)]
        self._used.update(out)
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._used:
                raise ValueError(f"double free or foreign block {b}")
            self._used.remove(b)
            self._free.append(b)


def blocks_needed(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


class KVPool:
    """The physical cache: per layer, a K and a V tensor of shape (num_blocks*block_size, Hkv, D)."""

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 num_kv_heads: int, head_dim: int, device, dtype):
        self.block_size = block_size
        self.num_blocks = num_blocks
        slots = num_blocks * block_size
        # One allocation for everything, then views. A single tensor makes the memory
        # accounting exact and keeps the pool from fragmenting the caching allocator.
        self._storage = torch.zeros((num_layers, 2, slots, num_kv_heads, head_dim), device=device, dtype=dtype)
        self.layers = [(self._storage[i, 0], self._storage[i, 1]) for i in range(num_layers)]

    @property
    def nbytes(self) -> int:
        return self._storage.numel() * self._storage.element_size()
