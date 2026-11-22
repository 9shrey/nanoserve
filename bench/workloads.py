"""Benchmark workloads. Everything is seeded so every arm sees the identical request list."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    prompt_ids: list[int]
    max_tokens: int


def random_workload(n: int, seed: int = 0, prompt_range=(32, 512), output_range=(32, 256),
                    vocab_lo: int = 1000, vocab_hi: int = 100_000) -> list[Request]:
    """Ragged random-token requests, the same construction as vLLM's `--dataset random`.

    Token *content* does not change throughput when outputs are forced to a fixed length
    (ignore_eos), but the *spread* of lengths does: it is what padding and static batching
    waste. Uniform ranges here put that spread at 16x for prompts and 8x for outputs.
    """
    rng = random.Random(seed)
    return [
        Request([rng.randrange(vocab_lo, vocab_hi) for _ in range(rng.randint(*prompt_range))],
                rng.randint(*output_range))
        for _ in range(n)
    ]


# ---------------------------------------------------------------- speculative decoding
CODE = '''def load_orders(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            row = dict(zip(header, values))
            row["quantity"] = int(row["quantity"])
            row["price"] = float(row["price"])
            rows.append(row)
    return rows


def total_revenue(rows):
    total = 0.0
    for row in rows:
        total += row["quantity"] * row["price"]
    return total


def top_customers(rows, n=5):
    spend = {}
    for row in rows:
        spend[row["customer"]] = spend.get(row["customer"], 0.0) + row["quantity"] * row["price"]
    ranked = sorted(spend.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]
'''

PROSE = (
    "The library opened in 1924 with fewer than two thousand books, most of them donated by "
    "families in the town. For its first decade it was run entirely by volunteers, who kept "
    "the doors open three evenings a week and catalogued every book by hand. In 1936 the town "
    "council agreed to pay for a full-time librarian, and the collection grew quickly after "
    "that: by 1950 it held more than twenty thousand volumes and a small archive of local "
    "newspapers. The original building was replaced in 1972, but the reading room's oak tables "
    "were moved to the new site, where they are still used today."
)

#: Edit-style tasks: most of the answer is copied from the prompt. This is the workload
#: prompt-lookup decoding exists for (code edits, rewrites, extraction, RAG quoting).
COPY_TASKS = [
    "Rename the parameter `rows` to `orders` everywhere in this code. Return the complete code only.\n\n" + CODE,
    "Add a docstring to each function in this code. Return the complete code only.\n\n" + CODE,
    "Rename the function `total_revenue` to `revenue` and `top_customers` to `best_customers`. Return the complete code.\n\n" + CODE,
    "Rewrite this paragraph, replacing every number written as digits with words. Keep everything else identical.\n\n" + PROSE,
    "Copy this paragraph exactly, but change 'library' to 'museum' and 'books' to 'objects'.\n\n" + PROSE,
    "Convert this code to use type hints on every function signature. Return the complete code only.\n\n" + CODE,
]

#: Open-ended generation: little to copy. The control that shows what speculation costs
#: when it has nothing to work with.
OPEN_TASKS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Give me five ideas for a weekend trip near the mountains, with a sentence on each.",
    "What are the trade-offs between microservices and a monolith?",
    "Describe the water cycle in detail.",
    "Write a poem about autumn in a city.",
]

CORRECTNESS_PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "In 1969, the first humans to walk on the moon were",
    "A haiku about the sea:\n",
    "The three laws of thermodynamics are",
    "Once upon a time, in a small village by the river,",
    "To make a good cup of tea, you should",
    "The difference between TCP and UDP is",
]
