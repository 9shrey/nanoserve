from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    ignore_eos: bool = False  # benchmarks use this to force exact output lengths
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


_ids = itertools.count()


@dataclass
class Sequence:
    prompt_ids: list[int]
    params: SamplingParams
    request_id: str = field(default_factory=lambda: f"req-{next(_ids)}")
    output_ids: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    #: Tokens whose K/V are in the cache. The invariant the whole engine rests on: positions
    #: [0, num_computed) are cached and valid, everything at or past it is not.
    num_computed: int = 0
    status: Status = Status.WAITING
    finish_reason: str | None = None
    num_preemptions: int = 0
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    def token_slice(self, start: int, end: int) -> list[int]:
        p = len(self.prompt_ids)
        if end <= p:
            return self.prompt_ids[start:end]
        if start >= p:
            return self.output_ids[start - p:end - p]
        return self.prompt_ids[start:] + self.output_ids[:end - p]

    @property
    def in_decode(self) -> bool:
        """Prompt fully cached and exactly one known token not yet run through the model."""
        return self.num_computed == self.num_tokens - 1 and len(self.output_ids) > 0

    @property
    def remaining_tokens(self) -> int:
        return self.params.max_tokens - len(self.output_ids)
