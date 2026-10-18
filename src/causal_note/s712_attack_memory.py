"""Short, past-only alignment of attack/presence evidence for inactive notes."""
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MemoryPolicy:
    onset_blocks: int = 0
    presence_blocks: int = 0
    current_presence_floor: float = .25


class AttackMemory:
    def __init__(self, selector, policy):
        if not 0 <= policy.onset_blocks <= 3 or not 0 <= policy.presence_blocks <= 2:
            raise ValueError("Unsupported history length")
        self.selector = selector
        self.policy = policy
        self.history = deque(maxlen=max(policy.onset_blocks, policy.presence_blocks) + 1)

    def step(self, post, count, end):
        p = self.policy
        self.history.append(np.asarray(post).copy())
        if not (p.onset_blocks or p.presence_blocks):
            return self.selector.step(post, count, end)
        values = list(self.history)
        pooled = np.asarray(post).copy()
        inactive = ~self.selector.active
        for column, blocks in ((0, p.onset_blocks), (1, p.presence_blocks)):
            best = np.maximum.reduce([x[:, column] for x in values[-blocks - 1:]])
            pooled[inactive, column] = best[inactive]
        pooled[inactive & (post[:, 1] < p.current_presence_floor), 1] = 0.
        # Active notes keep the current heads: no delayed release or backdating.
        return self.selector.step(pooled, count, end)

    def finish(self, sample):
        return self.selector.finish(sample)


def decode_memory(ends, post, count, selector, policy, samples):
    memory = AttackMemory(selector, policy)
    events = []
    for end, p, k in zip(ends, post, count):
        events.extend(memory.step(p, k, int(end)))
    events.extend(memory.finish(samples))
    return events
