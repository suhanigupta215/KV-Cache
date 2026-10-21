"""Deterministic event-driven async engine (spec sections 08, 25, 54).

Events have time, priority, type, stream ID, generation, and payload.
The scheduler runs normal order or adversarial order: adversarial mode
deliberately places stale corrections and late commits after newer
generations. Transport delays model control latency, not physical CDC
timing. A run ends only when the queue is drained.
"""
from __future__ import annotations

import heapq
import random
from typing import Callable, Optional

from .config import Config
from .data_objects import Event


class EventEngine:
    """Priority queue with configurable delays and reordering modes."""

    EVENT_TYPES = (
        "DECODE_READY",
        "MKAD_READY",
        "PAGE_RESERVED",
        "PAGE_COMMITTED",
        "KV_READY",
        "SEMANTIC_UPDATE",
        "CORRECTION_COMMITTED",
        "TRANSITION_PREFETCH",
        "FAULT",
    )

    def __init__(self, cfg: Config):
        self.ec = cfg.events
        self.rng = random.Random(self.ec.seed)
        self._queue: list[tuple] = []
        self._seq = 0
        self._handlers: dict[str, Callable[[Event], None]] = {}
        self.now = 0.0
        self.trace: list[Event] = []
        self.adversarial_bias = self.ec.adversarial_delay_jitter

    def register(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._handlers[event_type] = handler

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def schedule(
        self,
        time: float,
        event_type: str,
        stream_id: int,
        generation: int,
        payload: Optional[dict] = None,
        priority: int = 10,
        jitter: bool = False,
    ) -> Event:
        """Schedule an event; jitter models variable transport delay and
        equal-time reorderings (spec section 44)."""
        t = time
        if jitter:
            t += self.rng.uniform(0.0, self.adversarial_bias)
        event = Event(
            time=t,
            priority=priority,
            seq=self._next_seq(),
            type=event_type,
            stream_id=stream_id,
            generation=generation,
            payload=payload or {},
        )
        heapq.heappush(self._queue, event.sort_key() + (event,))
        return event

    def step(self) -> Optional[Event]:
        """Pop and dispatch the next event, or None when drained."""
        if not self._queue:
            return None
        key = heapq.heappop(self._queue)
        event: Event = key[-1]
        self.now = max(self.now, event.time)
        self.trace.append(event)
        handler = self._handlers.get(event.type)
        if handler is not None:
            handler(event)
        return event

    def run(self, max_events: int = 1_000_000) -> int:
        """Drain the queue deterministically."""
        processed = 0
        while self._queue and processed < max_events:
            self.step()
            processed += 1
        return processed

    def pending(self) -> int:
        return len(self._queue)

    def export_trace(self) -> list[dict]:
        return [e.to_dict() for e in self.trace]
