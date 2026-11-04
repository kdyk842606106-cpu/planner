"""Opt-in, process-local observation; default engine behaviour is unchanged."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

from .models import CandidatePath

_observer: ContextVar[Callable[[CandidatePath], None] | None] = ContextVar("candidate_observer", default=None)
_trace_memory: ContextVar[bool] = ContextVar("trace_memory", default=True)


@contextmanager
def observe_candidates(observer: Callable[[CandidatePath], None] | None = None, *, trace_memory: bool = True) -> Iterator[None]:
    observer_token = _observer.set(observer)
    memory_token = _trace_memory.set(trace_memory)
    try:
        yield
    finally:
        _observer.reset(observer_token)
        _trace_memory.reset(memory_token)


def emit_candidate(candidate: CandidatePath) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(candidate)


def memory_tracing_enabled() -> bool:
    return _trace_memory.get()
