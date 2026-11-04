"""Uniform-cost enumeration of the full bounded event/action model."""
from __future__ import annotations

import heapq
import itertools
import time

from ..models import to_primitive
from ..scenario import scenario_from_dict
from ..simulator import InvalidAction, PathSimulator
from .generator import candidate_from_actions


def exact_answer(payload: dict, seconds: float = 90, *, max_states: int = 200000) -> dict:
    started = time.perf_counter()
    simulator = PathSimulator(scenario_from_dict(payload))
    initial = simulator.initial_state()
    sequence = itertools.count()
    frontier = [(initial.time, next(sequence), initial)]
    visited = set()
    expanded = 0
    reason = "EXHAUSTED"
    while frontier:
        if time.perf_counter() - started >= seconds:
            reason = "TIME_LIMIT"
            break
        _, _, state = heapq.heappop(frontier)
        # Retain time, all bounded counters, full running records, facts and
        # canonical-start marker. Histories and accumulated wait cannot change
        # future feasibility or the sole objective (terminal absolute time).
        key = (state.time, state.active_state_ids, state.facts, state.counts, state.running, state.last_started_id)
        if key in visited:
            continue
        visited.add(key)
        if simulator.is_goal(state):
            actions = [to_primitive(action) for action in state.actions]
            candidate = candidate_from_actions(payload, actions)
            return {"status": "PROVEN_OPTIMAL", "optimal_makespan": candidate.metrics.makespan,
                    "lower_bound": candidate.metrics.makespan, "actions": actions,
                    "method": "uniform-cost-full-bounded-state", "expanded": expanded,
                    "elapsed_seconds": time.perf_counter() - started}
        if len(visited) + len(frontier) >= max_states:
            reason = "STATE_LIMIT"
            break
        expanded += 1
        for action in simulator.enabled_actions(state):
            try:
                child = simulator.transition(state, action)
            except InvalidAction:
                continue
            heapq.heappush(frontier, (child.time, next(sequence), child))
    return {"status": "PROVEN_INFEASIBLE" if not frontier and reason == "EXHAUSTED" else "UNKNOWN",
            "method": "uniform-cost-full-bounded-state", "stop_reason": reason,
            "expanded": expanded, "elapsed_seconds": time.perf_counter() - started}
