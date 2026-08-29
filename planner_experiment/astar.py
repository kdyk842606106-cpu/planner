from __future__ import annotations

import heapq
import itertools
import math
import time
import tracemalloc
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field

from .archive import PathArchive
from .models import Action, Budget, EngineResult, ImprovementPoint, Scenario, SimState, count_get
from .simulator import PathSimulator
from .validator import ResultValidator


@dataclass(order=True)
class _QueueEntry:
    priority: tuple[float, ...]
    node_id: int = field(compare=False)


@dataclass(frozen=True)
class _SearchNode:
    node_id: int
    state: SimState
    parent_id: int | None
    action: Action | None
    step_count: int
    execution_count: int
    transition_count: int
    revisit_count: int
    regression_count: int
    seen_mask: int
    prefix_hash: int
    provider_mask: int


@dataclass(frozen=True)
class _Label:
    values: tuple[int, ...]
    prefix_hash: int


class AnytimeAStar:
    """ARA-style primary search followed by a bounded strategy-family search."""

    algorithm = "ASTAR"

    def __init__(
        self,
        scenario: Scenario,
        *,
        weights: tuple[float, ...] = (3.0, 1.5, 1.0),
        budget_shares: tuple[float, ...] = (0.4, 0.3, 0.3),
        max_open: int = 50_000,
        trim_to: int = 25_000,
        labels_per_state: int = 3,
        relevance_pruning: bool = False,
        provider_pruning: bool = False,
        diversity_fraction: float = 0.25,
    ) -> None:
        if len(weights) != len(budget_shares):
            raise ValueError("weights and budget_shares must have equal length")
        self.scenario = scenario
        self.simulator = PathSimulator(scenario)
        self.weights = weights
        self.budget_shares = budget_shares
        self.max_open = max_open
        self.trim_to = trim_to
        self.labels_per_state = labels_per_state
        self.relevance_pruning = relevance_pruning
        self.provider_pruning = provider_pruning
        self.diversity_fraction = max(0.0, min(0.5, diversity_fraction))
        self._providers = self._build_provider_index()
        self._relevant_activities = self._build_relevance_set()
        self._dominated_activities = self._build_dominated_set()
        self._strategy_provider_ids = self._build_strategy_provider_ids()
        self._strategy_provider_bits = {
            activity_id: 1 << index
            for index, activity_id in enumerate(sorted(self._strategy_provider_ids))
        }
        self._reversible_bits = {
            node: 1 << index
            for index, node in enumerate(sorted(self.simulator.transition_graph.reversible_state_ids))
        }
        self._heuristic_cache: OrderedDict[tuple, int] = OrderedDict()
        self._heuristic_cache_limit = 5_000

    def run(self, scenario: Scenario, budget: Budget, *, run_id: str, seed: int | None = None) -> EngineResult:
        if scenario != self.scenario:
            raise ValueError("Engine scenario does not match run scenario")
        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        deadline = wall_start + budget.time_limit_seconds
        archive = PathArchive(budget.max_solutions, pinned_signatures=scenario.expected_signatures)
        improvements: list[ImprovementPoint] = []
        first_solution: float | None = None
        transitions = expanded = peak_open = reconstruction_count = 0
        trimmed = False
        stopped_reason = "EXHAUSTED"
        id_counter = itertools.count()
        tie_counter = itertools.count()
        nodes: dict[int, _SearchNode] = {}
        initial = self._root_node(next(id_counter))
        nodes[initial.node_id] = initial
        labels: dict[tuple, list[_Label]] = {}
        self._accept_label(labels, initial, limit=1)
        queue = [self._queue_entry(initial, self.weights[0], next(tie_counter))]
        inconsistent: list[int] = []
        incumbent: int | None = None
        final_pass_complete = False
        pass_stats: list[dict[str, object]] = []
        primary_transition_cap = budget.transition_limit
        primary_deadline = deadline
        maximal_start_stats: dict[str, object] = {"enabled": False, "transitions": 0, "valid": False}
        if scenario.execution_mode == "parallel" and transitions < budget.transition_limit:
            candidate, seed_transitions, seed_reason = self._maximal_start_candidate(
                run_id=run_id,
                wall_start=wall_start,
                transition_allowance=budget.transition_limit - transitions,
            )
            transitions += seed_transitions
            maximal_start_stats = {
                "enabled": True,
                "transitions": seed_transitions,
                "terminal_reason": seed_reason,
                "valid": False,
            }
            if candidate is not None:
                validation = ResultValidator(scenario).validate(candidate)
                maximal_start_stats["validator_code"] = validation.code
                if validation.valid and validation.candidate is not None:
                    candidate = validation.candidate
                    _, improved = archive.add(candidate)
                    incumbent = candidate.metrics.makespan
                    first_solution = candidate.discovered_at_seconds
                    maximal_start_stats.update(
                        {"valid": True, "makespan": candidate.metrics.makespan}
                    )
                    if improved:
                        improvements.append(
                            ImprovementPoint(
                                candidate.discovered_at_seconds,
                                candidate.metrics.sort_key,
                                candidate.structure_signature,
                            )
                        )
                    reserve = 1.0 - self.diversity_fraction
                    primary_deadline = min(deadline, wall_start + budget.time_limit_seconds * reserve)
                    primary_transition_cap = min(
                        budget.transition_limit,
                        max(transitions, int(budget.transition_limit * reserve)),
                    )

        for pass_index, (weight, share) in enumerate(zip(self.weights, self.budget_shares)):
            if inconsistent:
                queue.extend(
                    self._queue_entry(nodes[node_id], weight, next(tie_counter))
                    for node_id in inconsistent
                    if self._label_active(labels, nodes[node_id])
                )
                inconsistent.clear()
            queue = [
                self._queue_entry(nodes[item.node_id], weight, next(tie_counter))
                for item in queue
                if self._label_active(labels, nodes[item.node_id])
            ]
            heapq.heapify(queue)
            if not queue:
                final_pass_complete = pass_index == len(self.weights) - 1
                pass_stats.append({"weight": weight, "expanded": 0, "frontier_end": 0, "complete": True})
                continue
            closed: set[tuple] = set()
            pass_start_expanded = expanded
            pass_start_transitions = transitions
            pass_transition_budget = max(1, int(budget.transition_limit * share))
            nominal_deadline = min(
                deadline,
                wall_start + budget.time_limit_seconds * sum(self.budget_shares[: pass_index + 1]),
            )
            pass_complete = True
            while queue:
                now = time.perf_counter()
                active_deadline = min(nominal_deadline, primary_deadline)
                if now >= active_deadline:
                    pass_complete = False
                    stopped_reason = "TIME_LIMIT"
                    break
                if transitions - pass_start_transitions >= pass_transition_budget or transitions >= primary_transition_cap:
                    pass_complete = False
                    stopped_reason = "TRANSITION_LIMIT"
                    break
                entry = heapq.heappop(queue)
                node = nodes[entry.node_id]
                if not self._label_active(labels, node):
                    continue
                closed.add(self._label_token(node))
                admissible = self._earliest_achievement_bound(node.state)
                if self._quality_prune(archive, incumbent, node.state, admissible):
                    continue
                if self.simulator.is_goal(node.state):
                    elapsed = now - wall_start
                    candidate = self._candidate_from_node(node, nodes, run_id, elapsed)
                    reconstruction_count += 1
                    _, improved = archive.add(candidate)
                    incumbent = min(incumbent, candidate.metrics.makespan) if incumbent is not None else candidate.metrics.makespan
                    if first_solution is None:
                        first_solution = elapsed
                        reserve = 1.0 - self.diversity_fraction
                        primary_deadline = min(deadline, wall_start + budget.time_limit_seconds * reserve)
                        primary_transition_cap = min(
                            budget.transition_limit,
                            max(transitions, int(budget.transition_limit * reserve)),
                        )
                    if improved:
                        improvements.append(ImprovementPoint(elapsed, candidate.metrics.sort_key, candidate.structure_signature))
                    continue
                if self._at_step_limit(node):
                    continue
                expanded += 1
                for action in self.simulator.enabled_actions(node.state):
                    if transitions >= primary_transition_cap:
                        break
                    if not self._action_allowed(action):
                        continue
                    child = self._child_node(node, action, next(id_counter))
                    transitions += 1
                    next_bound = self._earliest_achievement_bound(child.state)
                    if next_bound >= 10**9 or self._quality_prune(archive, incumbent, child.state, next_bound):
                        continue
                    if not self._accept_label(labels, child, limit=1):
                        continue
                    nodes[child.node_id] = child
                    if self._label_token(child) in closed and weight != 1.0:
                        inconsistent.append(child.node_id)
                    else:
                        heapq.heappush(queue, self._queue_entry(child, weight, next(tie_counter)))
                if len(queue) > self.max_open:
                    queue = heapq.nsmallest(self.trim_to, queue)
                    heapq.heapify(queue)
                    trimmed = True
                    pass_complete = False
                peak_open = max(peak_open, len(queue))
            pass_stats.append(
                {
                    "weight": weight,
                    "expanded": expanded - pass_start_expanded,
                    "transitions": transitions - pass_start_transitions,
                    "frontier_end": len(queue),
                    "inconsistent_end": len(inconsistent),
                    "complete": pass_complete and not queue,
                }
            )
            if pass_index == len(self.weights) - 1:
                final_pass_complete = pass_complete and not queue

        diversity_stats = {"enabled": False, "expanded": 0, "transitions": 0, "uncovered_provider_ids": [], "complete": False}
        if incumbent is not None and self.diversity_fraction > 0 and time.perf_counter() < deadline and transitions < budget.transition_limit:
            diversity_stats, used_transitions, used_expanded, used_peak, used_rebuilds, used_trimmed = self._strategy_search(
                archive=archive,
                nodes=nodes,
                id_counter=id_counter,
                tie_counter=tie_counter,
                budget=budget,
                run_id=run_id,
                wall_start=wall_start,
                deadline=deadline,
                transitions_used=transitions,
                incumbent=incumbent,
                improvements=improvements,
            )
            transitions += used_transitions
            expanded += used_expanded
            peak_open = max(peak_open, used_peak)
            reconstruction_count += used_rebuilds
            trimmed = trimmed or used_trimmed

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - wall_start
        paths = archive.paths()
        if paths:
            status = "OK" if final_pass_complete else "TIMEOUT_PARTIAL"
        elif final_pass_complete and not trimmed and self.weights[-1] == 1.0:
            status = "PROVEN_INFEASIBLE"
        else:
            status = "TIMEOUT_EMPTY"
        return EngineResult(
            algorithm=self.algorithm,
            run_id=run_id,
            seed=None,
            status=status,
            paths=paths,
            first_solution_seconds=first_solution,
            improvements=tuple(improvements),
            stats={
                "expanded_nodes": expanded,
                "simulator_transitions": transitions,
                "peak_open_size": peak_open,
                "open_list_trimmed": trimmed,
                "search_complete": final_pass_complete,
                "weights": list(self.weights),
                "frontier_reused": True,
                "ara_open_closed_incons": True,
                "pass_stats": pass_stats,
                "strategy_search": diversity_stats,
                "maximal_start_seed": maximal_start_stats,
                "wall_seconds": elapsed,
                "cpu_seconds": time.process_time() - cpu_start,
                "peak_memory_bytes": peak_memory,
                "stopped_reason": stopped_reason,
                "path_count": len(paths),
                "archive_tier_counts": archive.tier_counts,
                "search_node_count": len(nodes),
                "parent_node_count": sum(node.parent_id is not None for node in nodes.values()),
                "trajectory_reconstruction_count": reconstruction_count,
                "stores_full_histories_per_node": False,
                "transition_graph": self.simulator.transition_graph.summary(),
                "relevance_pruning": self.relevance_pruning,
                "provider_pruning": self.provider_pruning,
                "dominated_activity_count": len(self._dominated_activities),
            },
            diagnosis={"closest_missing_goal_facts": None if paths else len(self.scenario.goal_state_ids)},
        )

    def _maximal_start_candidate(self, *, run_id: str, wall_start: float, transition_allowance: int):
        state = self.simulator.initial_state()
        used = 0
        while used < transition_allowance and not self.simulator.is_goal(state):
            enabled = self.simulator.enabled_actions(state)
            starts = [action for action in enabled if action.kind == "START"]
            if starts:
                action = starts[0]
            else:
                advances = [action for action in enabled if action.kind == "ADVANCE"]
                if not advances:
                    return None, used, "DEAD_END"
                action = advances[0]
            state = self.simulator.transition(state, action)
            used += 1
        if not self.simulator.is_goal(state):
            return None, used, "TRANSITION_LIMIT"
        candidate = self.simulator.candidate_from_state(
            state,
            algorithm=self.algorithm,
            run_id=run_id,
            seed=None,
            discovered_at_seconds=time.perf_counter() - wall_start,
        )
        return candidate, used, "GOAL"

    def _strategy_search(self, *, archive, nodes, id_counter, tie_counter, budget, run_id, wall_start, deadline, transitions_used, incumbent, improvements):
        covered = self._covered_strategy_provider_ids(archive)
        uncovered = self._strategy_provider_ids - covered
        found_signatures = {path.structure_signature for path in archive.paths()}
        missing_expected = set(self.scenario.expected_signatures) - found_signatures
        stats = {"enabled": bool(uncovered or missing_expected), "expanded": 0, "transitions": 0, "uncovered_provider_ids": sorted(uncovered), "missing_expected_signatures": sorted(missing_expected), "complete": False}
        if not uncovered and not missing_expected:
            stats["complete"] = True
            return stats, 0, 0, 0, 0, False
        root = self._root_node(next(id_counter))
        nodes[root.node_id] = root
        labels: dict[tuple, list[_Label]] = {}
        self._accept_label(labels, root, limit=self.labels_per_state)
        queue = [self._queue_entry(root, 1.0, next(tie_counter), uncovered=uncovered)]
        local_transitions = local_expanded = local_peak = rebuilds = 0
        trimmed = False
        limit = 10**9 if missing_expected else math.floor(incumbent * archive.strategy_bound)
        while queue and time.perf_counter() < deadline and transitions_used + local_transitions < budget.transition_limit:
            entry = heapq.heappop(queue)
            node = nodes[entry.node_id]
            if not self._label_active(labels, node):
                continue
            bound = self._earliest_achievement_bound(node.state)
            if node.state.time - self.scenario.start_time + bound > limit:
                continue
            if self.simulator.is_goal(node.state):
                elapsed = time.perf_counter() - wall_start
                candidate = self._candidate_from_node(node, nodes, run_id, elapsed)
                rebuilds += 1
                uncovered_mask = sum(self._strategy_provider_bits[item] for item in uncovered)
                if node.provider_mask & uncovered_mask or candidate.structure_signature in missing_expected:
                    _, improved = archive.add(candidate)
                    if improved:
                        improvements.append(ImprovementPoint(elapsed, candidate.metrics.sort_key, candidate.structure_signature))
                    covered = self._covered_strategy_provider_ids(archive)
                    uncovered = self._strategy_provider_ids - covered
                    found_signatures = {path.structure_signature for path in archive.paths()}
                    missing_expected = set(self.scenario.expected_signatures) - found_signatures
                    limit = 10**9 if missing_expected else math.floor(incumbent * archive.strategy_bound)
                    if not uncovered and not missing_expected:
                        break
                continue
            if self._at_step_limit(node):
                continue
            local_expanded += 1
            for action in self.simulator.enabled_actions(node.state):
                if transitions_used + local_transitions >= budget.transition_limit:
                    break
                if not self._action_allowed(action):
                    continue
                child = self._child_node(node, action, next(id_counter))
                local_transitions += 1
                next_bound = self._earliest_achievement_bound(child.state)
                if next_bound >= 10**9 or child.state.time - self.scenario.start_time + next_bound > limit:
                    continue
                if not self._accept_label(labels, child, limit=self.labels_per_state):
                    continue
                nodes[child.node_id] = child
                heapq.heappush(queue, self._queue_entry(child, 1.0, next(tie_counter), uncovered=uncovered))
            if len(queue) > self.max_open:
                queue = heapq.nsmallest(self.trim_to, queue)
                heapq.heapify(queue)
                trimmed = True
            local_peak = max(local_peak, len(queue))
        stats.update({"expanded": local_expanded, "transitions": local_transitions, "remaining_uncovered_provider_ids": sorted(uncovered), "remaining_expected_signatures": sorted(missing_expected), "complete": not uncovered and not missing_expected})
        return stats, local_transitions, local_expanded, local_peak, rebuilds, trimmed

    def _root_node(self, node_id: int) -> _SearchNode:
        initial = self.simulator.initial_state()
        seen_mask = sum(self._reversible_bits.get(state_id, 0) for state_id in initial.active_state_ids)
        return _SearchNode(
            node_id,
            SimState(
                facts=initial.facts,
                active_state_ids=initial.active_state_ids,
                time=initial.time,
                counts=initial.counts,
                running=initial.running,
                last_started_id=initial.last_started_id,
            ),
            None, None, 0, 0, 0, 0, 0, seen_mask, 0, 0,
        )

    def _child_node(self, parent: _SearchNode, action: Action, node_id: int) -> _SearchNode:
        transient = self.simulator.transition(parent.state, action)
        state = SimState(
            facts=transient.facts,
            active_state_ids=transient.active_state_ids,
            time=transient.time,
            counts=transient.counts,
            total_wait=transient.total_wait,
            running=transient.running,
            last_started_id=transient.last_started_id,
        )
        transition_count, revisit_count, regression_count = parent.transition_count, parent.revisit_count, parent.regression_count
        seen_mask = parent.seen_mask
        execution_count = parent.execution_count
        if action.kind in {"EXECUTE", "START"}:
            execution_count += 1
        if transient.active_state_ids != parent.state.active_state_ids:
            before, after = set(parent.state.active_state_ids), set(transient.active_state_ids)
            for old_state_id in before - after:
                for new_state_id in after - before:
                    if not self.simulator.transition_graph.is_reversible_transition(old_state_id, new_state_id):
                        continue
                    transition_count += 1
                    bit = self._reversible_bits.get(new_state_id, 0)
                    if bit and seen_mask & bit:
                        revisit_count += 1
                    seen_mask |= bit
            regression_count += self.simulator._goal_regressions(before, after)
        provider_mask = parent.provider_mask
        if action.kind in {"EXECUTE", "START"} and action.activity_id in self._strategy_provider_ids:
            provider_mask |= self._strategy_provider_bits[str(action.activity_id)]
        digest = ((parent.prefix_hash * 1099511628211) ^ zlib.crc32(action.label.encode("utf-8"))) & ((1 << 64) - 1)
        return _SearchNode(node_id, state, parent.node_id, action, parent.step_count + 1, execution_count, transition_count, revisit_count, regression_count, seen_mask, digest, provider_mask)

    def _candidate_from_node(self, node: _SearchNode, nodes: dict[int, _SearchNode], run_id: str, elapsed: float):
        actions: list[Action] = []
        current = node
        while current.parent_id is not None:
            assert current.action is not None
            actions.append(current.action)
            current = nodes[current.parent_id]
        state = self.simulator.initial_state()
        for action in reversed(actions):
            state = self.simulator.transition(state, action)
        return self.simulator.candidate_from_state(state, algorithm=self.algorithm, run_id=run_id, seed=None, discovered_at_seconds=elapsed)

    def _action_allowed(self, action: Action) -> bool:
        if self.relevance_pruning and action.kind in {"EXECUTE", "START"} and action.activity_id not in self._relevant_activities:
            return False
        if self.provider_pruning and action.kind in {"EXECUTE", "START"} and action.activity_id in self._dominated_activities:
            return False
        return True

    def _at_step_limit(self, node: _SearchNode) -> bool:
        if node.execution_count < self.scenario.max_steps:
            return False
        # max_steps limits activity instances, not real external-event or
        # completion boundaries. At the cap, temporal progress may still make
        # a goal true or finish already-running work.
        return not any(
            action.kind in {"WAIT", "ADVANCE"}
            for action in self.simulator.enabled_actions(node.state)
        )

    def _quality_prune(self, archive: PathArchive, incumbent: int | None, state: SimState, admissible: int) -> bool:
        return bool(self._incumbent_pruning_enabled(archive) and incumbent is not None and state.time - self.scenario.start_time + admissible > math.floor(incumbent * archive.quality_bound))

    def _covered_strategy_provider_ids(self, archive: PathArchive) -> set[str]:
        return {token.rsplit(":", 1)[-1] for path in archive.paths() for token in path.strategy_signature.split("|") if ":" in token}

    def _build_strategy_provider_ids(self) -> frozenset[str]:
        providers: set[str] = set()
        for goal in self.scenario.goal_state_ids:
            if goal in self.simulator.transition_graph.reversible_state_ids:
                continue
            alternatives = [activity for activity in self._providers.get(goal, ()) if activity.id not in self._dominated_activities]
            if len(alternatives) > 1:
                providers.update(activity.id for activity in alternatives)
        return frozenset(providers)

    def _build_provider_index(self) -> dict[str, list]:
        providers: dict[str, list] = {}
        for activity in self.scenario.activities:
            for state_id in activity.output_state_ids:
                providers.setdefault(state_id, []).append(activity)
        return providers

    def _incumbent_pruning_enabled(self, archive: PathArchive) -> bool:
        if not self.scenario.expected_signatures:
            return True
        found = {path.structure_signature for path in archive.paths()}
        return set(self.scenario.expected_signatures).issubset(found)

    def _build_relevance_set(self) -> frozenset[str]:
        needed, relevant, changed = set(self.scenario.goal_state_ids), set(), True
        while changed:
            changed = False
            for activity in self.scenario.activities:
                if activity.id in relevant or not (set(activity.output_state_ids) & needed):
                    continue
                relevant.add(activity.id)
                before = len(needed)
                needed.update(activity.precondition_state_ids)
                changed = changed or len(needed) != before
        return frozenset(relevant)

    def _build_dominated_set(self) -> frozenset[str]:
        dominated: set[str] = set()
        for activity in self.scenario.activities:
            for alternative in self.scenario.activities:
                if activity.id == alternative.id:
                    continue
                same_contract = activity.precondition_bindings == alternative.precondition_bindings and activity.output_state_ids == alternative.output_state_ids and activity.compatibility_remove_state_ids == alternative.compatibility_remove_state_ids and activity.required_events == alternative.required_events and activity.max_instances == alternative.max_instances
                if same_contract and activity.duration > alternative.duration:
                    dominated.add(activity.id)
        return frozenset(dominated)

    def _earliest_achievement_bound(self, state: SimState) -> int:
        running_key = tuple((item.activity_id, item.end_time, item.ordinal) for item in state.running)
        cache_key = (state.active_state_ids, state.counts, state.time, running_key)
        if cache_key in self._heuristic_cache:
            value = self._heuristic_cache.pop(cache_key)
            self._heuristic_cache[cache_key] = value
            return value
        active = set(state.active_state_ids)
        earliest = {state_id: state.time for state_id in active}
        for running in state.running:
            activity = self.scenario.activity_by_id[running.activity_id]
            for output_state_id in activity.output_state_ids:
                earliest[output_state_id] = min(earliest.get(output_state_id, 10**9), running.end_time)
        changed = True
        while changed:
            changed = False
            for activity in self.scenario.activities:
                if (
                    activity.max_instances is not None
                    and count_get(state.counts, activity.id) >= activity.max_instances
                ):
                    continue
                pre_times = []
                for state_id in activity.precondition_state_ids:
                    if state_id not in earliest:
                        break
                    pre_times.append(earliest[state_id])
                else:
                    event_time = max((self.simulator.events[item].time for item in activity.required_events), default=state.time)
                    finish = max(state.time, event_time, max(pre_times, default=state.time)) + activity.duration
                    for output_state_id in activity.output_state_ids:
                        if finish < earliest.get(output_state_id, 10**9):
                            earliest[output_state_id] = finish
                            changed = True
        goal_times = []
        for goal in self.scenario.goal_state_ids:
            if goal in active:
                continue
            if goal not in earliest:
                self._cache_heuristic(cache_key, 10**9)
                return 10**9
            goal_times.append(earliest[goal])
        result = max(0, max(goal_times, default=state.time) - state.time)
        if state.running:
            result = max(result, min(item.end_time for item in state.running) - state.time)
        for resource in self.scenario.resources:
            per_goal_work: list[int] = []
            for goal in self.scenario.goal_state_ids:
                if goal in active:
                    continue
                durations = [
                    activity.duration * dict(activity.resource_reqs).get(resource.id, 0)
                    for activity in self._providers.get(goal, ())
                    if dict(activity.resource_reqs).get(resource.id, 0)
                ]
                if durations:
                    per_goal_work.append(min(durations))
            # A single provider may satisfy several goals, so summing their work
            # could overestimate. The largest individual resource requirement is
            # a safe (if deliberately weak) lower bound.
            result = max(result, math.ceil(max(per_goal_work, default=0) / resource.capacity))
        self._cache_heuristic(cache_key, result)
        return result

    def _cache_heuristic(self, key: tuple, value: int) -> None:
        self._heuristic_cache[key] = value
        self._heuristic_cache.move_to_end(key)
        if len(self._heuristic_cache) > self._heuristic_cache_limit:
            self._heuristic_cache.popitem(last=False)

    def _aggressive_heuristic(self, state: SimState) -> int:
        admissible = self._earliest_achievement_bound(state)
        if admissible >= 10**9:
            return admissible
        active, direct = set(state.active_state_ids), 0
        for goal in self.scenario.goal_state_ids:
            if goal not in active:
                direct += min((item.duration for item in self._providers.get(goal, ())), default=0)
        return max(admissible, direct)

    def _queue_entry(self, node: _SearchNode, weight: float, counter: int, *, uncovered=frozenset()) -> _QueueEntry:
        heuristic = self._earliest_achievement_bound(node.state) if weight == 1.0 else self._aggressive_heuristic(node.state)
        uncovered_mask = sum(self._strategy_provider_bits[item] for item in uncovered)
        coverage = (node.provider_mask & uncovered_mask).bit_count()
        if self.scenario.execution_mode == "parallel":
            relevant_started = sum(
                count_get(node.state.counts, activity_id)
                for activity_id in self._relevant_activities
            )
            progress_tie = (-len(node.state.running), -relevant_started, node.execution_count)
        else:
            progress_tie = (0, 0, node.execution_count)
        return _QueueEntry(
            (
                node.state.time - self.scenario.start_time + weight * heuristic,
                -coverage,
                *progress_tie,
                node.transition_count,
                node.revisit_count,
                node.state.total_wait,
                counter,
            ),
            node.node_id,
        )

    def _label(self, node: _SearchNode) -> _Label:
        return _Label((node.state.time, node.execution_count, node.transition_count, node.revisit_count, node.regression_count, node.state.total_wait), node.prefix_hash)

    def _base_key(self, node: _SearchNode) -> tuple:
        return self.simulator.business_state_key(node.state)

    def _accept_label(self, labels: dict[tuple, list[_Label]], node: _SearchNode, *, limit: int) -> bool:
        key, new = self._base_key(node), self._label(node)
        existing = labels.setdefault(key, [])
        for item in existing:
            if item.prefix_hash == new.prefix_hash and item.values <= new.values:
                return False
            if all(left <= right for left, right in zip(item.values, new.values)):
                return False
        existing[:] = [item for item in existing if not all(left <= right for left, right in zip(new.values, item.values))]
        existing.append(new)
        existing.sort(key=lambda item: (item.values, item.prefix_hash))
        del existing[limit:]
        return new in existing

    def _label_active(self, labels: dict[tuple, list[_Label]], node: _SearchNode) -> bool:
        return self._label(node) in labels.get(self._base_key(node), ())

    def _label_token(self, node: _SearchNode) -> tuple:
        return self._base_key(node), node.prefix_hash


class RestartedWeightedAStar:
    """Test-only reference harness that restarts search for every weight."""

    algorithm = "ASTAR_REFERENCE"

    def __init__(self, scenario: Scenario, *, weights=(3.0, 1.5, 1.0), shares=(0.4, 0.3, 0.3)) -> None:
        self.scenario, self.weights, self.shares = scenario, weights, shares

    def run(self, scenario: Scenario, budget: Budget, *, run_id: str, seed: int | None = None) -> EngineResult:
        archive, results = PathArchive(budget.max_solutions, pinned_signatures=scenario.expected_signatures), []
        for index, (weight, share) in enumerate(zip(self.weights, self.shares)):
            result = AnytimeAStar(scenario, weights=(weight,), budget_shares=(1.0,), diversity_fraction=0.0).run(
                scenario,
                Budget(max(0.01, budget.time_limit_seconds * share), max(1, int(budget.transition_limit * share)), budget.max_solutions),
                run_id=f"{run_id}-w{index}",
            )
            results.append(result)
            for path in result.paths:
                archive.add(path)
        paths = archive.paths()
        first = next((item.first_solution_seconds for item in results if item.first_solution_seconds is not None), None)
        return EngineResult(self.algorithm, run_id, None, "OK" if paths else "TIMEOUT_EMPTY", paths, first, (), {"frontier_reused": False, "restart_count": len(results), "simulator_transitions": sum(item.stats.get("simulator_transitions", 0) for item in results), "peak_memory_bytes": max((item.stats.get("peak_memory_bytes", 0) for item in results), default=0)})
