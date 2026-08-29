from __future__ import annotations

import random
import time
import tracemalloc
from collections import OrderedDict
from dataclasses import dataclass, replace

from .archive import PathArchive
from .models import Action, Budget, CandidatePath, EngineResult, ImprovementPoint, Scenario, SimState
from .simulator import PathSimulator


FEATURES = (
    "goal_gain",
    "unlock_gain",
    "event_window_use",
    "event_unlock_gain",
    "duration",
    "transition",
    "revisit",
    "goal_regression",
    "redundant_effect",
    "resource_utilization_gain",
    "resource_remaining_capacity",
    "conflict_blocking",
    "projected_goal_gain",
    "critical_path_contribution",
)


@dataclass(frozen=True)
class _EvaluatedGene:
    genes: tuple[float, ...]
    fitness: tuple
    candidate: CandidatePath | None
    final_state: SimState
    terminal_reason: str
    candidates: tuple[CandidatePath, ...] = ()
    decision_sequences: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _DecisionFrame:
    state: SimState
    ranking: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _PhenotypeRecord:
    frames: tuple[_DecisionFrame, ...]
    evaluated: _EvaluatedGene


class GeneticExplorer:
    algorithm = "GA"

    def __init__(
        self,
        scenario: Scenario,
        *,
        population_size: int = 64,
        elite_count: int = 4,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        max_generations: int = 200,
        stagnation_generations: int = 15,
    ) -> None:
        self.scenario = scenario
        self.simulator = PathSimulator(scenario)
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.stagnation_generations = stagnation_generations
        self.activity_ids = tuple(activity.id for activity in sorted(scenario.activities, key=lambda item: item.id))
        self.advance_id = "__ADVANCE__" if scenario.execution_mode == "parallel" else "__WAIT__"
        self.action_ids = (*self.activity_ids, self.advance_id)
        self.activity_index = {activity_id: index for index, activity_id in enumerate(self.action_ids)}
        self.base_gene_count = len(self.action_ids)
        self.gene_count = self.base_gene_count + len(FEATURES)
        self.max_duration = max(activity.duration for activity in scenario.activities)
        self._relevant = self._build_relevance_set()

    def run(self, scenario: Scenario, budget: Budget, *, run_id: str, seed: int | None = None) -> EngineResult:
        if scenario != self.scenario:
            raise ValueError("Engine scenario does not match run scenario")
        actual_seed = 0 if seed is None else seed
        rng = random.Random(actual_seed)
        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        deadline = wall_start + budget.time_limit_seconds
        archive = PathArchive(budget.max_solutions, pinned_signatures=scenario.expected_signatures)
        improvements: list[ImprovementPoint] = []
        first_solution: float | None = None
        transitions = evaluations = generations = cache_hits = branch_rollouts = 0
        stopped_reason = "MAX_GENERATIONS"
        rule_seeds = self._rule_seed_genes()
        population = rule_seeds + [tuple(rng.random() for _ in range(self.gene_count)) for _ in range(self.population_size - len(rule_seeds))]
        phenotype_cache: OrderedDict[tuple, list[_PhenotypeRecord]] = OrderedDict()
        phenotype_record_count = 0
        global_best: tuple | None = None
        stagnant = 0

        for generation in range(self.max_generations):
            if time.perf_counter() >= deadline:
                stopped_reason = "TIME_LIMIT"
                break
            if transitions >= budget.transition_limit:
                stopped_reason = "TRANSITION_LIMIT"
                break
            evaluated: list[_EvaluatedGene] = []
            for genes in population:
                if time.perf_counter() >= deadline or transitions >= budget.transition_limit:
                    stopped_reason = "BUDGET"
                    break
                cache_key = self._coarse_policy_key(genes)
                item = self._reuse_phenotype(genes, phenotype_cache.get(cache_key, ()), run_id, actual_seed)
                if item is None:
                    item, used, frames = self._decode(
                        genes,
                        run_id=run_id,
                        seed=actual_seed,
                        wall_start=wall_start,
                        transition_allowance=budget.transition_limit - transitions,
                    )
                    transitions += used
                    branch_rollouts += len(item.decision_sequences)
                    phenotype_cache.setdefault(cache_key, []).append(_PhenotypeRecord(frames, item))
                    phenotype_cache.move_to_end(cache_key)
                    phenotype_record_count += 1
                    while phenotype_record_count > 256:
                        _, removed = phenotype_cache.popitem(last=False)
                        phenotype_record_count -= len(removed)
                else:
                    cache_hits += 1
                    phenotype_cache.move_to_end(cache_key)
                evaluations += 1
                evaluated.append(item)
                for candidate in item.candidates:
                    elapsed = time.perf_counter() - wall_start
                    _, improved = archive.add(candidate)
                    if first_solution is None:
                        first_solution = elapsed
                    if improved:
                        improvements.append(ImprovementPoint(elapsed, candidate.metrics.sort_key, candidate.structure_signature))
            if not evaluated:
                break
            generations = generation + 1
            evaluated.sort(key=lambda item: item.fitness)
            generation_best = evaluated[0].fitness
            if global_best is None or generation_best < global_best:
                global_best = generation_best
                stagnant = 0
            else:
                stagnant += 1
            next_population = [item.genes for item in self._diverse_elites(evaluated)]
            mutation_boost = 2.5 if stagnant >= self.stagnation_generations else 1.0
            while len(next_population) < self.population_size:
                parent_a = self._tournament(evaluated, rng).genes
                parent_b = self._tournament(evaluated, rng).genes
                child_a, child_b = self._crossover(parent_a, parent_b, rng)
                next_population.append(self._mutate(child_a, rng, mutation_boost))
                if len(next_population) < self.population_size:
                    next_population.append(self._mutate(child_b, rng, mutation_boost))
            if stagnant >= self.stagnation_generations:
                replace_count = max(1, self.population_size // 5)
                for index in range(self.population_size - replace_count, self.population_size):
                    next_population[index] = tuple(rng.random() for _ in range(self.gene_count))
                stagnant = 0
            population = next_population

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - wall_start
        paths = archive.paths()
        status = "TIMEOUT_PARTIAL" if paths and stopped_reason != "MAX_GENERATIONS" else "OK" if paths else "TIMEOUT_EMPTY"
        return EngineResult(
            algorithm=self.algorithm,
            run_id=run_id,
            seed=actual_seed,
            status=status,
            paths=paths,
            first_solution_seconds=first_solution,
            improvements=tuple(improvements),
            stats={
                "population_size": self.population_size,
                "generations": generations,
                "individual_evaluations": evaluations,
                "phenotype_cache_hits": cache_hits,
                "phenotype_cache_records": phenotype_record_count,
                "branch_rollouts": branch_rollouts,
                "max_rollouts_per_gene": 2,
                "simulator_transitions": transitions,
                "wall_seconds": elapsed,
                "cpu_seconds": time.process_time() - cpu_start,
                "peak_memory_bytes": peak_memory,
                "stopped_reason": stopped_reason,
                "path_count": len(paths),
                "archive_tier_counts": archive.tier_counts,
                "seed": actual_seed,
                "feature_names": list(FEATURES),
                "rule_seed_count": len(rule_seeds),
                "maximal_start_rule_seed": self.scenario.execution_mode == "parallel",
                "transition_graph": self.simulator.transition_graph.summary(),
            },
            diagnosis={"best_invalid_missing_goal_facts": None if paths else "not_recorded"},
        )

    def _decode(
        self,
        genes: tuple[float, ...],
        *,
        run_id: str,
        seed: int,
        wall_start: float,
        transition_allowance: int,
    ) -> tuple[_EvaluatedGene, int, tuple[_DecisionFrame, ...]]:
        primary, alternate = self.simulator.initial_state(), None
        primary_sequence: list[str] = []
        alternate_sequence: list[str] | None = None
        frames: list[_DecisionFrame] = []
        used = 0
        forked = False
        primary_seen = {self.simulator.business_state_key(primary)}
        primary, primary_reason, used, fork_state, fork_seen, frames = self._rollout(
            primary,
            genes,
            transition_allowance,
            used,
            frames,
            primary_sequence,
            allow_fork=True,
            seen=primary_seen,
        )
        if fork_state is not None and used < transition_allowance:
            forked = True
            alternate, alternate_sequence = fork_state, [action.label for action in fork_state.actions]
            alternate, alternate_reason, used, _, _, frames = self._rollout(
                alternate,
                genes,
                transition_allowance,
                used,
                frames,
                alternate_sequence,
                allow_fork=False,
                seen=fork_seen,
            )
        else:
            alternate_reason = "NOT_FORKED"
        outcomes = [(primary, primary_reason, tuple(primary_sequence))]
        if forked and alternate is not None and alternate_sequence is not None:
            outcomes.append((alternate, alternate_reason, tuple(alternate_sequence)))
        evaluated_outcomes = [self._evaluate_rollout(state, reason, run_id, seed, wall_start) for state, reason, _ in outcomes]
        best_index = min(range(len(evaluated_outcomes)), key=lambda index: evaluated_outcomes[index][0])
        fitness, candidate = evaluated_outcomes[best_index]
        candidates = tuple(
            result_candidate
            for _, result_candidate in evaluated_outcomes
            if result_candidate is not None
        )
        best_state, best_reason, _ = outcomes[best_index]
        item = _EvaluatedGene(
            genes,
            fitness,
            candidate,
            best_state,
            best_reason,
            candidates,
            tuple(sequence for _, _, sequence in outcomes),
        )
        return item, used, tuple(frames)

    def _rollout(self, state, genes, allowance, used, frames, sequence, *, allow_fork, seen=None):
        fork_state = None
        fork_seen = None
        seen = set(seen or (self.simulator.business_state_key(state),))
        terminal_reason = "DEAD_END"
        while used < allowance and self._can_continue_rollout(state):
            if self.simulator.is_goal(state):
                terminal_reason = "GOAL"
                break
            ranked = self._ranked_actions(state, genes)
            if not ranked:
                terminal_reason = "DEAD_END"
                break
            frames.append(_DecisionFrame(state, self._ranking_signature(ranked)))
            if allow_fork and fork_state is None and len(ranked) > 1 and self._scores_are_close(ranked[0][0], ranked[1][0]) and used + 2 <= allowance:
                fork_probe = self.simulator.transition(state, ranked[1][1])
                used += 1
                fork_key = self.simulator.business_state_key(fork_probe)
                if fork_key not in seen:
                    fork_state = fork_probe
                    fork_seen = {*seen, fork_key}
            chosen = ranked[0][1]
            next_state = self.simulator.transition(state, chosen)
            sequence.append(chosen.label)
            used += 1
            next_key = self.simulator.business_state_key(next_state)
            state = next_state
            if next_key in seen:
                terminal_reason = "REPEATED_STATE"
                break
            seen.add(next_key)
        if not self._can_continue_rollout(state) and not self.simulator.is_goal(state):
            terminal_reason = "MAX_STEPS"
        elif self.simulator.is_goal(state):
            terminal_reason = "GOAL"
        return state, terminal_reason, used, fork_state, fork_seen, frames

    def _evaluate_rollout(self, state, terminal_reason, run_id, seed, wall_start):
        if self.simulator.is_goal(state):
            candidate = self.simulator.candidate_from_state(
                state,
                algorithm=self.algorithm,
                run_id=run_id,
                seed=seed,
                discovered_at_seconds=time.perf_counter() - wall_start,
            )
            return candidate.metrics.sort_key, candidate
        metrics = self.simulator.metrics(state)
        return (
            1,
            self.simulator.missing_goal_facts(state),
            self._event_relaxed_distance(state),
            0 if terminal_reason == "MAX_STEPS" else 1,
            metrics.goal_regression_count,
            self._irrelevant_execution_count(state),
            len(state.executions),
            state.time,
        ), None

    def _action_score(self, action: Action, state: SimState, genes: tuple[float, ...]) -> float:
        base_id = action.activity_id if action.kind in {"EXECUTE", "START"} else self.advance_id
        base = genes[self.activity_index[str(base_id)]]
        weights = tuple((value - 0.5) * 4.0 for value in genes[self.base_gene_count :])
        features = self._action_features(action, state)
        return base + sum(weight * value for weight, value in zip(weights, features))

    def _ranked_actions(self, state: SimState, genes: tuple[float, ...]) -> list[tuple[float, Action]]:
        actions = self.simulator.enabled_actions(state)
        scored = [(self._action_score(action, state, genes), action) for action in actions]
        if not scored:
            return []
        regressions = {
            action.label: self._action_features(action, state)[7] > 0
            for _, action in scored
        }
        non_regressing = [
            (self._estimated_distance_after(action, state), action)
            for _, action in scored
            if not regressions[action.label]
        ]
        protected: list[tuple[float, Action]] = []
        dynamically_needed = self._needed_activity_ids(set(state.active_state_ids))
        useful_now = {
            action.activity_id
            for _, action in scored
            if action.kind in {"EXECUTE", "START"}
            and action.activity_id in dynamically_needed
            and not regressions[action.label]
        }
        for score, action in scored:
            if action.kind in {"EXECUTE", "START"} and action.activity_id not in dynamically_needed:
                score -= 4.0
            if action.kind in {"WAIT", "ADVANCE"} and useful_now and self.scenario.execution_mode == "serial":
                score -= 4.0
            if self._premature_goal_transition(action, state):
                score -= 4.0
            if regressions[action.label] and non_regressing:
                regressing_distance = self._estimated_distance_after(action, state)
                if min(distance for distance, _ in non_regressing) <= regressing_distance:
                    score -= 4.0
            protected.append((score, action))
        return sorted(protected, key=lambda item: (-item[0], item[1].label))

    def _can_continue_rollout(self, state: SimState) -> bool:
        return bool(self.simulator.enabled_actions(state))

    def _premature_goal_transition(self, action: Action, state: SimState) -> bool:
        if action.kind not in {"EXECUTE", "START"} or action.activity_id is None:
            return False
        activity = self.scenario.activity_by_id[action.activity_id]
        before = set(state.active_state_ids)
        after = (
            before - set(activity.transition_state_ids) - set(activity.compatibility_remove_state_ids)
        ) | set(activity.output_state_ids)
        needed_after = self._needed_activity_ids(after)
        for old_state_id in activity.transition_state_ids:
            for new_state_id in activity.output_state_ids:
                if new_state_id not in self.scenario.goal_state_ids:
                    continue
                if not self.simulator.transition_graph.is_reversible_transition(old_state_id, new_state_id):
                    continue
                if any(
                    old_state_id in self.scenario.activity_by_id[activity_id].output_state_ids
                    for activity_id in needed_after
                ):
                    return True
        return False

    def _estimated_distance_after(self, action: Action, state: SimState) -> int:
        active = set(state.active_state_ids)
        through_time = state.time
        if action.kind in {"WAIT", "ADVANCE"}:
            assert action.target_time is not None
            through_time = action.target_time
            self.simulator._apply_events(active, state.time, through_time)
            for running in state.running:
                if running.end_time == through_time:
                    activity = self.scenario.activity_by_id[running.activity_id]
                    active.difference_update((*activity.transition_state_ids, *activity.compatibility_remove_state_ids))
                    active.update(activity.output_state_ids)
        else:
            assert action.activity_id is not None
            activity = self.scenario.activity_by_id[action.activity_id]
            through_time = state.time if action.kind == "START" else state.time + activity.duration
            if action.kind == "EXECUTE":
                self.simulator._apply_events(active, state.time, through_time)
            active.difference_update((*activity.transition_state_ids, *activity.compatibility_remove_state_ids))
            active.update(activity.output_state_ids)
        frozen = tuple(sorted(active))
        probe = SimState(
            facts=self.scenario.compatibility_facts(frozen),
            active_state_ids=frozen,
            time=through_time,
            counts=state.counts,
            running=state.running,
        )
        return self._event_relaxed_distance(probe)

    @staticmethod
    def _scores_are_close(left: float, right: float) -> bool:
        return abs(left - right) <= 0.05 * max(1.0, abs(left), abs(right))

    @staticmethod
    def _ranking_signature(ranked: list[tuple[float, Action]]) -> tuple[tuple[str, float], ...]:
        if not ranked:
            return ()
        top = ranked[0][0]
        return tuple((action.label, round(top - score, 4)) for score, action in ranked)

    def _coarse_policy_key(self, genes: tuple[float, ...]) -> tuple:
        base_order = tuple(
            self.action_ids[index]
            for index in sorted(range(self.base_gene_count), key=lambda index: (-genes[index], self.action_ids[index]))
        )
        weights = tuple(round((value - 0.5) * 4.0, 1) for value in genes[self.base_gene_count :])
        return base_order, weights

    def _reuse_phenotype(
        self,
        genes: tuple[float, ...],
        records,
        run_id: str,
        seed: int,
    ) -> _EvaluatedGene | None:
        for record in records:
            if any(
                self._ranking_signature(self._ranked_actions(frame.state, genes)) != frame.ranking
                for frame in record.frames
            ):
                continue
            cached = record.evaluated
            candidates = tuple(replace(candidate, run_id=run_id, seed=seed) for candidate in cached.candidates)
            best = None
            if cached.candidate is not None:
                best = next(
                    candidate
                    for candidate in candidates
                    if candidate.structure_signature == cached.candidate.structure_signature
                    and candidate.metrics == cached.candidate.metrics
                )
            return replace(cached, genes=genes, candidate=best, candidates=candidates)
        return None

    def _action_features(self, action: Action, state: SimState) -> tuple[float, ...]:
        before = set(state.active_state_ids)
        after = set(before)
        if action.kind in {"WAIT", "ADVANCE"}:
            assert action.target_time is not None
            events = [event for event in self.scenario.external_events if state.time < event.time <= action.target_time]
            for event in events:
                after.difference_update(event.remove_state_ids)
                after.update(event.add_state_ids)
            for running in state.running:
                if running.end_time == action.target_time:
                    completing = self.scenario.activity_by_id[running.activity_id]
                    after.difference_update((*completing.transition_state_ids, *completing.compatibility_remove_state_ids))
                    after.update(completing.output_state_ids)
            duration = action.target_time - state.time
            event_unlock = len(
                self._enabled_relevant_ids(after, action.target_time)
                - self._enabled_relevant_ids(before, state.time)
            )
            transition = revisit = regression = redundant = 0
            window_use = 0.0
        else:
            assert action.activity_id is not None
            activity = self.scenario.activity_by_id[action.activity_id]
            after.difference_update((*activity.transition_state_ids, *activity.compatibility_remove_state_ids))
            after.update(activity.output_state_ids)
            duration = activity.duration
            event_unlock = 0
            transition_pairs = {
                (old_state_id, new_state_id)
                for old_state_id in activity.transition_state_ids
                for new_state_id in activity.output_state_ids
                if self.simulator.transition_graph.is_reversible_transition(old_state_id, new_state_id)
            }
            transition = len(transition_pairs)
            history = set(self.scenario.initial_state_ids)
            for record in state.executions:
                history.update(record.after_state_ids)
            revisit = sum(new_state_id in history for _, new_state_id in transition_pairs)
            regression = self.simulator._goal_regressions(before, after)
            redundant = len(set(activity.output_state_ids) & before)
            future = [event.time for event in self.scenario.external_events if event.time > state.time]
            gap = min(future) - state.time if future else 0
            window_use = min(duration, gap) / max(1, gap) if gap else 0.0
        goal_gain = len((set(self.scenario.goal_state_ids) & after) - before)
        # Fact unlocks and time/event unlocks are separate features so long
        # activities do not receive artificial credit merely for crossing an event.
        unlock_gain = len(
            self._enabled_relevant_ids(after, state.time)
            - self._enabled_relevant_ids(before, state.time)
        )
        parallel = self._parallel_features(action, state, before, after, duration)
        return (
            float(goal_gain),
            float(max(0, unlock_gain)),
            float(window_use),
            float(max(0, event_unlock)),
            duration / self.max_duration,
            float(transition),
            float(revisit),
            float(regression),
            float(redundant),
            *parallel,
        )

    def _parallel_features(
        self,
        action: Action,
        state: SimState,
        before: set[str],
        after: set[str],
        duration: int,
    ) -> tuple[float, ...]:
        if self.scenario.execution_mode != "parallel" or action.kind not in {"START", "EXECUTE"}:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        assert action.activity_id is not None
        activity = self.scenario.activity_by_id[action.activity_id]
        requirements = dict(activity.resource_reqs)
        total_capacity = sum(item.capacity for item in self.scenario.resources) or 1
        used = sum(requirements.values())
        occupied: dict[str, int] = {}
        for running in state.running:
            for resource_id, quantity in running.resource_allocations:
                occupied[resource_id] = occupied.get(resource_id, 0) + quantity
        remaining = sum(
            item.capacity - occupied.get(item.id, 0) - requirements.get(item.id, 0)
            for item in self.scenario.resources
        )
        reads = set(activity.required_state_ids)
        writes = set(activity.transition_state_ids) | set(activity.compatibility_remove_state_ids) | set(activity.output_state_ids)
        blocked = 0
        for other in self.scenario.activities:
            if other.id == activity.id:
                continue
            other_reads = set(other.required_state_ids)
            other_writes = set(other.transition_state_ids) | set(other.compatibility_remove_state_ids) | set(other.output_state_ids)
            if writes & (other_reads | other_writes) or reads & other_writes:
                blocked += 1
        projected_gain = len((set(self.scenario.goal_state_ids) & after) - before)
        return (
            used / total_capacity,
            max(0, remaining) / total_capacity,
            blocked / max(1, len(self.scenario.activities)),
            float(projected_gain),
            duration / self.max_duration,
        )

    def _enabled_relevant_ids(self, active: set[str], at_time: int) -> set[str]:
        result: set[str] = set()
        needed_activities = self._needed_activity_ids(active)
        for activity in self.scenario.activities:
            if activity.id not in needed_activities:
                continue
            if not set(activity.precondition_state_ids).issubset(active):
                continue
            if any(self.simulator.events[item].time > at_time for item in activity.required_events):
                continue
            result.add(activity.id)
        return result

    def _needed_activity_ids(self, active: set[str]) -> set[str]:
        needed = set(self.scenario.goal_state_ids) - active
        relevant: set[str] = set()
        changed = True
        while changed:
            changed = False
            for activity in self.scenario.activities:
                if activity.id in relevant or not (set(activity.output_state_ids) & needed):
                    continue
                relevant.add(activity.id)
                for state_id in activity.precondition_state_ids:
                    if state_id not in active and state_id not in needed:
                        needed.add(state_id)
                        changed = True
        return relevant

    def _event_relaxed_distance(self, state: SimState) -> int:
        active = set(state.active_state_ids)
        distance = 0
        for goal in self.scenario.goal_state_ids:
            if goal in active:
                continue
            running_finishes = [
                item.end_time - state.time
                for item in state.running
                if goal in self.scenario.activity_by_id[item.activity_id].output_state_ids
            ]
            if running_finishes:
                distance += min(running_finishes)
                continue
            providers = [activity for activity in self.scenario.activities if goal in activity.output_state_ids]
            if not providers:
                return 10**9
            distance += min(
                activity.duration
                + max((max(0, self.simulator.events[item].time - state.time) for item in activity.required_events), default=0)
                for activity in providers
            )
        return distance

    def _irrelevant_execution_count(self, state: SimState) -> int:
        return sum(record.activity_id not in self._relevant for record in state.executions)

    def _build_relevance_set(self) -> frozenset[str]:
        needed = set(self.scenario.goal_state_ids)
        relevant: set[str] = set()
        changed = True
        while changed:
            changed = False
            for activity in self.scenario.activities:
                if activity.id in relevant or not (set(activity.output_state_ids) & needed):
                    continue
                relevant.add(activity.id)
                old = len(needed)
                needed.update(activity.precondition_state_ids)
                changed = changed or len(needed) != old
        return frozenset(relevant)

    def _rule_seed_genes(self) -> list[tuple[float, ...]]:
        preferred_policy = (1.0, 1.0, 0.5, 2.0, -0.5, -0.5, -2.0, -2.0, -2.0, 1.0, 0.5, -1.0, 1.0, 1.0)
        policies = [
            (1.5, 2.0, 0.5, 1.0, -0.5, -0.5, -1.0, -2.0, -1.0, 1.0, 0.5, -1.0, 1.0, 1.0),
            (1.0, 1.0, 2.0, 1.5, -0.5, -0.5, -1.0, -2.0, -1.0, 0.5, 1.0, -2.0, 1.0, 0.5),
            (1.0, 1.0, 0.5, 2.0, -0.5, -0.5, -1.0, -2.0, -1.0, 2.0, 0.5, -1.0, 1.0, 1.0),
            (1.0, 1.0, 0.5, 1.0, -0.5, -2.0, -2.0, -2.0, -1.0, 0.5, 1.0, -2.0, 0.5, 0.5),
            (1.0, 1.0, 0.5, 1.0, -1.5, -0.5, -1.0, -2.0, -1.0, 1.0, 0.5, -1.0, 1.0, 2.0),
            (1.0, 1.5, 1.5, 1.0, -0.5, -0.5, -1.0, -2.0, -1.5, 1.5, 0.5, -1.0, 1.0, 1.0),
            (1.0, 1.0, 0.5, 1.0, -0.5, -0.5, -1.0, -2.0, -2.0, 0.5, 1.0, -2.0, 1.0, 0.5),
        ]
        bases = [0.25 + 0.5 * index / max(1, self.base_gene_count - 1) for index in range(self.base_gene_count)]
        targeted = [
            self._provider_seed_genes(provider_set, preferred_policy)
            for provider_set in self._provider_portfolios(limit=4)
        ]
        fallback = [
            tuple((*bases, *(min(1.0, max(0.0, weight / 4.0 + 0.5)) for weight in policy)))
            for policy in policies
        ]
        result = []
        if self.scenario.execution_mode == "parallel":
            result.append(self._maximal_start_seed_genes())
        result.extend((*targeted, *fallback))
        while len(result) < 8:
            result.append(tuple((*bases, *(0.5 for _ in FEATURES))))
        return result[:8]

    def _maximal_start_seed_genes(self) -> tuple[float, ...]:
        """Source-independent ASAP policy: fill each canonical start batch before advancing."""
        activity_count = max(1, len(self.activity_ids))
        bases = [
            1.0 - 0.8 * index / activity_count
            for index, _ in enumerate(self.activity_ids)
        ]
        bases.append(0.0)
        return tuple((*bases, *(0.5 for _ in FEATURES)))

    def _provider_seed_genes(
        self,
        preferred: set[str],
        policy: tuple[float, ...],
    ) -> tuple[float, ...]:
        bases: list[float] = []
        for activity_id in self.action_ids:
            if activity_id == self.advance_id:
                # Waiting is always available as a fallback; keep its base key
                # below executable providers so event-window seeds exploit work
                # released at the current boundary before waiting again.
                bases.append(0.05)
                continue
            activity = self.scenario.activity_by_id[activity_id]
            changes_reversible_state = any(
                self.simulator.transition_graph.is_reversible_transition(old_state_id, new_state_id)
                for old_state_id in activity.transition_state_ids
                for new_state_id in activity.output_state_ids
            )
            if changes_reversible_state:
                base = 0.05
            elif activity_id in preferred:
                base = 0.85
            elif activity_id in self._relevant:
                base = 0.15
            else:
                base = 0.0
            bases.append(min(1.0, base + 0.1 * (self.max_duration - activity.duration) / self.max_duration))
        return tuple((*bases, *(min(1.0, max(0.0, weight / 4.0 + 0.5)) for weight in policy)))

    def _provider_portfolios(self, *, limit: int) -> list[set[str]]:
        initial = set(self.scenario.initial_state_ids)
        goals = [goal for goal in self.scenario.goal_state_ids if goal not in initial]
        options = {
            goal: [activity for activity in self.scenario.activities if goal in activity.output_state_ids]
            for goal in goals
        }
        selectors = (
            lambda activity: (len(activity.required_events), -len(set(activity.output_state_ids) & set(goals)), activity.duration, activity.id),
            lambda activity: (-len(set(activity.output_state_ids) & set(goals)), activity.duration, activity.id),
            lambda activity: (len(set(activity.output_state_ids) & set(goals)), activity.duration, activity.id),
            lambda activity: (activity.duration, len(activity.precondition_state_ids), activity.id),
        )
        portfolios: list[set[str]] = []
        for selector in selectors:
            chosen = {
                min(candidates, key=selector).id
                for candidates in options.values()
                if candidates
            }
            portfolios.append(self._expand_provider_dependencies(chosen))
        unique: list[set[str]] = []
        for portfolio in portfolios:
            if portfolio and portfolio not in unique:
                unique.append(portfolio)
            if len(unique) >= limit:
                break
        return unique

    def _expand_provider_dependencies(self, selected: set[str]) -> set[str]:
        initial = set(self.scenario.initial_state_ids)
        result = set(selected)
        pending = list(selected)
        while pending:
            activity = self.scenario.activity_by_id[pending.pop()]
            available = initial | {
                state_id
                for activity_id in result
                for state_id in self.scenario.activity_by_id[activity_id].output_state_ids
            }
            for state_id in activity.precondition_state_ids:
                if state_id in available:
                    continue
                providers = [item for item in self.scenario.activities if state_id in item.output_state_ids]
                if not providers:
                    continue
                provider = min(
                    providers,
                    key=lambda item: (
                        sum(required not in initial for required in item.precondition_state_ids),
                        item.duration,
                        item.id,
                    ),
                )
                if provider.id not in result:
                    result.add(provider.id)
                    pending.append(provider.id)
        return result

    def _preferred_provider_set(self) -> set[str]:
        """Greedy relaxed provider cover used only to create an independent GA seed."""
        initial = set(self.scenario.initial_state_ids)
        needed = set(self.scenario.goal_state_ids) - initial
        selected: set[str] = set()
        for _ in range(len(self.scenario.activities) * 2):
            if not needed:
                break
            candidates = []
            for activity in self.scenario.activities:
                covered = len(set(activity.output_state_ids) & needed)
                if covered:
                    candidates.append((covered, -activity.duration, -len(activity.precondition_state_ids), activity.id, activity))
            if not candidates:
                break
            activity = max(candidates)[-1]
            selected.add(activity.id)
            needed.difference_update(activity.output_state_ids)
            needed.update(state_id for state_id in activity.precondition_state_ids if state_id not in initial)
        return selected

    def _diverse_elites(self, evaluated: list[_EvaluatedGene]) -> list[_EvaluatedGene]:
        selected: list[_EvaluatedGene] = []
        family_counts: dict[str, int] = {}
        family_cap = max(1, self.elite_count // 2)
        for item in evaluated:
            family = (
                item.candidate.strategy_signature or item.candidate.provider_signature
                if item.candidate is not None
                else f"invalid:{item.terminal_reason}:{self.simulator.missing_goal_facts(item.final_state)}"
            )
            if family_counts.get(family, 0):
                continue
            selected.append(item)
            family_counts[family] = 1
            if len(selected) >= self.elite_count:
                return selected
        for item in evaluated:
            family = (
                item.candidate.strategy_signature or item.candidate.provider_signature
                if item.candidate is not None
                else f"invalid:{item.terminal_reason}:{self.simulator.missing_goal_facts(item.final_state)}"
            )
            if item not in selected and family_counts.get(family, 0) < family_cap:
                selected.append(item)
                family_counts[family] = family_counts.get(family, 0) + 1
                if len(selected) >= self.elite_count:
                    break
        return selected

    def _tournament(self, population: list[_EvaluatedGene], rng: random.Random) -> _EvaluatedGene:
        contenders = [population[rng.randrange(len(population))] for _ in range(self.tournament_size)]
        return min(contenders, key=lambda item: item.fitness)

    def _crossover(
        self,
        parent_a: tuple[float, ...],
        parent_b: tuple[float, ...],
        rng: random.Random,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if rng.random() >= self.crossover_rate:
            return parent_a, parent_b
        child_a: list[float] = []
        child_b: list[float] = []
        for index, (left, right) in enumerate(zip(parent_a, parent_b)):
            if index < self.base_gene_count:
                if rng.random() < 0.5:
                    child_a.append(left)
                    child_b.append(right)
                else:
                    child_a.append(right)
                    child_b.append(left)
            else:
                alpha = rng.random()
                child_a.append(alpha * left + (1 - alpha) * right)
                child_b.append(alpha * right + (1 - alpha) * left)
        return tuple(child_a), tuple(child_b)

    def _mutate(self, genes: tuple[float, ...], rng: random.Random, boost: float) -> tuple[float, ...]:
        probability = min(0.5, boost / len(genes))
        mutated = list(genes)
        for index, value in enumerate(mutated):
            if rng.random() < probability:
                if rng.random() < 0.15:
                    value = rng.random()
                else:
                    value += rng.gauss(0.0, 0.15 * boost)
                mutated[index] = min(1.0, max(0.0, value))
        if rng.random() < probability and self.base_gene_count > 1:
            left = rng.randrange(self.base_gene_count)
            right = rng.randrange(self.base_gene_count)
            mutated[left], mutated[right] = mutated[right], mutated[left]
        return tuple(mutated)
