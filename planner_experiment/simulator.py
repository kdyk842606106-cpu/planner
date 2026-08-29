from __future__ import annotations

import hashlib
from dataclasses import replace

from .models import (
    Action,
    CandidatePath,
    ExecutionRecord,
    FactsDelta,
    PathMetrics,
    PlanEdge,
    RunningActivity,
    Scenario,
    SimState,
    count_get,
    count_increment,
    freeze_mapping,
    freeze_state_ids,
    thaw_mapping,
)
from .transition_graph import StateTransitionGraph


class InvalidAction(ValueError):
    pass


class PathSimulator:
    """Deterministic serial/temporal event simulator shared by both engines."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.activities = scenario.activity_by_id
        self.events = scenario.event_by_id
        self.transition_graph = StateTransitionGraph.build(scenario)

    def initial_state(self) -> SimState:
        state_ids = set(self.scenario.initial_state_ids)
        self._apply_events(state_ids, None, self.scenario.start_time)
        frozen = freeze_state_ids(state_ids)
        return SimState(
            facts=self.scenario.compatibility_facts(frozen),
            active_state_ids=frozen,
            time=self.scenario.start_time,
        )

    def is_goal(self, state: SimState) -> bool:
        if state.running:
            return False
        active = set(state.active_state_ids)
        if not set(self.scenario.goal_state_ids).issubset(active):
            return False
        if set(self.scenario.forbidden_state_ids) & active:
            return False
        return True

    def missing_goal_facts(self, state: SimState) -> int:
        active = set(state.active_state_ids)
        missing = len(set(self.scenario.goal_state_ids) - active)
        missing += len(set(self.scenario.forbidden_state_ids) & active)
        return int(missing)

    def business_state_key(self, state: SimState) -> tuple:
        """Semantic state used for search dominance and loop detection.

        Absolute time and counters for unlimited activities are deliberately
        excluded. Search labels compare time and path costs separately, while
        the event phase and remaining running durations retain time-dependent
        behavior.
        """
        occurred_events = tuple(
            event.id
            for event in self.scenario.external_events
            if event.time <= state.time
        )
        running = tuple(
            sorted(
                (
                    item.activity_id,
                    item.end_time - state.time,
                    item.resource_allocations,
                )
                for item in state.running
            )
        )
        resource_usage: dict[str, int] = {}
        for item in state.running:
            for resource_id, quantity in item.resource_allocations:
                resource_usage[resource_id] = resource_usage.get(resource_id, 0) + quantity
        limited_counts = tuple(
            (activity.id, count_get(state.counts, activity.id))
            for activity in sorted(self.scenario.activities, key=lambda item: item.id)
            if activity.max_instances is not None
        )
        return (
            state.active_state_ids,
            occurred_events,
            running,
            tuple(sorted(resource_usage.items())),
            state.last_started_id,
            limited_counts,
        )

    @staticmethod
    def activity_instance_count(state: SimState) -> int:
        return sum(value for _, value in state.counts)

    @staticmethod
    def _within_activity_limit(state: SimState, activity) -> bool:
        return (
            activity.max_instances is None
            or count_get(state.counts, activity.id) < activity.max_instances
        )

    def enabled_actions(self, state: SimState) -> tuple[Action, ...]:
        if self.scenario.execution_mode == "parallel":
            return self._enabled_parallel(state)
        if self.is_goal(state):
            return ()
        active = set(state.active_state_ids)
        actions: list[Action] = []
        if self.activity_instance_count(state) < self.scenario.max_steps:
            for activity in sorted(self.scenario.activities, key=lambda item: item.id):
                if not self._within_activity_limit(state, activity):
                    continue
                if not set(activity.precondition_state_ids).issubset(active):
                    continue
                if any(self.events[event_id].time > state.time for event_id in activity.required_events):
                    continue
                actions.append(Action(kind="EXECUTE", activity_id=activity.id))
        future_events = sorted({event.time for event in self.events.values() if event.time > state.time})
        if future_events:
            actions.append(Action(kind="WAIT", target_time=future_events[0]))
        return tuple(actions)

    def transition(self, state: SimState, action: Action) -> SimState:
        if self.scenario.execution_mode == "parallel":
            return self._transition_parallel(state, action)
        if action not in self.enabled_actions(state):
            raise InvalidAction(f"Action {action.label} is not enabled at time {state.time}")
        if action.kind == "WAIT":
            assert action.target_time is not None
            wait = action.target_time - state.time
            if wait <= 0:
                raise InvalidAction("WAIT target must be in the future")
            active = set(state.active_state_ids)
            self._apply_events(active, state.time, action.target_time)
            frozen = freeze_state_ids(active)
            return SimState(
                facts=self.scenario.compatibility_facts(frozen),
                active_state_ids=frozen,
                time=action.target_time,
                counts=state.counts,
                actions=(*state.actions, action),
                executions=state.executions,
                total_wait=state.total_wait + wait,
            )

        assert action.activity_id is not None
        activity = self.activities[action.activity_id]
        before_ids = set(state.active_state_ids)
        after_ids = set(before_ids)
        start = state.time
        end = start + activity.duration
        self._apply_events(after_ids, start, end)
        after_ids.difference_update((*activity.transition_state_ids, *activity.compatibility_remove_state_ids))
        after_ids.update(activity.output_state_ids)
        before_frozen = freeze_state_ids(before_ids)
        after_frozen = freeze_state_ids(after_ids)
        ordinal = count_get(state.counts, activity.id) + 1
        record = ExecutionRecord(
            instance_id=f"{activity.id}#{ordinal}",
            activity_id=activity.id,
            activity_name=activity.name,
            ordinal=ordinal,
            start_time=start,
            end_time=end,
            before_facts=self.scenario.compatibility_facts(before_frozen),
            after_facts=self.scenario.compatibility_facts(after_frozen),
            trigger_reason=self._trigger_reason(activity.id, state),
            before_state_ids=before_frozen,
            after_state_ids=after_frozen,
        )
        return SimState(
            facts=self.scenario.compatibility_facts(after_frozen),
            active_state_ids=after_frozen,
            time=end,
            counts=count_increment(state.counts, activity.id),
            actions=(*state.actions, action),
            executions=(*state.executions, record),
            total_wait=state.total_wait,
        )

    def _enabled_parallel(self, state: SimState) -> tuple[Action, ...]:
        if self.is_goal(state):
            return ()
        active = set(state.active_state_ids)
        started_count = self.activity_instance_count(state)
        actions: list[Action] = []
        if started_count < self.scenario.max_steps:
            for activity in sorted(self.scenario.activities, key=lambda item: item.id):
                if state.last_started_id is not None and activity.id < state.last_started_id:
                    continue
                if not self._within_activity_limit(state, activity):
                    continue
                if not set(activity.precondition_state_ids).issubset(active):
                    continue
                if any(self.events[event_id].time > state.time for event_id in activity.required_events):
                    continue
                if not self._resources_available(state, activity.resource_reqs):
                    continue
                if self._has_running_lock_conflict(state, activity.id):
                    continue
                if self._event_breaks_invariant(activity.id, state.time, state.time + activity.duration):
                    continue
                actions.append(Action(kind="START", activity_id=activity.id))
        next_times = [item.end_time for item in state.running if item.end_time > state.time]
        next_times.extend(event.time for event in self.events.values() if event.time > state.time)
        if next_times:
            actions.append(Action(kind="ADVANCE", target_time=min(next_times)))
        return tuple(actions)

    def _transition_parallel(self, state: SimState, action: Action) -> SimState:
        # Legacy actions remain replayable. EXECUTE is a convenience for one isolated activity.
        if action.kind == "EXECUTE":
            if state.running:
                raise InvalidAction("legacy EXECUTE requires an empty running set in parallel mode")
            started = self._transition_parallel(state, Action("START", activity_id=action.activity_id))
            target = next(item.end_time for item in started.running if item.activity_id == action.activity_id)
            current = started
            while current.time < target:
                next_time = min(
                    [item.end_time for item in current.running if item.end_time > current.time]
                    + [event.time for event in self.events.values() if current.time < event.time <= target]
                )
                current = self._transition_parallel(current, Action("ADVANCE", target_time=next_time))
            return replace(current, actions=(*state.actions, action))
        if action.kind == "WAIT":
            action = Action("ADVANCE", target_time=action.target_time)
        if action not in self._enabled_parallel(state):
            raise InvalidAction(f"Action {action.label} is not enabled at time {state.time}")
        if action.kind == "START":
            assert action.activity_id is not None
            activity = self.activities[action.activity_id]
            ordinal = count_get(state.counts, activity.id) + 1
            running = RunningActivity(
                instance_id=f"{activity.id}#{ordinal}",
                activity_id=activity.id,
                ordinal=ordinal,
                start_time=state.time,
                end_time=state.time + activity.duration,
                start_facts=state.facts,
                start_state_ids=state.active_state_ids,
                resource_allocations=activity.resource_reqs,
            )
            return SimState(
                facts=state.facts,
                active_state_ids=state.active_state_ids,
                time=state.time,
                counts=count_increment(state.counts, activity.id),
                actions=(*state.actions, action),
                executions=state.executions,
                total_wait=state.total_wait,
                running=tuple(sorted((*state.running, running), key=lambda item: (item.end_time, item.activity_id, item.ordinal))),
                last_started_id=activity.id,
            )

        assert action.kind == "ADVANCE" and action.target_time is not None
        target = action.target_time
        before_ids = set(state.active_state_ids)
        after_events = set(before_ids)
        self._apply_events(after_events, state.time, target)
        completing = tuple(item for item in state.running if item.end_time == target)
        survivors = tuple(item for item in state.running if item.end_time != target)
        for running in survivors:
            activity = self.activities[running.activity_id]
            if not set(activity.precondition_state_ids).issubset(after_events):
                raise InvalidAction(f"invariant broken for {running.instance_id} at {target}")
        writes: set[str] = set()
        for running in completing:
            activity = self.activities[running.activity_id]
            activity_writes = set(activity.transition_state_ids) | set(activity.compatibility_remove_state_ids) | set(activity.output_state_ids)
            overlap = writes & activity_writes
            if overlap:
                raise InvalidAction(f"simultaneous completions write {sorted(overlap)[0]}")
            writes.update(activity_writes)
        after = set(after_events)
        for running in completing:
            activity = self.activities[running.activity_id]
            after.difference_update((*activity.transition_state_ids, *activity.compatibility_remove_state_ids))
        for running in completing:
            after.update(self.activities[running.activity_id].output_state_ids)
        for running in survivors:
            activity = self.activities[running.activity_id]
            if not set(activity.precondition_state_ids).issubset(after):
                raise InvalidAction(f"completion breaks invariant for {running.instance_id} at {target}")
        final_state_ids = freeze_state_ids(after)
        final_facts = self.scenario.compatibility_facts(final_state_ids)
        before_completion_state_ids = freeze_state_ids(after_events)
        completed_records = tuple(
            ExecutionRecord(
                instance_id=item.instance_id,
                activity_id=item.activity_id,
                activity_name=self.activities[item.activity_id].name,
                ordinal=item.ordinal,
                start_time=item.start_time,
                end_time=item.end_time,
                before_facts=self.scenario.compatibility_facts(before_completion_state_ids),
                after_facts=final_facts,
                trigger_reason=self._trigger_reason(
                    item.activity_id,
                    replace(
                        state,
                        facts=item.start_facts,
                        active_state_ids=item.start_state_ids,
                        time=item.start_time,
                    ),
                ),
                before_state_ids=before_completion_state_ids,
                after_state_ids=final_state_ids,
            )
            for item in sorted(completing, key=lambda item: (item.activity_id, item.ordinal))
        )
        idle = target - state.time if not state.running else 0
        return SimState(
            facts=final_facts,
            active_state_ids=final_state_ids,
            time=target,
            counts=state.counts,
            actions=(*state.actions, action),
            executions=(*state.executions, *completed_records),
            total_wait=state.total_wait + idle,
            running=survivors,
            last_started_id=None,
        )

    def _resources_available(self, state: SimState, requirements: tuple[tuple[str, int], ...]) -> bool:
        used: dict[str, int] = {}
        for running in state.running:
            for resource_id, quantity in running.resource_allocations:
                used[resource_id] = used.get(resource_id, 0) + quantity
        return all(
            used.get(resource_id, 0) + quantity <= self.scenario.resource_by_id[resource_id].capacity
            for resource_id, quantity in requirements
        )

    def _has_running_lock_conflict(self, state: SimState, activity_id: str) -> bool:
        activity = self.activities[activity_id]
        reads = set(activity.required_state_ids)
        writes = set(activity.transition_state_ids) | set(activity.compatibility_remove_state_ids) | set(activity.output_state_ids)
        for running in state.running:
            other = self.activities[running.activity_id]
            other_reads = set(other.required_state_ids)
            other_writes = set(other.transition_state_ids) | set(other.compatibility_remove_state_ids) | set(other.output_state_ids)
            if writes & (other_reads | other_writes) or reads & other_writes:
                return True
        return False

    def _event_breaks_invariant(self, activity_id: str, start: int, end: int) -> bool:
        required = set(self.activities[activity_id].precondition_state_ids)
        for event in self.events.values():
            if not start < event.time < end:
                continue
            if required & set(event.remove_state_ids):
                return True
        return False

    def _trigger_reason(self, activity_id: str, state: SimState) -> str:
        activity = self.activities[activity_id]
        reasons = [
            f"{item.relation_role}:{item.state_id}" for item in activity.precondition_bindings
        ]
        reasons.extend(f"event:{item}@{self.events[item].time}" for item in activity.required_events)
        return ", ".join(reasons) if reasons else "no_precondition"

    def _apply_events(self, state_ids: set[str], after_time: int | None, through_time: int) -> None:
        for event in sorted(self.events.values(), key=lambda item: (item.time, item.id)):
            if event.time > through_time or (after_time is not None and event.time <= after_time):
                continue
            state_ids.difference_update(event.remove_state_ids)
            state_ids.update(event.add_state_ids)

    def normalize_state(self, state: SimState) -> SimState:
        """Replay an activity sequence at its deterministic earliest schedule."""
        if self.scenario.execution_mode == "parallel":
            return state
        normalized = self.initial_state()
        for execution in state.executions:
            action = Action("EXECUTE", activity_id=execution.activity_id)
            while action not in self.enabled_actions(normalized):
                waits = [item for item in self.enabled_actions(normalized) if item.kind == "WAIT"]
                if not waits:
                    raise InvalidAction(f"Cannot normalize activity {execution.activity_id}")
                normalized = self.transition(normalized, waits[0])
            normalized = self.transition(normalized, action)
        while not self.is_goal(normalized):
            waits = [item for item in self.enabled_actions(normalized) if item.kind == "WAIT"]
            if not waits:
                break
            normalized = self.transition(normalized, waits[0])
        return normalized

    def candidate_from_state(
        self,
        state: SimState,
        *,
        algorithm: str,
        run_id: str,
        seed: int | None,
        discovered_at_seconds: float,
        normalize: bool = True,
    ) -> CandidatePath:
        raw_makespan = state.time - self.scenario.start_time
        if normalize and self.scenario.execution_mode == "serial":
            state = self.normalize_state(state)
        ordered_executions = tuple(sorted(state.executions, key=lambda item: (item.start_time, item.activity_id, item.ordinal)))
        signature = ">".join(item.activity_id for item in ordered_executions)
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        edges, causal_core, provider_signature, strategy_signature = self._causal_analysis(state.executions)
        metrics = self.metrics(state, non_causal_action_count=len(state.executions) - len(causal_core))
        if self.scenario.execution_mode == "parallel":
            groups: dict[int, list[str]] = {}
            for record in ordered_executions:
                groups.setdefault(record.start_time, []).append(record.activity_id)
            schedule_signature = ">".join(
                f"t{time}:[{','.join(sorted(activity_ids))}]"
                for time, activity_ids in sorted(groups.items())
            )
        else:
            schedule_signature = ">".join(
                action.activity_id if action.kind == "EXECUTE" else f"WAIT@{action.target_time}"
                for action in state.actions
            )
        causal_core_signature = ">".join(
            record.activity_id for record in state.executions if record.instance_id in causal_core
        )
        return CandidatePath(
            path_id=f"{algorithm.lower()}-{digest}",
            algorithm=algorithm,
            run_id=run_id,
            seed=seed,
            actions=state.actions,
            executions=state.executions,
            partial_order=edges,
            state_trajectory=self._derive_trajectory(state.executions),
            final_facts=state.facts,
            final_state_ids=state.active_state_ids,
            structure_signature=signature,
            schedule_signature=schedule_signature,
            causal_core_signature=causal_core_signature,
            provider_signature=provider_signature,
            strategy_signature=strategy_signature,
            metrics=metrics,
            raw_makespan=raw_makespan,
            normalization_saved_time=max(0, raw_makespan - metrics.makespan) if self.scenario.execution_mode == "serial" else 0,
            discovered_at_seconds=discovered_at_seconds,
        )

    def metrics(self, state: SimState, *, non_causal_action_count: int = 0) -> PathMetrics:
        transition_count = 0
        revisit_count = 0
        regression_count = 0
        current = set(self.scenario.initial_state_ids)
        self._apply_events(current, None, self.scenario.start_time)
        seen_state_ids = set(current)
        timeline: dict[int, list[tuple[str, object]]] = {}
        for event in self.scenario.external_events:
            if self.scenario.start_time < event.time <= state.time:
                timeline.setdefault(event.time, []).append(("event", event))
        for record in state.executions:
            timeline.setdefault(record.end_time, []).append(("activity", record))
        for time in sorted(timeline):
            event_after = set(current)
            for kind, item in sorted(timeline[time], key=lambda value: (0 if value[0] == "event" else 1, getattr(value[1], "id", getattr(value[1], "activity_id", "")))):
                if kind == "event":
                    event_after.difference_update(item.remove_state_ids)
                    event_after.update(item.add_state_ids)
            transition_count, revisit_count, regression_count = self._accumulate_change_metrics(
                current, event_after, seen_state_ids, transition_count, revisit_count, regression_count
            )
            current = event_after
            activity_after = set(current)
            for kind, item in timeline[time]:
                if kind == "activity":
                    activity = self.activities[item.activity_id]
                    activity_after.difference_update((*activity.transition_state_ids, *activity.compatibility_remove_state_ids))
            for kind, item in timeline[time]:
                if kind == "activity":
                    activity_after.update(self.activities[item.activity_id].output_state_ids)
            transition_count, revisit_count, regression_count = self._accumulate_change_metrics(
                current, activity_after, seen_state_ids, transition_count, revisit_count, regression_count
            )
            current = activity_after
        makespan = state.time - self.scenario.start_time
        duration_sum = sum(self.activities[item.activity_id].duration for item in state.executions)
        serial_baseline = self._serial_baseline(state.executions) if self.scenario.execution_mode == "parallel" else makespan
        peak_parallelism, average_parallelism = self._parallelism(state.executions, makespan)
        resource_peak, resource_utilization = self._resource_metrics(state.executions, makespan)
        return PathMetrics(
            goal_check=self.is_goal(state),
            makespan=makespan,
            execution_count=len(state.executions),
            transition_count=transition_count,
            state_revisit_count=revisit_count,
            goal_regression_count=regression_count,
            non_causal_action_count=non_causal_action_count,
            total_wait=state.total_wait,
            missing_goal_facts=self.missing_goal_facts(state),
            activity_duration_sum=duration_sum,
            serial_baseline_makespan=serial_baseline,
            parallel_savings=max(0, serial_baseline - makespan),
            compression_ratio=(max(0, serial_baseline - makespan) / serial_baseline) if serial_baseline else 0.0,
            peak_parallelism=peak_parallelism,
            average_parallelism=average_parallelism,
            resource_peak=resource_peak,
            resource_utilization=resource_utilization,
            critical_path_length=self._critical_path_length(state.executions),
            idle_wait_time=state.total_wait,
        )

    def _accumulate_change_metrics(
        self,
        before: set[str],
        after: set[str],
        seen_state_ids: set[str],
        transition_count: int,
        revisit_count: int,
        regression_count: int,
    ) -> tuple[int, int, int]:
        removed, added = before - after, after - before
        reversible_added: set[str] = set()
        for old_state_id in removed:
            for new_state_id in added:
                if self.transition_graph.is_reversible_transition(old_state_id, new_state_id):
                    transition_count += 1
                    reversible_added.add(new_state_id)
        revisit_count += len(reversible_added & seen_state_ids)
        seen_state_ids.update(added)
        regression_count += self._goal_regressions(before, after)
        return transition_count, revisit_count, regression_count

    def _serial_baseline(self, executions: tuple[ExecutionRecord, ...]) -> int:
        serial = PathSimulator(replace(self.scenario, execution_mode="serial"))
        state = serial.initial_state()
        try:
            for record in sorted(executions, key=lambda item: (item.start_time, item.activity_id, item.ordinal)):
                action = Action("EXECUTE", activity_id=record.activity_id)
                while action not in serial.enabled_actions(state):
                    waits = [item for item in serial.enabled_actions(state) if item.kind == "WAIT"]
                    if not waits:
                        raise InvalidAction(f"cannot serially replay {record.activity_id}")
                    state = serial.transition(state, waits[0])
                state = serial.transition(state, action)
            return state.time - self.scenario.start_time
        except InvalidAction:
            # Keep reporting useful for a valid temporal path even when its
            # arbitrary sibling order is not directly serial-replayable.
            return sum(self.activities[item.activity_id].duration for item in executions)

    @staticmethod
    def _parallelism(executions: tuple[ExecutionRecord, ...], makespan: int) -> tuple[int, float]:
        points: list[tuple[int, int]] = []
        for record in executions:
            points.extend(((record.start_time, 1), (record.end_time, -1)))
        active = peak = area = 0
        previous: int | None = None
        for time, delta in sorted(points, key=lambda item: (item[0], item[1])):
            if previous is not None:
                area += active * (time - previous)
            active += delta
            peak = max(peak, active)
            previous = time
        return max(peak, 1 if executions else 0), (area / makespan if makespan else 0.0)

    def _resource_metrics(
        self, executions: tuple[ExecutionRecord, ...], makespan: int
    ) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, float], ...]]:
        peak: dict[str, int] = {}
        area: dict[str, int] = {}
        for resource in self.scenario.resources:
            points: list[tuple[int, int]] = []
            for record in executions:
                quantity = dict(self.activities[record.activity_id].resource_reqs).get(resource.id, 0)
                if quantity:
                    points.extend(((record.start_time, quantity), (record.end_time, -quantity)))
                    area[resource.id] = area.get(resource.id, 0) + quantity * (record.end_time - record.start_time)
            current = 0
            for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
                current += delta
                peak[resource.id] = max(peak.get(resource.id, 0), current)
        utilization = {
            item.id: (area.get(item.id, 0) / (item.capacity * makespan) if makespan else 0.0)
            for item in self.scenario.resources
        }
        return tuple(sorted(peak.items())), tuple(sorted(utilization.items()))

    def _critical_path_length(self, executions: tuple[ExecutionRecord, ...]) -> int:
        if not executions:
            return 0
        ordered = sorted(executions, key=lambda item: (item.end_time, item.instance_id))
        longest: dict[str, int] = {}
        for record in ordered:
            activity = self.activities[record.activity_id]
            predecessors: list[int] = []
            for state_id in activity.precondition_state_ids:
                providers = [
                    item for item in ordered
                    if item.end_time <= record.start_time
                    and state_id in self.activities[item.activity_id].output_state_ids
                ]
                if providers:
                    provider = max(providers, key=lambda item: (item.end_time, item.instance_id))
                    predecessors.append(longest.get(provider.instance_id, 0))
            release_bound = max(
                (self.events[item].time - self.scenario.start_time for item in activity.required_events),
                default=0,
            )
            longest[record.instance_id] = activity.duration + max([release_bound, *predecessors], default=0)
        return max(longest.values(), default=0)

    def _goal_regressions(self, before: set[str], after: set[str]) -> int:
        result = sum(state_id in before and state_id not in after for state_id in self.scenario.goal_state_ids)
        result += sum(state_id not in before and state_id in after for state_id in self.scenario.forbidden_state_ids)
        return int(result)

    @staticmethod
    def _derive_trajectory(executions: tuple[ExecutionRecord, ...]) -> tuple[FactsDelta, ...]:
        trajectory: list[FactsDelta] = []
        for record in executions:
            before, after = set(record.before_state_ids), set(record.after_state_ids)
            changes = freeze_mapping(
                {state_id: "active" for state_id in sorted(after - before)}
            )
            trajectory.append(
                FactsDelta(
                    record.end_time,
                    record.activity_id,
                    changes,
                    added_state_ids=freeze_state_ids(after - before),
                    removed_state_ids=freeze_state_ids(before - after),
                )
            )
        return tuple(trajectory)

    def _causal_analysis(
        self, executions: tuple[ExecutionRecord, ...]
    ) -> tuple[tuple[PlanEdge, ...], set[str], str, str]:
        executions = tuple(sorted(executions, key=lambda item: (item.end_time, item.activity_id, item.ordinal)))
        edges: list[PlanEdge] = []
        latest_writer: dict[str, str] = {}
        seen: set[tuple[str, str, str, str | None]] = set()
        activity_by_instance = {record.instance_id: record.activity_id for record in executions}
        for index, record in enumerate(executions):
            activity = self.activities[record.activity_id]
            for state_id in activity.precondition_state_ids:
                eligible = [
                    item for item in executions[:index]
                    if item.end_time <= record.start_time
                    and state_id in self.activities[item.activity_id].output_state_ids
                ]
                provider = max(eligible, key=lambda item: (item.end_time, item.instance_id)).instance_id if eligible else None
                if provider:
                    marker = (provider, record.instance_id, "state_causal", state_id)
                    if marker not in seen:
                        edges.append(PlanEdge(provider, record.instance_id, "state_causal", state_id))
                        seen.add(marker)
            if self.scenario.execution_mode == "serial" and index > 0:
                previous = executions[index - 1]
                marker = (previous.instance_id, record.instance_id, "serial_experiment", None)
                if marker not in seen:
                    edges.append(PlanEdge(previous.instance_id, record.instance_id, "serial_experiment"))
                    seen.add(marker)
            for state_id in activity.output_state_ids:
                latest_writer[state_id] = record.instance_id
        if self.scenario.execution_mode == "parallel":
            self._append_temporal_order_edges(executions, edges, seen)
        goal_provider = {
            goal: latest_writer[goal]
            for goal in self.scenario.goal_state_ids
            if goal in latest_writer
        }
        roots = set(goal_provider.values())
        incoming: dict[str, list[PlanEdge]] = {}
        for edge in edges:
            if edge.edge_type == "state_causal":
                incoming.setdefault(edge.to_instance, []).append(edge)
        core = set(roots)
        pending = list(roots)
        while pending:
            target = pending.pop()
            for edge in incoming.get(target, []):
                if edge.from_instance not in core:
                    core.add(edge.from_instance)
                    pending.append(edge.from_instance)
        providers: set[str] = set()
        for goal in self.scenario.goal_state_ids:
            provider = latest_writer.get(goal)
            if provider:
                providers.add(f"{goal}:{activity_by_instance[provider]}")
        for edge in edges:
            if edge.edge_type == "state_causal" and edge.to_instance in core and edge.fact_key:
                providers.add(f"{edge.fact_key}:{activity_by_instance[edge.from_instance]}")
        strategy = self._strategy_providers(
            goal_provider=goal_provider,
            incoming=incoming,
            activity_by_instance=activity_by_instance,
        )
        return tuple(edges), core, "|".join(sorted(providers)), "|".join(sorted(strategy))

    def _append_temporal_order_edges(
        self,
        executions: tuple[ExecutionRecord, ...],
        edges: list[PlanEdge],
        seen: set[tuple[str, str, str, str | None]],
    ) -> None:
        ordered = sorted(executions, key=lambda item: (item.start_time, item.end_time, item.instance_id))
        for index, current in enumerate(ordered):
            current_activity = self.activities[current.activity_id]
            current_reads = set(current_activity.required_state_ids)
            current_writes = set(current_activity.transition_state_ids) | set(current_activity.compatibility_remove_state_ids) | set(current_activity.output_state_ids)
            for previous in reversed(ordered[:index]):
                if previous.end_time > current.start_time:
                    continue
                previous_activity = self.activities[previous.activity_id]
                shared_resources = set(dict(previous_activity.resource_reqs)) & set(dict(current_activity.resource_reqs))
                if shared_resources:
                    key = sorted(shared_resources)[0]
                    marker = (previous.instance_id, current.instance_id, "resource_order", key)
                    if marker not in seen:
                        edges.append(PlanEdge(previous.instance_id, current.instance_id, "resource_order", key))
                        seen.add(marker)
                previous_reads = set(previous_activity.required_state_ids)
                previous_writes = set(previous_activity.transition_state_ids) | set(previous_activity.compatibility_remove_state_ids) | set(previous_activity.output_state_ids)
                mutex = (current_writes & (previous_reads | previous_writes)) | (current_reads & previous_writes)
                if mutex:
                    key = sorted(mutex)[0]
                    marker = (previous.instance_id, current.instance_id, "state_mutex_order", key)
                    if marker not in seen:
                        edges.append(PlanEdge(previous.instance_id, current.instance_id, "state_mutex_order", key))
                        seen.add(marker)

    def _strategy_providers(
        self,
        *,
        goal_provider: dict[str, str],
        incoming: dict[str, list[PlanEdge]],
        activity_by_instance: dict[str, str],
    ) -> set[str]:
        """Build a business-provider family without reversible repair cycles."""
        reversible_state_ids = set(self.transition_graph.reversible_state_ids)
        selected_goals = {
            goal: instance
            for goal, instance in goal_provider.items()
            if goal not in reversible_state_ids
        }
        if not selected_goals:
            selected_goals = dict(goal_provider)
        providers = {
            f"goal:{state_id}:{activity_by_instance[instance]}"
            for state_id, instance in selected_goals.items()
        }
        pending = list(selected_goals.values())
        visited = set(pending)
        while pending:
            target = pending.pop()
            for edge in incoming.get(target, ()):
                if not edge.fact_key or edge.fact_key in reversible_state_ids:
                    continue
                provider_id = activity_by_instance[edge.from_instance]
                providers.add(f"dep:{edge.fact_key}:{provider_id}")
                if edge.from_instance not in visited:
                    visited.add(edge.from_instance)
                    pending.append(edge.from_instance)
        if not providers:
            return {"initial-state-strategy"}
        return providers
