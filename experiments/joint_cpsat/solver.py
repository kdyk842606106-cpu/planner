"""Generic causal-link compilation; no task metadata, path or witness inputs.

Scope: positive-duration parallel actions, Boolean states, completion effects,
full Simulator read/write interval locks, renewable resources, release-only
events, positive/negative final goals, bounded execution counts. State-changing
external events and serial-mode inputs are explicitly unsupported in v1.

All occurrence slots are optional. A selected precondition chooses an initial
or action-produced fact and protects it from intervening net deletions. Final
negative goals use the dual construction. Only original input count bounds are
used. A temporal relaxation is compiled; every reported incumbent is converted
to an event-driven schedule and replayed before it can be credited.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time

from ortools.sat.python import cp_model

from planner_experiment.models import Action, to_primitive
from planner_experiment.scenario import scenario_from_dict
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator
from planner_experiment.benchmark_suite.generator import independent_resource_check


class UnsupportedModel(ValueError):
    pass


@dataclass
class Slot:
    index: int
    aid: str
    ordinal: int
    duration: int
    present: object
    start: object
    end: object
    interval: object


def build(payload, *, deadline=None):
    # Reject wrappers instead of accidentally consuming generator metadata.
    if 'scenario' in payload or 'answer' in payload or 'metadata' in payload:
        raise UnsupportedModel('Only a raw scenario is accepted')
    scenario = scenario_from_dict(payload)
    if scenario.execution_mode != 'parallel':
        raise UnsupportedModel('v1 supports parallel, non-preemptive execution only')
    if any(e.add_state_ids or e.remove_state_ids for e in scenario.external_events):
        raise UnsupportedModel('State-changing external events require a separate encoding')
    if any(a.duration <= 0 for a in scenario.activities):
        raise UnsupportedModel('v1 requires positive durations')
    def check():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('MODEL_BUILD_TIME_LIMIT')
    t0 = scenario.start_time
    # Every event-driven trajectory waits at most until the last event plus
    # the sum of all executed durations. No oracle/witness upper bound is read.
    horizon = max([t0, *[e.time for e in scenario.external_events]]) + scenario.max_steps * max(
        [1, *[a.duration for a in scenario.activities]])
    model = cp_model.CpModel()
    finish = model.new_int_var(t0, horizon, 'finish')
    slots = []
    by_activity = defaultdict(list)
    acts = scenario.activity_by_id
    for a in scenario.activities:
        check()
        limit = min(scenario.max_steps, a.max_instances if a.max_instances is not None else scenario.max_steps)
        for k in range(limit):
            i = len(slots)
            p = model.new_bool_var(f'use:{i}')
            s = model.new_int_var(t0, horizon, f's:{i}')
            e = model.new_int_var(t0, horizon, f'e:{i}')
            interval = model.new_optional_interval_var(s, a.duration, e, p, f'I:{i}')
            slot = Slot(i, a.id, k + 1, a.duration, p, s, e, interval)
            if by_activity[a.id]:
                prev = by_activity[a.id][-1]
                model.add(p <= prev.present)
                model.add(s >= prev.end).only_enforce_if(p)
            model.add(s == t0).only_enforce_if(p.Not())
            model.add(e == t0).only_enforce_if(p.Not())
            release = max([t0, *[scenario.event_by_id[x].time for x in a.required_events]])
            model.add(s >= release).only_enforce_if(p)
            model.add(finish >= e).only_enforce_if(p)
            slots.append(slot)
            by_activity[a.id].append(slot)
    model.add(sum(x.present for x in slots) <= scenario.max_steps)
    producers, deleters = defaultdict(list), defaultdict(list)
    for x in slots:
        a = acts[x.aid]
        for fact in a.output_state_ids:
            producers[fact].append(x)
        for fact in set(a.removed_state_ids) - set(a.output_state_ids):
            deleters[fact].append(x)
    support_count = threat_count = lock_count = 0

    def support(fact, consumer=None, negative=False):
        nonlocal support_count, threat_count
        check()
        target = consumer.start if consumer is not None else finish
        truth = fact in scenario.initial_state_ids
        sources = deleters[fact] if negative else producers[fact]
        threats = producers[fact] if negative else deleters[fact]
        options = ([None] if truth != negative else []) + [x for x in sources if x is not consumer]
        choices = []
        for producer in options:
            selected = model.new_bool_var(f'support:{support_count}')
            support_count += 1
            choices.append(selected)
            if producer is not None:
                model.add_implication(selected, producer.present)
                model.add(producer.end <= target).only_enforce_if(selected)
            for destroyer in threats:
                if destroyer is consumer or destroyer is producer:
                    continue
                threat_count += 1
                if producer is None:
                    model.add(destroyer.end > target).only_enforce_if([selected, destroyer.present])
                elif consumer is None:
                    model.add(destroyer.end <= producer.end).only_enforce_if([selected, destroyer.present])
                else:
                    before = model.new_bool_var(f'threat_before:{threat_count}')
                    model.add(destroyer.end <= producer.end).only_enforce_if([selected, destroyer.present, before])
                    model.add(destroyer.end > target).only_enforce_if([selected, destroyer.present, before.Not()])
        model.add(sum(choices) == (consumer.present if consumer is not None else 1))

    for x in slots:
        for fact in acts[x.aid].precondition_state_ids:
            support(fact, x)
    for fact in scenario.goal_state_ids:
        support(fact)
    for fact in scenario.forbidden_state_ids:
        support(fact, negative=True)
    reads = {a.id: set(a.required_state_ids) for a in scenario.activities}
    writes = {a.id: set(a.written_state_ids) for a in scenario.activities}
    for i, a in enumerate(scenario.activities):
        check()
        for b in scenario.activities[i:]:
            if not (writes[a.id] & (reads[b.id] | writes[b.id]) or reads[a.id] & writes[b.id]):
                continue
            for x in by_activity[a.id]:
                for y in by_activity[b.id]:
                    if x.index >= y.index and a.id == b.id:
                        continue
                    model.add_no_overlap([x.interval, y.interval])
                    lock_count += 1
    for resource in scenario.resources:
        selected = [(x, dict(acts[x.aid].resource_reqs).get(resource.id, 0)) for x in slots]
        selected = [(x, q) for x, q in selected if q]
        model.add_cumulative([x.interval for x, _ in selected], [q for _, q in selected], resource.capacity)
    model.minimize(finish)
    check()
    error = model.validate()
    if error:
        raise ValueError(error)
    stats = {'candidate_activities': len(acts), 'optional_occurrence_slots': len(slots),
             'state_count': len(scenario.state_definitions), 'horizon': horizon,
             'support_choices': support_count, 'threat_checks': threat_count, 'interval_locks': lock_count,
             'variables': len(model.proto.variables), 'constraints': len(model.proto.constraints)}
    return model, scenario, slots, stats


def rows_from(reader, slots):
    return [{'activity_id': x.aid, 'ordinal': x.ordinal, 'start': reader.value(x.start),
             'end': reader.value(x.end)} for x in slots if reader.value(x.present)]


def certificate(payload, rows):
    """Left-shift preserving every original non-overlap, then original replay.

    Preserving all original disjoint interval orders preserves causal threats
    and locks. Every new overlapping set was a clique of original intervals;
    interval Helly therefore preserves cumulative resource feasibility too.
    Earliest starts are initial time, release events, or predecessor ends.
    """
    scenario = scenario_from_dict(payload)
    acts = scenario.activity_by_id
    ordered = sorted(rows, key=lambda r: (r['start'], r['activity_id'], r['ordinal']))
    shifted = []
    for i, row in enumerate(ordered):
        a = acts[row['activity_id']]
        release = max([scenario.start_time, *[scenario.event_by_id[e].time for e in a.required_events]])
        start = max([release, *[shifted[j]['end'] for j in range(i) if ordered[j]['end'] <= row['start']]])
        if start > row['start']:
            raise AssertionError('Left-shift delayed an activity')
        shifted.append(dict(row, start=start, end=start + a.duration))
    pending = sorted(shifted, key=lambda r: (r['start'], r['activity_id'], r['ordinal']))
    sim = PathSimulator(scenario)
    state = sim.initial_state()
    while pending or state.running:
        if sim.is_goal(state):
            break
        while pending and pending[0]['start'] == state.time:
            row = pending.pop(0)
            state = sim.transition(state, Action('START', activity_id=row['activity_id']))
        if not pending and not state.running:
            break
        advances = [a for a in sim.enabled_actions(state) if a.kind == 'ADVANCE']
        if not advances or (pending and pending[0]['start'] < advances[0].target_time):
            raise AssertionError('Schedule cannot be replayed by event-driven semantics')
        state = sim.transition(state, advances[0])
    if not sim.is_goal(state):
        raise AssertionError('Decoded schedule does not reach goal')
    candidate = sim.candidate_from_state(state, algorithm='JOINT_CP_SAT', run_id=scenario.id,
                                         seed=11, discovered_at_seconds=0, normalize=False)
    checked = ResultValidator(scenario).validate(candidate)
    if not checked.valid:
        raise AssertionError(checked.code)
    independent_resource_check(payload, checked.candidate)
    return {'makespan': candidate.metrics.makespan, 'executions': [to_primitive(x) for x in candidate.executions],
            'actions': [to_primitive(x) for x in candidate.actions], 'validator': checked.code,
            'selected_definitions': len({x.activity_id for x in candidate.executions}),
            'execution_count': len(candidate.executions)}


def solve(payload, seconds=30, *, on_candidate=None, on_model=None, started=None):
    started = time.perf_counter() if started is None else started
    deadline = started + seconds
    model, scenario, slots, stats = build(payload, deadline=deadline)
    built = time.perf_counter() - started
    if on_model is not None:
        on_model({'build_seconds': built, 'model_stats': stats})
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 11
    solver.parameters.max_time_in_seconds = max(0.001, deadline - time.perf_counter())
    incumbents, rejected = [], []

    class Observe(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            found = time.perf_counter() - started
            if time.perf_counter() >= deadline:
                self.stop_search()
                return
            try:
                cert = certificate(payload, rows_from(self, slots))
                elapsed = time.perf_counter() - started
                record = {'discovered_seconds': found, 'validated_seconds': elapsed,
                          'raw_makespan': self.objective_value - scenario.start_time,
                          'lower_bound': self.best_objective_bound - scenario.start_time,
                          'certificate': cert, 'within_budget': elapsed <= seconds}
                if record['within_budget']:
                    incumbents.append(record)
                    if on_candidate is not None:
                        on_candidate(record)
            except Exception as exc:
                rejected.append({'elapsed': time.perf_counter() - started, 'error': repr(exc)})
                self.stop_search()
            if time.perf_counter() >= deadline:
                self.stop_search()

    status = solver.solve(model, Observe())
    elapsed = time.perf_counter() - started
    best = min(incumbents, key=lambda x: x['certificate']['makespan']) if incumbents else None
    bound = max(0, math.ceil(solver.best_objective_bound - scenario.start_time - 1e-7))
    proved = bool(best and best['certificate']['makespan'] == bound and not rejected and elapsed <= seconds)
    return {'solver_status': solver.status_name(status), 'proven_optimal': proved,
            'status': 'ENCODING_ERROR' if rejected else 'PROVEN_OPTIMAL' if proved else
            'FEASIBLE' if best else 'PROVEN_INFEASIBLE' if status == cp_model.INFEASIBLE and elapsed <= seconds else 'UNKNOWN',
            'best': best, 'lower_bound': bound, 'build_seconds': built, 'elapsed_seconds': elapsed,
            'first_solution_seconds': incumbents[0]['validated_seconds'] if incumbents else None,
            'branches': solver.num_branches, 'conflicts': solver.num_conflicts,
            'accepted_candidates': len(incumbents), 'rejections': rejected, 'model_stats': stats,
            'proof_scope': 'original finite counts; parallel actions; release-only external events'}
