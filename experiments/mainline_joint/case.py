"""Deterministic scenario and private validation witnesses (never solver hints)."""
from collections import Counter
import math
import random

from experiments.joint_cpsat.cases import activity, base_payload
from planner_experiment.benchmark_suite.complexity import complexity, dag_properties
from planner_experiment.benchmark_suite.generator import candidate_from_actions
from planner_experiment.models import Action, to_primitive
from planner_experiment.scenario import scenario_from_dict
from planner_experiment.simulator import PathSimulator


def make_case(seed=101):
    rng = random.Random(seed)
    main = [f'M{i:03d}' for i in range(1, 181)]
    branches = [[f'B{b}_{i:02d}' for i in range(1, 21)] for b in range(1, 4)]
    nodes = main + sum(branches, [])
    durations = {aid: rng.randint(5, 10) for aid in main}
    durations.update({aid: rng.randint(2, 6) for aid in sum(branches, [])})
    edges = list(zip(main, main[1:]))
    for chain, fork, join in zip(branches, (20, 40, 60), (110, 145, 175)):
        edges += list(zip(chain, chain[1:]))
        edges += [(f'M{fork:03d}', chain[0]), (chain[-1], f'M{join:03d}'), ('M081', chain[12])]
    edges = [list(edge) for edge in edges]
    dag = dag_properties(nodes, edges, durations)
    predecessors = {aid: [] for aid in nodes}
    for left, right in edges:
        predecessors[right].append(left)
    ends = {}
    for aid in dag['topological_order']:
        ends[aid] = max((ends[p] for p in predecessors[aid]), default=0) + durations[aid]
    h0 = dag['critical_path_minutes']
    delay = math.ceil(h0 / 10)
    installation_ids = [chain[7] for chain in branches]
    reference_install_starts = {aid: ends[aid] - durations[aid] for aid in installation_ids}
    events = [{'id': f'material_{b}', 'time': reference_install_starts[aid] + delay,
               'add_state_ids': [], 'remove_state_ids': []}
              for b, aid in enumerate(installation_ids, 1)]
    acts = []
    for i, aid in enumerate(main, 1):
        required = ['done:' + p for p in predecessors[aid]]
        if i < 80:
            required += ['mode:isolated']
            resources = {'electricians': 1}
            if i % 5 == 0:
                resources['install_team'] = 1
        else:
            required += ['mode:on']
            resources = {'electricians': 1}
            if i % 3 == 0 or i in (80, 81):
                resources['instrument'] = 1
            if i > 81:
                required += ['valid:test']
        transition, outputs = [], []
        if i == 80:
            transition, outputs = ['dirty:cal'], ['valid:cal']
        if i == 81:
            required += ['valid:cal']
            transition, outputs = ['dirty:test'], ['valid:test']
        acts.append(activity(aid, durations[aid], required=required, transition=transition,
                             outputs=outputs, limit=4 if i in (80, 81) else 1, resources=resources))
    for b, chain in enumerate(branches, 1):
        for i, aid in enumerate(chain, 1):
            required = ['done:' + p for p in predecessors[aid]]
            required += ['mode:isolated'] if i <= 12 else ['mode:on', 'valid:test']
            resources = {'electricians': 1}
            if i <= 8:
                resources['install_team'] = 1
            elif i >= 11:
                resources['instrument'] = 1
            acts.append(activity(aid, durations[aid], required=required, resources=resources,
                                 events=[f'material_{b}'] if i == 8 else [],
                                 outputs=['dirty:cal', 'dirty:test'] if i == 8 else [],
                                 delete=['valid:cal', 'valid:test'] if i == 8 else []))
    # Switching changes the mode only; installation explicitly invalidates both
    # qualifications. This separates mode feasibility from qualification loss.
    for aid, before, after, duration in [('unground', 'isolated', 'ready', 3),
                                         ('power_on', 'ready', 'on', 5),
                                         ('power_off', 'on', 'off', 4),
                                         ('ground', 'off', 'isolated', 3)]:
        acts.append(activity(aid, duration, transition=['mode:' + before],
                             outputs=['mode:' + after], limit=6, resources={'electricians': 1}))
    # Extended-scope checks are optional business definitions, not substitutes
    # for any mainline or branch task. Their scope is explicit in metadata.
    for b, chain in enumerate(branches, 1):
        for i in (5, 10, 15, 20):
            aid = f'CHECK_B{b}_{i:02d}'
            acts.append(activity(aid, 2, required=['done:' + chain[i-1]],
                                 resources={'instrument': 1}))
    payload = base_payload(f'mainline256-tight-invalidate-moderate-g{seed}', acts,
                           ['mode:isolated', 'dirty:cal', 'dirty:test'],
                           ['done:M180', *['done:' + c[-1] for c in branches],
                            'mode:on', 'valid:cal', 'valid:test'], events,
                           {'electricians': 2, 'instrument': 1, 'install_team': 1}, 282)
    # This runner does not invoke the budget-based engine interface.
    payload.pop('default_budget')
    metadata = {'generator_seed': seed, 'task_activities': {aid: [aid] for aid in nodes},
                'edges': edges, 'reference_activity_ids': nodes, 'reference_h0_minutes': h0,
                'reference_install_starts': reference_install_starts, 'uniform_delay': delay,
                'mandatory_process_task_count': 240, 'mode_definition_count': 4,
                'optional_extended_checks': 12, 'invalidating_activities': installation_ids,
                'calibration_activity': 'M080', 'testing_activity': 'M081',
                'mode_switch_invalidation': False,
                'goal_policy': 'mainline and all branch histories; current on/calibrated/tested',
                'mandatory_map_policy': 'private input analysis only; never supplied to solver',
                'resource_definition_counts': dict(Counter(r for a in acts for r in a['resource_reqs']))}
    return payload, metadata


def witness(payload, *, recover=False):
    sim = PathSimulator(scenario_from_dict(payload))
    state = sim.initial_state()
    sequence = [f'M{i:03d}' for i in range(1, 80)]
    if recover:
        sequence += ['unground', 'power_on', 'M080', 'M081', 'power_off', 'ground']
    sequence += [f'B{b}_{i:02d}' for b in range(1, 4) for i in range(1, 13)]
    sequence += ['unground', 'power_on', 'M080', 'M081']
    sequence += [f'B{b}_{i:02d}' for b in range(1, 4) for i in range(13, 21)]
    sequence += [f'M{i:03d}' for i in range(82, 181)]
    invalidation_checks = []
    for aid in sequence:
        action = Action('START', activity_id=aid)
        while action not in sim.enabled_actions(state):
            advance = next((a for a in sim.enabled_actions(state) if a.kind == 'ADVANCE'), None)
            if advance is None:
                raise AssertionError('Witness blocked: ' + aid)
            state = sim.transition(state, advance)
        state = sim.transition(state, action)
        while state.running:
            state = sim.transition(state, next(a for a in sim.enabled_actions(state) if a.kind == 'ADVANCE'))
        if aid in ('B1_08', 'B2_08', 'B3_08'):
            active = set(state.active_state_ids)
            assert not active.intersection(('valid:cal', 'valid:test'))
            assert {'dirty:cal', 'dirty:test'} <= active
            invalidation_checks.append({'installation': aid, 'both_qualifications_invalid': True})
    actions = [to_primitive(a) for a in state.actions]
    checked = candidate_from_actions(payload, actions)
    rows = []
    ordinals = Counter()
    for execution in checked.executions:
        ordinals[execution.activity_id] += 1
        rows.append({'activity_id': execution.activity_id, 'ordinal': ordinals[execution.activity_id],
                     'start': execution.start_time, 'end': execution.end_time})
    return {'makespan': checked.metrics.makespan, 'actions': actions, 'rows': rows,
            'execution_counts': dict(ordinals), 'invalidation_checks': invalidation_checks}


def describe(payload, metadata):
    result = complexity(payload, metadata)
    result['skeleton'].pop('topological_order')
    result['scope'] = 'DAG statistics cover 240 immutable process histories; mode and restoration cycles are separate'
    return result
