"""Deterministic mandatory-process DAGs and independently replayed witnesses."""
from __future__ import annotations

import random
from collections import Counter

from planner_experiment.benchmark_suite.complexity import dag_properties
from planner_experiment.benchmark_suite.generator import candidate_from_actions, finish_instance
from planner_experiment.models import Action, to_primitive
from planner_experiment.scenario import scenario_from_dict
from planner_experiment.simulator import PathSimulator

EARLY = ('config', 'base', 'install_a', 'wire_a', 'insulate_a', 'communication', 'cal_a', 'test_a')
LATE = ('install_b', 'wire_b', 'insulate_b', 'cal_b', 'test_b', 'accept')
RECOVER = ('cal_a', 'test_a', 'cal_b', 'test_b', 'accept')
POWER = ('off', 'ground', 'unground', 'on')
ONLINE = {'communication', *RECOVER}
RANGES = {'config': (8, 20), 'base': (15, 30), 'install_a': (15, 30), 'install_b': (15, 30),
          'wire_a': (8, 20), 'wire_b': (8, 20), 'insulate_a': (5, 10), 'insulate_b': (5, 10),
          'communication': (8, 15), 'cal_a': (10, 20), 'cal_b': (10, 20),
          'test_a': (20, 40), 'test_b': (20, 40), 'accept': (10, 20)}
REQUIREMENTS = {'config': ('electrical',), 'base': ('mechanical', 'hoist'),
                'install_a': ('mechanical',), 'install_b': ('mechanical', 'hoist'),
                'wire_a': ('electrical',), 'wire_b': ('electrical',),
                'insulate_a': ('electrical', 'instrument'), 'insulate_b': ('electrical', 'instrument'),
                'communication': ('electrical', 'instrument'),
                'cal_a': ('calibrator', 'instrument'), 'cal_b': ('calibrator', 'instrument'),
                'test_a': ('tester', 'test_bench'), 'test_b': ('tester', 'test_bench'),
                'accept': ('tester', 'test_bench')}
CAPACITIES = {'mechanical': 4, 'electrical': 4, 'calibrator': 4, 'tester': 2,
              'instrument': 4, 'test_bench': 2, 'hoist': 2, 'power_team': 2}


def aid(module, kind):
    return f'm{module:02d}:{kind}'


def reduce_edges(nodes, edges):
    graph = dag_properties(nodes, [list(e) for e in edges], {n: 1 for n in nodes})
    successors = {n: set() for n in nodes}
    reach = {n: set() for n in nodes}
    for a, b in edges:
        successors[a].add(b)
    for a in reversed(graph['topological_order']):
        for b in successors[a]:
            reach[a].add(b)
            reach[a].update(reach[b])
    return {(a, b) for a, b in edges if not any(b in reach[c] for c in successors[a] if c != b)}


def structure(width, density, seed):
    """W disjoint covering chains plus genuine adjacent-layer interface gates.

    Early and late legs never get a late->early gate. This preserves feasibility
    when a late installation additionally needs an early test history.
    """
    lanes = width // 2
    nodes = [aid(m, k) for m in range(16) for k in (*EARLY, *LATE)]
    chains = []
    for kinds in (EARLY, LATE):
        for lane in range(lanes):
            chains.append([aid(m, k) for m in range(lane, 16, lanes) for k in kinds])
    edges = {(a, b) for chain in chains for a, b in zip(chain, chain[1:])}
    # Installation prerequisites and final convergence, all history-based.
    edges.update((aid(m, 'base'), aid(m, 'install_b')) for m in range(16))
    edges.update((aid(m, 'test_a'), aid(m, 'accept')) for m in range(16))
    base = set(edges)
    candidates = []
    for leg in (chains[:lanes], chains[lanes:]):
        for level in range(1, len(leg[0])):
            for left in range(lanes):
                for right in range(lanes):
                    if left != right:
                        candidates.append((leg[left][level - 1], leg[right][level]))
    random.Random(seed + width * 1000).shuffle(candidates)
    target = density * len(nodes)
    # Within one leg all added edges join adjacent levels: none can hide a
    # same-leg transitive path. Cross-leg edges are reduced explicitly.
    for edge in candidates:
        if len(edges) >= target:
            break
        edges.add(edge)
    reduced = reduce_edges(nodes, edges)
    for edge in candidates:
        if len(reduced) >= target:
            break
        if edge not in edges:
            edges.add(edge)
            reduced = reduce_edges(nodes, edges)
    graph = dag_properties(nodes, [list(e) for e in reduced], {n: 1 for n in nodes})
    if graph['width'] != width:
        raise AssertionError((width, density, graph['width']))
    # Even at maximum density the chain cover and first early/late branch
    # antichain certify the requested width; the generic matching checks it.
    return nodes, sorted(reduced), chains, sorted(base)


def make_instance(width=16, density=2, condition='T0', seed=101):
    if width not in (8, 16, 32) or density not in (2, 4) or condition not in ('T0', 'T1', 'T2', 'T3'):
        raise ValueError('Unsupported business study factor')
    nodes, edges, chains, base_edges = structure(width, density, seed)
    rng = random.Random(seed)
    durations = {aid(m, k): rng.randint(*RANGES[k]) for m in range(16) for k in (*EARLY, *LATE)}
    power_durations = {f'p{g:02d}:{k}': rng.randint(2, 5) for g in range(8) for k in POWER}
    reference = dag_properties(nodes, [list(e) for e in edges], durations)
    ref_nodes, ref_edges, _, _ = structure(16, 2, seed)
    h0 = dag_properties(ref_nodes, [list(e) for e in ref_edges], durations)['critical_path_minutes']
    parents = {n: [] for n in nodes}
    for a, b in edges:
        parents[b].append(a)
    restore = condition in ('T2', 'T3')
    initial = [f'p{g:02d}:isolated' for g in range(8)]
    initial += [f'invalid:{aid(m, k)}' for m in range(16) for k in RECOVER]
    activities = []
    for m in range(16):
        for kind in (*EARLY, *LATE):
            identifier = aid(m, kind)
            pre = [{'state_id': f'history:{p}', 'relation_role': 'required'} for p in parents[identifier]]
            mode = 'on' if kind in ONLINE else 'isolated'
            if kind != 'config':
                pre.append({'state_id': f'p{m//2:02d}:{mode}', 'relation_role': 'required'})
            output = [f'valid:{identifier}'] if kind in RECOVER else []
            delete = []
            if kind in RECOVER:
                pre.append({'state_id': f'invalid:{identifier}', 'relation_role': 'transition'})
            if kind in ('test_a', 'test_b'):
                pre.append({'state_id': f'valid:{aid(m, kind.replace("test", "cal"))}', 'relation_role': 'required'})
            if kind == 'accept':
                pre += [{'state_id': f'valid:{aid(m, k)}', 'relation_role': 'required'} for k in ('test_a', 'test_b')]
            if restore and kind == 'install_b':
                # One selected module per power domain must have been tested
                # before retrofit. A history is not an up-to-date certificate.
                if m % 2 == 0:
                    pre.append({'state_id': f'history:{aid(m, "test_a")}', 'relation_role': 'required'})
                output += [f'invalid:{aid(m, k)}' for k in RECOVER]
                delete = [f'valid:{aid(m, k)}' for k in RECOVER]
            activities.append({'id': identifier, 'name': identifier, 'duration': durations[identifier],
                               'output_state_id': f'history:{identifier}', 'additional_output_state_ids': output,
                               'preconditions': pre, 'delete_effect_state_ids': delete,
                               'resource_reqs': {r: 1 for r in REQUIREMENTS[kind]},
                               'event_reqs': [f'arrival:{m:02d}'] if kind == 'install_b' else [],
                               'max_instances': 6 if kind in RECOVER else 1, 'is_milestone': kind not in RECOVER})
    changes = {'off': ('on', 'off'), 'ground': ('off', 'isolated'),
               'unground': ('isolated', 'ready'), 'on': ('ready', 'on')}
    for g in range(8):
        for kind in POWER:
            old, new = changes[kind]
            identifier = f'p{g:02d}:{kind}'
            activities.append({'id': identifier, 'name': identifier, 'duration': power_durations[identifier],
                               'output_state_id': f'history:{identifier}',
                               'additional_output_state_ids': [f'p{g:02d}:{new}'],
                               'preconditions': [{'state_id': f'p{g:02d}:{old}', 'relation_role': 'transition'}],
                               'delete_effect_state_ids': [], 'resource_reqs': {'power_team': 1},
                               'max_instances': 6, 'is_milestone': False})
    iid = f'b256-w{width}-d{density}-{condition}-g{seed}'
    payload = {'id': iid, 'name': iid, 'schema_version': 2, 'execution_mode': 'parallel',
               'initial_state_ids': initial,
               'goal_state_ids': [f'valid:{aid(m, "accept")}' for m in range(16)] + [f'p{g:02d}:on' for g in range(8)],
               'forbidden_state_ids': [], 'activities': activities,
               'external_events': [{'id': f'arrival:{m:02d}', 'time': 0 if condition == 'T0' else (h0 * (1 if m < 8 else 3) + 3)//4,
                                    'add_state_ids': [], 'remove_state_ids': []} for m in range(16)],
               'resources': [{'id': r, 'capacity': cap if condition == 'T3' else (8 if r == 'power_team' else 16)}
                             for r, cap in CAPACITIES.items()],
               'max_steps': sum(a['max_instances'] for a in activities)}
    metadata = {'id': iid, 'group': condition, 'scale': 256, 'generator_seed': seed,
                'width_target': width, 'density_target': density, 'task_activities': {n: [n] for n in nodes},
                'edges': [list(e) for e in edges], 'reference_activity_ids': nodes,
                'reference_h0_minutes': h0, 'chain_cover': chains, 'base_edges': base_edges,
                'static_dag_scope': '224 mandatory first-completion tasks; excludes mode transitions and retrofit-history gates',
                'dynamic_history_gates': [[aid(m, 'test_a'), aid(m, 'install_b')] for m in range(0, 16, 2)] if restore else [],
                'condition': condition, 'direct_dependency_mean': len(edges)/224,
                'density_target_attained': len(edges) == density * 224,
                'all_activities_non_substitutable': True}
    actions = witness(payload, metadata)
    candidate = candidate_from_actions(payload, actions)
    # This relaxation drops resource, event, mode and restoration constraints.
    answer = {'status': 'UNKNOWN', 'method': 'mandatory-history-DAG-lower-bound+validated-serial-witness',
              'lower_bound': reference['critical_path_minutes'], 'witness_makespan': candidate.metrics.makespan, 'actions': actions,
              'optimal_makespan': None}
    result = finish_instance(payload, metadata, answer)
    result['complexity']['dependency_edges_after_transitive_reduction'] = len(edges)
    result['complexity']['direct_dependency_mean'] = len(edges)/224
    result['complexity']['restoration_gate_count'] = len(metadata['dynamic_history_gates'])
    assert len(activities) == 256
    return result


def witness(payload, metadata):
    simulator = PathSimulator(scenario_from_dict(payload))
    state = simulator.initial_state()
    activities = {a['id']: a for a in payload['activities']}

    def execute(identifier):
        nonlocal state
        action = Action('START', identifier)
        for _ in range(100):
            enabled = simulator.enabled_actions(state)
            if action in enabled:
                state = simulator.transition(state, action)
                break
            advance = next((a for a in enabled if a.kind == 'ADVANCE'), None)
            if advance is None:
                missing = set(p['state_id'] for p in activities[identifier]['preconditions']) - set(state.active_state_ids)
                raise ValueError(f'Witness blocked {identifier}: {sorted(missing)}')
            state = simulator.transition(state, advance)
        else:
            raise ValueError('Witness wait limit')
        while state.running:
            state = simulator.transition(state, next(a for a in simulator.enabled_actions(state) if a.kind == 'ADVANCE'))

    def mode(g, desired):
        sequence = {'on': ('off', 'ground'), 'isolated': ('unground', 'on')}
        active = set(state.active_state_ids)
        if f'p{g:02d}:{desired}' in active:
            return
        opposite = 'isolated' if desired == 'on' else 'on'
        for k in sequence[opposite]:
            execute(f'p{g:02d}:{k}')

    def recover(m, kind):
        if f'valid:{aid(m, kind)}' in state.active_state_ids:
            return
        mode(m//2, 'on')
        if kind in ('test_a', 'test_b'):
            recover(m, kind.replace('test', 'cal'))
        execute(aid(m, kind))

    # All early legs first is a constructive choice, not an engine hint.
    early = {aid(m, k) for m in range(16) for k in EARLY}
    order = dag_properties(list(metadata['task_activities']), metadata['edges'],
                           {n: 1 for n in metadata['task_activities']})['topological_order']
    for identifier in [n for n in order if n in early] + [n for n in order if n not in early]:
        module, kind = identifier.split(':')
        m = int(module[1:])
        if kind != 'config':
            mode(m//2, 'on' if kind in ONLINE else 'isolated')
        if kind == 'accept':
            recover(m, 'test_a')
            recover(m, 'test_b')
        execute(identifier)
    if not simulator.is_goal(state):
        raise ValueError('Witness did not attain all current-valid goals')
    return [to_primitive(a) for a in state.actions]
