"""Deterministic, feasibility-controlled prerequisite density intervention."""
import copy
from collections import Counter

from experiments.mainline_joint.case import make_case as original_case, describe
from experiments.window_joint.adapter import certificate
from planner_experiment.benchmark_suite.complexity import dag_properties
from planner_experiment.benchmark_suite.io import digest


def reachability(nodes, edges, durations):
    successors = {n: set() for n in nodes}
    for a, b in edges:
        successors[a].add(b)
    reach = {n: set() for n in nodes}
    order = dag_properties(nodes, sorted(edges), durations)['topological_order']
    for a in reversed(order):
        for b in sorted(successors[a]):
            reach[a].add(b)
            reach[a].update(reach[b])
    return reach


def reduction(nodes, edges, durations):
    reach = reachability(nodes, edges, durations)
    successors = {n: set() for n in nodes}
    for a, b in edges:
        successors[a].add(b)
    return {(a, b) for a, b in edges
            if not any(c != b and b in reach[c] for c in successors[a])}


def degree_summary(nodes, edges):
    counts = Counter(b for a, b in edges)
    return {'mean': len(edges) / len(nodes), 'maximum': max(counts.values()),
            'histogram': dict(sorted(Counter(counts[n] for n in nodes).items()))}


def make_case(baseline, baseline_certificate):
    payload = copy.deepcopy(baseline)
    original, metadata = original_case()
    for key in original:
        if key not in ('id', 'name'):
            assert baseline[key] == original[key], key
    nodes = list(metadata['task_activities'])
    durations = {a['id']: a['duration'] for a in payload['activities']}
    executions = baseline_certificate['executions']
    rows = {e['activity_id']: e for e in executions}
    assert all(sum(e['activity_id'] == n for e in executions) == 1 for n in nodes)
    old = set(map(tuple, metadata['edges']))
    # Same-stage handoffs between adjacent branches are the initial intervention.
    edges = old | {(f'B{b}_{i:02d}', f'B{b+1}_{i:02d}')
                   for b in (1, 2) for i in range(1, 21)}
    assert all(rows[a]['end_time'] <= rows[b]['start_time'] for a, b in edges)
    reach = reachability(nodes, edges, durations)
    # Add cross-chain handoffs only when they introduce an incomparable direct
    # predecessor and preserve the frozen witness. No algorithm timings used.
    for b in sorted(nodes, key=lambda a: (rows[a]['start_time'], a)):
        for a in sorted(nodes, key=lambda a: (-rows[a]['end_time'], a)):
            if a == b or rows[a]['end_time'] > rows[b]['start_time'] or b in reach[a]:
                continue
            predecessors = {x for x, y in edges if y == b}
            if any(x in reach[a] or a in reach[x] for x in predecessors):
                continue
            edges.add((a, b))
            reach = reachability(nodes, edges, durations)
    # Retain every old edge, but discard newly added edges rendered redundant
    # by later additions. Every final new edge is in the transitive reduction.
    edges = old | (reduction(nodes, edges, durations) - old)
    added = sorted(edges - old)
    assert all(edge in reduction(nodes, edges, durations) for edge in added)
    activities = {a['id']: a for a in payload['activities']}
    for a, b in added:
        activities[b]['preconditions'].append({'state_id': 'done:' + a, 'relation_role': 'required'})
    payload['id'] += '-dense-prerequisites'
    payload['name'] = payload['id']
    metadata = copy.deepcopy(metadata)
    metadata['edges'] = [list(edge) for edge in sorted(edges)]
    metrics = describe(payload, metadata)
    metrics.update(time_window_count=len(payload['time_windows']),
                   process_predecessors=degree_summary(nodes, edges),
                   mainline_predecessors=degree_summary(nodes[:180], {(a,b) for a,b in edges if b.startswith('M')}),
                   branch_predecessors=degree_summary(nodes[180:], {(a,b) for a,b in edges if b.startswith('B')}),
                   added_process_edges=len(added), nonredundant_process_edges=len(reduction(nodes, edges, durations)),
                   reference_load_policy='Original fixed reference activities and H0=1336 retained; actual DAG critical path separately measured')
    raw = [{'activity_id': e['activity_id'], 'ordinal': e['ordinal'],
            'start': e['start_time'], 'end': e['end_time']} for e in executions]
    cert = certificate(payload, raw)
    assert cert['makespan'] == baseline_certificate['makespan'] == 1451
    evidence = {
        'baseline_input_hash': digest(baseline), 'added_edges': [list(e) for e in added],
        'metadata': metadata,
        'baseline_process_predecessors': degree_summary(nodes, old),
        'baseline_skeleton': {k:v for k,v in dag_properties(nodes, sorted(old), durations).items() if k != 'topological_order'},
        'generation_policy': 'Feasibility-controlled add-only graph: same-stage B1->B2->B3 handoffs, then deterministic incomparable-predecessor augmentation compatible with the frozen 1451 witness; retain only nonredundant added edges. This is not an independent business sample.',
        'witness_raw_rows': raw, 'witness_certificate': cert,
        'answer': {'status': 'PROVEN_OPTIMAL_BY_SUBSET_AND_ATTAINING_WITNESS',
                   'optimal_makespan': 1451, 'lower_bound': 1451,
                   'proof': 'Original instance has certified optimum 1451. Adding only prerequisites cannot reduce it. Independently validated original schedule remains feasible and attains 1451.'},
        'formal_solver_policy': 'raw input only; no witness, hints, mandatory map or supplied bounds',
        'comparison_scope': 'One paired case/seed; measures density changes at unchanged optimal value, not population performance or width increase.'}
    return payload, metrics, evidence
