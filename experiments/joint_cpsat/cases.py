"""Raw inputs and separate, private generation evidence for the joint study."""
from __future__ import annotations

import copy
import math
import random

from planner_experiment.models import Action, to_primitive
from planner_experiment.scenario import scenario_from_dict
from planner_experiment.simulator import PathSimulator
from planner_experiment.benchmark_suite.generator import candidate_from_actions


def activity(aid, duration=1, *, required=(), transition=(), outputs=(), delete=(), limit=1, resources=None, events=()):
    return {'id': aid, 'name': aid, 'duration': duration, 'output_state_id': 'done:'+aid,
            'additional_output_state_ids': list(outputs), 'delete_effect_state_ids': list(delete),
            'preconditions': [{'state_id': s, 'relation_role': role} for role, states in
                              [('required', required), ('transition', transition)] for s in states],
            'max_instances': limit, 'resource_reqs': resources or {}, 'event_reqs': list(events)}


def base_payload(iid, acts, initial, goals, events, capacities, max_steps):
    return {'schema_version': 2, 'id': iid, 'name': iid, 'execution_mode': 'parallel',
            'start_time': 0, 'activities': acts, 'initial_state_ids': initial,
            'goal_state_ids': goals, 'forbidden_state_ids': [], 'external_events': events,
            'resources': [{'id': r, 'capacity': c} for r, c in capacities.items()],
            'max_steps': max_steps, 'default_budget': {'time_limit_seconds': 30, 'transition_limit': 2**60}}


def small_case(iid='small', *, arrival=2, downstream=2, destroy=True, forced=False,
               capacity=2, service=False, invalidators=1):
    r = {'instrument': 1}
    acts = [
        activity('base', required=['off']),
        activity('on', required=['done:base'], transition=['off'], outputs=['on'], limit=invalidators+1),
        activity('off', transition=['on'], outputs=['off'], limit=invalidators),
        activity('install', required=['off', 'done:base', *(['history:test'] if forced else [])],
                 events=['material'], outputs=['dirty:cal', 'dirty:test'] if destroy else [],
                 delete=['valid:cal', 'valid:test'] if destroy else []),
        activity('cal', required=['on'], transition=['dirty:cal'], outputs=['valid:cal'],
                 limit=invalidators+1, resources=r),
        activity('test', required=['on', 'valid:cal'], transition=['dirty:test'],
                 outputs=['valid:test', 'history:test'], limit=invalidators+1, resources=r),
        # A short dispatch reads the history. Long downstream work reads its
        # own release flag, so restoration cannot violate a held history lock.
        activity('dispatch', required=['history:test']),
        activity('downstream', downstream, required=['done:dispatch']),
        activity('final', required=['on', 'valid:test', 'done:install', 'done:downstream']),
        activity('optional_inspection', 20, required=['done:base'], resources=r),
        activity('optional_archive', required=['done:optional_inspection']),
    ]
    events = [{'id': 'material', 'time': arrival, 'add_state_ids': [], 'remove_state_ids': []}]
    if service:
        acts.append(activity('service', 5, events=['service_release'], resources=r))
        events.append({'id': 'service_release', 'time': 3, 'add_state_ids': [], 'remove_state_ids': []})
        acts[8]['preconditions'].append({'state_id': 'done:service', 'relation_role': 'required'})
    if invalidators == 2:
        acts.append(activity('install_c', required=['off', 'done:install'],
                             outputs=['dirty:cal', 'dirty:test'], delete=['valid:cal', 'valid:test'],
                             events=['material_c']))
        events.append({'id': 'material_c', 'time': arrival+3, 'add_state_ids': [], 'remove_state_ids': []})
        acts[8]['preconditions'].append({'state_id': 'done:install_c', 'relation_role': 'required'})
    return base_payload(iid, acts, ['off', 'dirty:cal', 'dirty:test'], ['done:final'], events,
                        {'instrument': capacity}, 24)


def small_suite():
    definitions = [
        ('arrival', {'arrival': 0}, {'arrival': 8}),
        ('downstream', {'arrival': 2, 'downstream': 1}, {'arrival': 2, 'downstream': 8}),
        ('resource', {'arrival': 2, 'downstream': 8, 'service': True, 'capacity': 2},
                     {'arrival': 2, 'downstream': 8, 'service': True, 'capacity': 1}),
        ('invalidation', {'arrival': 8, 'destroy': False}, {'arrival': 8, 'destroy': True}),
        ('pretest', {'arrival': 0, 'forced': False}, {'arrival': 0, 'forced': True}),
        ('invalidators', {'arrival': 2, 'invalidators': 1}, {'arrival': 2, 'invalidators': 2}),
    ]
    return [(small_case(f'small-{factor}-{variant}', **params), {'factor': factor, 'variant': variant, **params})
            for factor, first, second in definitions for variant, params in [('a', first), ('b', second)]]


def serial_witness(payload):
    """Private generator witness; never passed to build()/solve()."""
    sim = PathSimulator(scenario_from_dict(payload))
    state = sim.initial_state()
    if len(payload['activities']) == 256:
        sequence = [f'm{m:02d}:{kind}' for m in range(16) for kind in
                    ('base', 'wire_a', 'insulate_a', 'delivery_check', 'install', 'wire_b', 'insulate_b',
                     'on', 'cal', 'test', 'dispatch', 'downstream', 'final')]
    else:
        acts = {a['id']: a for a in payload['activities']}
        forced = any(p['state_id'] == 'history:test' for p in acts['install']['preconditions'])
        sequence = ['base']
        if forced:
            sequence += ['on', 'cal', 'test', 'dispatch', 'downstream', 'off']
        sequence += ['install']
        if 'install_c' in acts:
            sequence += ['install_c']
        sequence += ['on', 'cal', 'test']
        if not forced:
            sequence += ['dispatch', 'downstream']
        if 'service' in acts:
            sequence += ['service']
        sequence += ['final']
    for aid in sequence:
        action = Action('START', activity_id=aid)
        while action not in sim.enabled_actions(state):
            advances = [x for x in sim.enabled_actions(state) if x.kind == 'ADVANCE']
            if not advances:
                raise AssertionError(f'No witness continuation: {aid}')
            state = sim.transition(state, advances[0])
        state = sim.transition(state, action)
        while state.running:
            state = sim.transition(state, next(x for x in sim.enabled_actions(state) if x.kind == 'ADVANCE'))
    actions = [to_primitive(a) for a in state.actions]
    checked = candidate_from_actions(payload, actions)
    return {'actions': actions, 'makespan': checked.metrics.makespan}


def large_case(arrival_level, tight, length, seed):
    rng = random.Random(seed)
    kinds = ('base', 'on', 'off', 'install', 'cal', 'test', 'dispatch', 'downstream', 'final',
             'optional_inspection', 'optional_archive', 'delivery_check', 'wire_a', 'insulate_a', 'wire_b', 'insulate_b')
    durations = {f'm{m:02d}:{k}': rng.randint(2, 5) for m in range(16) for k in kinds}
    # Fixed seed-specific reference scale; neither tested length nor arrival
    # nor capacity is used to recompute it. This is generation metadata only.
    reference = max(sum(durations[f'm{m:02d}:{k}'] for k in
                        ('base','wire_a','insulate_a','delivery_check','install','wire_b','insulate_b','on','cal','test','dispatch','final'))
                    + 20 + sum(durations[f'm{j:02d}:base'] for j in range(m % 4, m, 4)) for m in range(16))
    release = math.ceil(reference * (0, .5, 1.5)[arrival_level])
    acts, initial, goals, events = [], [], [], []
    for m in range(16):
        prefix = f'm{m:02d}:'
        p = small_case(arrival=release)
        replacements = {}
        for a in p['activities']:
            def rename(s): return prefix+s
            b = copy.deepcopy(a)
            b['id'] = prefix+a['id']; b['name'] = b['id']; b['duration'] = durations[b['id']]
            b['output_state_id'] = rename(a['output_state_id'])
            b['additional_output_state_ids'] = [rename(s) for s in a['additional_output_state_ids']]
            b['delete_effect_state_ids'] = [rename(s) for s in a['delete_effect_state_ids']]
            b['preconditions'] = [dict(q, state_id=rename(q['state_id'])) for q in a['preconditions']]
            b['event_reqs'] = [prefix+e for e in a['event_reqs']]
            # Retain business input bounds; no scenario-derived tightening.
            if a['id'] in ('on', 'off', 'cal', 'test'): b['max_instances'] = 4
            if a['id'] == 'downstream': b['duration'] = (5 if length == 'short' else 80) + durations[b['id']]
            if a['id'] in ('base', 'install'): b['resource_reqs'] = {'mechanical': 1}
            if a['id'] in ('on','off'): b['resource_reqs'] = {'power_team': 1}
            acts.append(b); replacements[a['id']] = b
        def req(kind, state):
            replacements[kind]['preconditions'].append({'state_id': prefix+state, 'relation_role': 'required'})
        if m >= 4:
            replacements['base']['preconditions'].append({'state_id': f'm{m-4:02d}:done:base', 'relation_role': 'required'})
        req('on','done:insulate_a'); req('install','done:delivery_check'); req('final','done:insulate_b')
        for kind, predecessors, resource in (
            ('delivery_check', [], 'mechanical'),
            ('wire_a', ['done:base','off'], 'electrical'),
            ('insulate_a', ['done:wire_a','off'], 'electrical'),
            ('wire_b', ['done:install','off'], 'electrical'),
            ('insulate_b', ['done:wire_b','off'], 'electrical')):
            b = activity(prefix+kind, durations[prefix+kind], required=[prefix+s for s in predecessors],
                         resources={resource:1}, events=[prefix+'material'] if kind=='delivery_check' else [])
            b['output_state_id'] = prefix+'done:'+kind
            acts.append(b)
        initial += [prefix+s for s in p['initial_state_ids']]
        goals += [prefix+'done:final']
        events.append({'id': prefix+'material', 'time': release, 'add_state_ids': [], 'remove_state_ids': []})
    iid = f'joint256-a{arrival_level}-r{int(tight)}-{length}-g{seed}'
    payload = base_payload(iid, acts, initial, goals, events,
                           {r: 2 if tight else 16 for r in ('instrument','mechanical','electrical','power_team')}, 512)
    assert len(acts) == 256
    return payload, {'arrival_level': arrival_level, 'release_time': release, 'resource_tight': tight,
                     'downstream_length': length, 'generator_seed': seed, 'reference_time': reference,
                     'activity_count':256, 'optional_business_checks':32,
                     'note':'No required-task set or phase assignment is supplied to the solver'}
