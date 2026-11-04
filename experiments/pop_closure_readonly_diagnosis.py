"""Replay unchanged POP once per selected input, inspect its terminal object.

Does not change any planner methods/constants, resume closure, or run CP-SAT.
Runs on a non-experiment CPU and saves evidence outside the formal batch.
"""
from __future__ import annotations

from collections import Counter
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path('D:/planner/runs/business256-20261102')
OUT = Path('D:/planner/runs/business256-pop-diagnosis-20261102')
SOLVER = Path('D:/Solver_demo_project_planner_exploration')


def isolate():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    handle = k.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise OSError('Cannot inspect affinity')
    available = process_mask.value & ~1
    if not available:
        raise RuntimeError('No separate CPU available; skip diagnostic replay')
    cpu = available.bit_length() - 1
    if not k.SetProcessAffinityMask(handle, 1 << cpu):
        raise OSError('Cannot isolate diagnostic CPU')
    k.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    return cpu


def main():
    cpu = isolate()
    sys.path.insert(0, str(SOLVER))
    from app.core.planner.partial_order import PartialOrderPlanner, MAX_CLOSURE_ITERATIONS, MAX_CLOSURE_INSTANCES
    from app.db.models import StateFeatureDef
    from app.services.planner_legacy_engine import _build_rules
    from planner_experiment.scenario import scenario_from_dict
    from planner_experiment.benchmark_suite.io import read_json, write_json, digest
    from experiments.business256.__main__ import verify
    verify(ROOT)
    results = []
    for iid in ('b256-w16-d4-T0-g101', 'b256-w16-d4-T1-g101',
                'b256-w16-d4-T2-g101', 'b256-w16-d4-T3-g101', 'b256-w32-d2-T0-g101'):
        item = read_json(ROOT / 'instances' / f'{iid}.json')
        original = read_json(ROOT / 'runs' / f'{iid}-t30-s11' / 'LEGACY' / 'engine_result.json')
        scenario = scenario_from_dict(item['scenario'])
        states = sorted(scenario.state_by_id)
        active = set(scenario.initial_state_ids)
        for event in scenario.external_events:
            active.update(event.add_state_ids)
        current = {s: 'active' if s in active else 'inactive' for s in states}
        target = {s: 'active' for s in scenario.goal_state_ids}
        target.update({s: 'inactive' for s in scenario.forbidden_state_ids})
        defs = {s: StateFeatureDef(id=i, machine_type_id=0, feature_key=s, feature_name=s, value_type='enum')
                for i, s in enumerate(states, 1)}
        rules, _, limits = _build_rules(scenario)
        planner = PartialOrderPlanner(current, target, rules, defs, limits)
        fallbacks, reused = [], []
        def profile(frame, event, arg):
            if event != 'return' or not frame.f_code.co_filename.replace('\\', '/').endswith('/planner/partial_order.py'):
                return
            name, loc = frame.f_code.co_name, frame.f_locals
            if name == '_resolve_open_precondition' and arg is True:
                caller = frame.f_back
                if caller is None or caller.f_code.co_name != '_create_rule_instance':
                    return
                ctx = caller.f_locals.get('context_state')
                obligation = loc.get('open_precond')
                if ctx is None or obligation is None:
                    return
                fact = obligation.fact
                if current.get(fact.feature_key) == fact.value and ctx.get(fact.feature_key) != fact.value:
                    link = planner._causal_link_for_consumer_fact(obligation.consumer_id, fact)
                    fallbacks.append({'consumer': obligation.consumer_id, 'fact': fact.feature_key,
                                      'initial_value': current.get(fact.feature_key), 'context_value': ctx.get(fact.feature_key),
                                      'context_writer': (caller.f_locals.get('context_writers') or {}).get(fact.feature_key),
                                      'linked_provider': link.provider_id if link else None})
            elif name == '_best_existing_provider' and isinstance(arg, str):
                caller = frame.f_back
                owner = caller.f_back if caller else None
                if owner is None or owner.f_code.co_name != '_create_rule_instance':
                    return
                ctx, fact = owner.f_locals.get('context_state'), loc.get('fact')
                if ctx is not None and fact is not None and ctx.get(fact.feature_key) != fact.value:
                    reused.append({'consumer': loc.get('consumer_id'), 'fact': fact.feature_key,
                                   'context_value': ctx.get(fact.feature_key), 'selected_provider': arg,
                                   'context_writer': (owner.f_locals.get('context_writers') or {}).get(fact.feature_key)})
        try:
            if iid == 'b256-w16-d4-T2-g101':
                sys.setprofile(profile)
            result = planner.plan()
        finally:
            sys.setprofile(None)
        # Exact diagnostic equality proves this replay reached the same stopped
        # planner object as the unmodified formal engine (no altered policy).
        assert result.diagnostics == original['stats']['pathfinder'], iid
        before = digest(result.diagnostics)
        steps, final_state, writers = planner._replay_plan()
        unmet = []
        for step in steps:
            for p in planner.instances[step.instance_id].rule.preconditions:
                if not planner.evaluator.evaluate_precondition(step.before_state, p):
                    unmet.append({'consumer': step.instance_id, 'fact': p.feature_key,
                                  'expected': p.feature_value, 'observed': step.before_state.get(p.feature_key),
                                  'latest_writer': step.latest_writers_before.get(p.feature_key)})
        causal = result.diagnostics['causal_window_orderings']
        inserted = result.diagnostics['reprovider_insertions']
        counts = Counter((x['consumer'], x['fact'], x.get('before_writer', x.get('latest_writer'))) for x in [*causal, *inserted])
        power_counts = {s.rule.code: planner.instance_counts.get(s.rule.id, 0) for s in planner.instances.values() if s.rule.code.startswith('p')}
        datum = {'instance_id': iid, 'matches_formal_diagnostics': True, 'status': result.status, 'error_code': result.error_code,
                 'initial_fact_fallback_despite_inactive_context': fallbacks,
                 'existing_provider_selected_despite_inactive_context': reused,
                 'closure_iterations': result.diagnostics['closure_iterations'], 'selected_instances': len(planner.instances),
                 'ordering_repairs': len(causal), 'provider_repairs': len(inserted),
                 'distinct_repair_keys': len(counts), 'duplicate_repair_keys': [{'key': list(k), 'count': c} for k, c in counts.items() if c > 1],
                 'unmet_after_final_repair': unmet,
                 'unmet_fact_counts': dict(Counter(u['fact'] for u in unmet)),
                 'unmet_final_goals': {k: {'expected': v, 'observed': final_state.get(k)} for k,v in target.items() if final_state.get(k) != v},
                 'power_instance_counts': power_counts,
                 'power_instances_at_input_cap': {k: v for k,v in power_counts.items() if v >= 6},
                 'first_repairs': causal[:3], 'last_repairs': causal[-5:], 'provider_repair_trace': inserted,
                 'remaining_isolation_tasks': sorted({u['consumer'] for u in unmet if u['fact'].endswith(':isolated')}),
                 'initial_isolation_consumers': sum(any(p.state_id.endswith(':isolated') and p.relation_role == 'required'
                                                       for p in a.precondition_bindings) for a in scenario.activities)}
        assert before == digest(result.diagnostics)
        results.append(datum)
        print(iid, {'repairs': (len(causal),len(inserted)), 'remaining_unmet':len(unmet),
                    'remaining_goals':len(datum['unmet_final_goals']), 'power_at_cap':datum['power_instances_at_input_cap']}, flush=True)
    verify(ROOT)
    write_json(OUT / 'diagnosis.json', {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'cpu': cpu,
                'algorithm_source_guard': 'PASS', 'formal_batch_untouched': True,
                'constants_unchanged': {'MAX_CLOSURE_ITERATIONS':MAX_CLOSURE_ITERATIONS,'MAX_CLOSURE_INSTANCES':MAX_CLOSURE_INSTANCES},
                'method': 'unchanged original POP; inspect terminal replay only; do not resume or enlarge limits', 'results': results})


if __name__ == '__main__':
    main()
