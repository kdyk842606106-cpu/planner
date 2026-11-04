"""Force a complete restoration episode without changing solver algorithms."""
import copy
from collections import Counter

from experiments.window_joint.adapter import certificate
from experiments.mainline_joint.case import describe
from planner_experiment.benchmark_suite.io import digest

REPAIR = 'CHECK_B1_10'
CYCLE = ['power_off', 'ground', REPAIR, 'unground', 'power_on', 'M080', 'M081']


def cycle_audit(cert):
    executions = cert['executions']
    repairs = [e for e in executions if e['activity_id'] == REPAIR]
    counts = Counter(e['activity_id'] for e in executions)
    assert len(repairs) == 1
    repair = repairs[0]
    before, after = {}, {}
    for aid in ('M080', 'M081'):
        before[aid] = [e for e in executions if e['activity_id'] == aid and e['end_time'] <= repair['start_time']]
        after[aid] = [e for e in executions if e['activity_id'] == aid and e['start_time'] >= repair['end_time']]
        assert before[aid] and after[aid], 'Qualification must be achieved both before and after repair'
    assert counts['power_on'] >= 2 and counts['power_off'] >= 1 and counts['ground'] >= 1
    return {'execution_counts':dict(counts), 'repair_start':repair['start_time'], 'repair_end':repair['end_time'],
            'calibration_before_and_after':True, 'testing_before_and_after':True,
            'complete_power_cycle_present':True,
            'qualification_rows':[e for e in executions if e['activity_id'] in CYCLE]}


def make_case(baseline, baseline_certificate, baseline_metrics):
    payload = copy.deepcopy(baseline)
    payload['id'] += '-forced-recovery1'
    payload['name'] = payload['id']
    acts = {a['id']:a for a in payload['activities']}
    repair = acts[REPAIR]
    repair['name'] = 'M100后离线整改：使校准与调测失效'
    repair['preconditions'] += [{'state_id':'done:M100','relation_role':'required'},
                                {'state_id':'mode:isolated','relation_role':'required'}]
    repair['additional_output_state_ids'] += ['dirty:cal', 'dirty:test']
    repair['delete_effect_state_ids'] += ['valid:cal', 'valid:test']
    acts['M101']['preconditions'].append({'state_id':'done:'+REPAIR,'relation_role':'required'})
    # All original resource requests, durations, count limits and windows stay fixed.
    # Move the formerly optional check; do not give the solver this witness.
    cut = next(e['end_time'] for e in baseline_certificate['executions'] if e['activity_id']=='M100')
    cycle_duration = sum(acts[a]['duration'] for a in CYCLE)
    raw=[]
    for e in baseline_certificate['executions']:
        if e['activity_id'].startswith('CHECK_'):
            continue
        assert not (e['start_time'] < cut < e['end_time']), 'Insertion point must be idle for all other work'
        shift=cycle_duration if e['start_time'] >= cut else 0
        raw.append({'activity_id':e['activity_id'],'ordinal':e['ordinal'],
                    'start':e['start_time']+shift,'end':e['end_time']+shift})
    counts=Counter(r['activity_id'] for r in raw)
    t=cut
    for aid in CYCLE:
        counts[aid]+=1
        raw.append({'activity_id':aid,'ordinal':counts[aid],'start':t,'end':t+acts[aid]['duration']})
        t+=acts[aid]['duration']
    raw.sort(key=lambda r:(r['start'],r['end'],r['activity_id']))
    cert=certificate(payload,raw)
    audit=cycle_audit(cert)
    # M090's unchanged release plus its mandatory serial suffix and mandatory
    # complete recovery between M100/M101 gives an independent lower bound.
    release=next(w['earliest_start'] for w in payload['time_windows'] if w['activity_id']=='M090')
    suffix=sum(acts[f'M{i:03d}']['duration'] for i in range(90,181))
    lower=max(1451,release+suffix+cycle_duration)
    assert lower <= cert['makespan']
    process_ids=[a['id'] for a in payload['activities'] if a['id'].startswith(('M','B'))]
    history_ids=process_ids+[REPAIR]
    history_set=set(history_ids)
    history_edges=[[p['state_id'][5:],a['id']] for a in payload['activities'] if a['id'] in history_set
                   for p in a['preconditions'] if p['state_id'].startswith('done:') and p['state_id'][5:] in history_set]
    metadata={'task_activities':{a:[a] for a in history_ids},'edges':history_edges,
              'reference_activity_ids':process_ids,
              'reference_h0_minutes':sum(acts[f'M{i:03d}']['duration'] for i in range(1,181)),
              'witness_activity_ids':[e['activity_id'] for e in cert['executions']]}
    metrics=describe(payload,metadata)
    metrics.update(precondition_count=sum(len(a['preconditions']) for a in payload['activities']),
                   time_window_count=len(payload['time_windows']),
                   mandatory_process_task_count=241, optional_extended_checks=11,
                   restoration_requirement={'minimum_calibrations':2,'minimum_tests':2,
                                            'minimum_power_on':2,'minimum_power_off':1,
                                            'minimum_complete_recovery_episodes':1},
                   cycle_activity_ids=CYCLE, minimum_serial_recovery_duration=cycle_duration)
    metrics['mean_preconditions']=metrics['precondition_count']/len(payload['activities'])
    metrics['independent_delete_activity_fraction']=sum(bool(a['delete_effect_state_ids']) for a in payload['activities'])/len(payload['activities'])
    metrics['scope']='History DAG covers 241 required task definitions including repair; repeated calibration/testing occurrences and mode cycles are separately reported, not represented as a cyclic task DAG.'
    metrics['reference_load_policy']='Same original 240 reference tasks and H0=1336; excludes repeat workload and repair, whose demands are reported separately.'
    metrics['witness_resource_work']={r['id']:sum((e['end_time']-e['start_time'])*acts[e['activity_id']]['resource_reqs'].get(r['id'],0) for e in cert['executions']) for r in payload['resources']}
    evidence={'baseline_input_hash':digest(baseline),
              'generation_policy':'One mandatory offline repair between M100 and M101 using existing CHECK_B1_10; retain resource requests/durations/windows and the original 332 process edges. Add 3 preconditions and qualification invalidation effects. Known witness is validation only.',
              'changed_activity_ids':[REPAIR,'M101'], 'metadata':metadata, 'witness_raw_rows':raw,'witness_certificate':cert,
              'witness_cycle_audit':audit,
              'answer':{'status':'PROVEN_OPTIMAL_BY_CHAIN_AND_WITNESS' if cert['makespan']==lower else 'OPTIMUM_UNKNOWN',
                        'lower_bound':lower,'witness_makespan':cert['makespan'],
                        'optimal_makespan':lower if cert['makespan']==lower else None,
                        'release_activity':'M090','release_time':release,'mainline_suffix_duration':suffix,
                        'serial_recovery_duration':cycle_duration,
                        'proof':'M090 starts no earlier than its window. M090..M180 form a mandatory serial chain. Between M100 (on, valid test) and M101 (on, valid test, repair done), repair requires isolated mode and deletes both qualifications. Unique mode transitions and unique qualification producers force power_off, ground, repair, unground, power_on, M080, M081 in serial order, costing 34 beyond the mainline suffix. The input requires at least two calibrations and tests, not merely repeated history labels.'},
              'formal_solver_policy':'Unchanged solver reads raw input only; no witness, cycle sequence, lower bound or supplied hints'}
    return payload,metrics,evidence
