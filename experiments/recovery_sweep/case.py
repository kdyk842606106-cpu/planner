"""Nested forced repair cases, with unchanged original time windows."""
import copy
from collections import Counter

from experiments.mainline_joint.case import describe
from experiments.window_joint.adapter import certificate
from planner_experiment.benchmark_suite.io import digest

REPAIRS = [('CHECK_B1_10',100), ('CHECK_B2_10',140), ('CHECK_B3_10',160)]


def episode(aid):
    return ['power_off','ground',aid,'unground','power_on','M080','M081']


def audit_cycles(payload, cert, count):
    executions=cert['executions']
    counts=Counter(e['activity_id'] for e in executions)
    repairs=[]
    previous_end=-1
    for aid, milestone in REPAIRS[:count]:
        rows=[e for e in executions if e['activity_id']==aid]
        assert len(rows)==1
        row=rows[0]
        prior=next(e for e in executions if e['activity_id']==f'M{milestone:03d}')
        next_task=next(e for e in executions if e['activity_id']==f'M{milestone+1:03d}')
        assert prior['end_time']<=row['start_time']<row['end_time']<=next_task['start_time']
        recovered={}
        for qual in ('M080','M081'):
            before=[e for e in executions if e['activity_id']==qual and previous_end<=e['start_time'] and e['end_time']<=row['start_time']]
            after=[e for e in executions if e['activity_id']==qual and row['end_time']<=e['start_time'] and e['end_time']<=next_task['start_time']]
            assert before and after, 'Each episode must destroy and restore qualifications separately'
            recovered[qual]={'before':before[-1], 'after':after[0]}
        repairs.append({'repair':aid,'milestone':milestone,'start':row['start_time'],'end':row['end_time'],'qualifications':recovered})
        previous_end=row['end_time']
    for aid in ('M080','M081','unground','power_on'):
        assert counts[aid]>=count+1
    for aid in ('ground','power_off'):
        assert counts[aid]>=count
    assert len(executions)<=payload['max_steps']
    return {'episode_count':count,'execution_counts':dict(counts),'episodes':repairs}


def make_case(r1, r1_certificate, count):
    if count not in (2,3):
        raise ValueError('Only R2/R3 are in this stage')
    payload=copy.deepcopy(r1)
    payload['id']=r1['id'].removesuffix('-forced-recovery1')+f'-forced-recovery{count}'
    payload['name']=payload['id']
    acts={a['id']:a for a in payload['activities']}
    for aid,milestone in REPAIRS[1:count]:
        repair=acts[aid]
        repair['name']=f'M{milestone:03d}后离线整改：使校准与调测失效'
        repair['preconditions'] += [{'state_id':f'done:M{milestone:03d}','relation_role':'required'},
                                    {'state_id':'mode:isolated','relation_role':'required'}]
        repair['additional_output_state_ids'] += ['dirty:cal','dirty:test']
        repair['delete_effect_state_ids'] += ['valid:cal','valid:test']
        acts[f'M{milestone+1:03d}']['preconditions'].append({'state_id':'done:'+aid,'relation_role':'required'})
    assert all(sum(acts[a]['duration'] for a in episode(aid))==34 for aid,_ in REPAIRS[:count])
    # Preserve the validated one-recovery execution order, remove optional checks,
    # and insert later episodes into globally idle boundaries. This is private.
    rows=[]
    cuts=[(next(e['end_time'] for e in r1_certificate['executions'] if e['activity_id']==f'M{milestone:03d}'),aid)
          for aid,milestone in REPAIRS[1:count]]
    for e in r1_certificate['executions']:
        if e['activity_id'].startswith('CHECK_') and e['activity_id']!=REPAIRS[0][0]:
            continue
        assert all(not(e['start_time']<cut<e['end_time']) for cut,_ in cuts)
        shift=34*sum(e['start_time']>=cut for cut,_ in cuts)
        rows.append({'activity_id':e['activity_id'],'ordinal':e['ordinal'],
                     'start':e['start_time']+shift,'end':e['end_time']+shift})
    ordinals=Counter(r['activity_id'] for r in rows)
    for index,(cut,aid) in enumerate(cuts):
        t=cut+34*index
        for action in episode(aid):
            ordinals[action]+=1
            rows.append({'activity_id':action,'ordinal':ordinals[action],'start':t,'end':t+acts[action]['duration']})
            t+=acts[action]['duration']
    rows.sort(key=lambda r:(r['start'],r['end'],r['activity_id']))
    release=next(w['earliest_start'] for w in payload['time_windows'] if w['activity_id']=='M090')
    deadline=next(w['latest_finish'] for w in payload['time_windows'] if w['activity_id']=='M180')
    suffix=sum(acts[f'M{i:03d}']['duration'] for i in range(90,181))
    lower=release+suffix+34*count
    proof={'release_time':release,'mandatory_mainline_suffix':suffix,'episode_count':count,
           'minimum_duration_per_episode':34,'completion_lower_bound':lower,'final_latest_finish':deadline,
           'deadline_excess':max(0,lower-deadline),
           'reason':'M090..M180 is mandatory and serial. Each separate milestone gate requires offline repair between two on/valid-test mainline tasks. The unique mode transitions and qualification producers force a distinct 34-unit recovery episode. Recovery cannot overlap mainline progress or another gated episode.'}
    evidence={'baseline_input_hash':digest(r1),'episode_specs':[{'activity_id':a,'after_mainline':m} for a,m in REPAIRS[:count]],
              'generation_policy':'Nested required offline repairs after M100, M140, M160; only add gates/invalidation to existing checks. Keep all 30 windows and resource/duration/count bounds. No timing-based tuning of milestones.',
              'independent_proof':proof,'formal_solver_policy':'Raw input only; no private proof, bounds, witness, selected tasks or hints'}
    if lower<=deadline:
        cert=certificate(payload,rows)
        assert cert['makespan']==lower
        evidence.update(answer={'status':'PROVEN_OPTIMAL_BY_CHAIN_AND_WITNESS','optimal_makespan':lower},
                        witness_raw_rows=rows,witness_certificate=cert,witness_cycle_audit=audit_cycles(payload,cert,count))
    else:
        # A separate diagnostic removes only the final deadline conflict. Never
        # give this relaxed input to a formal worker or call it an R3 solution.
        diagnostic=copy.deepcopy(payload)
        for window in diagnostic['time_windows']:
            if window['activity_id']=='M180':window['latest_finish']=lower
        cert=certificate(diagnostic,rows)
        assert cert['makespan']==lower
        evidence.update(answer={'status':'PROVEN_INFEASIBLE_BY_WINDOW_CONTRADICTION','optimal_makespan':None},
                        relaxed_diagnostic={'not_a_formal_R3_solution':True,
                          'changed_window':'M180 latest_finish only','original_latest_finish':deadline,
                          'diagnostic_latest_finish':lower,'diagnostic_input_hash':digest(diagnostic),
                          'witness_raw_rows':rows,'witness_certificate':cert,
                          'cycle_audit':audit_cycles(diagnostic,cert,count)})
    process=[a['id'] for a in payload['activities'] if a['id'].startswith(('M','B'))]
    nodes=process+[a for a,_ in REPAIRS[:count]]
    edges=[[p['state_id'][5:],a['id']] for a in payload['activities'] if a['id'] in nodes
           for p in a['preconditions'] if p['state_id'].startswith('done:') and p['state_id'][5:] in nodes]
    metadata={'task_activities':{a:[a] for a in nodes},'edges':edges,'reference_activity_ids':process,
              'reference_h0_minutes':sum(acts[f'M{i:03d}']['duration'] for i in range(1,181)),
              'witness_activity_ids':[e['activity_id'] for e in cert['executions']] if lower<=deadline else []}
    metrics=describe(payload,metadata)
    metrics.update(time_window_count=30,required_recovery_episodes=count,
                   minimum_calibration_and_test_counts=count+1,
                   optional_extended_checks=12-count,mandatory_process_task_count=240+count,
                   feasibility=evidence['answer']['status'],
                   reference_load_policy='Same 240 reference tasks and H0=1336; repeats are separate from static reference load')
    metrics['scope']=f'History DAG includes {240+count} required task definitions; repeated executions/mode-state cycles are separate.'
    if lower>deadline:metrics['witness_repeat_count']=None
    evidence['metadata']=metadata
    return payload,metrics,evidence
