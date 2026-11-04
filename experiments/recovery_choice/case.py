"""Two required offline repairs; only their second opening differs by case."""
import copy
from collections import Counter

from experiments.mainline_joint.case import describe
from experiments.window_joint.adapter import certificate
from planner_experiment.benchmark_suite.io import digest

REPAIRS=('CHECK_B1_10','CHECK_B2_10')


def raw_rows(cert):
    return [{'activity_id':e['activity_id'],'ordinal':e['ordinal'],
             'start':e['start_time'],'end':e['end_time']} for e in cert['executions']]


def audit_cycles(payload,cert):
    executions=cert['executions']
    counts=Counter(e['activity_id'] for e in executions)
    groups={}
    for aid in REPAIRS:
        repairs=[e for e in executions if e['activity_id']==aid]
        assert len(repairs)==1
        r=repairs[0]
        off=max((e for e in executions if e['activity_id']=='power_off' and e['end_time']<=r['start_time']),key=lambda e:e['end_time'])
        on=min((e for e in executions if e['activity_id']=='power_on' and e['start_time']>=r['end_time']),key=lambda e:e['start_time'])
        key=(off['ordinal'],on['ordinal'])
        groups.setdefault(key,{'power_off':off,'power_on':on,'repairs':[]})['repairs'].append(r)
    episodes=[]
    for group in sorted(groups.values(),key=lambda g:g['power_off']['start_time']):
        last=max(e['end_time'] for e in group['repairs'])
        next_off=min((e['start_time'] for e in executions if e['activity_id']=='power_off' and e['start_time']>=group['power_on']['end_time']),default=float('inf'))
        qualifications={}
        for aid in ('M080','M081'):
            after=[e for e in executions if e['activity_id']==aid and e['start_time']>=last and e['end_time']<=next_off]
            assert after,'Both qualifications must be restored within each powered episode'
            qualifications[aid]=min(after,key=lambda e:e['start_time'])
        group['qualifications']=qualifications
        episodes.append(group)
    assert counts['M080']>=len(groups)+1 and counts['M081']>=len(groups)+1
    first_test=min(e['end_time'] for e in executions if e['activity_id']=='M081')
    assert all(first_test<=e['start_time'] for e in executions if e['activity_id'] in REPAIRS)
    assert len(executions)<=payload['max_steps']
    return {'repair_count':2,'repair_episode_count':len(groups),
            'actual_power_off_count':counts['power_off'],'calibration_count':counts['M080'],
            'testing_count':counts['M081'],'execution_counts':dict(counts),'episodes':episodes}


def make_case(baseline,r1_certificate,r2_certificate,variant):
    if variant not in ('R4','R5'):raise ValueError('Unknown variant')
    payload=copy.deepcopy(baseline)
    payload['id']+=f'-{variant.lower()}-repair-grouping'
    payload['name']=payload['id']
    acts={a['id']:a for a in payload['activities']}
    for aid in REPAIRS:
        a=acts[aid]
        a['name']='M100后可择时的离线整改：使校准与调测失效'
        a['preconditions'] += [{'state_id':'done:M100','relation_role':'required'},
                               {'state_id':'mode:isolated','relation_role':'required'}]
        a['additional_output_state_ids'] += ['dirty:cal','dirty:test']
        a['delete_effect_state_ids'] += ['valid:cal','valid:test']
        payload['goal_state_ids'].append('done:'+aid)
    # All original thirty windows are unchanged. Both cases add the same two
    # repair windows; their sole difference is RW2's release, 830 versus 1150.
    payload['time_windows'] += [
        {'id':'RW1','activity_id':REPAIRS[0],'earliest_start':830,'latest_finish':900},
        {'id':'RW2','activity_id':REPAIRS[1],'earliest_start':830 if variant=='R4' else 1150,'latest_finish':1200}]
    # Build a one-outage private witness by inserting repair 2 immediately
    # after repair 1 in the already validated R1 schedule.
    cut=next(e['end_time'] for e in r1_certificate['executions'] if e['activity_id']==REPAIRS[0])
    duration=acts[REPAIRS[1]]['duration']
    merged=[]
    for e in r1_certificate['executions']:
        if e['activity_id'].startswith('CHECK_') and e['activity_id']!=REPAIRS[0]:continue
        assert not(e['start_time']<cut<e['end_time'])
        shift=duration if e['start_time']>=cut else 0
        merged.append({'activity_id':e['activity_id'],'ordinal':e['ordinal'],
                       'start':e['start_time']+shift,'end':e['end_time']+shift})
    merged.append({'activity_id':REPAIRS[1],'ordinal':1,'start':cut,'end':cut+duration})
    merged.sort(key=lambda r:(r['start'],r['end'],r['activity_id']))
    split=[r for r in raw_rows(r2_certificate) if not r['activity_id'].startswith('CHECK_') or r['activity_id'] in REPAIRS]
    split_cert=certificate(payload,split)
    split_audit=audit_cycles(payload,split_cert)
    assert split_audit['repair_episode_count']==2 and split_cert['makespan']==1494
    rows=merged if variant=='R4' else split
    cert=certificate(payload,rows)
    audit=audit_cycles(payload,cert)
    minimum_episodes=1 if variant=='R4' else 2
    assert audit['repair_episode_count']==minimum_episodes
    release=next(w['earliest_start'] for w in payload['time_windows'] if w['activity_id']=='M090')
    suffix=sum(acts[f'M{i:03d}']['duration'] for i in range(90,181))
    overhead=sum(acts[a]['duration'] for a in ('power_off','ground','unground','power_on','M080','M081'))
    repairs=sum(acts[a]['duration'] for a in REPAIRS)
    lower=release+suffix+overhead*minimum_episodes+repairs
    assert cert['makespan']==lower==(1462 if variant=='R4' else 1494)
    evidence={'baseline_input_hash':digest(baseline),'variant':variant,
              'generation_policy':'Two required offline repairs after M100, no immediate successor gates. Preserve all original 30 windows; add RW1 [830,900] and RW2 [830,1200] in R4 or [1150,1200] in R5. Only RW2 opening differs. Existing M130 online window [1079,1131] lies strictly between R5 repair windows and forces separate restored powered episodes.',
              'witness_raw_rows':rows,'witness_certificate':cert,'witness_cycle_audit':audit,
              'split_witness_raw_rows':split,'split_witness_certificate':split_cert,'split_cycle_audit':split_audit,
              'answer':{'status':'PROVEN_OPTIMAL_BY_CHAIN_AND_WITNESS','optimal_makespan':lower,
                        'lower_bound':lower,'minimum_repair_episodes':minimum_episodes},
              'independent_proof':{'release_time':release,'mainline_suffix_duration':suffix,
                'overhead_per_recovery_episode':overhead,'total_repair_duration':repairs,
                'minimum_repair_episodes':minimum_episodes,
                'mandatory_online_separator':'M130 within [1079,1131]' if variant=='R5' else None,
                'reason':'Both repairs require completed M100, so occur after the initial test. Offline repairs and mode/qualification restoration cannot overlap the mandatory on/valid-test mainline suffix. Instrument capacity1 serializes the two repairs. One merged episode costs32+4=36. In R5 the M130 online window between repair windows forces two episodes, costing64+4=68. Add these to M090 release755 plus serial suffix671. Attaining replayed witnesses certify the bounds.'},
              'formal_solver_policy':'Raw input only; no grouping decision, episode count, witness, bound or hint provided',
              'comparison_scope':'One pair of synthetic models, three run seeds; R4 genuinely admits merged and split schedules. R5 only changes RW2 opening, not task availability via a fixed successor gate.'}
    process=[a['id'] for a in payload['activities'] if a['id'].startswith(('M','B'))]
    nodes=process+list(REPAIRS)
    edges=[[p['state_id'][5:],a['id']] for a in payload['activities'] if a['id'] in nodes
           for p in a['preconditions'] if p['state_id'].startswith('done:') and p['state_id'][5:] in nodes]
    metadata={'task_activities':{a:[a] for a in nodes},'edges':edges,'reference_activity_ids':process,
              'reference_h0_minutes':sum(acts[f'M{i:03d}']['duration'] for i in range(1,181)),
              'witness_activity_ids':[e['activity_id'] for e in cert['executions']]}
    metrics=describe(payload,metadata)
    metrics.update(time_window_count=32,original_time_window_count=30,repair_time_window_count=2,
                   required_repairs=2,minimum_recovery_episodes=minimum_episodes,
                   witness_recovery_episodes=audit['repair_episode_count'],
                   mandatory_process_task_count=242,optional_extended_checks=10,
                   reference_load_policy='Original 240 reference tasks and H0=1336, repeat workload reported separately')
    metrics['scope']='History DAG covers242 required definitions; repeated restoration instances and temporal window implications are separate.'
    evidence['metadata']=metadata
    return payload,metrics,evidence
