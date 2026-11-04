"""Independent exact business256 oracle, outside the frozen three-engine benchmark.

One B installation is the only local certificate invalidator. A calibration
and testing may occur once before and once after B; other process activities
occur once. Optional power phases cover isolated->ready->on->off->isolated,
with at most six on phases. Historical and current-valid prerequisites differ.
No tested algorithm or input is changed; this is a separate reference oracle.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import heapq
from pathlib import Path
import time
from types import SimpleNamespace

from ortools.sat.python import cp_model

from experiments.pop_closure_readonly_diagnosis import isolate
from experiments.business256.__main__ import verify
from planner_experiment.benchmark_suite.io import read_json, write_json
from planner_experiment.benchmark_suite.generator import candidate_from_actions
from planner_experiment.models import Action, to_primitive
from planner_experiment.scenario import scenario_from_dict
from planner_experiment.simulator import PathSimulator

ROOT = Path('D:/planner/runs/business256-20261102')
OUT = Path('D:/planner/runs/business256-optimum-check-20261102')


def build(item):
    restore = item['metadata']['condition'] in ('T2', 'T3')
    payload = item['scenario']
    acts = {a['id']: a for a in payload['activities']}
    for a in acts.values():
        if a['delete_effect_state_ids'] and not (restore and a['id'].endswith(':install_b')):
            raise ValueError('Unsupported invalidator')
    tasks = list(item['metadata']['task_activities'])
    groups = sorted({int(a[1:3]) for a in acts if a.startswith('p')})
    modules = sorted({int(a.split(':')[0][1:]) for a in tasks})
    horizon = item['answer']['witness_makespan']
    model = cp_model.CpModel()
    starts, ends, intervals, active, duration = {}, {}, {}, {}, {}
    pre_present, first_end = {}, {}
    constraints = set()
    release = {}
    assigned = {}
    event_time = {e['id']: e['time'] for e in payload['external_events']}

    def interval(key, d, presence=None):
        starts[key] = model.new_int_var(0, horizon, 's:'+key)
        ends[key] = model.new_int_var(0, horizon, 'e:'+key)
        duration[key] = d
        if presence is None:
            intervals[key] = model.new_interval_var(starts[key], d, ends[key], key)
        else:
            intervals[key] = model.new_optional_interval_var(starts[key], d, ends[key], presence, key)

    for aid in tasks:
        interval(aid, acts[aid]['duration'])
        release[aid] = max((event_time[e] for e in acts[aid].get('event_reqs', [])), default=0)
        model.add(starts[aid] >= release[aid])
    if restore:
        for m in modules:
            for kind in ('cal_a', 'test_a'):
                aid = f'm{m:02d}:{kind}'
                key = aid+'#pre'
                present = model.new_bool_var('present:'+key)
                pre_present[aid] = present
                interval(key, acts[aid]['duration'], present)
                model.add(ends[key] <= starts[f'm{m:02d}:install_b']).only_enforce_if(present)
                model.add(starts[aid] >= ends[f'm{m:02d}:install_b'])
                first_end[aid] = model.new_int_var(0, horizon, 'first_end:'+aid)
                model.add(first_end[aid] == ends[key]).only_enforce_if(present)
                model.add(first_end[aid] == ends[aid]).only_enforce_if(present.Not())
            model.add(pre_present[f'm{m:02d}:test_a'] <= pre_present[f'm{m:02d}:cal_a'])
            if m % 2 == 0:
                model.add(pre_present[f'm{m:02d}:test_a'] == 1)
    for aid in tasks:
        first_end.setdefault(aid, ends[aid])
    makespan = model.new_int_var(item['answer']['lower_bound'], horizon, 'makespan')
    for g in groups:
        for phase in range(6):
            active[g, phase] = model.new_bool_var(f'active:{g}:{phase}')
            if phase == 0:
                model.add(active[g, phase] == 1)
            else:
                model.add(active[g, phase] <= active[g, phase-1])
            for kind in ('unground', 'on'):
                key = f'p{g:02d}:{kind}#{phase}'
                interval(key, acts[f'p{g:02d}:{kind}']['duration'], active[g, phase])
            model.add(starts[f'p{g:02d}:on#{phase}'] >= ends[f'p{g:02d}:unground#{phase}']).only_enforce_if(active[g, phase])
            model.add(makespan >= ends[f'p{g:02d}:on#{phase}']).only_enforce_if(active[g, phase])
            if phase:
                for kind in ('off', 'ground'):
                    key = f'p{g:02d}:{kind}#{phase-1}'
                    interval(key, acts[f'p{g:02d}:{kind}']['duration'], active[g, phase])
                chain = [f'p{g:02d}:on#{phase-1}', f'p{g:02d}:off#{phase-1}',
                         f'p{g:02d}:ground#{phase-1}', f'p{g:02d}:unground#{phase}']
                for before, after in zip(chain, chain[1:]):
                    model.add(starts[after] >= ends[before]).only_enforce_if(active[g, phase])
    for key in [*tasks, *[a+'#pre' for a in pre_present]]:
        aid = key.split('#')[0]
        optional = pre_present.get(aid) if key.endswith('#pre') else None
        def enforce(constraint):
            if optional is not None:
                constraint.only_enforce_if(optional)
            return constraint
        enforce(model.add(makespan >= ends[key]))
        mode = None
        for pre in acts[aid]['preconditions']:
            fact = pre['state_id']
            if fact.startswith('history:'):
                provider = fact.split(':', 1)[1]
                if provider not in tasks:
                    raise ValueError('Unexpected prerequisite provider')
                enforce(model.add(starts[key] >= first_end[provider]))
            elif fact.startswith('valid:'):
                provider = fact.split(':', 1)[1]
                if key.endswith('#pre') and provider in pre_present:
                    provider += '#pre'
                enforce(model.add(starts[key] >= ends[provider]))
            elif fact.startswith('invalid:'):
                assert fact == 'invalid:'+aid and pre['relation_role'] == 'transition'
            elif fact.startswith('p'):
                mode = fact
            else:
                raise ValueError(f'Unhandled prerequisite {fact}')
        if mode:
            group, state = mode.split(':')
            g = int(group[1:])
            choices = []
            for phase in range(6):
                choice = model.new_bool_var(f'{key}@{phase}')
                choices.append(choice)
                assigned[key, phase] = choice
                model.add(choice <= active[g, phase])
                if state == 'isolated':
                    if phase:
                        model.add(starts[key] >= ends[f'{group}:ground#{phase-1}']).only_enforce_if(choice)
                    model.add(ends[key] <= starts[f'{group}:unground#{phase}']).only_enforce_if(choice)
                elif state == 'on':
                    model.add(starts[key] >= ends[f'{group}:on#{phase}']).only_enforce_if(choice)
                    if phase < 5:
                        model.add(ends[key] <= starts[f'{group}:off#{phase}']).only_enforce_if([choice, active[g, phase+1]])
                else:
                    raise ValueError('Unexpected task mode')
            if optional is not None:
                model.add(sum(choices) == optional)
            else:
                model.add_exactly_one(choices)
    for r in payload['resources']:
        resource_intervals, demands = [], []
        for key in intervals:
            activity = acts[key.split('#')[0]]
            quantity = activity['resource_reqs'].get(r['id'], 0)
            if quantity:
                resource_intervals.append(intervals[key])
                demands.append(quantity)
        model.add_cumulative(resource_intervals, demands, r['capacity'])
    # Simulator protects full activity intervals, even when a repeated write
    # merely reasserts an already true history. Those conflicts matter after
    # retrofit when other modules still read the earlier completion history.
    definitions = scenario_from_dict(payload).activity_by_id
    keys = list(intervals)
    locks = []
    reads = {k: set(definitions[k.split('#')[0]].required_state_ids) for k in keys}
    writes = {k: set(definitions[k.split('#')[0]].written_state_ids) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if writes[a] & (reads[b] | writes[b]) or reads[a] & writes[b]:
                model.add_no_overlap([intervals[a], intervals[b]])
                locks.append((a,b))
    model.minimize(makespan)
    return model, (starts, ends, active, assigned, duration, constraints, release, pre_present, locks)


def extract(reader, data):
    starts, ends, active, assigned, duration, edges, release, pre_present = data[:8]
    phases = {g: sum(reader.value(active[g,p]) for p in range(6)) for g in sorted({g for g,p in active})}
    rows = []
    for key in starts:
        if key.endswith('#pre') and not reader.value(pre_present[key.split('#')[0]]):
            continue
        if key.startswith('p'):
            base, number = key.split('#')
            g = int(base[1:3]); phase = int(number)
            count = phases[g] if base.split(':')[1] in ('unground','on') else phases[g]-1
            if phase >= count:
                continue
        rows.append({'key':key,'activity_id':key.split('#')[0], 'start':reader.value(starts[key]),'end':reader.value(ends[key])})
    allocations = {aid: p for (aid,p), variable in assigned.items() if reader.value(variable)}
    return {'rows':rows, 'phases':phases, 'allocations':allocations}


def certificate(item, schedule, data):
    """Left shift along explicit phase/resource orders into event-driven form."""
    payload = item['scenario']
    acts = {a['id']:a for a in payload['activities']}
    rows = {r['key']:r for r in schedule['rows']}
    edges = set()
    for a,b in data[8]:
        if a not in rows or b not in rows:
            continue
        if rows[a]['end'] <= rows[b]['start']:
            edges.add((a,b))
        elif rows[b]['end'] <= rows[a]['start']:
            edges.add((b,a))
        else:
            raise AssertionError('Oracle state-lock conflict')
    producers = defaultdict(list)
    for key,row in rows.items():
        producers[row['activity_id']].append(key)
    for key,row in rows.items():
        if key.startswith('p'):
            continue
        aid=row['activity_id']
        for pre in acts[aid]['preconditions']:
            fact=pre['state_id']
            if fact.startswith('history:'):
                provider=min(producers[fact.split(':',1)[1]], key=lambda p:rows[p]['end'])
                edges.add((provider,key))
            elif fact.startswith('valid:'):
                provider=fact.split(':',1)[1]
                if key.endswith('#pre') and provider+'#pre' in rows:
                    provider+='#pre'
                edges.add((provider,key))
        if key.endswith('#pre'):
            edges.add((key,aid.split(':')[0]+':install_b'))
        elif item['metadata']['condition'] in ('T2','T3') and aid.endswith((':cal_a',':test_a')):
            edges.add((aid.split(':')[0]+':install_b',key))
    for g, count in schedule['phases'].items():
        chain=[]
        for p in range(count):
            if p:chain.extend((f'p{g:02d}:off#{p-1}',f'p{g:02d}:ground#{p-1}'))
            chain.extend((f'p{g:02d}:unground#{p}',f'p{g:02d}:on#{p}'))
        edges.update(zip(chain,chain[1:]))
    for key,p in schedule['allocations'].items():
        aid=key.split('#')[0]
        mode=next(x['state_id'] for x in acts[aid]['preconditions'] if x['state_id'].startswith('p'))
        group,state=mode.split(':');g=int(group[1:])
        if state=='isolated':
            if p:edges.add((f'{group}:ground#{p-1}',key))
            edges.add((key,f'{group}:unground#{p}'))
        else:
            edges.add((f'{group}:on#{p}',key))
            if p+1<schedule['phases'][g]:edges.add((key,f'{group}:off#{p}'))
    for resource in payload['resources']:
        lanes=[(0,i,None) for i in range(resource['capacity'])]
        heapq.heapify(lanes)
        for row in sorted(rows.values(),key=lambda r:(r['start'],r['key'])):
            q=acts[row['activity_id']]['resource_reqs'].get(resource['id'],0)
            for _ in range(q):
                end,lane,previous=heapq.heappop(lanes)
                if end>row['start']:raise AssertionError('Oracle schedule resource conflict')
                if previous:edges.add((previous,row['key']))
                heapq.heappush(lanes,(row['end'],lane,row['key']))
    parents={k:set() for k in rows};children={k:set() for k in rows}
    for a,b in edges:
        parents[b].add(a);children[a].add(b)
    degrees={k:len(v) for k,v in parents.items()}
    ready=[k for k,v in degrees.items() if not v];heapq.heapify(ready)
    shifted={};ends={}
    while ready:
        key=heapq.heappop(ready)
        start=max([data[6].get(key,0),*[ends[p] for p in parents[key]]])
        shifted[key]=start;ends[key]=start+acts[rows[key]['activity_id']]['duration']
        for child in children[key]:
            degrees[child]-=1
            if degrees[child]==0:heapq.heappush(ready,child)
    if len(shifted)!=len(rows):raise AssertionError('Oracle order cycle')
    simulator=PathSimulator(scenario_from_dict(payload));state=simulator.initial_state()
    pending=dict(shifted)
    while pending or state.running:
        if simulator.is_goal(state):break
        keys=sorted((k for k,t in pending.items() if t==state.time),key=lambda k:rows[k]['activity_id'])
        for key in keys:
            state=simulator.transition(state,Action('START',rows[key]['activity_id']))
            del pending[key]
        if pending or state.running:
            advance=next((a for a in simulator.enabled_actions(state) if a.kind=='ADVANCE'),None)
            if advance is None:raise AssertionError('Oracle left-shift is not event-driven')
            if pending and min(pending.values())<advance.target_time:
                raise AssertionError('Missed left-shifted start')
            state=simulator.transition(state,advance)
    actions=[to_primitive(a) for a in state.actions]
    checked=candidate_from_actions(payload,actions,algorithm='EXACT_REFERENCE')
    return {'makespan':checked.metrics.makespan,'actions':actions,'execution_count':len(checked.executions)}


def run_case(instance, seconds, output=None, prior=None):
    args=SimpleNamespace(instance=instance,seconds=seconds)
    cpu=isolate();verify(ROOT)
    output=Path(output) if output else OUT/f'{args.instance}-t{args.seconds:g}'
    output.mkdir(parents=True,exist_ok=True)
    item=read_json(ROOT/'instances'/f'{args.instance}.json')
    if prior and prior.get('certificate'):
        import copy, math
        item=copy.deepcopy(item)
        item['answer']['witness_makespan']=min(item['answer']['witness_makespan'],prior['certificate']['makespan'])
        item['answer']['lower_bound']=max(item['answer']['lower_bound'],math.ceil(prior['best_bound']-1e-7))
    started=time.perf_counter();model,data=build(item);build_seconds=time.perf_counter()-started
    snapshots=[]
    first_schedule=None
    class Observer(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            nonlocal first_schedule
            record={'elapsed_total_seconds':time.perf_counter()-started,'solver_seconds':self.wall_time,
                    'objective':self.objective_value,'bound':self.best_objective_bound}
            if first_schedule is None:
                first_schedule=extract(self,data)
            snapshots.append(record)
            if len(snapshots)==1 or len(snapshots)%25==0:
                print('INCUMBENT',args.instance,record['objective'],record['bound'],round(record['elapsed_total_seconds'],3),flush=True)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=args.seconds
    solver.parameters.random_seed=11
    status=solver.solve(model,Observer());search_seconds=time.perf_counter()-started
    result={'instance_id':args.instance,'cpu':cpu,'budget_seconds':args.seconds,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':solver.status_name(status),'build_seconds':build_seconds,'search_including_build_seconds':search_seconds,
            'solver_seconds':solver.wall_time,'best_bound':solver.best_objective_bound,'branches':solver.num_branches,
            'conflicts':solver.num_conflicts,'scope':'exact bounded business256 dependency/mode/state-lock/resource/restoration model; tested engines unchanged',
            'input_scenario_hash':item['metadata']['scenario_hash'],'formal_batch_untouched':True}
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        plan=certificate(item,extract(solver,data),data)
        first_plan=certificate(item,first_schedule,data)
        result.update(objective=solver.objective_value,certificate=plan,
                      first_candidate_discovery_seconds=snapshots[0]['elapsed_total_seconds'],
                      first_candidate_validated_makespan=first_plan['makespan'],
                      proven_optimal=status==cp_model.OPTIMAL and plan['makespan']==round(solver.best_objective_bound))
        if status==cp_model.OPTIMAL and not result['proven_optimal']:raise AssertionError('Optimum certificate mismatch')
        result['first_final_objective_seconds']=next(x['elapsed_total_seconds'] for x in snapshots if x['objective']==solver.objective_value)
    result['total_including_validation_seconds']=time.perf_counter()-started
    write_json(output/'incumbents.json',snapshots);write_json(output/'result.json',result)
    verify(ROOT)
    print({k:v for k,v in result.items() if k!='certificate'},flush=True)
    return result


def fix_witness(item, model, data):
    """For completeness checks only: fix all oracle choices to a known witness."""
    checked=candidate_from_actions(item['scenario'],item['answer']['actions'])
    executions=list(checked.executions)
    restore=item['metadata']['condition'] in ('T2','T3')
    bstart={e.activity_id.split(':')[0]:e.start_time for e in executions if e.activity_id.endswith(':install_b')}
    mapped={}
    for e in executions:
        aid=e.activity_id
        if aid.startswith('p'):
            key=aid+f'#{e.ordinal-1}'
        elif restore and aid.endswith((':cal_a',':test_a')) and e.end_time<=bstart[aid.split(':')[0]]:
            key=aid+'#pre'
        else:key=aid
        if key in mapped:raise AssertionError('Witness exceeds exact occurrence reduction')
        mapped[key]=e
    for key,start in data[0].items():
        if key in mapped:
            model.add(start==mapped[key].start_time)
            model.add(data[1][key]==mapped[key].end_time)
    for (g,p),active in data[2].items():
        model.add(active==int(f'p{g:02d}:on#{p}' in mapped))
    for aid,present in data[7].items():
        model.add(present==int(aid+'#pre' in mapped))
    for (key,p),choice in data[3].items():
        expected=False
        if key in mapped:
            e=mapped[key]
            mode=next(x['state_id'] for x in next(a for a in item['scenario']['activities'] if a['id']==e.activity_id)['preconditions'] if x['state_id'].startswith('p'))
            group,state=mode.split(':')
            count=sum(x.end_time<=e.start_time for x in executions if x.activity_id==group+(':on' if state=='on' else ':ground'))
            phase=count-1 if state=='on' else count
            expected=p==phase
        model.add(choice==int(expected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance',default='b256-w32-d2-T0-g101')
    parser.add_argument('--seconds',type=float,default=90)
    args=parser.parse_args()
    run_case(args.instance,args.seconds)


if __name__=='__main__':main()
