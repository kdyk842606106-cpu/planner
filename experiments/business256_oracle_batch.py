"""Staged, resumable optimum-reference batch; original benchmark stays live."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import time
import traceback

from ortools.sat.python import cp_model

from experiments.business256_oracle import build, fix_witness, run_case, ROOT
from experiments.pop_closure_readonly_diagnosis import isolate
from experiments.business256.__main__ import verify
from planner_experiment.benchmark_suite.io import read_json,write_json,digest
from planner_experiment.benchmark_suite.report import csv_file

OUT=Path('D:/planner/runs/business256-optimum-all-20261102')
STAGES=(60,300,1800)
CAP=86400


def sources():
    paths=[Path(__file__),Path(__file__).with_name('business256_oracle.py'),Path(__file__).with_name('pop_closure_readonly_diagnosis.py')]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def audit(root):
    cpu=isolate();manifest=verify(ROOT)
    root.mkdir(parents=True,exist_ok=True)
    checks=[]
    for iid in manifest['instance_ids']:
        item=read_json(ROOT/'instances'/f'{iid}.json')
        model,data=build(item)
        fix_witness(item,model,data)
        solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
        solver.parameters.max_time_in_seconds=10
        status=solver.solve(model)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            raise AssertionError(f'Exact-model witness check failed {iid}: {solver.status_name(status)}')
        checks.append({'instance_id':iid,'fixed_original_witness_feasible':True,'status':solver.status_name(status)})
        print('AUDIT',iid,flush=True)
        write_json(root/'audit-progress.json',{'cpu':cpu,'completed':len(checks),'checks':checks})
    freeze={'created_at_utc':datetime.now(timezone.utc).isoformat(),'cpu':cpu,'sources':sources(),
            'formal_source_hashes':manifest['environment']['source_hashes'],
            'instance_hashes':{k:v for k,v in manifest['instance_hashes'].items() if k in manifest['instance_ids']},
            'ids':manifest['instance_ids'],'stages_seconds':list(STAGES),'batch_cap_seconds':CAP,
            'scope':'24 conditions x 2 frozen generator seeds; independent oracle, not three-engine performance',
            'proof_requirements':'CP-SAT integer bound equals validated candidate; state locks and bounded mode/recovery occurrences included',
            'small_hand_proofs':{'single_module_T0':11,'single_module_T2':20},'audit':checks}
    write_json(root/'manifest.json',freeze)
    for name in freeze['sources']:
        (root/('source-'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    return freeze


def report(root):
    freeze=read_json(root/'manifest.json')
    rows=[]
    for iid in freeze['ids']:
        item=read_json(ROOT/'instances'/f'{iid}.json')
        attempts=[]
        for seconds in STAGES:
            p=root/'attempts'/iid/f't{seconds}'/'result.json'
            if p.exists():attempts.append(read_json(p))
        feasible=[x for x in attempts if x.get('certificate')]
        best=min(feasible,key=lambda x:x['certificate']['makespan']) if feasible else None
        proved=next((x for x in attempts if x.get('proven_optimal')),None)
        lb=max([item['answer']['lower_bound'],*[x.get('best_bound',0) for x in attempts]])
        ub=best['certificate']['makespan'] if best else item['answer']['witness_makespan']
        rows.append({'instance_id':iid,'condition':item['metadata']['condition'],'width':item['metadata']['width_target'],
                     'density':item['metadata']['density_target'],'generator_seed':item['metadata']['generator_seed'],
                     'attempts_completed':len(attempts),'status':'PROVEN_OPTIMAL' if proved else 'FEASIBLE_OPTIMUM_UNKNOWN' if best else 'SEARCH_UNKNOWN' if attempts else 'NOT_RUN',
                     'lower_bound':lb,'best_validated_makespan':ub,'relative_bound_gap':(ub-lb)/lb if lb else None,
                     'first_candidate_search_seconds':attempts[0].get('first_candidate_discovery_seconds') if attempts else None,
                     'proof_attempt_seconds':proved['search_including_build_seconds'] if proved else None,
                     'proof_attempt_candidate_first_hit_seconds':proved['first_final_objective_seconds'] if proved else None,
                     'total_compute_including_validation_seconds':sum(x.get('total_including_validation_seconds',0) for x in attempts),
                     'best_attempt_budget_seconds':best['budget_seconds'] if best else None})
    state=read_json(root/'state.json') if (root/'state.json').exists() else {'status':'NOT_STARTED'}
    summary={'state':state,'models':len(rows),'conditions':24,'proven_optimal':sum(r['status']=='PROVEN_OPTIMAL' for r in rows),
             'oracle_feasible':sum(r['status'] in ('PROVEN_OPTIMAL','FEASIBLE_OPTIMUM_UNKNOWN') for r in rows),'rows':rows,
             'caveats':['Reference oracle timings are not measured timings of the three tested engines.',
                        'First-hit time refers to candidate generation; replay validation happens after the solve.',
                        'Reported compute times are observations, not a universally minimal budget.',
                        'Unclosed bounds remain optimum unknown, even after the longest stage.']}
    write_json(root/'summary.json',summary);csv_file(root/'results.csv',rows)
    text=['# 24 组业务模型的独立最优答案验证','',f'状态：{state["status"]}；独立求解得到合法方案 {summary["oracle_feasible"]}/48；已证明最优 {summary["proven_optimal"]}/48。','',
          '这是独立精确参考模型，原三算法及其正式实验保持不变。首次候选时间是搜索发现时间，合法性在该次求解后回放确认；没有上下界闭合就不标记最优。','',
          '|模型|状态|下界|最好合法工期|证明轮次耗时秒|累计计算秒|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        text.append('|'+ '|'.join('—' if r[k] is None else f'{r[k]:.3f}' if isinstance(r[k],float) else str(r[k]) for k in
                                   ('instance_id','status','lower_bound','best_validated_makespan','proof_attempt_seconds','total_compute_including_validation_seconds'))+'|')
    (root/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary


def run(root):
    freeze=read_json(root/'manifest.json')
    if freeze['sources']!=sources():raise ValueError('Oracle source changed after validation/freeze')
    verify(ROOT)
    previous=read_json(root/'state.json') if (root/'state.json').exists() else {}
    consumed=previous.get('consumed_seconds',0)
    if previous.get('status')=='RUNNING' and previous.get('active'):
        consumed+=previous['active']['seconds']+60
    began=time.perf_counter()
    state={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),'consumed_seconds':consumed,
           'cpu':isolate(),'active':None,'batch_cap_seconds':CAP}
    ids=list(freeze['ids']);random.Random(81173).shuffle(ids)
    write_json(root/'state.json',state)
    try:
        for seconds in STAGES:
            for iid in ids:
                attempts=[root/'attempts'/iid/f't{s}'/'result.json' for s in STAGES]
                existing=[read_json(p) for p in attempts if p.exists()]
                if any(r.get('proven_optimal') for r in existing):continue
                directory=root/'attempts'/iid/f't{seconds}'
                if (directory/'result.json').exists():continue
                if consumed+time.perf_counter()-began+seconds+60>CAP:
                    state['status']='BUDGET_STOPPED';return
                prior=min((x for x in existing if x.get('certificate')),key=lambda x:x['certificate']['makespan'],default=None)
                state['active']={'instance_id':iid,'seconds':seconds}
                state['consumed_seconds']=consumed+time.perf_counter()-began
                state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
                write_json(root/'state.json',state)
                run_case(iid,seconds,directory,prior)
                state['active']=None
                state['consumed_seconds']=consumed+time.perf_counter()-began
                write_json(root/'state.json',state)
                report(root)
        state['status']='COMPLETED'
    except BaseException as exc:
        state.update(status='INTERRUPTED',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        state['consumed_seconds']=consumed+time.perf_counter()-began
        state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        write_json(root/'state.json',state);report(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','run','report'))
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    {'audit':audit,'run':run,'report':report}[args.mode](args.output)


if __name__=='__main__':main()
