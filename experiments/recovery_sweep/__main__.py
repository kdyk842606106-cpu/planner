"""Sequential fresh-process R2/R3 runs with immutable inputs and 3 seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from experiments.cycle_joint.__main__ import DEFAULT as PREVIOUS, files as previous_files, verify as verify_previous
from experiments.window_joint.__main__ import write_json, fixed_witness_check, now, append
from experiments.business256.observe import cpu_seconds
from experiments.recovery_sweep.case import make_case, audit_cycles
from experiments.recovery_sweep.runtime import unlimited
from planner_experiment.benchmark_suite.io import read_json, digest
from planner_experiment.benchmark_suite.runner import batch_lock
from planner_experiment.benchmark_suite.runtime import process_memory

DEFAULT=Path('D:/planner/runs/mainline-joint-cpsat-recovery-r2-r3-20261103')


def files():
    return [*previous_files(),
            *[(f'recovery_sweep/{p.name}',p) for p in sorted(Path(__file__).parent.glob('*')) if p.is_file()],
            ('test_recovery_sweep.py',Path('D:/planner/tests/test_recovery_sweep.py'))]


def hashes():
    return {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files()}


def prepare(root):
    verify_previous(PREVIOUS)
    if root.exists():raise ValueError('Refuse to overwrite existing batch')
    baseline=read_json(PREVIOUS/'input.json')
    previous_result=read_json(PREVIOUS/'attempt/result.json')
    assert previous_result['proven_optimal'] and previous_result['best_makespan']==1460
    basecert=read_json(PREVIOUS/'attempt/best.json')['certificate']
    prepared={}
    for count in (2,3):
        p,m,e=make_case(baseline,basecert,count)
        if count==2:
            e['compiler_feasibility']=fixed_witness_check(p,e['witness_raw_rows'])
        else:
            # Private diagnostic only: preserve exact R3 formal windows.
            diagnostic=json.loads(json.dumps(p))
            for w in diagnostic['time_windows']:
                if w['activity_id']=='M180':w['latest_finish']=1528
            e['relaxed_diagnostic']['compiler_feasibility']=fixed_witness_check(diagnostic,e['relaxed_diagnostic']['witness_raw_rows'])
        prepared[f'R{count}']=(p,m,e)
    root.mkdir(parents=True)
    inputs={}
    for name,(p,m,e) in prepared.items():
        for filename,value in [('input',p),('complexity',m),('private-evidence',e)]:
            relative=f'cases/{name}/{filename}.json'
            write_json(root/relative,value)
            inputs[relative]=digest(value)
    jobs=[{'id':f'{case}-s{seed}','case':case,'seed':seed}
          for seed in (11,23,37) for case in ('R2','R3')]
    config={'jobs':jobs,'workers':1,'cpu':4,'time_limit_seconds':None,'parent_deadline_seconds':None,
            'objective':'makespan only','baseline':str(PREVIOUS),
            'parameters':'Frozen CP-SAT/compiler/validator/search loop; only the planned run seed varies',
            'formal_worker_input':'raw cases/<case>/input.json only; no proof, witness or hints',
            'protocol':'6 fresh processes, sequential, fixed CPU; R2 feasible-performance and R3 infeasibility-boundary results reported separately',
            'recovery_scope':'R2 after M100/M140; R3 after M100/M140/M160, cannot merge episodes',
            'resume_policy':'skip successfully completed runs; never automatically restart an existing incomplete attempt',
            'baseline_repetitions':'R0/R1 existing seed 11 only; not three-seed matched baselines',
            'next_stage':'R4/R5 deferred until R2/R3 findings are available'}
    write_json(root/'config.json',config)
    write_json(root/'baseline-result.json',previous_result)
    write_json(root/'manifest.json',{'created_at':now(),'sources':hashes(),'data_hashes':inputs,
                                   'config_hash':digest(config),'python':sys.version})
    with zipfile.ZipFile(root/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in files():archive.write(path,name)
    write_json(root/'batch-state.json',{'status':'PREPARED','completed_runs':0,'total_runs':len(jobs),'updated_at':now()})
    print(json.dumps({'prepared':str(root),'runs':len(jobs),'answers':{k:e['answer'] for k,(_,_,e) in prepared.items()}},ensure_ascii=False),flush=True)


def verify(root):
    manifest=read_json(root/'manifest.json')
    assert manifest['sources']==hashes(),'Frozen sources changed'
    assert manifest['config_hash']==digest(read_json(root/'config.json')),'Configuration changed'
    for relative,expected in manifest['data_hashes'].items():
        assert digest(read_json(root/relative))==expected,relative
    verify_previous(PREVIOUS)


def summarize(root):
    jobs=read_json(root/'config.json')['jobs']
    rows=[]
    for job in jobs:
        directory=root/'attempts'/job['id']
        path=directory/'result.json'
        if not path.exists():path=directory/'state.json'
        row={**job,'status':'NOT_RUN','finished':False}
        if path.exists():
            state=read_json(path)
            row.update({k:state.get(k) for k in ('status','solver_status','best_makespan','lower_bound',
                'proven_optimal','first_solution_seconds','first_optimal_seconds','elapsed_seconds','cpu_seconds',
                'branches','conflicts','search_stats_sample_seconds','memory','updated_at')})
            row['finished']=(directory/'result.json').exists()
        rows.append(row)
    summary={'updated_at':now(),'runs':rows,'completed_runs':sum(r['finished'] for r in rows),'total_runs':len(jobs),
             'interpretation':'R3 infeasible by unchanged final window; no-solution is not algorithm failure. Each condition is one model with three run seeds. Branches/conflicts are not explicit Simulator state counts.'}
    write_json(root/'summary.json',summary)
    return summary


def run_one(root,job):
    directory=root/'attempts'/job['id']
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        supervisor=read_json(directory/'supervisor-result.json')
        assert supervisor['exit_code']==0
        return
    if (directory/'ready.json').exists() or (directory/'supervisor.json').exists():
        raise ValueError('Existing incomplete attempt: inspect it; never silently restart '+job['id'])
    with (directory/'worker-stdout.log').open('w',encoding='utf8') as out, (directory/'worker-stderr.log').open('w',encoding='utf8') as err:
        launched=time.perf_counter()
        process=subprocess.Popen([sys.executable,'-u','-m','experiments.recovery_sweep','worker',
                                  '--root',str(root),'--case',job['case'],'--seed',str(job['seed'])],
                                 cwd='D:/planner',stdout=out,stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write_json(directory/'supervisor.json',{'pid':os.getpid(),'child_launcher_pid':process.pid,'started_at':now(),'deadline':None})
        ready=None
        last_summary=0
        while process.poll() is None:
            if ready is None and (directory/'ready.json').exists():ready=read_json(directory/'ready.json')
            if ready:
                cpu=cpu_seconds(ready['pid'])
                append(directory/'resources.jsonl',{'utc':now(),'worker_pid':ready['pid'],
                       'wall_seconds_including_worker_startup':time.perf_counter()-launched,
                       'cpu_seconds':None if cpu is None else cpu-ready['cpu_start'],'memory':process_memory(ready['pid'])})
            if time.perf_counter()-last_summary>=5:
                summarize(root);last_summary=time.perf_counter()
            time.sleep(1)
        write_json(directory/'supervisor-result.json',{'exit_code':process.returncode,'finished_at':now(),
                                                      'full_call_seconds':time.perf_counter()-launched})
        if process.returncode:raise RuntimeError('Worker failed: '+job['id'])
    result=read_json(directory/'result.json')
    if result['status'] in ('ERROR','ENCODING_ERROR'):raise RuntimeError('Invalid worker result: '+job['id'])
    if (directory/'best.json').exists():
        cert=read_json(directory/'best.json')['certificate']
        audit=audit_cycles(read_json(root/'cases'/job['case']/'input.json'),cert,int(job['case'][1:]))
        write_json(directory/'cycle-audit.json',audit)
    if job['case']=='R2':
        assert result['proven_optimal'] and result['best_makespan']==1494,result
    else:
        assert result['status']=='PROVEN_INFEASIBLE' and result['best_makespan'] is None,result


def run(root):
    verify(root)
    with batch_lock(root):
        config=read_json(root/'config.json')
        try:
            for job in config['jobs']:
                summary=summarize(root)
                write_json(root/'batch-state.json',{'status':'RUNNING','active_run':job['id'],
                           'completed_runs':summary['completed_runs'],'total_runs':6,'updated_at':now()})
                run_one(root,job)
            summary=summarize(root)
            write_json(root/'batch-state.json',{'status':'COMPLETE','active_run':None,
                       'completed_runs':summary['completed_runs'],'total_runs':6,'updated_at':now()})
        except BaseException as error:
            write_json(root/'batch-state.json',{'status':'ERROR','error':str(error),'updated_at':now()})
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','verify','run','worker','summarize'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--case',choices=['R2','R3'])
    parser.add_argument('--seed',type=int,choices=[11,23,37])
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root)
    elif args.command=='verify':verify(args.root);print('VERIFIED')
    elif args.command=='run':run(args.root)
    elif args.command=='summarize':print(json.dumps(summarize(args.root),ensure_ascii=False))
    else:
        if args.case is None or args.seed is None:parser.error('worker requires --case and --seed')
        # The solver reads only raw input. Evidence/answer files are not read.
        unlimited(args.root/'cases'/args.case/'input.json',args.root/'attempts'/f'{args.case}-s{args.seed}',cpu=4,seed=args.seed)


if __name__=='__main__':main()
