"""Gate, freeze, and run the isolated fourth-engine experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile

from planner_experiment.benchmark_suite.io import digest, read_json, write_json, read_events
from planner_experiment.benchmark_suite.runtime import process_memory
from planner_experiment.benchmark_suite.runner import source_files, batch_lock
from planner_experiment.benchmark_suite.report import csv_file
from experiments.business256.observe import cpu_seconds
from experiments.joint_cpsat.cases import small_suite, large_case, serial_witness
from experiments.joint_cpsat.solver import solve

DEFAULT = Path('D:/planner/runs/joint-cpsat-validation-20261102')
SOLVER_ROOT = Path('D:/Solver_demo_project_planner_exploration')
OLD_ROOT = Path('D:/planner/runs/business256-20261102')
OLD_ORACLE = Path('D:/planner/runs/business256-optimum-all-20261102')


def now(): return datetime.now(timezone.utc).isoformat()


def isolate(cpu=2):
    if os.name != 'nt':
        os.sched_setaffinity(0,{cpu})
        return cpu
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(),1<<cpu):
        raise OSError('Unable to pin separate CPU')
    k.SetPriorityClass(k.GetCurrentProcess(),0x00004000)
    return cpu


def files():
    return [*source_files(SOLVER_ROOT),
            *[(f'joint_cpsat/{p.name}',p) for p in sorted(Path(__file__).parent.glob('*.py'))],
            ('test_joint_cpsat.py',Path('D:/planner/tests/test_joint_cpsat.py')),
            ('joint_cpsat/README.md',Path(__file__).with_name('README.md')),
            ('business256_observe.py',Path('D:/planner/experiments/business256/observe.py'))]


def hashes(): return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files()}


def guard_originals():
    from experiments.business256.__main__ import verify
    verify(OLD_ROOT)
    frozen=read_json(OLD_ORACLE/'manifest.json')
    for name,expected in frozen['sources'].items():
        if hashlib.sha256((Path('D:/planner/experiments')/name).read_bytes()).hexdigest()!=expected:
            raise AssertionError('Existing oracle changed: '+name)


def prepare(root):
    isolate()
    guard_originals()
    if (root/'manifest.json').exists():
        raise ValueError('Already frozen; use run or another output directory')
    root.mkdir(parents=True,exist_ok=True)
    small=[]
    for payload,meta in small_suite():
        iid=payload['id']
        write_json(root/'inputs'/f'{iid}.json',payload)
        write_json(root/'private-evidence'/f'{iid}.json',{'generation':meta,'witness':serial_witness(payload)})
        small.append(iid)
    large=[]
    for seed in (101,202):
        for arrival in range(3):
            for tight in (False,True):
                for length in ('short','long'):
                    payload,meta=large_case(arrival,tight,length,seed)
                    iid=payload['id']
                    write_json(root/'inputs'/f'{iid}.json',payload)
                    write_json(root/'private-evidence'/f'{iid}.json',{'generation':meta,'witness':serial_witness(payload)})
                    large.append(iid)
    random.Random(81173).shuffle(large)
    config={'version':1,'engine':'JOINT_CP_SAT','cpu':2,'workers':1,'solver_seed':11,
            'stages_seconds':[30,300],'large_batch_cap_seconds':3*3600,'small_budget_seconds':30,
            'exact_per_small_seconds':90,'exact_state_limit':200000,
            'input_policy':'raw scenario only; no generator metadata, required-task list, witness or oracle bound',
            'objective':'makespan only; equal-makespan extra operations are not disallowed',
            'support':'parallel positive-duration Boolean-state actions, arbitrary activity deletions, bounded repeats, resources, release-only events, positive/negative goals',
            'unsupported':'state-changing external events, serial mode, zero-duration activities',
            'timing':'parsing/build/search/synchronous validation included; imports and OS startup separate',
            'memory':'OS peak working set including imports; 0.25-second external sampling',
            'stop_policy':'only pre-deadline validated candidates count; 300s reruns only unproved models',
            'proof_scope':'input execution bounds, not unbounded planning'}
    write_json(root/'config.json',config)
    manifest={'created_at_utc':now(),'small_ids':small,'large_ids':large,'sources':hashes(),
              'input_hashes':{iid:digest(read_json(root/'inputs'/f'{iid}.json')) for iid in small+large},
              'config_hash':digest(config),'private_evidence_hashes':{iid:digest(read_json(root/'private-evidence'/f'{iid}.json')) for iid in small+large},
              'python':sys.version,'executable':sys.executable}
    write_json(root/'manifest.json',manifest)
    with zipfile.ZipFile(root/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in files(): z.write(path,name)
    print('PREPARED',len(small),'small',len(large),'large',flush=True)
    return manifest


def verify(root):
    m=read_json(root/'manifest.json')
    if hashes()!=m['sources']: raise AssertionError('Frozen source changed')
    if digest(read_json(root/'config.json'))!=m['config_hash']: raise AssertionError('Config changed')
    for iid,expected in m['input_hashes'].items():
        if digest(read_json(root/'inputs'/f'{iid}.json'))!=expected: raise AssertionError('Input changed: '+iid)
    guard_originals()
    return m


def worker(input_path,directory,seconds):
    isolate()
    directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();cpu_start=time.process_time()
    write_json(directory/'ready.json',{'pid':os.getpid(),'started_perf':started,'cpu_start':cpu_start,'utc':now()})
    def accept(record):
        with (directory/'candidates.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    try:
        payload=read_json(input_path)
        result=solve(payload,seconds,on_candidate=accept,started=started,
                     on_model=lambda info:write_json(directory/'model.json',info))
    except TimeoutError as exc:
        result={'status':'UNKNOWN','stop_reason':str(exc),'best':None,'proven_optimal':False}
    except Exception as exc:
        result={'status':'ERROR','error':repr(exc),'traceback':traceback.format_exc(),'best':None,'proven_optimal':False}
    result.update(cpu_seconds=time.process_time()-cpu_start,elapsed_seconds=time.perf_counter()-started,
                  memory=process_memory(os.getpid()))
    write_json(directory/'worker-result.json',result)


def supervise(root,iid,seconds,directory):
    directory.mkdir(parents=True,exist_ok=True)
    # Each attempt is immutable. An interrupted attempt is archived on resume.
    launched=time.perf_counter()
    with (directory/'stdout.log').open('w',encoding='utf-8') as out,(directory/'stderr.log').open('w',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','-m','experiments.joint_cpsat','worker',
                               '--input',str(root/'inputs'/f'{iid}.json'),'--directory',str(directory),
                               '--seconds',str(seconds)],stdout=out,stderr=err,cwd='D:/planner',
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        ready=None;peak=0;last_cpu=None;killed=False
        with (directory/'resources.jsonl').open('w',encoding='utf-8') as log:
            while proc.poll() is None:
                if ready is None and (directory/'ready.json').exists(): ready=read_json(directory/'ready.json')
                # Windows venv python.exe is a launcher. Its PID is not the
                # solver PID; the actual child identifies itself after imports.
                observed_pid=ready['pid'] if ready else proc.pid
                memory=process_memory(observed_pid)
                cpu=cpu_seconds(observed_pid)
                if memory:
                    peak=max(peak,memory['peak_working_set_bytes'])
                    log.write(json.dumps({'wall_seconds':time.perf_counter()-launched,'sample_pid':observed_pid,
                                          'solver_process':bool(ready),'cpu_seconds_total':cpu,**memory})+'\n');log.flush()
                if cpu is not None and ready: last_cpu=cpu
                deadline=(ready['started_perf']+seconds+.25) if ready else launched+30
                if time.perf_counter()>=deadline:
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW,timeout=10)
                    else:
                        proc.kill()
                    proc.wait(timeout=10);killed=True;break
                time.sleep(.25)
        proc.wait()
    path=directory/'worker-result.json'
    result=read_json(path) if path.exists() else {'status':'UNKNOWN' if killed else 'ERROR','proven_optimal':False}
    if ready is None and (directory/'ready.json').exists(): ready=read_json(directory/'ready.json')
    # A short solve can finish between samples. The worker's OS high-water
    # counter is authoritative and also includes its imported modules.
    peak=max(peak,(result.get('memory') or {}).get('peak_working_set_bytes',0))
    if (directory/'model.json').exists():
        for key,value in read_json(directory/'model.json').items(): result.setdefault(key,value)
    candidates=[x for x in read_events(directory/'candidates.jsonl') if x['within_budget'] and x['validated_seconds']<=seconds]
    if candidates:
        result['best']=min(candidates,key=lambda x:x['certificate']['makespan'])
        result['first_solution_seconds']=candidates[0]['validated_seconds']
        if result['status']=='UNKNOWN': result['status']='FEASIBLE'
    result.update(instance_id=iid,budget_seconds=seconds,recorded_at_utc=now(),process_exit_code=proc.returncode,
                  hard_killed=killed,call_wall_seconds=time.perf_counter()-launched,
                  solver_pid=ready['pid'] if ready else None,launcher_pid=proc.pid,
                  startup_seconds=ready['started_perf']-launched if ready else None,
                  accepted_candidates=len(candidates),peak_working_set_bytes=peak or None)
    if 'cpu_seconds' not in result:
        result['cpu_seconds']=max(0,last_cpu-ready['cpu_start']) if last_cpu is not None and ready else None
    result.setdefault('stop_reason','PARENT_DEADLINE' if killed else result.get('solver_status',result['status']))
    write_json(directory/'result.json',result)
    return result


def audit(root):
    from planner_experiment.benchmark_suite.oracle import exact_answer
    isolate();manifest=verify(root)
    rows=[]
    for iid in manifest['small_ids']:
        payload=read_json(root/'inputs'/f'{iid}.json')
        directory=root/'small'/iid
        directory.mkdir(parents=True,exist_ok=True)
        exact_path=directory/'exact.json'
        exact=read_json(exact_path) if exact_path.exists() else exact_answer(payload,90,max_states=200000)
        write_json(exact_path,exact)
        run_path=directory/'t30'/'result.json'
        result=read_json(run_path) if run_path.exists() else supervise(root,iid,30,directory/'t30')
        value=result.get('best',{}).get('certificate',{}).get('makespan') if result.get('best') else None
        okay=exact['status']=='PROVEN_OPTIMAL' and result.get('proven_optimal') and value==exact['optimal_makespan'] and not result.get('rejections')
        row={'instance_id':iid,'passed':okay,'optimal_makespan':value,
             'exact_status':exact['status'],'exact_seconds':exact['elapsed_seconds'],
             'cp_status':result['status'],'cp_seconds':result.get('elapsed_seconds'),
             'executions':Counter(x['activity_id'] for x in result['best']['certificate']['executions']) if result.get('best') else {}}
        rows.append(row)
        write_json(root/'gate.json',{'status':'RUNNING','checks':rows,'sources':manifest['sources']})
        print('GATE',iid,okay,value,flush=True)
        if not okay:
            write_json(root/'gate.json',{'status':'FAILED','checks':rows,'sources':manifest['sources']})
            raise AssertionError('Small-model completeness/optimality gate failed: '+iid)
    indexed={x['instance_id']:x for x in rows}
    switches={}
    for factor in ('arrival','downstream','pretest'):
        left=indexed[f'small-{factor}-a']['executions'].get('test',0)
        right=indexed[f'small-{factor}-b']['executions'].get('test',0)
        switches[factor]={'a_test_count':left,'b_test_count':right,'passed':(left,right)==(1,2)}
    passed=all(x['passed'] for x in rows) and all(x['passed'] for x in switches.values())
    gate={'status':'PASSED' if passed else 'FAILED','checks':rows,'path_switches':switches,
          'sources':manifest['sources'],'completed_at_utc':now(),
          'limits':'Independent exhaustive search shares Simulator semantics; hand proofs and lock/resource tests supplement it'}
    write_json(root/'gate.json',gate)
    if not passed: raise AssertionError('Expected path changes were not observed')
    guard_originals()
    return gate


def report(root):
    m=read_json(root/'manifest.json');rows=[]
    for iid in m['large_ids']:
        meta=read_json(root/'private-evidence'/f'{iid}.json')['generation']
        for seconds in (30,300):
            p=root/'attempts'/iid/f't{seconds}'/'result.json'
            r=read_json(p) if p.exists() else {}
            cert=(r.get('best') or {}).get('certificate',{})
            executions=cert.get('executions',[])
            installs={x['activity_id'].split(':')[0]:x['start_time'] for x in executions if x['activity_id'].endswith(':install')}
            early={x['activity_id'].split(':')[0] for x in executions if x['activity_id'].endswith(':test')
                   and x['end_time']<=installs.get(x['activity_id'].split(':')[0],-1)}
            rows.append({'instance_id':iid,'budget_seconds':seconds,**meta,
                         'status':r.get('status','NOT_RUN'),'makespan':cert.get('makespan'),
                         'lower_bound':r.get('lower_bound'),'proven_optimal':r.get('proven_optimal',False),
                         'first_solution_seconds':r.get('first_solution_seconds'),'cpu_seconds':r.get('cpu_seconds'),
                         'peak_memory_mib':r.get('peak_working_set_bytes')/2**20 if r.get('peak_working_set_bytes') else None,
                         'build_seconds':r.get('build_seconds'),'selected_definitions':cert.get('selected_definitions'),
                         'execution_count':cert.get('execution_count'),'branches':r.get('branches'),'conflicts':r.get('conflicts'),
                         'modules_tested_before_install':len(early) if cert else None,
                         'test_executions':sum(x['activity_id'].endswith(':test') for x in executions) if cert else None,
                         'off_executions':sum(x['activity_id'].endswith(':off') for x in executions) if cert else None,
                         'model_stats':r.get('model_stats'),'stop_reason':r.get('stop_reason')})
    state=read_json(root/'state.json') if (root/'state.json').exists() else {'status':'NOT_STARTED'}
    by_id={iid:[x for x in rows if x['instance_id']==iid] for iid in m['large_ids']}
    summary={'state':state,'models':24,'completed_attempts':sum(x['status']!='NOT_RUN' for x in rows),
             'feasible_models':sum(any(x['makespan'] is not None for x in xs) for xs in by_id.values()),
             'proven_models':sum(any(x['proven_optimal'] for x in xs) for xs in by_id.values()),'rows':rows,
             'gate':read_json(root/'gate.json') if (root/'gate.json').exists() else None}
    for iid,xs in by_id.items():
        if xs[0]['proven_optimal'] and xs[1]['status']=='NOT_RUN': xs[1]['status']='SKIPPED_ALREADY_PROVEN'
    write_json(root/'summary.json',summary);csv_file(root/'results.csv',rows)
    pairs=[]
    # Pair only identical seed/budget and all other experimental factors.
    for factor,levels in [('arrival_level',(0,1,2)),('resource_tight',(False,True)),('downstream_length',('short','long'))]:
        others=[x for x in ('generator_seed','budget_seconds','arrival_level','resource_tight','downstream_length') if x!=factor]
        grouped={}
        for row in rows:
            grouped.setdefault(tuple(row[k] for k in others),{})[row[factor]]=row
        for key,group in grouped.items():
            for low,high in zip(levels,levels[1:]):
                a,b=group[low],group[high]
                complete=a['makespan'] is not None and b['makespan'] is not None
                pairs.append({'factor':factor,**dict(zip(others,key)),'from':low,'to':high,
                              'from_status':a['status'],'to_status':b['status'],'both_feasible':complete,
                              'makespan_change':b['makespan']-a['makespan'] if complete else None,
                              'early_module_change':b['modules_tested_before_install']-a['modules_tested_before_install'] if complete else None,
                              'both_proven':a['proven_optimal'] and b['proven_optimal']})
    csv_file(root/'paired-results.csv',pairs)
    lines=['# 第四种方案：从原始候选活动联合规划与排程','',
           f"状态 {state['status']}；完成 {summary['completed_attempts']} 次；有合法解 {summary['feasible_models']}/24；已证明最优 {summary['proven_models']}/24。",'',
           '全部执行实例均为可选；输入不含必要任务、阶段、见证或答案。构建、搜索与同步校验计入预算。CPU 2 单线程，与原批次分开；仍可能共享缓存/内存带宽。',
           '首版明确拒绝改变状态的外部事件；当前使用物料释放事件。最优只针对输入执行次数边界。只优化工期，不以最少操作数破除平局。',
           'CP 分支、冲突和变量数不是 A* 状态数，不作同口径比较。NOT_RUN 不计为失败；已证明实例跳过 300 秒。','',
           '|实例|预算|结果|工期|下界|首解秒|CPU 秒|峰值 MiB|',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append('|'+ '|'.join('—' if x[k] is None else f'{x[k]:.3f}' if isinstance(x[k],float) else str(x[k]) for k in
                                  ('instance_id','budget_seconds','status','makespan','lower_bound','first_solution_seconds','cpu_seconds','peak_memory_mib'))+'|')
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(root):
    with batch_lock(root):
        return _run(root)


def _run(root):
    isolate();m=verify(root)
    gate=read_json(root/'gate.json')
    if gate['status']!='PASSED' or gate['sources']!=m['sources']: raise AssertionError('Small-model gate is not passed')
    config=read_json(root/'config.json')
    previous=read_json(root/'state.json') if (root/'state.json').exists() else {}
    consumed=previous.get('consumed_seconds',0)
    if previous.get('active'): consumed+=previous['active']['seconds']+30
    began=time.perf_counter()
    state={'status':'RUNNING','pid':os.getpid(),'started_at_utc':now(),'cpu':2,
           'consumed_seconds':consumed,'active':None,'batch_cap_seconds':config['large_batch_cap_seconds']}
    def save():
        state['consumed_seconds']=consumed+time.perf_counter()-began
        state['updated_at_utc']=now();write_json(root/'state.json',state)
    save()
    try:
        for seconds in config['stages_seconds']:
            for iid in m['large_ids']:
                verify(root)
                paths=[root/'attempts'/iid/f't{s}'/'result.json' for s in config['stages_seconds']]
                if any(p.exists() and read_json(p).get('proven_optimal') for p in paths): continue
                directory=root/'attempts'/iid/f't{seconds}'
                if (directory/'result.json').exists(): continue
                if state['consumed_seconds']+seconds+30>state['batch_cap_seconds']:
                    state['status']='BATCH_TIME_LIMIT';save();report(root);return
                if directory.exists():
                    archived=directory.with_name(directory.name+'-interrupted-'+str(time.time_ns()))
                    directory.rename(archived)
                state['active']={'instance_id':iid,'seconds':seconds};save()
                result=supervise(root,iid,seconds,directory)
                state['active']=None;save();report(root)
                print('RESULT',iid,seconds,result['status'],(result.get('best') or {}).get('certificate',{}).get('makespan'),flush=True)
                if result['status'] in ('ERROR','ENCODING_ERROR'):
                    raise AssertionError('Stop for investigation: '+iid)
        state['status']='COMPLETE';save();report(root);guard_originals()
    except BaseException as exc:
        state['status']='ERROR';state['error']=repr(exc);save();report(root);raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','audit','run','report','verify','worker'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--input',type=Path);parser.add_argument('--directory',type=Path)
    parser.add_argument('--seconds',type=float,default=30)
    args=parser.parse_args()
    if args.command=='worker': worker(args.input,args.directory,args.seconds)
    else: globals()[args.command](args.root)


if __name__=='__main__': main()
