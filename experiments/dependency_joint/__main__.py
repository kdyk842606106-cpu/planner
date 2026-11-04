"""Freeze and supervise add-only dependency experiments without solver changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from experiments.window_joint.__main__ import (
    DEFAULT as PREVIOUS, files as previous_files, verify as verify_previous,
    unlimited, fixed_witness_check, write_json, now, append)
from experiments.dependency_joint.case import make_case
from experiments.business256.observe import cpu_seconds
from planner_experiment.benchmark_suite.io import digest, read_json
from planner_experiment.benchmark_suite.runner import batch_lock
from planner_experiment.benchmark_suite.runtime import process_memory

DEFAULT = Path('D:/planner/runs/mainline-joint-cpsat-dense-prerequisites-20261103')


def files():
    return [*previous_files(),
            *[(f'dependency_joint/{p.name}', p) for p in sorted(Path(__file__).parent.glob('*')) if p.is_file()],
            ('test_dependency_joint.py', Path('D:/planner/tests/test_dependency_joint.py'))]


def hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files()}


def prepare(root):
    verify_previous(PREVIOUS)
    if root.exists():
        raise ValueError('Refuse to overwrite existing experiment directory')
    previous_result = read_json(PREVIOUS/'attempt/result.json')
    assert previous_result['proven_optimal'] and previous_result['best_makespan'] == 1451
    baseline = read_json(PREVIOUS/'input.json')
    payload, metrics, evidence = make_case(baseline, read_json(PREVIOUS/'attempt/best.json')['certificate'])
    evidence['compiler_feasibility'] = fixed_witness_check(payload, evidence['witness_raw_rows'])
    root.mkdir(parents=True)
    config = {'engine':'JOINT_CP_SAT_WINDOWS', 'solver_seed':11, 'workers':1, 'cpu':4,
              'time_limit_seconds':None, 'parent_deadline_seconds':None,
              'objective':'makespan only', 'proof_scope':'finite input execution bounds',
              'input_policy':'raw input.json only; no hints, witness, mandatory map or supplied bounds',
              'prior_study':str(PREVIOUS), 'changed_factor':'added activity prerequisites only',
              'generation_policy':evidence['generation_policy'],
              'stop_policy':'optimality/infeasibility proof, error or external interruption',
              'timing':'parse/build/search/certificate/persistence; imports excluded; external startup measured separately',
              'memory':'actual worker PID OS peak and external samples including imports',
              'states':'CP-SAT branches/conflicts, not explicit Simulator states'}
    for name, value in [('input',payload),('config',config),('complexity',metrics),('private-evidence',evidence),
                        ('baseline-result',previous_result)]:
        write_json(root/f'{name}.json', value)
    write_json(root/'manifest.json', {'created_at':now(), 'sources':hashes(),
               'input_hash':digest(payload), 'config_hash':digest(config),
               'complexity_hash':digest(metrics), 'private_evidence_hash':digest(evidence), 'python':sys.version})
    with zipfile.ZipFile(root/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in files():
            archive.write(path,name)
    print(json.dumps({'prepared':str(root),'edges_added':metrics['added_process_edges'],
                      'skeleton':metrics['skeleton'],'compiler_feasibility':evidence['compiler_feasibility']},ensure_ascii=False),flush=True)


def verify(root):
    manifest = read_json(root/'manifest.json')
    assert manifest['sources'] == hashes(), 'Frozen source changed'
    for name, key in [('input','input_hash'),('config','config_hash'),('complexity','complexity_hash'),('private-evidence','private_evidence_hash')]:
        assert digest(read_json(root/f'{name}.json')) == manifest[key], name
    verify_previous(PREVIOUS)


def run(root):
    verify(root)
    with batch_lock(root):
        if (root/'attempt/ready.json').exists():
            raise ValueError('Do not restart an existing attempt')
        with (root/'worker-stdout.log').open('w',encoding='utf-8') as out, (root/'worker-stderr.log').open('w',encoding='utf-8') as err:
            launched = time.perf_counter()
            process = subprocess.Popen([sys.executable,'-u','-m','experiments.dependency_joint','worker','--root',str(root)],
                                       cwd='D:/planner',stdout=out,stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            write_json(root/'supervisor.json', {'pid':os.getpid(),'child_launcher_pid':process.pid,'started_at':now(),'deadline':None})
            ready = None
            while process.poll() is None:
                if ready is None and (root/'attempt/ready.json').exists():
                    ready = read_json(root/'attempt/ready.json')
                if ready:
                    cpu = cpu_seconds(ready['pid'])
                    append(root/'resources.jsonl',{'utc':now(),'worker_pid':ready['pid'],
                        'wall_seconds_including_worker_startup':time.perf_counter()-launched,
                        'cpu_seconds':None if cpu is None else cpu-ready['cpu_start'], 'memory':process_memory(ready['pid'])})
                time.sleep(1)
            write_json(root/'supervisor-result.json',{'exit_code':process.returncode,'finished_at':now(),'full_call_seconds':time.perf_counter()-launched})
            if process.returncode:
                raise RuntimeError('Worker failed; see worker-stderr.log')
        print(json.dumps(read_json(root/'attempt/result.json'),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','verify','run','worker'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.root)
    elif args.command=='verify': verify(args.root); print('VERIFIED')
    elif args.command=='run': run(args.root)
    else: unlimited(args.root/'input.json',args.root/'attempt',cpu=4)


if __name__=='__main__':
    main()
