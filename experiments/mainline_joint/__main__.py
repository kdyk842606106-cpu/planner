"""Freeze and run one unlimited attempt, without changing the joint compiler."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile

from ortools.sat.python import cp_model

from experiments.business256.observe import cpu_seconds
from experiments.joint_cpsat.__main__ import DEFAULT as PREVIOUS, files as previous_files, isolate, verify as verify_previous
from experiments.joint_cpsat.solver import build, certificate, rows_from
from experiments.mainline_joint.case import make_case, witness, describe
from planner_experiment.benchmark_suite.io import digest, read_json
from planner_experiment.benchmark_suite.runner import batch_lock
from planner_experiment.benchmark_suite.runtime import process_memory

DEFAULT = Path('D:/planner/runs/mainline-joint-cpsat-unlimited-20261103')


def write_json(path, value):
    """Atomic persistence, tolerating transient Windows reader sharing locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, item):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        stream.flush()


def files():
    return [*previous_files(),
            *[(f'mainline_joint/{p.name}', p) for p in sorted(Path(__file__).parent.glob('*')) if p.is_file()],
            ('test_mainline_joint.py', Path('D:/planner/tests/test_mainline_joint.py'))]


def hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files()}


def fixed_witness_check(payload, rows):
    """Audit only: all occurrence decisions/times fixed, never formal hints."""
    model, _, slots, stats = build(payload)
    chosen = {(r['activity_id'], r['ordinal']): r for r in rows}
    for slot in slots:
        row = chosen.get((slot.aid, slot.ordinal))
        model.add(slot.present == int(row is not None))
        if row:
            model.add(slot.start == row['start'])
            model.add(slot.end == row['end'])
    model.clear_objective()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 60  # feasibility gate, not the experiment
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise AssertionError('Compiler excludes witness or audit incomplete: ' + solver.status_name(status))
    return {'status': solver.status_name(status), 'seconds': solver.wall_time, 'model_stats': stats}


def prepare(root):
    verify_previous(PREVIOUS)
    if (root / 'manifest.json').exists():
        raise ValueError('Already frozen')
    payload, metadata = make_case()
    metrics = describe(payload, metadata)
    assert len(payload['activities']) == 256
    assert (metrics['skeleton']['depth'], metrics['skeleton']['width']) == (180, 4)
    evidence = {}
    for recover in (False, True):
        key = 'early_test_then_restore' if recover else 'wait_then_test'
        proof = witness(payload, recover=recover)
        replay = certificate(payload, proof['rows'])
        assert replay['makespan'] == proof['makespan']
        proof['compiler_feasibility'] = fixed_witness_check(payload, proof['rows'])
        evidence[key] = proof
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'input.json', payload)
    write_json(root / 'complexity.json', metrics)
    write_json(root / 'private-evidence.json', {'generation': metadata, 'witnesses': evidence})
    config = {'engine': 'JOINT_CP_SAT', 'generator_seed': 101, 'solver_seed': 11, 'workers': 1, 'cpu': 2,
              'time_limit_seconds': None, 'parent_deadline_seconds': None,
              'stop_policy': 'optimality proof / infeasibility proof / encoding error / external interruption',
              'input_policy': 'raw input.json only; no private witness, mandatory map, hint or supplied bound',
              'objective': 'makespan only', 'proof_scope': 'finite input execution bounds',
              'timing': 'input parse, build, search, synchronous certificate validation and persistence; imports excluded',
              'memory': 'actual worker PID external 1-second samples and OS peak including imports',
              'states': 'CP-SAT branches and conflicts, not comparable to enumerated Simulator states',
              'prior_study': str(PREVIOUS)}
    write_json(root / 'config.json', config)
    manifest = {'created_at': now(), 'sources': hashes(), 'input_hash': digest(payload),
                'config_hash': digest(config), 'complexity_hash': digest(metrics),
                'private_evidence_sha256': hashlib.sha256((root / 'private-evidence.json').read_bytes()).hexdigest(),
                'python': sys.version}
    write_json(root / 'manifest.json', manifest)
    with zipfile.ZipFile(root / 'source-snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files():
            archive.write(path, name)
    print(json.dumps({'prepared': str(root), 'complexity': metrics['skeleton'],
                      'witness_makespans': {k: v['makespan'] for k, v in evidence.items()}}, ensure_ascii=False), flush=True)


def verify(root):
    manifest = read_json(root / 'manifest.json')
    assert manifest['sources'] == hashes(), 'Frozen source changed'
    assert manifest['input_hash'] == digest(read_json(root / 'input.json')), 'Input changed'
    assert manifest['config_hash'] == digest(read_json(root / 'config.json')), 'Config changed'
    verify_previous(PREVIOUS)


def unlimited(payload, directory, *, cpu=2):
    """No deadline. Shared frozen compilation/normalization/validation only."""
    isolate(cpu)
    directory.mkdir(parents=True, exist_ok=True)
    started, cpu_start = time.perf_counter(), time.process_time()
    write_json(directory / 'ready.json', {'pid': os.getpid(), 'utc': now(),
                                         'started_perf': started, 'cpu_start': cpu_start})
    lock, stopped = threading.RLock(), threading.Event()
    state = {'status': 'RUNNING', 'phase': 'BUILDING', 'pid': os.getpid(), 'started_at': now(),
             'time_limit_seconds': None, 'candidate_count': 0, 'best_makespan': None, 'lower_bound': None,
             'first_solution_seconds': None, 'first_optimal_seconds': None,
             'validation_seconds': 0.0, 'persistence_seconds': 0.0, 'rejections': [],
             'branches': None, 'conflicts': None, 'search_stats_sample_seconds': None}

    def save():
        with lock:
            state.update(updated_at=now(), elapsed_seconds=time.perf_counter()-started,
                         cpu_seconds=time.process_time()-cpu_start, memory=process_memory(os.getpid()))
            write_json(directory / 'state.json', state)

    def heartbeat():
        while not stopped.wait(5):
            save()

    save()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if isinstance(payload, Path):
            payload = read_json(payload)
        model, scenario, slots, stats = build(payload)
        built = time.perf_counter() - started
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 11
        assert math.isinf(solver.parameters.max_time_in_seconds)
        assert math.isinf(solver.parameters.max_deterministic_time)
        write_json(directory / 'model.json', {'build_seconds': built, 'stats': stats,
                                              'parameters_text': str(solver.parameters),
                                              'wall_time_limit': 'infinity', 'deterministic_time_limit': 'infinity'})
        with lock:
            state.update(phase='SOLVING', build_seconds=built, model_stats=stats)
        save()

        def accept_bound(value):
            bound = max(0, math.ceil(value - scenario.start_time - 1e-7))
            with lock:
                if state['lower_bound'] is None or bound > state['lower_bound']:
                    state['lower_bound'] = bound
                    append(directory / 'bounds.jsonl', {'seconds': time.perf_counter()-started, 'lower_bound': bound})
                best = state['best_makespan']
                if best is not None and bound >= best:
                    if bound > best:
                        state['rejections'].append({'error': 'Bound exceeds replayed incumbent', 'bound': bound, 'best': best})
                    else:
                        state['first_optimal_seconds'] = state['first_optimal_seconds'] or time.perf_counter()-started
                    solver.stop_search()

        solver.best_bound_callback = accept_bound

        class Observe(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                found = time.perf_counter() - started
                try:
                    raw = rows_from(self, slots)
                    cert = certificate(payload, raw)
                    validated = time.perf_counter() - started
                    bound = max(0, math.ceil(self.best_objective_bound - scenario.start_time - 1e-7))
                    counts = Counter(row['activity_id'] for row in cert['executions'])
                    record = {'discovered_seconds': found, 'validated_seconds': validated,
                              'raw_makespan': self.objective_value - scenario.start_time,
                              'makespan': cert['makespan'], 'lower_bound': bound,
                              'branches': self.num_branches, 'conflicts': self.num_conflicts,
                              'execution_count': cert['execution_count'], 'selected_definitions': cert['selected_definitions'],
                              'repeated_activities': {a: n for a, n in counts.items() if n > 1},
                              'raw_rows': raw,
                              'schedule': [{k: e[k] for k in ('activity_id', 'start_time', 'end_time')} for e in cert['executions']]}
                    persist_started = time.perf_counter()
                    with lock:
                        state['validation_seconds'] += validated - found
                        state['candidate_count'] += 1
                        if state['first_solution_seconds'] is None:
                            state['first_solution_seconds'] = validated
                        if state['best_makespan'] is None or cert['makespan'] < state['best_makespan']:
                            state.update(best_makespan=cert['makespan'], best_candidate_seconds=validated,
                                         best_execution_count=cert['execution_count'], best_repeated_activities=record['repeated_activities'])
                            write_json(directory / 'best.json', {'summary': record, 'certificate': cert})
                        state.update(lower_bound=bound, branches=self.num_branches, conflicts=self.num_conflicts,
                                     search_stats_sample_seconds=found)
                        append(directory / 'candidates.jsonl', record)
                        state['persistence_seconds'] += time.perf_counter() - persist_started
                    accept_bound(self.best_objective_bound)
                    save()
                except Exception:
                    with lock:
                        state['rejections'].append({'seconds': time.perf_counter()-started, 'error': traceback.format_exc()})
                    self.stop_search()

        result = solver.solve(model, Observe())
        with lock:
            state.update(solver_status=solver.status_name(result), branches=solver.num_branches,
                         conflicts=solver.num_conflicts, search_stats_sample_seconds=time.perf_counter()-started)
            bound = max(0, math.ceil(solver.best_objective_bound-scenario.start_time-1e-7))
            state['lower_bound'] = bound
            best = state['best_makespan']
            if best is not None and bound > best:
                state['rejections'].append({'error': 'Final bound exceeds replayed incumbent'})
            proven = best is not None and best == bound and not state['rejections']
            state.update(proven_optimal=proven,
                         status='ENCODING_ERROR' if state['rejections'] else 'PROVEN_OPTIMAL' if proven else
                         'PROVEN_INFEASIBLE' if result == cp_model.INFEASIBLE and best is None else
                         'FEASIBLE' if best is not None else 'UNKNOWN', phase='FINISHED',
                         response_stats=solver.response_stats())
            if proven and state['first_optimal_seconds'] is None:
                state['first_optimal_seconds'] = time.perf_counter()-started
    except BaseException:
        with lock:
            state.update(status='ERROR', phase='FINISHED', error=traceback.format_exc())
        raise
    finally:
        stopped.set()
        thread.join(timeout=6)
        save()
        write_json(directory / 'result.json', state)
    return state


def run(root):
    verify(root)
    with batch_lock(root):
        if (root / 'attempt' / 'ready.json').exists():
            raise ValueError('Attempt already started; do not silently restart unlimited search')
        with (root / 'worker-stdout.log').open('w', encoding='utf-8') as out, (root / 'worker-stderr.log').open('w', encoding='utf-8') as err:
            launched = time.perf_counter()
            process = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.mainline_joint', 'worker', '--root', str(root)],
                                       cwd='D:/planner', stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            write_json(root / 'supervisor.json', {'pid': os.getpid(), 'child_launcher_pid': process.pid,
                                                'started_at': now(), 'deadline': None})
            ready = None
            while process.poll() is None:
                if ready is None and (root / 'attempt' / 'ready.json').exists():
                    ready = read_json(root / 'attempt' / 'ready.json')
                if ready:
                    total_cpu = cpu_seconds(ready['pid'])
                    append(root / 'resources.jsonl', {'utc': now(), 'worker_pid': ready['pid'],
                           'wall_seconds_including_worker_startup': time.perf_counter()-launched,
                           'cpu_seconds': None if total_cpu is None else total_cpu-ready['cpu_start'],
                           'memory': process_memory(ready['pid'])})
                time.sleep(1)
            write_json(root / 'supervisor-result.json', {'exit_code': process.returncode, 'finished_at': now(),
                                                       'full_call_seconds': time.perf_counter()-launched})
            if process.returncode:
                raise RuntimeError('Worker failed; see worker-stderr.log')
        print(json.dumps(read_json(root / 'attempt' / 'result.json'), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'run', 'worker'])
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root)
    elif args.command == 'verify':
        verify(args.root)
        print('VERIFIED')
    elif args.command == 'run':
        run(args.root)
    else:
        # No metadata/witness read by the formal solving process.
        unlimited(args.root / 'input.json', args.root / 'attempt')


if __name__ == '__main__':
    main()
