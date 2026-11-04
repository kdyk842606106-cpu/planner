from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import random
import shutil
import sys
import zipfile
from unittest.mock import patch

from planner_experiment.benchmark_suite import runner
from planner_experiment.benchmark_suite.golden import golden_instances
from planner_experiment.benchmark_suite.io import digest, read_json, write_json
from planner_experiment.benchmark_suite.profiles import assert_same_engines, file_hash
from planner_experiment.benchmark_suite.report import csv_file
from .model import make_instance
from .observe import observed_worker, supervise
from .report import build_report

DEFAULT_OUTPUT = Path('D:/planner/runs/business256-20261102')
SOLVER = Path('D:/Solver_demo_project_planner_exploration')
BASELINE = Path('D:/planner/runs/three-engine-long-quick-20261102')
ENGINES = ('LEGACY', 'ASTAR', 'GA')


def study_sources():
    return {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}


def prepare(root, solver=SOLVER):
    root = Path(root).resolve()
    if (root / 'manifest.json').exists():
        raise ValueError('Frozen suite exists; use run/report')
    baseline = read_json(BASELINE / 'manifest.json')
    env = runner.environment(solver)
    assert_same_engines(baseline['environment']['source_hashes'], env['source_hashes'])
    for k in ('python', 'python_executable', 'ortools'):
        if env[k] != baseline['environment'][k]:
            raise ValueError(f'Baseline runtime changed: {k}')
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(solver))
    from app.services.planner_scenarios import validate_scenario
    hashes, ids, complexities = {}, [], []
    for seed in (101, 202):
        for width in (8, 16, 32):
            for density in (2, 4):
                for condition in ('T0', 'T1', 'T2', 'T3'):
                    item = make_instance(width, density, condition, seed)
                    issues = validate_scenario(copy.deepcopy(item['scenario']))
                    if issues:
                        raise ValueError(f'Shared input preflight rejected {item["metadata"]["id"]}: {issues}')
                    iid = item['metadata']['id']
                    write_json(root / 'instances' / f'{iid}.json', item)
                    ids.append(iid)
                    hashes[iid] = digest(item)
                    c = item['complexity']
                    complexities.append({'id': iid, 'seed': seed, 'width': c['skeleton']['width'], 'density_target': density,
                                         'density_actual': c['direct_dependency_mean'], 'condition': condition,
                                         'activities': c['activity_count'], 'states': c['state_count'],
                                         'preconditions': c['precondition_count'], 'depth': c['skeleton']['depth'],
                                         'order_strength': c['skeleton']['order_strength'], 'lower_bound': item['answer']['lower_bound'],
                                         'witness_repeats': c['witness_repeat_count'], 'witness_makespan': item['answer']['witness_makespan']})
                    print(f'CERTIFIED {iid} width={c["skeleton"]["width"]} density={c["direct_dependency_mean"]:.4f}', flush=True)
    warmup = next(i for i in golden_instances() if i['metadata']['id'] == 'G01')
    write_json(root / 'instances/G01.json', warmup)
    hashes['G01'] = digest(warmup)
    rng = random.Random(81173)
    schedule = []
    for seconds in (30., 120.):
        # Complete 24 conditions for one seed before replication; rotate all
        # factors rather than leaving the widest/restoration conditions last.
        for seed in (101, 202):
            conditions = [(w, d, t) for w in (8, 16, 32) for d in (2, 4) for t in ('T0','T1','T2','T3')]
            rng.shuffle(conditions)
            for w, d, t in conditions:
                engines = list(ENGINES)
                rng.shuffle(engines)
                schedule.append({'phase': f'{seconds:g}s', 'instance_id': f'b256-w{w}-d{d}-{t}-g{seed}',
                                 'seconds': seconds, 'seed': 11, 'engines': engines})
    config = {'profile': 'business256', 'engines': list(ENGINES), 'generator_seeds': [101,202], 'run_seeds': [11],
              'budgets_seconds': [30,120], 'batch_limit_seconds': 28800, 'warmup_instance_id': 'G01',
              'answers_ready': True, 'oracle_mode': 'validated_witness_and_relaxation_bound', 'report_each_block': True,
              'search_defaults': 'unchanged constructors; prior protocol: 1 CPU, 1 CP-SAT thread, tracemalloc off, transition protection 2**60, max_solutions 20',
              'observation': 'OS CPU/RSS at 2 Hz; read-only native frame lengths at 1 Hz, no search mutations',
              'reference_event_clock': 'same g-seed width16 density2 reference CP across all structures',
              'effect_thresholds': {'memory_ratio': 2, 'memory_delta_mib': 128, 'first_solution_ratio': 2},
              'state_count_scope': 'engine-specific native counts, not common reachable-state enumeration',
              'condition_scope': 'T2 adds retrofit history gates and local invalidation jointly; T3 only capacities',
              'business_assumptions': 'off/on preserves certificates; installing B invalidates local certificates in T2/T3; eight modules require A test history before retrofit'}
    write_json(root / 'config.json', config)
    write_json(root / 'schedule.json', schedule)
    with zipfile.ZipFile(root / 'source-snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in runner.source_files(solver):
            archive.write(path, name)
        for path in Path(__file__).parent.glob('*.py'):
            archive.write(path, 'study/business256/' + path.name)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'profile': 'business256',
                'solver_root': str(solver), 'instance_ids': ids, 'instance_hashes': hashes,
                'auxiliary_instance_ids': ['G01'], 'environment': env,
                'config_hash': digest(config), 'schedule_hash': digest(schedule), 'study_hashes': study_sources(),
                'planned_run_count': len(schedule)*3, 'baseline_protected_source_guard': 'PASS', 'baseline': str(BASELINE)}
    write_json(root / 'manifest.json', manifest)
    csv_file(root / 'complexity.csv', complexities)
    build_report(root)
    return manifest


def verify(root, solver=SOLVER):
    manifest = runner.verify_frozen(root, solver)
    if manifest['study_hashes'] != study_sources():
        raise ValueError('Study code changed since freeze')
    return manifest


def refreeze_prelaunch(root, solver=SOLVER):
    """Preserve earlier harness revision; prohibited after any formal launch."""
    root = Path(root)
    if (root / 'batch_state.json').exists() or any((root / 'runs').rglob('run.json')):
        raise ValueError('Cannot refreeze an experiment after formal launch')
    manifest = runner.verify_frozen(root, solver)
    revision = root / 'prelaunch-revisions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    revision.mkdir(parents=True)
    shutil.copy2(root / 'manifest.json', revision / 'manifest.json')
    shutil.copy2(root / 'source-snapshot.zip', revision / 'source-snapshot.zip')
    with zipfile.ZipFile(root / 'source-snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in runner.source_files(solver):
            archive.write(path, name)
        for path in Path(__file__).parent.glob('*.py'):
            archive.write(path, 'study/business256/' + path.name)
    manifest['study_hashes'] = study_sources()
    manifest['prelaunch_refrozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(root / 'manifest.json', manifest)
    return manifest


def run(root, solver=SOLVER):
    verify(root, solver)
    with patch.object(runner, 'engine_worker', observed_worker), patch.object(runner, 'supervise', supervise):
        return runner.run_suite(root, solver, report_builder=build_report)


def main():
    parser = argparse.ArgumentParser(description='Frozen 24-condition non-substitutable business benchmark')
    parser.add_argument('mode', choices=('generate', 'run', 'report', 'verify', 'smoke', 'refreeze-prelaunch'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.mode == 'generate':
        result = prepare(args.output)
    elif args.mode == 'run':
        result = run(args.output)
    elif args.mode == 'report':
        result = build_report(args.output)
    elif args.mode == 'verify':
        result = verify(args.output)
    elif args.mode == 'refreeze-prelaunch':
        result = refreeze_prelaunch(args.output)
    else:
        verify(args.output)
        with patch.object(runner, 'engine_worker', observed_worker), patch.object(runner, 'supervise', supervise):
            result = {engine: runner.run_one(args.output, SOLVER,
                       {'instance_id': 'b256-w16-d2-T2-g101', 'seconds': 3., 'seed': 11}, engine, warmup=True)
                      for engine in ENGINES}
    print({k: result[k] for k in ('status', 'completed_runs', 'planned_run_count') if k in result}, flush=True)


if __name__ == '__main__':
    main()
