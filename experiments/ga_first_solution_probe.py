"""Read-only GA diagnosis: process-local timing wrappers preserve all return values."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import ExitStack
from datetime import datetime, timezone
import faulthandler
from functools import wraps
import json
from pathlib import Path
import pstats
import time
from unittest.mock import patch

from planner_experiment.benchmark_suite.io import read_events, read_json, write_json
from planner_experiment.benchmark_suite.runner import supervise

ROOT = Path('D:/planner/runs/ga-first-solution-diagnosis-20261102')
STUDY = Path('D:/planner/runs/c0-width-sweep-20261102')
SOLVER = 'D:/Solver_demo_project_planner_exploration'


def state_summary(state):
    return {'simulated_time': state.time, 'completed': len(state.executions),
            'started': sum(v for _, v in state.counts), 'running': len(state.running),
            'active_states': len(state.active_state_ids),
            'repeat_starts': sum(max(0, value - 1) for _, value in state.counts)}


def probe_worker(instance_path, engine, seconds, seed, solver_root, directory, pipe):
    from planner_experiment.archive import PathArchive
    from planner_experiment.ga import GeneticExplorer
    from planner_experiment.simulator import PathSimulator
    from planner_experiment.validator import ResultValidator
    from planner_experiment.benchmark_suite.worker import engine_worker
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / 'probe-events.jsonl').open('a', encoding='utf-8', buffering=1)
    stacks = (directory / 'sampled-stacks.txt').open('w', encoding='utf-8')
    counters = {'decode_enter': 0, 'decode_exit': 0, 'primary_enter': 0, 'primary_exit': 0,
                'alternate_enter': 0, 'alternate_exit': 0, 'ranking_enter': 0, 'ranking_exit': 0,
                'evaluate_enter': 0, 'evaluate_exit': 0, 'candidate_build_enter': 0,
                'candidate_build_exit': 0, 'validator_enter': 0, 'validator_exit': 0,
                'archive_enter': 0, 'archive_exit': 0}
    begin = time.perf_counter()
    last_log = begin

    def emit(kind, **data):
        event = {'elapsed_seconds': time.perf_counter() - begin, 'event': kind,
                 'counters': dict(counters), **data}
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        stream.flush()

    def ga_run(original):
        @wraps(original)
        def wrapped(self, *args, **kwargs):
            nonlocal begin, last_log
            begin = last_log = time.perf_counter()
            emit('ga_run_enter', population_size=self.population_size, max_generations=self.max_generations,
                 budget_seconds=args[1].time_limit_seconds, gene_count=self.gene_count)
            result = original(self, *args, **kwargs)
            emit('ga_run_exit', stats=result.stats)
            return result
        return wrapped

    def phase(original, prefix):
        @wraps(original)
        def wrapped(*args, **kwargs):
            counters[prefix + '_enter'] += 1
            t0 = time.perf_counter()
            emit(prefix + '_enter')
            result = original(*args, **kwargs)
            counters[prefix + '_exit'] += 1
            detail = {}
            if prefix == 'decode':
                item, used, frames = result
                detail = {'candidate_count': len(item.candidates), 'transitions': used,
                          'makespans': [c.metrics.makespan for c in item.candidates], 'frames': len(frames)}
            elif prefix == 'candidate_build':
                detail = {'makespan': result.metrics.makespan}
            emit(prefix + '_exit', call_seconds=time.perf_counter() - t0, **detail)
            return result
        return wrapped

    original_rollout = GeneticExplorer._rollout
    @wraps(original_rollout)
    def rollout(self, *args, **kwargs):
        prefix = 'primary' if kwargs['allow_fork'] else 'alternate'
        counters[prefix + '_enter'] += 1
        t0 = time.perf_counter()
        emit(prefix + '_enter', state=state_summary(args[0]))
        result = original_rollout(self, *args, **kwargs)
        counters[prefix + '_exit'] += 1
        emit(prefix + '_exit', call_seconds=time.perf_counter() - t0,
             terminal_reason=result[1], state=state_summary(result[0]))
        return result

    original_rank = GeneticExplorer._ranked_actions
    @wraps(original_rank)
    def ranked(self, state, genes):
        nonlocal last_log
        counters['ranking_enter'] += 1
        start = time.perf_counter()
        emit_now = start - last_log >= 5 or counters['ranking_enter'] <= 2
        if emit_now:
            emit('ranking_enter', state=state_summary(state))
        result = original_rank(self, state, genes)
        counters['ranking_exit'] += 1
        if emit_now:
            counts = dict(state.counts)
            repeat_options = sum(action.kind == 'START' and counts.get(action.activity_id, 0) > 0
                                 for _, action in result)
            emit('ranking_exit', call_seconds=time.perf_counter() - start, state=state_summary(state),
                 action_count=len(result), repeat_start_options=repeat_options,
                 top_action=result[0][1].label if result else None)
            last_log = time.perf_counter()
        return result

    try:
        faulthandler.dump_traceback_later(30, repeat=True, file=stacks)
        with ExitStack() as context:
            context.enter_context(patch.object(GeneticExplorer, 'run', ga_run(GeneticExplorer.run)))
            context.enter_context(patch.object(GeneticExplorer, '_decode', phase(GeneticExplorer._decode, 'decode')))
            context.enter_context(patch.object(GeneticExplorer, '_rollout', rollout))
            context.enter_context(patch.object(GeneticExplorer, '_ranked_actions', ranked))
            context.enter_context(patch.object(GeneticExplorer, '_evaluate_rollout', phase(GeneticExplorer._evaluate_rollout, 'evaluate')))
            context.enter_context(patch.object(PathSimulator, 'candidate_from_state', phase(PathSimulator.candidate_from_state, 'candidate_build')))
            context.enter_context(patch.object(ResultValidator, 'validate', phase(ResultValidator.validate, 'validator')))
            context.enter_context(patch.object(PathArchive, 'add', phase(PathArchive.add, 'archive')))
            engine_worker(instance_path, engine, seconds, seed, solver_root, directory, pipe)
    finally:
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        stream.close()


def replay(output=ROOT):
    from experiments.c0_width_sweep import verify_study
    verify_study(STUDY, SOLVER)
    output.mkdir(parents=True, exist_ok=True)
    source = STUDY / 'instances/n256-C0-w1-g101.json'
    started = datetime.now(timezone.utc).isoformat()
    write_json(output / 'state.json', {'status': 'RUNNING', 'started_at_utc': started,
               'mode': 'instrumented unchanged full engine via original worker', 'budget_seconds': 300})
    run = supervise(probe_worker, (str(source), 'GA', 300, 11, SOLVER), output / 'reproduction', 300)
    events = read_events(output / 'reproduction/probe-events.jsonl')
    run['last_probe_event'] = events[-1] if events else None
    run['candidate_events'] = read_events(output / 'reproduction/candidates.jsonl')
    write_json(output / 'reproduction/run.json', run)
    verify_study(STUDY, SOLVER)
    write_json(output / 'state.json', {'status': 'COMPLETED', 'started_at_utc': started,
               'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'source_guard': 'PASS',
               'process_status': run['process_status'], 'candidate_count': len(run['candidate_events'])})
    print(json.dumps({'process_status': run['process_status'], 'last_probe_event': run['last_probe_event']}, ensure_ascii=False), flush=True)


def snapshots(output=ROOT):
    from planner_experiment.benchmark_suite.runtime import pin_one_cpu
    from planner_experiment.ga import GeneticExplorer
    from planner_experiment.models import Action
    from planner_experiment.scenario import scenario_from_dict
    pin_one_cpu()
    output.mkdir(parents=True, exist_ok=True)
    item = read_json(STUDY / 'instances/n256-C0-w1-g101.json')
    explorer = GeneticExplorer(scenario_from_dict(item['scenario']))
    genes = explorer._rule_seed_genes()[0]
    state = explorer.simulator.initial_state()
    checkpoints = {0, 16, 64, 128, 192, 240, 255}
    saved = {}
    for action_dict in item['answer']['actions']:
        if not state.running and len(state.executions) in checkpoints:
            saved[len(state.executions)] = state
        state = explorer.simulator.transition(state, Action(**action_dict))
    results = []
    for completed, state in sorted(saved.items()):
        before = time.perf_counter()
        ranked = explorer._ranked_actions(state, genes)
        wall = time.perf_counter() - before
        counts = dict(state.counts)
        record = {'completed_tasks': completed, 'state': state_summary(state), 'action_count': len(ranked),
                  'repeat_start_options': sum(a.kind == 'START' and counts.get(a.activity_id, 0) > 0 for _, a in ranked),
                  'top_action': ranked[0][1].label, 'unprofiled_ranking_seconds': wall}
        if completed in (0, 128):
            profile = cProfile.Profile()
            profiled_ranked = profile.runcall(explorer._ranked_actions, state, genes)
            assert ranked == profiled_ranked
            profile.dump_stats(str(output / f'ranking-completed-{completed}.pstats'))
            stats = pstats.Stats(profile)
            functions = [{'file': file, 'line': line, 'function': name, 'calls': calls,
                          'self_seconds': own, 'cumulative_seconds': cumulative}
                         for (file, line, name), (_, calls, own, cumulative, _) in stats.stats.items()]
            functions.sort(key=lambda r: r['cumulative_seconds'], reverse=True)
            record.update(profile_total_seconds=stats.total_tt, profile_total_calls=stats.total_calls,
                          functions=functions)
        results.append(record)
        write_json(output / 'snapshots.json', results)
        print(json.dumps({k: v for k, v in record.items() if k != 'functions'}, ensure_ascii=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay', 'snapshots'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.mode == 'replay':
        replay(args.output)
    else:
        snapshots(args.output)


if __name__ == '__main__':
    main()
