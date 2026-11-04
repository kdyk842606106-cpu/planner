"""C0/N256 width intervention; solver/worker code and search defaults stay frozen."""
from __future__ import annotations

import argparse
import copy
import html
import random
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from planner_experiment.benchmark_suite.complexity import dag_properties
from planner_experiment.benchmark_suite.generator import analytic_certificate, finish_instance
from planner_experiment.benchmark_suite.io import digest, read_events, read_json, write_json
from planner_experiment.benchmark_suite.profiles import assert_same_engines, file_hash
from planner_experiment.benchmark_suite.report import csv_file
from planner_experiment.benchmark_suite.runner import (block_key, environment, run_one, run_suite,
                                                       source_files, verify_frozen)

ENGINES = ("GA", "ASTAR")
WIDTHS = (8, 4, 2, 1)
SOLVER = Path('D:/Solver_demo_project_planner_exploration')
BASELINE = Path('D:/planner/runs/three-engine-long-quick-20261102')
DEFAULT_OUTPUT = Path('D:/planner/runs/c0-width-sweep-20261102')


def narrower_chains(baseline, width):
    """Concatenate whole old chains, retaining all old edges and all old goals."""
    item = copy.deepcopy(baseline)
    payload, metadata = item['scenario'], item['metadata']
    n = len(payload['activities'])
    old_width = baseline['complexity']['skeleton']['width']
    if width not in WIDTHS or old_width % width or n % old_width:
        raise ValueError('Width must divide the original independent-chain width')
    iid = f'n{n}-C0-w{width}-g101'
    payload.update(id=iid, name=f'C0 N={n}, original chains concatenated to width {width}')
    old_length = n // old_width
    activities = payload['activities']
    tasks = list(metadata['task_activities'])
    added = []
    for lane in range(width):
        old_lanes = list(range(lane, old_width, width))
        for previous, following in zip(old_lanes, old_lanes[1:]):
            tail = (old_length - 1) * old_width + previous
            edge = [tasks[tail], tasks[following]]
            added.append(edge)
            activities[following]['preconditions'].append({
                'state_id': activities[tail]['additional_output_state_ids'][0], 'relation_role': 'required'})
    metadata['edges'] = sorted([*metadata['edges'], *added])
    graph = dag_properties(tasks, metadata['edges'],
                           {task: activities[i]['duration'] for i, task in enumerate(tasks)})
    assert graph['width'] == width and graph['depth'] == n // width
    metadata.update(id=iid, width_parameter=width, baseline_width=old_width,
                    added_chain_links=added, baseline_reference_h0_minutes=metadata['reference_h0_minutes'],
                    reference_h0_minutes=graph['critical_path_minutes'], paired_baseline=baseline['metadata']['id'])
    proof = analytic_certificate(payload, metadata)
    result = finish_instance(payload, metadata, answer=proof)
    assert_pairing(baseline, result)
    return result


def assert_pairing(baseline, variant):
    original, changed = baseline['scenario'], variant['scenario']
    for key in original:
        if key not in ('id', 'name', 'activities'):
            assert original[key] == changed[key], f'Unexpected scenario change: {key}'
    for before, after in zip(original['activities'], changed['activities'], strict=True):
        assert {k: v for k, v in before.items() if k != 'preconditions'} == {
            k: v for k, v in after.items() if k != 'preconditions'}
        assert all(relation in after['preconditions'] for relation in before['preconditions'])
    old_edges = {tuple(e) for e in baseline['metadata']['edges']}
    new_edges = {tuple(e) for e in variant['metadata']['edges']}
    assert old_edges <= new_edges
    c = variant['complexity']
    assert c['structurally_relevant_count'] == c['activity_count'] == 256
    assert c['state_count'] == 512 and c['witness_repeat_count'] == 0


def prepare(output, solver_root=SOLVER, baseline=BASELINE):
    output, baseline, solver_root = Path(output).resolve(), Path(baseline).resolve(), Path(solver_root).resolve()
    if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError('Output must be separate from baseline')
    if (output / 'manifest.json').exists():
        raise ValueError('Frozen output already exists; use run/report')
    old = read_json(baseline / 'manifest.json')
    if read_json(baseline / 'batch_state.json')['status'] != 'COMPLETED':
        raise ValueError('Baseline must be completed')
    current = environment(solver_root)
    assert_same_engines(old['environment']['source_hashes'], current['source_hashes'])
    for key in ('python', 'python_executable', 'ortools'):
        if old['environment'][key] != current[key]:
            raise ValueError(f'Baseline environment changed: {key}')
    iid = 'n256-C0-g101'
    original = read_json(baseline / 'instances' / f'{iid}.json')
    if digest(original) != old['instance_hashes'][iid]:
        raise ValueError('Baseline model changed')
    output.mkdir(parents=True, exist_ok=True)
    references = {}

    def frozen_copy(source, relative):
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        references[relative] = file_hash(dest)

    frozen_copy(baseline / 'instances' / f'{iid}.json', f'reference/{iid}.json')
    frozen_copy(baseline / 'manifest.json', 'reference/baseline-manifest.json')
    for engine in ENGINES:
        source = baseline / 'runs' / f'{iid}-t300-s11' / engine
        raw = read_json(source / 'run.json')
        assert (raw['engine'], raw['instance_id'], raw['seed'], raw['budget_seconds']) == (engine, iid, 11, 300)
        for name in ('run.json', 'candidates.jsonl', 'best_candidate.json', 'engine_result.json'):
            if (source / name).exists():
                frozen_copy(source / name, f'reference/{engine}/{name}')
    hashes, ids = {}, []
    for width in WIDTHS:
        item = narrower_chains(original, width)
        identifier = item['metadata']['id']
        write_json(output / 'instances' / f'{identifier}.json', item)
        hashes[identifier] = digest(item)
        ids.append(identifier)
        print(f'Certified width={width}, depth={item["complexity"]["skeleton"]["depth"]}, optimum={item["answer"]["optimal_makespan"]}', flush=True)
    warmup = read_json(baseline / 'instances/G01.json')
    if digest(warmup) != old['instance_hashes']['G01']:
        raise ValueError('Baseline warmup changed')
    write_json(output / 'instances/G01.json', warmup)
    hashes['G01'] = digest(warmup)
    rng = random.Random(81173)
    blocks = []
    for width in (1, 8, 2, 4):
        engines = list(ENGINES)
        rng.shuffle(engines)
        blocks.append({'phase': f'WIDTH_{width}', 'instance_id': f'n256-C0-w{width}-g101',
                       'seconds': 300.0, 'seed': 11, 'engines': engines})
    config = {'profile': 'c0-width-sweep', 'engines': list(ENGINES), 'widths': list(WIDTHS),
              'reference_width': 16, 'budgets_seconds': [300], 'run_seeds': [11], 'generator_seeds': [101],
              'batch_limit_seconds': 3000, 'warmup_instance_id': 'G01', 'answers_ready': True,
              'oracle_mode': 'precomputed_analytic_certificates', 'report_each_block': True,
              'intervention': 'Concatenate old chains by adding 16-W links. Preserve every original dependency, all goals and all activity attributes.',
              'quality_comparison': 'Compare to each width-specific proved optimum; raw makespans are not comparable across changed chain lengths.',
              'repeat_count_per_condition': 1, 'reference_source': str(baseline)}
    write_json(output / 'config.json', config)
    write_json(output / 'schedule.json', blocks)
    own_source = Path(__file__).resolve()
    with zipfile.ZipFile(output / 'source-snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as snapshot:
        for name, path in source_files(solver_root):
            snapshot.write(path, name)
        snapshot.write(own_source, 'study/experiments/c0_width_sweep.py')
    manifest = {'profile': 'c0-width-sweep', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'solver_root': str(solver_root), 'instance_ids': ids, 'auxiliary_instance_ids': ['G01'],
                'instance_hashes': hashes, 'config_hash': digest(config), 'schedule_hash': digest(blocks),
                'reference_hashes': references, 'planned_run_count': 8, 'reference_run_count': 2,
                'environment': current, 'study_source_hash': file_hash(own_source)}
    write_json(output / 'manifest.json', manifest)
    build_report(output)
    return manifest


def verify_study(output, solver_root):
    manifest = verify_frozen(output, solver_root)
    if manifest['study_source_hash'] != file_hash(Path(__file__).resolve()):
        raise ValueError('Frozen study entry changed')
    return manifest


def build_report(output):
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    baseline = read_json(output / 'reference/n256-C0-g101.json')
    cases = [(16, baseline, output / 'reference', True)]
    for iid in manifest['instance_ids']:
        item = read_json(output / 'instances' / f'{iid}.json')
        block = {'instance_id': iid, 'seconds': 300, 'seed': 11}
        cases.append((item['complexity']['skeleton']['width'], item, output / 'runs' / block_key(block), False))
    rows = []
    for width, item, directory, is_reference in sorted(cases, key=lambda c: -c[0]):
        optimum = item['answer']['optimal_makespan']
        graph = item['complexity']['skeleton']
        for engine in ENGINES:
            path = directory / engine
            exists = (path / 'run.json').exists()
            raw = read_json(path / 'run.json') if exists else {}
            accepted = [e for e in read_events(path / 'candidates.jsonl') if e['valid'] and e['elapsed_seconds'] <= 300] if exists else []
            first = min(accepted, key=lambda e: e['elapsed_seconds'], default=None)
            best = min((e['makespan'] for e in accepted), default=None)
            candidate = read_json(path / 'best_candidate.json') if best is not None else None
            stats = raw.get('engine_stats') or {}
            ids = [e['activity_id'] for e in candidate['executions']] if candidate else []
            rows.append({'width': width, 'depth': graph['depth'], 'preconditions': item['complexity']['precondition_count'],
                         'order_strength': graph['order_strength'], 'engine': engine, 'reference': is_reference,
                         'executed': exists, 'status': raw.get('process_status', 'UNRUN'),
                         'optimum': optimum, 'best_makespan': best,
                         'optimal_gap': (best - optimum) / optimum if best is not None else None,
                         'first_seconds': first['elapsed_seconds'] if first else None,
                         'first_optimal_seconds': min((e['elapsed_seconds'] for e in accepted if e['makespan'] == optimum), default=None),
                         'repeat_count': len(ids) - len(set(ids)) if candidate else None,
                         'non_causal_count': candidate['metrics']['non_causal_action_count'] if candidate else None,
                         'first_equals_final': first['actions'] == candidate['actions'] if first and candidate else None,
                         'seed_makespan': (stats.get('maximal_start_seed') or {}).get('makespan'),
                         'expanded_nodes': stats.get('expanded_nodes'), 'reconstruction_count': stats.get('trajectory_reconstruction_count'),
                         'valid_candidates': len(accepted), 'rejected_candidates': raw.get('rejected_candidate_count', 0),
                         'stop_reason': stats.get('stopped_reason') or raw.get('engine_status') or raw.get('process_status', 'UNRUN'),
                         'measured_seconds': raw.get('measured_elapsed_seconds'),
                         'peak_memory_mib': raw['peak_working_set_bytes'] / 2**20 if raw.get('peak_working_set_bytes') else None})
    state = read_json(output / 'batch_state.json') if (output / 'batch_state.json').exists() else {'status': 'NOT_STARTED'}
    summary = {'status': state['status'], 'planned_runs': 8, 'reference_runs': 2,
               'completed_runs': sum(r['executed'] for r in rows if not r['reference']), 'rows': rows,
               'limits': ['One generated model and one solve seed per width.', 'Width16 is a frozen historical reference, not a new run.',
                          'Width/depth/order/critical-path changes are coupled. Compare solution quality to the width-specific proved optimum.']}
    write_json(output / 'summary.json', summary)
    csv_file(output / 'results.csv', rows)
    csv_file(output / 'complexity.csv', [{k: r[k] for k in ('width', 'depth', 'preconditions', 'order_strength', 'optimum')}
                                        for r in rows if r['engine'] == ENGINES[0]])
    columns = [('width', '宽度'), ('depth', '深度'), ('engine', '引擎'), ('best_makespan', '最终工期'),
               ('optimum', '已证明最优'), ('optimal_gap', '工期差距'), ('first_seconds', '首解秒'),
               ('repeat_count', '重复执行'), ('stop_reason', '停止原因')]
    def display(key, value):
        if value is None:
            return '—'
        return f'{value:.1%}' if key == 'optimal_gap' else f'{value:.3f}' if isinstance(value, float) else str(value)
    table = '<table><tr>' + ''.join(f'<th>{label}</th>' for _, label in columns) + '</tr>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{html.escape(display(key, row[key]))}</td>' for key, _ in columns) + '</tr>'
    table += '</table>'
    page = f'''<!doctype html><meta charset="utf-8"><title>C0 256 活动宽度实验</title>
<style>body{{font:15px system-ui;max-width:1150px;margin:32px auto;padding:0 18px;color:#203049}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:right;border-bottom:1px solid #ddd}}p{{line-height:1.7}}</style>
<h1>C0：256 活动，降低偏序宽度</h1><p>状态：{html.escape(state['status'])} · 新运行 {summary['completed_runs']}/8 · 宽度16为已完成的历史基线。</p>
<p>每次300秒，求解种子11。通过增加链间先后关系，将宽度16降至8、4、2、1；活动、时长、资源、目标、执行边界保持不变。宽度降低同时增加深度和关键路径，工期差距按各自已证明最优值计算。每条件一次运行。</p>
<p><a href="results.csv">逐项结果</a> · <a href="complexity.csv">复杂度</a> · <a href="batch_state.json">实时进度</a> · <a href="manifest.json">冻结清单</a></p><div style="overflow:auto">{table}</div>'''
    (output / 'report.html').write_text(page, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'run', 'report', 'smoke'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--solver-root', type=Path, default=SOLVER)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    args = parser.parse_args()
    if args.command == 'generate':
        prepare(args.output, args.solver_root, args.baseline)
    elif args.command == 'report':
        build_report(args.output)
    else:
        verify_study(args.output, args.solver_root)
        if args.command == 'smoke':
            for engine in ENGINES:
                raw = run_one(args.output, args.solver_root, {'instance_id': 'G01', 'seconds': 0.2, 'seed': 0}, engine, warmup=True)
                assert raw['best_makespan'] == 10
                print(engine, raw['process_status'], raw['best_makespan'], flush=True)
        else:
            run_suite(args.output, args.solver_root, report_builder=build_report)


if __name__ == '__main__':
    main()
