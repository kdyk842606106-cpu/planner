"""Incremental coverage, native counters and paired-factor comparisons."""
from __future__ import annotations

from collections import defaultdict
import html
from pathlib import Path
import statistics

from planner_experiment.benchmark_suite.io import read_events, read_json, write_json
from planner_experiment.benchmark_suite.report import csv_file
from planner_experiment.benchmark_suite.runner import block_key


def telemetry(directory, raw):
    values = {}
    last_time = None
    for event in read_events(directory / 'state-samples.jsonl'):
        last_time = event.get('elapsed_seconds')
        values['observer_cpu_seconds'] = max(values.get('observer_cpu_seconds', 0), event.get('observer_cpu_seconds', 0))
        for row in event.get('frames', []):
            phase = row['phase']
            for k, v in row.items():
                if k in ('engine', 'phase', 'native_stats') or not isinstance(v, (int, float)):
                    continue
                key = f'{phase}:{k}'
                values[key] = max(values.get(key, 0), v)
            for k, v in row.get('native_stats', {}).items():
                if isinstance(v, (int, float)):
                    key = f'POP:{k}'
                    values[key] = max(values.get(key, 0), v)
    stats = raw.get('engine_stats') or {}
    pop = stats.get('pathfinder') or {}
    return {'sampled_maxima': values, 'last_state_sample_seconds': last_time,
            'expanded_nodes': stats.get('expanded_nodes', values.get('run:expanded')),
            'simulator_transitions': stats.get('simulator_transitions', values.get('run:transitions')),
            'astar_retained_nodes_sampled': values.get('run:nodes_count'),
            'astar_retained_nodes_final': stats.get('search_node_count'),
            'astar_business_keys_sampled': values.get('run:labels_count'),
            'astar_open_peak': stats.get('peak_open_size', values.get('run:peak_open')),
            'ga_individual_evaluations': stats.get('individual_evaluations', values.get('run:evaluations')),
            'ga_rollout_unique_states_sampled': values.get('_rollout:seen_count'),
            'ga_rollout_completed_sampled': values.get('_rollout:completed'),
            'ga_rollout_transitions_sampled': values.get('_rollout:used'),
            'pop_instances_sampled': values.get('POP:instances_count'),
            'pop_instances_final': pop.get('selected_instance_count'),
            'pop_provider_branches_final': pop.get('provider_branches'),
            'pop_closure_iterations_final': pop.get('closure_iterations'),
            'pop_open_preconditions_sampled': values.get('POP:open_preconditions_count'),
            'cp_sat_branches': (stats.get('scheduler') or {}).get('branches'),
            'observer_cpu_seconds': values.get('observer_cpu_seconds')}


def build_report(output):
    root = Path(output)
    manifest = read_json(root / 'manifest.json')
    schedule = read_json(root / 'schedule.json')
    rows, evidence = [], []
    for block in schedule:
        item = read_json(root / 'instances' / f'{block["instance_id"]}.json')
        meta, complexity = item['metadata'], item['complexity']
        for engine in block['engines']:
            directory = root / 'runs' / block_key(block) / engine
            exists = (directory / 'run.json').exists()
            raw = read_json(directory / 'run.json') if exists else {}
            stats = raw.get('engine_stats') or {}
            metrics = telemetry(directory, raw) if exists else {}
            evidence.append({'instance_id': block['instance_id'], 'engine': engine, 'budget_seconds': block['seconds'], **metrics})
            rows.append({'instance_id': block['instance_id'], 'engine': engine, 'budget_seconds': block['seconds'],
                         'generator_seed': meta['generator_seed'], 'run_seed': block['seed'],
                         'width': complexity['skeleton']['width'], 'density': meta['density_target'],
                         'dependency_mean_actual': complexity['direct_dependency_mean'],
                         'condition': meta['condition'], 'executed': exists,
                         'status': raw.get('process_status', 'UNRUN'), 'engine_status': raw.get('engine_status'),
                         'success': bool(raw.get('valid_candidate_count')), 'first_seconds': raw.get('first_solution_seconds'),
                         'cpu_seconds': raw.get('cpu_seconds_measured'), 'wall_seconds': raw.get('measured_elapsed_seconds'),
                         'budget_censored': bool(exists and raw.get('measured_elapsed_seconds', 0) >= block['seconds'] * .95),
                         'peak_memory_mib': raw['peak_working_set_bytes'] / 2**20 if raw.get('peak_working_set_bytes') else None,
                         'makespan': raw.get('best_makespan'), 'lower_bound': item['answer']['lower_bound'],
                         'witness_makespan': item['answer']['witness_makespan'],
                         'rejected': raw.get('rejected_candidate_count'),
                         'stop_reason': stats.get('stopped_reason') or stats.get('stop_reason') or
                                        (raw.get('diagnosis') or {}).get('error_code') or raw.get('engine_status') or raw.get('process_status'),
                         **{k: v for k, v in metrics.items() if k != 'sampled_maxima'}})
    index = {(r['width'], r['density'], r['condition'], r['generator_seed'], r['engine'], r['budget_seconds']): r for r in rows}
    comparisons = []
    for after in rows:
        common = (after['generator_seed'], after['engine'], after['budget_seconds'])
        factors = []
        w, d, t = after['width'], after['density'], after['condition']
        if d == 4:
            factors.append(('dependency_density', (w, 2, t, *common)))
        if w in (16, 32):
            factors.append(('width', (w//2, d, t, *common)))
        if t != 'T0':
            factors.append(({'T1': 'late_material', 'T2': 'required_cycle_and_invalidation', 'T3': 'resource_bottleneck'}[t],
                            (w, d, f'T{int(t[1])-1}', *common)))
        for factor, key in factors:
            before = index[key]
            if not (before['executed'] and after['executed']):
                continue
            pair = {'factor': factor, 'before': before['instance_id'], 'after': after['instance_id'],
                    'engine': after['engine'], 'budget_seconds': after['budget_seconds'],
                    'after_width': w, 'after_density': d, 'after_condition': t,
                    'generator_seed': after['generator_seed'], 'before_success': before['success'], 'after_success': after['success'],
                    'before_budget_censored': before['budget_censored'], 'after_budget_censored': after['budget_censored']}
            for metric in ('cpu_seconds', 'peak_memory_mib', 'first_seconds', 'expanded_nodes',
                           'astar_business_keys_sampled', 'ga_rollout_unique_states_sampled', 'ga_rollout_completed_sampled'):
                a, b = before.get(metric), after.get(metric)
                pair[metric + '_before'], pair[metric + '_after'] = a, b
                pair[metric + '_ratio'] = b/a if a is not None and b is not None and a > 0 else None
            pair['memory_growth_flag'] = bool((pair['peak_memory_mib_ratio'] or 0) >= 2 and
                                               after['peak_memory_mib'] - before['peak_memory_mib'] >= 128)
            pair['first_solution_growth_flag'] = bool((pair['first_seconds_ratio'] or 0) >= 2)
            pair['lost_solution'] = before['success'] and not after['success']
            comparisons.append(pair)
    effect_groups = defaultdict(list)
    for pair in comparisons:
        effect_groups[tuple(pair[k] for k in ('factor', 'engine', 'budget_seconds', 'after_width', 'after_density', 'after_condition'))].append(pair)
    effects = []
    for key, members in sorted(effect_groups.items()):
        effect = dict(zip(('factor', 'engine', 'budget_seconds', 'after_width', 'after_density', 'after_condition'), key))
        effect.update(paired_instances=len(members), memory_growth_instances=sum(p['memory_growth_flag'] for p in members),
                      first_solution_growth_instances=sum(p['first_solution_growth_flag'] for p in members),
                      lost_solution_instances=sum(p['lost_solution'] for p in members))
        for metric in ('cpu_seconds_ratio', 'peak_memory_mib_ratio', 'first_seconds_ratio', 'astar_business_keys_sampled_ratio',
                       'ga_rollout_unique_states_sampled_ratio', 'ga_rollout_completed_sampled_ratio'):
            values = [p[metric] for p in members if p.get(metric) is not None and
                      (metric != 'cpu_seconds_ratio' or not (p['before_budget_censored'] or p['after_budget_censored']))]
            effect[metric + '_median'] = statistics.median(values) if values else None
        effects.append(effect)
    state = read_json(root / 'batch_state.json') if (root / 'batch_state.json').exists() else {'status': 'NOT_STARTED'}
    groups = defaultdict(list)
    for row in rows:
        if row['executed']:
            groups[(row['budget_seconds'], row['engine'], row['condition'])].append(row)
    aggregates = []
    for (budget, engine, condition), members in sorted(groups.items()):
        agg = {'budget_seconds': budget, 'engine': engine, 'condition': condition,
               'completed': len(members), 'successes': sum(r['success'] for r in members)}
        for metric in ('cpu_seconds', 'peak_memory_mib', 'first_seconds'):
            values = [r[metric] for r in members if r.get(metric) is not None]
            agg[metric + '_median'] = statistics.median(values) if values else None
        aggregates.append(agg)
    summary = {'status': state['status'], 'completed_runs': sum(r['executed'] for r in rows),
               'planned_runs': manifest['planned_run_count'], 'aggregates': aggregates,
               'factor_pair_count': len(comparisons),
               'limits': ['Two independent generated instances per condition; one solver seed.',
                          'Static width refers to 224 first-completion tasks, excluding dynamic mode/recovery gates.',
                          'Native state counters differ by engine and are sampled at 1 Hz; missing means unavailable, not zero.',
                          'CPU includes observer overhead; observer thread CPU is recorded. Timeout CPU ratios are censored.',
                          'No optimum proof for these 256-activity business instances.',
                          'T1->T2 jointly adds mandatory retrofit histories and invalidation; it cannot separate their individual effects.',
                          'A 2x resource flag is an exploratory effect threshold, not statistical significance.']}
    csv_file(root / 'results.csv', rows)
    csv_file(root / 'factor-comparisons.csv', comparisons)
    csv_file(root / 'aggregates.csv', aggregates)
    csv_file(root / 'factor-summary.csv', effects)
    write_json(root / 'summary.json', summary)
    write_json(root / 'state-count-evidence.json', evidence)
    lines = ['# 256 候选活动业务实验', '', f'状态：{state["status"]}；完成 {summary["completed_runs"]}/{summary["planned_runs"]} 次。', '',
             '三引擎默认搜索参数冻结；24 个条件 × 2 个独立生成种子 × 2 档预算 × 3 引擎。', '',
             '|预算秒|引擎|条件|完成|成功|CPU 中位秒|峰值内存中位 MiB|成功首解中位秒|', '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):
        return '—' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)
    for a in aggregates:
        lines.append('|' + '|'.join(fmt(a[k]) for k in ('budget_seconds', 'engine', 'condition', 'completed', 'successes',
                                                       'cpu_seconds_median', 'peak_memory_mib_median', 'first_seconds_median')) + '|')
    lines += ['', '## 成对因素变化', '',
              '以下只列两个生成实例均出现同方向高增长或均失去首解的条件。缺少配对、未完成或仅一个实例的变化不列为重复观察。', '',
              '|因素|引擎|预算|变化后宽度/密度/条件|内存倍数中位|首解倍数中位|失去首解实例数|',
              '|---|---|---:|---|---:|---:|---:|']
    repeated = [e for e in effects if e['paired_instances'] == 2 and max(e['memory_growth_instances'], e['first_solution_growth_instances'], e['lost_solution_instances']) == 2]
    for e in repeated:
        lines.append('|'+ '|'.join(fmt(v) for v in (e['factor'], e['engine'], e['budget_seconds'],
                       f'{e["after_width"]}/{e["after_density"]}/{e["after_condition"]}',
                       e['peak_memory_mib_ratio_median'], e['first_seconds_ratio_median'], e['lost_solution_instances']))+'|')
    if not repeated:
        lines.append('|尚无满足上述重复观察条件的完整配对|||||||')
    lines += ['', '全部配对（含持平、下降和单个实例变化）保存在 factor-summary.csv 与 factor-comparisons.csv，避免只展示增长。',
              '', '## 对照与解释', '',
              '完整逐项对照见 factor-comparisons.csv；逐次原始记录见 results.csv。只比较同引擎、同预算、同生成种子的成对模型。',
              '高增长标记：内存至少 2 倍且增加至少 128 MiB；成功首解时间至少 2 倍；或原条件成功、新条件没有候选。',
              '固定预算下 CPU 接近预算不证明计算成本相同；结合节点吞吐、首个解码进度和候选数量解释。',
              'A* 的 labels 是保留的业务状态键，nodes 是存储的搜索节点；GA 的 seen 是当前单条 rollout 的去重状态；POP 的实例及开放前置不等同于状态空间大小。', '',
              '## 局限', '', *['- '+s for s in summary['limits']]]
    (root / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Self-contained, readable incremental report. Data links stay local.
    table = '<table><thead><tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in ('预算','引擎','条件','完成','成功','CPU 秒','内存 MiB','首解秒')) + '</tr></thead><tbody>'
    for a in aggregates:
        table += '<tr>'+''.join(f'<td>{html.escape(fmt(v))}</td>' for v in a.values())+'</tr>'
    table += '</tbody></table>'
    (root / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>256 候选活动业务实验</title><style>body{font:16px system-ui;margin:24px;line-height:1.6}table{border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #aaa;text-align:left}.scroll{overflow:auto}</style>'
        f'<h1>256 候选活动业务实验</h1><p>{html.escape(state["status"])} · {summary["completed_runs"]}/{summary["planned_runs"]}</p>'
        '<p><a href="results.csv">逐次结果</a> · <a href="factor-comparisons.csv">因素配对比较</a> · <a href="complexity.csv">实际模型复杂度</a></p>'
        '<div class="scroll">'+table+'</div><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in summary['limits'])+'</ul></html>', encoding='utf-8')
    return summary
