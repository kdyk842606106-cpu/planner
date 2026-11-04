"""Instance-first summaries and a dependency-free interactive HTML readout."""
from __future__ import annotations

import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_events, read_json, write_json
from .runner import block_key


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def relative_gap(value, reference):
    if value is None or reference in (None, 0):
        return None
    return (value - reference) / reference


def observation_checkpoints(seconds):
    return [t for t in sorted({0.1, 0.25, 0.5, 1, 3, 10, 30, 60, 120, 180, 240, 300, seconds}) if t <= seconds]


def describe_run(run, instance, answer, events):
    optimum = answer.get("optimal_makespan") if answer["status"] == "PROVEN_OPTIMAL" else None
    valid = [event for event in events if event["valid"] and event["elapsed_seconds"] <= run["budget_seconds"]]
    infeasible = answer["status"] == "PROVEN_INFEASIBLE"
    engine_proof = run.get("engine_status") == "PROVEN_INFEASIBLE"
    if infeasible:
        verdict = "FALSE_FEASIBLE" if valid else "CORRECT_INFEASIBLE" if engine_proof else "UNKNOWN_INFEASIBLE"
    elif engine_proof:
        verdict = "FALSE_INFEASIBLE"
    else:
        verdict = "FEASIBLE" if valid else "NO_VALID_SOLUTION"
    best = min((event["makespan"] for event in valid), default=None)
    first = min((event["elapsed_seconds"] for event in valid), default=None)
    first_optimal = min((event["elapsed_seconds"] for event in valid if event["makespan"] == optimum), default=None)
    return {"instance_id": run["instance_id"], "scale": instance["metadata"]["scale"],
            "activity_count": instance["complexity"]["activity_count"], "group": instance["metadata"]["group"],
            "generator_seed": instance["metadata"].get("generator_seed"), "engine": run["engine"],
            "budget_seconds": run["budget_seconds"], "seed": run["seed"], "executed": True,
            "process_status": run["process_status"], "engine_status": run.get("engine_status"), "verdict": verdict,
            "answer_status": answer["status"], "success": bool(valid) if not infeasible else None,
            "first_solution_seconds": first, "par2_seconds": first if first is not None else 2 * run["budget_seconds"] if not infeasible else None,
            "best_makespan": best, "optimal_makespan": optimum, "lower_bound": answer.get("lower_bound"),
            "optimal_hit": first_optimal is not None if optimum is not None else None,
            "first_optimal_seconds": first_optimal, "optimal_gap": relative_gap(best, optimum),
            "optimal_absolute_gap": best - optimum if best is not None and optimum is not None else None,
            "lower_bound_gap": relative_gap(best, answer.get("lower_bound")),
            "rejected_candidate_count": run.get("rejected_candidate_count", 0), "valid_candidate_count": len(valid),
            "candidate_rejection_rate": run.get("rejected_candidate_count", 0) / (len(valid) + run.get("rejected_candidate_count", 0))
                                        if len(valid) + run.get("rejected_candidate_count", 0) else None,
            "peak_memory_mib": run["peak_working_set_bytes"] / 2**20 if run.get("peak_working_set_bytes") else None,
            "startup_seconds": run.get("startup_seconds"), "call_wall_seconds": run["call_wall_seconds"],
            "exception": run["process_status"] not in ("COMPLETED", "TIME_LIMIT"),
            "diagnosis": json.dumps(run.get("diagnosis", {}), ensure_ascii=False),
            "stop_reason": json.dumps(run.get("engine_stats", {}), ensure_ascii=False)}


def aggregate(rows):
    """First aggregate repeated runs within each independently generated input."""
    by_instance = defaultdict(list)
    for row in rows:
        if row["executed"]:
            by_instance[(row["instance_id"], row["engine"], row["budget_seconds"])].append(row)
    instances = []
    for (iid, engine, seconds), values in sorted(by_instance.items()):
        item = {"instance_id": iid, "engine": engine, "budget_seconds": seconds,
                "scale": values[0]["scale"], "group": values[0]["group"], "executed_repetitions": len(values),
                "success_rate": mean([float(v["success"]) if v["success"] is not None else None for v in values]),
                "first_solution_median_seconds": median([v["first_solution_seconds"] for v in values]),
                "par2_seconds": mean([v["par2_seconds"] for v in values]),
                "optimal_hit_rate": mean([float(v["optimal_hit"]) if v["optimal_hit"] is not None else None for v in values]),
                "first_optimal_median_seconds": median([v["first_optimal_seconds"] for v in values]),
                "makespan_median": median([v["best_makespan"] for v in values]),
                "optimal_gap": mean([v["optimal_gap"] for v in values]),
                "best_known_gap": mean([v.get("best_known_gap") for v in values]),
                "lower_bound_gap": mean([v["lower_bound_gap"] for v in values]),
                "peak_memory_mib": median([v["peak_memory_mib"] for v in values]),
                "exception_rate": mean([float(v["exception"]) for v in values]),
                "candidate_rejection_rate": mean([v["candidate_rejection_rate"] for v in values]),
                "verdicts": dict(Counter(v["verdict"] for v in values))}
        instances.append(item)
    cells = defaultdict(list)
    for item in instances:
        cells[(item["group"], item["scale"], item["engine"], item["budget_seconds"])].append(item)
    summaries = []
    metrics = ["success_rate", "first_solution_median_seconds", "par2_seconds", "optimal_hit_rate",
               "first_optimal_median_seconds", "makespan_median", "optimal_gap", "best_known_gap", "lower_bound_gap",
               "peak_memory_mib", "exception_rate", "candidate_rejection_rate"]
    for (group, scale, engine, seconds), items in sorted(cells.items()):
        summaries.append({"group": group, "scale": scale, "engine": engine, "budget_seconds": seconds,
                          "independent_instances": len(items), "executed_repetitions": sum(i["executed_repetitions"] for i in items),
                          **{metric: mean([i[metric] for i in items]) for metric in metrics}})
    return instances, summaries


def csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def findings(cells):
    """Only rank groups with all six scales and all planned repetitions."""
    statements = []
    for group in ("C0", "C1", "C2", "C3", "C4", "C5", "C6"):
        for seconds in (1, 3, 10):
            selected = [c for c in cells if c["group"] == group and c["budget_seconds"] == seconds]
            if len(selected) != 18 or any(c["independent_instances"] != 3 or c["executed_repetitions"] != 9 for c in selected):
                continue
            scores = {e: {metric: mean([c[metric] for c in selected if c["engine"] == e])
                          for metric in ("success_rate", "par2_seconds", "optimal_hit_rate")}
                      for e in ("LEGACY", "ASTAR", "GA")}
            leaders = {}
            for metric, maximize in (("success_rate", True), ("par2_seconds", False), ("optimal_hit_rate", True)):
                available = {e: s[metric] for e, s in scores.items() if s[metric] is not None}
                if available:
                    best = (max if maximize else min)(available.values())
                    leaders[metric] = {"engines": [e for e, value in available.items() if abs(value - best) < 1e-12], "value": best}
            statements.append({"group": group, "budget_seconds": seconds, "independent_instances": 18,
                               "scores": scores, "leaders": leaders})
    return statements


def build_report(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    config_path = output / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    schedule = read_json(output / "schedule.json")
    inputs = {iid: read_json(output / "instances" / f"{iid}.json") for iid in manifest["instance_ids"]}
    answers = {iid: item["answer"] for iid, item in inputs.items()}
    for iid in inputs:
        path = output / "oracles" / iid / "answer.json"
        if path.exists():
            candidate = read_json(path)
            if candidate["status"] in ("PROVEN_OPTIMAL", "PROVEN_INFEASIBLE"):
                answers[iid] = candidate
    rows, events_by_key = [], {}
    for block in schedule:
        for engine in block["engines"]:
            directory = output / "runs" / block_key(block) / engine
            path = directory / "run.json"
            if path.exists():
                run = read_json(path)
                events = read_events(directory / "candidates.jsonl")
                events_by_key[(block["instance_id"], engine, block["seconds"], block["seed"])] = events
                rows.append(describe_run(run, inputs[block["instance_id"]], answers[block["instance_id"]], events))
            else:
                item = inputs[block["instance_id"]]
                rows.append({"instance_id": block["instance_id"], "scale": item["metadata"]["scale"],
                             "group": item["metadata"]["group"], "engine": engine, "budget_seconds": block["seconds"],
                             "seed": block["seed"], "executed": False, "process_status": "UNRUN"})
    best_known = {}
    if config.get("profile") == "long-quick":
        for iid in inputs:
            for engine in ("LEGACY", "ASTAR", "GA"):
                path = output / "reference" / iid / engine / "run.json"
                if path.exists():
                    best = read_json(path)["best_makespan"]
                    if best is not None:
                        best_known[iid] = min(best_known.get(iid, float("inf")), best)
    for row in rows:
        if row.get("best_makespan") is not None:
            iid = row["instance_id"]
            best_known[iid] = min(best_known.get(iid, float("inf")), row["best_makespan"])
    for row in rows:
        if row["executed"]:
            row["best_known_makespan"] = best_known.get(row["instance_id"])
            row["best_known_gap"] = relative_gap(row["best_makespan"], row["best_known_makespan"])
    instance_stats, cells = aggregate(rows)
    by_instance = {(i["instance_id"], i["engine"], i["budget_seconds"]): i for i in instance_stats}
    pairs = []
    for item in instance_stats:
        iid = item["instance_id"]
        base = inputs[iid]["metadata"].get("paired_baseline")
        baseline = by_instance.get((base, item["engine"], item["budget_seconds"]))
        if baseline:
            pairs.append({"instance_id": iid, "baseline_instance_id": base, "engine": item["engine"],
                          "budget_seconds": item["budget_seconds"], "scale": item["scale"],
                          "baseline_activity_count": inputs[base]["complexity"]["activity_count"],
                          "alternative_activity_count": inputs[iid]["complexity"]["activity_count"],
                          "success_rate_delta": item["success_rate"] - baseline["success_rate"],
                          "par2_seconds_delta": item["par2_seconds"] - baseline["par2_seconds"],
                          "makespan_delta": item["makespan_median"] - baseline["makespan_median"]
                                            if item["makespan_median"] is not None and baseline["makespan_median"] is not None else None})
    # Curves are also instance-first. Quality is conditional on an available
    # solution, and coverage is emitted alongside it to expose survivor bias.
    curve_inputs = defaultdict(list)
    for row in rows:
        if not row["executed"] or row["group"] == "GOLDEN":
            continue
        iid, seconds = row["instance_id"], row["budget_seconds"]
        reference = answers[iid].get("optimal_makespan", best_known.get(iid))
        for checkpoint in observation_checkpoints(seconds):
            if checkpoint > seconds:
                continue
            found = [e["makespan"] for e in events_by_key[(iid, row["engine"], seconds, row["seed"])]
                     if e["valid"] and e["elapsed_seconds"] <= checkpoint]
            key = (iid, row["engine"], seconds, checkpoint)
            curve_inputs[key].append({"available": float(bool(found)),
                                      "gap": relative_gap(min(found) if found else None, reference)})
    curves_by_cell = defaultdict(list)
    for (iid, engine, seconds, checkpoint), values in curve_inputs.items():
        metadata = inputs[iid]["metadata"]
        curves_by_cell[(metadata["group"], metadata["scale"], engine, seconds, checkpoint)].append(
            {"coverage": mean([v["available"] for v in values]), "gap": mean([v["gap"] for v in values])})
    curves = [{"group": key[0], "scale": key[1], "engine": key[2], "budget_seconds": key[3], "seconds": key[4],
               "coverage": mean([v["coverage"] for v in values]), "gap": mean([v["gap"] for v in values])}
              for key, values in sorted(curves_by_cell.items())]
    executed = [r for r in rows if r["executed"]]
    input_hashes = defaultdict(set)
    for block in schedule:
        for engine in block["engines"]:
            run_path = output / "runs" / block_key(block) / engine / "run.json"
            if run_path.exists():
                ready = read_json(run_path).get("ready") or {}
                if ready.get("shared_input_hash"):
                    input_hashes[block["instance_id"]].add(ready["shared_input_hash"])
    summary = {"planned_runs": len(rows), "executed_runs": len(executed), "unrun_runs": len(rows) - len(executed),
               "answer_statuses": dict(Counter(a["status"] for a in answers.values())),
               "process_statuses": dict(Counter(r["process_status"] for r in executed)),
               "verdicts": dict(Counter(r["verdict"] for r in executed)),
               "rejected_unique_candidates": sum(r["rejected_candidate_count"] for r in executed),
               "proof_violations": [r for r in executed if r["verdict"] in ("FALSE_FEASIBLE", "FALSE_INFEASIBLE")
                                    or (r["optimal_absolute_gap"] is not None and r["optimal_absolute_gap"] < 0)],
               "input_hash_conflicts": {iid: sorted(values) for iid, values in input_hashes.items() if len(values) > 1},
               "findings": findings(cells),
               "cells": cells, "instances": instance_stats, "provider_pairs": pairs, "curves": curves}
    summary["profile"] = config.get("profile", "standard")
    summary["budget_options"] = sorted({b["seconds"] for b in schedule})
    summary["planned_cells"] = [{"group": inputs[b["instance_id"]]["metadata"]["group"],
                                  "scale": inputs[b["instance_id"]]["metadata"]["scale"],
                                  "budget_seconds": b["seconds"]} for b in schedule]
    if config.get("profile") == "long-quick":
        from .long_report import comparison_rows
        summary["long_quick"] = comparison_rows(output, rows, answers)
        summary["findings"] = []
        csv_file(output / "baseline_comparison.csv", summary["long_quick"]["comparisons"])
    write_json(output / "summary.json", summary)
    csv_file(output / "runs.csv", rows)
    csv_file(output / "instances.csv", instance_stats)
    csv_file(output / "cells.csv", cells)
    csv_file(output / "provider_pairs.csv", pairs)
    csv_file(output / "complexity.csv", [{"instance_id": iid, **item["metadata"], **item["complexity"]} for iid, item in inputs.items()])
    csv_file(output / "answers.csv", [{"instance_id": iid, **{k: v for k, v in a.items() if k != "actions"}} for iid, a in answers.items()])
    state = read_json(output / "batch_state.json") if (output / "batch_state.json").exists() else {"status": "NOT_STARTED"}
    report = HTML.replace("__DATA__", json.dumps(summary, ensure_ascii=False).replace("</", "<\\/"))
    status_label = state["status"] + (f" · 阶段 {state['phase']}" if state.get("phase") else "")
    if config.get("profile") == "long-quick":
        remaining = max(0, state.get("budget_seconds", 4200) - state.get("consumed_seconds", 0))
        status_label += f" · 批次剩余上限 {remaining/60:.1f} 分钟"
    report = report.replace("__STATUS__", html.escape(status_label))
    report = report.replace("__COVERAGE__", f"{len(executed)} / {len(rows)}")
    (output / "report.html").write_text(report, encoding="utf-8")
    lines = ["# 三算法实验报告", "", f"状态：{state['status']}；完成 {len(executed)}/{len(rows)} 次正式运行。", "",
             "统计按独立实例先汇总重复，再在同组内等权平均。未运行项只计覆盖率，不视为算法失败。", "",
             f"唯一候选校验拒绝数：{summary['rejected_unique_candidates']}；答案矛盾数：{len(summary['proof_violations'])}。", "",
             "交互图表与分组结果见 [report.html](report.html)；原始逐次数据见 [runs.csv](runs.csv)。", ""]
    for finding in summary["findings"]:
        leaders = finding["leaders"]
        success = leaders.get("success_rate")
        speed = leaders.get("par2_seconds")
        if success and speed:
            lines.append(f"- {finding['group']}，{finding['budget_seconds']} 秒预算：合法解成功率最高为 {'/'.join(success['engines'])}（{success['value']:.1%}）；首解 PAR-2 最低为 {'/'.join(speed['engines'])}（{speed['value']:.3f} 秒）。")
    if not summary["findings"]:
        lines.append("快速验证：各阶段样本不同，每条件一次运行，不作总体排名。" if config.get("profile") == "long-quick"
                     else "尚无覆盖完整六档规模和全部重复的分组，不作算法排名。")
    if summary.get("long_quick"):
        lines.extend(["", summary["long_quick"]["note"], "", "与冻结十秒结果的配对比较见 [baseline_comparison.csv](baseline_comparison.csv)。", "",
                      "| 阶段 | 预算秒 | 模型数 | 已运行/计划 | 配对变化 |", "|---|---:|---:|---:|---|"])
        for phase in summary["long_quick"]["phases"]:
            lines.append(f"| {phase['phase']} | {phase['budget_seconds']} | {phase['model_count']} | {phase['executed_runs']}/{phase['planned_runs']} | {phase['paired_outcomes']} |")
    lines.extend(["",
             "| 分组 | N | 引擎 | 预算秒 | 独立实例 | 已运行重复 | 合法解成功率 | PAR-2 秒 | 最优命中率 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
                 )
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    for cell in cells:
        lines.append(f"| {cell['group']} | {cell['scale']} | {cell['engine']} | {cell['budget_seconds']} | {cell['independent_instances']} | {cell['executed_repetitions']} | {fmt(cell['success_rate'])} | {fmt(cell['par2_seconds'])} | {fmt(cell['optimal_hit_rate'])} |")
    lines.extend(["", "各格独立实例数和重复数见表；结果是探索性比较。成功样本的首解时间和工期差距均需结合成功率阅读。",
                  "精确枚举未完成时保留 UNKNOWN；已知最好值来自已完成运行及本配置引用的冻结基线，是回顾性参考，不是最优证明。",
                  "内存为外部 OS 进程峰值工作集，包含模块加载；校验和候选观测开销包含在求解预算中。"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三算法实验</title>
<style>body{font:15px system-ui,sans-serif;color:#203049;background:#f4f6fa;margin:30px auto;max-width:1250px;padding:0 20px}h1{font-size:30px}section{background:white;border-radius:12px;padding:22px;margin:16px 0}select{padding:7px;margin:5px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:7px;border-bottom:1px solid #e4e8ed}th:first-child,td:first-child{text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.muted{color:#64748b}.legend{display:flex;gap:20px}.note{line-height:1.8}a{color:#2457bc}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>三种算法的正确性与求解性能</h1><p>状态：__STATUS__ · 正式运行覆盖：__COVERAGE__</p>
<p><a href="runs.csv">逐次结果</a> · <a href="cells.csv">分组汇总</a> · <a href="complexity.csv">复杂度向量</a> · <a href="answers.csv">答案与证明状态</a> · <a href="provider_pairs.csv">提供者配对</a> · <a href="manifest.json">冻结清单</a></p>
<section><label>分组 <select id="group"></select></label><label>预算 <select id="budget"></select></label><label>曲线规模 <select id="scale"></select></label><div class="legend"><span style="color:#2563eb">● LEGACY</span><span style="color:#e38420">● ASTAR</span><span style="color:#138579">● GA</span></div></section>
<div class="grid"><section><h3>合法解成功率随规模变化</h3><div id="success"></div></section><section><h3>首解 PAR-2（秒）</h3><div id="time"></div></section><section><h3>最终工期相对已知最好值的差距</h3><div id="quality"></div></section><section><h3>峰值进程工作集（MiB）</h3><div id="memory"></div></section><section><h3>最好工期差距随时间变化</h3><div id="curve"></div></section><section><h3>截止各时刻的合法解覆盖率</h3><div id="coverage"></div></section></div>
<section><h3>当前分组结果</h3><div id="table"></div></section>
<section id="long-comparison" hidden><h3>与冻结十秒结果配对</h3><p>阶段A有18个模型，B/C各有两个C0模型；每条件一次运行，不跨阶段混合排名。无最优证明时仅报告工期变化。</p><p><a href="baseline_comparison.csv">下载配对结果</a></p><div id="long-phases"></div><div id="long-table" style="overflow:auto"></div></section>
<section><h3>覆盖完整的分组结论</h3><div id="findings"></div></section>
<section class="note"><strong>阅读口径</strong><p>先在同一输入内汇总重复运行，再对独立输入等权平均；具体实例数和重复数见表。未运行项单列，不视为失败。PAR-2 对预算内未获得合法解的运行计两倍预算。</p><p>工期差距和质量曲线仅对已有合法解的样本计算，请结合成功率和时间覆盖曲线阅读。质量曲线优先采用已证明最优值，否则使用已知最好值；后者是回顾性参考，不能证明最优。没有成功样本时显示空白，不补零。</p><p>C2 改变宽度和深度；C6 组合多个因素。PROVIDER 按基线任务规模标记，实际活动数增加约25%，单列配对结果。样本有限，不能据此断言高精度总体排名。输入复杂度不是计算难度分数。</p><p>资源、事件、动作及最终目标均经统一重放校验。内存来自外部 OS 进程峰值，包含模块加载。预算内仍失败，只能说明当前预算和原算法限制下未找到合法解。</p><p id="flags"></p></section>
<script>const DATA=__DATA__;const colors={LEGACY:'#2563eb',ASTAR:'#e38420',GA:'#138579'};
const labels={C0:'C0 参考',C1:'C1 依赖增强',C2:'C2 宽度/深度',C3:'C3 状态恢复',C4:'C4 资源竞争',C5:'C5 事件等待',C6:'C6 混合压力',PROVIDER:'多提供者配对',GOLDEN:'黄金用例'};
function options(id,values,display){document.getElementById(id).innerHTML=values.map(v=>`<option value="${v}">${display?display(v):v}</option>`).join('')}
const planned=DATA.planned_cells;options('group',Object.keys(labels).filter(g=>planned.some(p=>p.group===g)),v=>labels[v]);options('budget',DATA.budget_options,v=>v+' 秒');
function updateScales(){const group=document.getElementById('group').value,budget=+document.getElementById('budget').value,old=document.getElementById('scale').value;const scales=[...new Set(planned.filter(p=>p.group===group&&p.budget_seconds===budget).map(p=>p.scale))].sort((a,b)=>a-b);options('scale',scales);if(scales.includes(+old))document.getElementById('scale').value=old;}
function updateBudgets(){const group=document.getElementById('group').value,old=document.getElementById('budget').value;const budgets=[...new Set(planned.filter(p=>p.group===group).map(p=>p.budget_seconds))].sort((a,b)=>a-b);options('budget',budgets,v=>v+' 秒');if(budgets.includes(+old))document.getElementById('budget').value=old;updateScales();}
updateBudgets();
function chart(id,points,field,xfield,percent=false){let all=points.filter(p=>p[field]!=null);const w=500,h=260,l=58,r=18,t=15,b=42;
if(!all.length){document.getElementById(id).innerHTML='<p class="muted">尚无可用数据</p>';return}
let xs=[...new Set(points.map(p=>p[xfield]))].sort((a,b)=>a-b),max=Math.max(...all.map(p=>p[field]),percent?1:0.001),min=Math.min(0,...all.map(p=>p[field]));
const xp=x=>l+(Math.log(x)-Math.log(xs[0]))/(Math.log(xs[xs.length-1])-Math.log(xs[0])||1)*(w-l-r),yp=y=>t+(max-y)/(max-min||1)*(h-t-b);let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${id}">`;
for(let i=0;i<=4;i++){let y=min+(max-min)*i/4;svg+=`<line x1="${l}" y1="${yp(y)}" x2="${w-r}" y2="${yp(y)}" stroke="#e4e8ed"/><text x="${l-7}" y="${yp(y)+4}" text-anchor="end" font-size="11" fill="#64748b">${percent?(100*y).toFixed(0)+'%':y.toFixed(2)}</text>`}
for(let x of xs)svg+=`<text x="${xp(x)}" y="${h-21}" text-anchor="middle" font-size="11" fill="#64748b">${x}</text>`;
svg+=`<text x="${(l+w-r)/2}" y="${h-3}" text-anchor="middle" font-size="10" fill="#64748b">${xfield==='seconds'?'时间（秒，对数刻度）':'规模档位（对数刻度）'}</text>`;
for(let engine of Object.keys(colors)){let ps=all.filter(p=>p.engine===engine).sort((a,b)=>a[xfield]-b[xfield]);svg+=`<polyline fill="none" stroke="${colors[engine]}" stroke-width="2" points="${ps.map(p=>xp(p[xfield])+','+yp(p[field])).join(' ')}"/>`;for(let p of ps)svg+=`<circle cx="${xp(p[xfield])}" cy="${yp(p[field])}" r="3" fill="${colors[engine]}"><title>${engine} · ${p[xfield]}: ${p[field].toFixed(4)}</title></circle>`}document.getElementById(id).innerHTML=svg+'</svg>'}
function render(){let group=document.getElementById('group').value,budget=+document.getElementById('budget').value,scale=+document.getElementById('scale').value;
let cells=DATA.cells.filter(p=>p.group===group&&p.budget_seconds===budget);chart('success',cells,'success_rate','scale',true);chart('time',cells,'par2_seconds','scale');chart('quality',cells,'best_known_gap','scale');chart('memory',cells,'peak_memory_mib','scale');
let curves=DATA.curves.filter(p=>p.group===group&&p.budget_seconds===budget&&p.scale===scale);chart('curve',curves,'gap','seconds');chart('coverage',curves,'coverage','seconds',true);
let f=(v,p=false)=>v==null?'—':p?(100*v).toFixed(1)+'%':v.toFixed(3);document.getElementById('table').innerHTML='<table><tr><th>引擎 / N</th><th>实例 / 重复</th><th>成功率</th><th>PAR-2 秒</th><th>最优命中率</th><th>最优差距</th></tr>'+cells.map(p=>`<tr><td>${p.engine} / ${p.scale}</td><td>${p.independent_instances} / ${p.executed_repetitions}</td><td>${f(p.success_rate,true)}</td><td>${f(p.par2_seconds)}</td><td>${f(p.optimal_hit_rate,true)}</td><td>${f(p.optimal_gap,true)}</td></tr>`).join('')+'</table>';
document.getElementById('flags').textContent=`唯一候选校验拒绝数：${DATA.rejected_unique_candidates}；答案矛盾数：${DATA.proof_violations.length}；未运行：${DATA.unrun_runs}。`;}
const baseRender=render;render=function(){baseRender();if(DATA.long_quick){const group=document.getElementById('group').value,budget=+document.getElementById('budget').value;document.getElementById('long-comparison').hidden=false;document.getElementById('long-phases').textContent=DATA.long_quick.phases.map(p=>`阶段${p.phase}：${p.executed_runs}/${p.planned_runs}`).join(' · ');const changes={UNRUN:'未运行',NEW_SOLUTION:'新增合法解',STILL_NO_SOLUTION:'仍未找到',LOST_SOLUTION:'本轮未找到',IMPROVED:'工期改善',UNCHANGED:'工期持平',WORSENED:'工期变差'};const v=x=>x==null?'—':typeof x==='number'?Number.isInteger(x)?x:x.toFixed(3):String(x);document.getElementById('long-table').innerHTML='<table><tr><th>实例 / 引擎</th><th>十秒工期</th><th>本轮工期</th><th>变化</th><th>首解秒</th><th>首次最优秒</th><th>重复执行变化</th><th>非因果动作变化</th><th>仍为初始种子</th><th>停止原因</th></tr>'+DATA.long_quick.comparisons.filter(p=>p.group===group&&p.budget_seconds===budget).map(p=>`<tr><td>${p.instance_id} / ${p.engine}</td><td>${v(p.baseline_makespan)}</td><td>${v(p.new_makespan)}</td><td>${changes[p.change]}</td><td>${v(p.new_first_solution_seconds)}</td><td>${v(p.first_optimal_seconds)}</td><td>${v(p.repeat_count_delta)}</td><td>${v(p.non_causal_count_delta)}</td><td>${p.seed_equals_final==null?'—':p.seed_equals_final?'是':'否'}</td><td>${v(p.stop_reason)}</td></tr>`).join('')+'</table>';}};
document.getElementById('findings').textContent=DATA.profile==='long-quick'?'快速验证只做同模型配对，不作跨阶段总体排名。':DATA.findings.length?DATA.findings.map(f=>`${f.group} / ${f.budget_seconds}秒：合法解成功率最高 ${f.leaders.success_rate.engines.join('/')}（${(100*f.leaders.success_rate.value).toFixed(1)}%），首解PAR-2最低 ${f.leaders.par2_seconds.engines.join('/')}（${f.leaders.par2_seconds.value.toFixed(3)}秒）。`).join(' '):'尚无覆盖完整六档规模和全部重复的分组，不作排名。';
document.getElementById('group').onchange=()=>{updateBudgets();render()};document.getElementById('budget').onchange=()=>{updateScales();render()};document.getElementById('scale').onchange=render;render();</script></html>'''
