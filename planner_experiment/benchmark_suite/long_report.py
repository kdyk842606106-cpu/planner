"""Paired observations against frozen ten-second results, without pooling phases."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io import read_events, read_json
from .runner import block_key


def classify_change(before, after, *, executed=True):
    if not executed:
        return "UNRUN"
    if before is None:
        return "NEW_SOLUTION" if after is not None else "STILL_NO_SOLUTION"
    if after is None:
        return "LOST_SOLUTION"
    return "IMPROVED" if after < before else "UNCHANGED" if after == before else "WORSENED"


def comparison_rows(output, rows, answers):
    output = Path(output)
    blocks = read_json(output / "schedule.json")
    phases = {(b["instance_id"], b["seconds"], b["seed"]): b.get("phase") for b in blocks}
    metric_cache = {}

    def metrics(directory):
        key = str(directory)
        if key not in metric_cache:
            path = directory / "best_candidate.json"
            if path.exists():
                candidate = read_json(path)
                ids = [e["activity_id"] for e in candidate["executions"]]
                metric_cache[key] = {"execution_count": len(ids), "repeat_count": len(ids) - len(set(ids)),
                                     "non_causal_count": candidate["metrics"]["non_causal_action_count"]}
            else:
                metric_cache[key] = {"execution_count": None, "repeat_count": None, "non_causal_count": None}
        return metric_cache[key]

    result = []
    baseline_cache = {}
    for row in rows:
        iid, engine = row["instance_id"], row["engine"]
        reference = output / "reference" / iid / engine
        key = (iid, engine)
        if key not in baseline_cache:
            baseline_cache[key] = read_json(reference / "run.json")
        before = baseline_cache[key]
        block = {"instance_id": iid, "seconds": row["budget_seconds"], "seed": row["seed"]}
        directory = output / "runs" / block_key(block) / engine
        after = read_json(directory / "run.json") if row["executed"] else None
        new_makespan = after["best_makespan"] if after else None
        old_makespan = before["best_makespan"]
        old_metrics, new_metrics = metrics(reference), metrics(directory)
        optimum = answers[iid].get("optimal_makespan") if answers[iid]["status"] == "PROVEN_OPTIMAL" else None
        expected_hash = (before.get("ready") or {}).get("shared_input_hash")
        actual_hash = ((after or {}).get("ready") or {}).get("shared_input_hash")
        if expected_hash and actual_hash and expected_hash != actual_hash:
            raise ValueError(f"Baseline/current input hashes differ: {iid}/{engine}")
        events = [e for e in read_events(directory / "candidates.jsonl")
                  if row["executed"] and e["valid"] and e["elapsed_seconds"] <= row["budget_seconds"]]
        first_event = min(events, key=lambda e: e["elapsed_seconds"], default=None)
        stats = (after or {}).get("engine_stats", {})
        seed = stats.get("maximal_start_seed") or {}
        candidate_path = directory / "best_candidate.json"
        final_actions = read_json(candidate_path)["actions"] if candidate_path.exists() else None
        first_equals_final = first_event["actions"] == final_actions if first_event and final_actions is not None else None
        entry = {"phase": phases[(iid, row["budget_seconds"], row["seed"])],
                 "instance_id": iid, "group": row["group"], "scale": row["scale"], "engine": engine,
                 "seed": row["seed"], "budget_seconds": row["budget_seconds"], "executed": row["executed"],
                 "baseline_budget_seconds": 10, "baseline_makespan": old_makespan, "new_makespan": new_makespan,
                 "baseline_first_solution_seconds": before["first_solution_seconds"],
                 "new_first_solution_seconds": (after or {}).get("first_solution_seconds"),
                 "change": classify_change(old_makespan, new_makespan, executed=row["executed"]),
                 "makespan_delta": new_makespan - old_makespan if new_makespan is not None and old_makespan is not None else None,
                 "relative_makespan_delta": (new_makespan - old_makespan) / old_makespan if new_makespan is not None and old_makespan else None,
                 "optimal_makespan": optimum, "baseline_optimal": old_makespan == optimum if optimum is not None else None,
                 "new_optimal": new_makespan == optimum if optimum is not None and row["executed"] else None,
                 "first_optimal_seconds": row.get("first_optimal_seconds"),
                 "first_candidate_equals_final": first_equals_final,
                 "seed_makespan_equals_final": seed.get("makespan") == new_makespan if seed.get("valid") and new_makespan is not None else None,
                 "seed_equals_final": first_equals_final if seed.get("valid") and first_event and first_event["makespan"] == seed.get("makespan") else None,
                 "stop_reason": stats.get("stopped_reason") or (after or {}).get("engine_status") or (after or {}).get("process_status"),
                 "input_hash_match": expected_hash == actual_hash if expected_hash and actual_hash else None}
        for metric in ("execution_count", "repeat_count", "non_causal_count"):
            entry[f"baseline_{metric}"] = old_metrics[metric]
            entry[f"new_{metric}"] = new_metrics[metric]
            entry[f"{metric}_delta"] = new_metrics[metric] - old_metrics[metric] if new_metrics[metric] is not None and old_metrics[metric] is not None else None
        result.append(entry)
    phases_summary = []
    for phase in dict.fromkeys(b.get("phase") for b in blocks):
        selected = [r for r in result if r["phase"] == phase]
        phases_summary.append({"phase": phase, "planned_runs": len(selected),
                               "executed_runs": sum(r["executed"] for r in selected),
                               "model_count": len({r["instance_id"] for r in selected}),
                               "budget_seconds": sorted({r["budget_seconds"] for r in selected}),
                               "paired_outcomes": dict(Counter(r["change"] for r in selected if r["executed"]))})
    return {"comparisons": result, "phases": phases_summary,
            "note": "阶段A有18个模型；B/C各只有两个C0模型。每条件一次运行，不跨阶段混合排名。"}
