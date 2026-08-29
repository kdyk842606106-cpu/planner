from __future__ import annotations

import json
import multiprocessing as mp
import platform
import queue
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .astar import AnytimeAStar
from .ga import GeneticExplorer
from .gantt import GANTT_RENDERER_VERSION, write_gantt_artifacts
from .models import Budget, EngineResult, Scenario, to_primitive
from .report import (
    benchmark_markdown,
    build_benchmark_summary,
    build_comparison,
    comparison_markdown,
)
from .scenario import scenario_hash
from .validator import ResultValidator


def _worker(
    engine_name: str,
    scenario: Scenario,
    budget: Budget,
    run_id: str,
    seed: int,
    output_queue,
    force_failure: str | None,
) -> None:
    try:
        if force_failure == engine_name:
            raise RuntimeError(f"Forced {engine_name} failure")
        if engine_name == "ASTAR":
            engine = AnytimeAStar(scenario)
            result = engine.run(scenario, budget, run_id=run_id, seed=None)
        elif engine_name == "GA":
            engine = GeneticExplorer(scenario)
            result = engine.run(scenario, budget, run_id=run_id, seed=seed)
        else:
            raise ValueError(f"Unknown engine {engine_name}")
        output_queue.put((engine_name, result, None))
    except BaseException as exc:  # worker boundary must report all failures
        output_queue.put((engine_name, None, f"{exc}\n{traceback.format_exc()}"))


def run_compare(
    scenario: Scenario,
    budget: Budget,
    *,
    seed: int,
    output_dir: str | Path,
    run_id: str | None = None,
    force_failure: str | None = None,
) -> dict[str, Any]:
    actual_run_id = run_id or _run_id(scenario.id, seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = {
        name: context.Process(
            target=_worker,
            args=(name, scenario, budget, actual_run_id, seed, result_queue, force_failure),
            name=f"planner-{name.lower()}-{actual_run_id}",
        )
        for name in ("ASTAR", "GA")
    }
    started = time.perf_counter()
    for process in processes.values():
        process.start()

    received: dict[str, EngineResult] = {}
    errors: dict[str, str] = {}
    parent_deadline = started + budget.time_limit_seconds + 10.0
    while len(received) + len(errors) < 2 and time.perf_counter() < parent_deadline:
        try:
            name, result, error = result_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if error:
            errors[name] = error
        elif result is not None:
            received[name] = result

    for name, process in processes.items():
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            errors.setdefault(name, "Worker exceeded parent deadline and was terminated")

    results = {
        name: _validated_result(
            received.get(name) or EngineResult(name, actual_run_id, seed if name == "GA" else None, "ERROR", error=errors.get(name, "Worker returned no result")),
            scenario,
        )
        for name in ("ASTAR", "GA")
    }
    comparison = build_comparison(
        results["ASTAR"],
        results["GA"],
        scenario_id=scenario.id,
        run_id=actual_run_id,
        provenance=scenario.provenance_dict,
    )
    comparison["astar"]["paths"] = [to_primitive(path) for path in results["ASTAR"].paths]
    comparison["ga"]["paths"] = [to_primitive(path) for path in results["GA"].paths]
    comparison["gantt"] = write_gantt_artifacts(scenario, results, destination)
    manifest = {
        "schema_version": 4,
        "run_id": actual_run_id,
        "scenario_id": scenario.id,
        "scenario_hash": scenario_hash(scenario),
        "scenario_provenance": scenario.provenance_dict,
        "execution_mode": scenario.execution_mode,
        "seed": seed,
        "budget": to_primitive(budget),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": datetime.now().astimezone().isoformat(),
        "engines_parallel": True,
        "engines_share_mutable_state": False,
        "gantt_renderer_version": GANTT_RENDERER_VERSION,
        "gantt_files": [item["svg_file"] for item in comparison["gantt"]],
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "astar_result.json", to_primitive(results["ASTAR"]))
    _write_json(destination / "ga_result.json", to_primitive(results["GA"]))
    _write_json(destination / "comparison.json", comparison)
    (destination / "report.md").write_text(comparison_markdown(comparison), encoding="utf-8")
    return comparison


def run_benchmark(
    scenario: Scenario,
    budget: Budget,
    *,
    seeds: list[int],
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for seed in seeds:
        run_id = _run_id(scenario.id, seed)
        comparisons.append(
            run_compare(
                scenario,
                budget,
                seed=seed,
                output_dir=destination / run_id,
                run_id=run_id,
            )
        )
    summary = build_benchmark_summary(comparisons, seeds)
    _write_json(destination / "benchmark.json", summary)
    (destination / "report.md").write_text(benchmark_markdown(summary, scenario.id), encoding="utf-8")
    return summary


def _validated_result(result: EngineResult, scenario: Scenario) -> EngineResult:
    validator = ResultValidator(scenario)
    valid_paths = []
    rejected = []
    for candidate in result.paths:
        validation = validator.validate(candidate)
        if validation.valid and validation.candidate is not None:
            valid_paths.append(validation.candidate)
        else:
            rejected.append({"path_id": candidate.path_id, "code": validation.code, "message": validation.message})
    stats = dict(result.stats)
    stats["validator_accepted"] = len(valid_paths)
    stats["validator_rejected"] = len(rejected)
    diagnosis = dict(result.diagnosis)
    if rejected:
        diagnosis["validator_rejections"] = rejected
    status = result.status
    if result.paths and not valid_paths:
        status = "ERROR"
    return replace(result, paths=tuple(valid_paths), stats=stats, diagnosis=diagnosis, status=status)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _run_id(scenario_id: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{scenario_id}-{stamp}-s{seed}"
