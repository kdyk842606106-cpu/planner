"""Importable spawn target; all observed candidates are independently replayed."""
from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from pathlib import Path

from .io import read_json, write_json
from .runtime import pin_one_cpu


def engine_worker(instance_path, engine, seconds, seed, solver_root, directory, pipe):
    directory = Path(directory)
    try:
        affinity = pin_one_cpu()
        sys.path.insert(0, solver_root)
        from app.services.planner_engine_bridge import _planner_modules, run_engine
        from app.services.planner_scenarios import scenario_hash, validate_scenario
        from ..models import to_primitive
        from ..scenario import scenario_from_dict
        from ..validator import ResultValidator

        instance = read_json(instance_path)
        payload = instance["scenario"]
        _planner_modules()
        issues = validate_scenario(copy.deepcopy(payload))
        if issues:
            raise ValueError(f"Shared preflight failed: {issues}")
        validator = ResultValidator(scenario_from_dict(payload))
        input_hash = scenario_hash(copy.deepcopy(payload))
        source_before = copy.deepcopy(payload)
        pipe.send(("ready", {"affinity": affinity, "shared_input_hash": input_hash}))
        if pipe.recv() != "go":
            return
        started = None
        seen = set()
        event_stream = (directory / "candidates.jsonl").open("a", encoding="utf-8", buffering=1)

        def on_start(stamp):
            nonlocal started
            started = stamp
            pipe.send(("started", stamp))

        def observe(candidate):
            # Exact content identity, not a trusted candidate-provided hash.
            key = (candidate.actions, candidate.executions, candidate.final_facts, candidate.final_state_ids,
                   candidate.metrics, candidate.structure_signature, candidate.schedule_signature,
                   candidate.causal_core_signature, candidate.provider_signature, candidate.strategy_signature,
                   candidate.raw_makespan, candidate.normalization_saved_time, candidate.schema_version)
            if key in seen:
                return
            checked = validator.validate(candidate)
            elapsed = time.perf_counter() - started
            if elapsed > seconds:
                return
            seen.add(key)
            event = {"elapsed_seconds": elapsed, "valid": checked.valid, "code": checked.code,
                     "path_id": candidate.path_id, "makespan": candidate.metrics.makespan,
                     "score": list(candidate.metrics.sort_key),
                     "actions": [to_primitive(action) for action in candidate.actions]}
            if not checked.valid:
                event["rejected_candidate"] = to_primitive(candidate)
            event_stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            event_stream.flush()

        try:
            result = run_engine(payload, engine=engine, run_id=directory.name, seed=seed,
                                budget_override={"time_limit_seconds": seconds, "transition_limit": 2**60, "max_solutions": 20},
                                run_options={"end_to_end_deadline": True, "worker_threads": 1,
                                             "trace_memory": False, "on_start": on_start}, candidate_observer=observe)
            if payload != source_before:
                raise AssertionError("Engine mutated its input")
            # Full plans are recoverable from streamed actions; avoid storing
            # quadratic state trajectories repeatedly for every archived path.
            result["paths"] = [{"path_id": p["path_id"], "metrics": p["metrics"],
                                "validator_status": p["validator_status"]} for p in result["paths"]]
            write_json(directory / "engine_result.json", result)
            pipe.send(("done", {"status": result["status"], "elapsed_seconds": time.perf_counter() - started}))
        finally:
            event_stream.close()
    except BaseException as exc:
        write_json(directory / "error.json", {"error": str(exc), "traceback": traceback.format_exc()})
        try:
            pipe.send(("error", str(exc)))
        except (OSError, EOFError):
            pass
    finally:
        pipe.close()


def oracle_worker(instance_path, seconds, directory, pipe):
    try:
        affinity = pin_one_cpu()
        from .oracle import exact_answer
        instance = read_json(instance_path)
        pipe.send(("ready", {"affinity": affinity}))
        if pipe.recv() == "go":
            started = time.perf_counter()
            pipe.send(("started", started))
            answer = exact_answer(instance["scenario"], seconds)
            write_json(Path(directory) / "answer.json", answer)
            pipe.send(("done", {"elapsed_seconds": time.perf_counter() - started}))
    except BaseException as exc:
        write_json(Path(directory) / "error.json", {"error": str(exc), "traceback": traceback.format_exc()})
        try:
            pipe.send(("error", str(exc)))
        except (OSError, EOFError):
            pass
    finally:
        pipe.close()
