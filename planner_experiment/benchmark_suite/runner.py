"""Frozen inputs, bounded fresh processes, balanced scheduling and resume."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import BUDGETS, ENGINES, GENERATOR_SEEDS, GROUPS, RUN_SEEDS, SCALES, SUITE_VERSION
from .generator import candidate_from_actions, make_instance
from .golden import golden_instances
from .io import digest, read_events, read_json, write_json
from .runtime import process_memory
from .worker import engine_worker, oracle_worker


def source_files(solver_root):
    planner = Path(__file__).resolve().parents[2]
    files = [(f"planner/{p.relative_to(planner).as_posix()}", p) for p in (planner / "planner_experiment").rglob("*.py")]
    solver = Path(solver_root)
    files.extend((f"solver/{p.relative_to(solver).as_posix()}", p) for p in (solver / "app").rglob("*.py"))
    return sorted(files)


def source_hashes(solver_root):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files(solver_root)}


def environment(solver_root):
    roots = [Path(__file__).resolve().parents[2], Path(solver_root)]
    git = {}
    for root in roots:
        def command(*args):
            completed = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root,
                                       capture_output=True, text=True, encoding="utf-8", errors="replace")
            if completed.returncode:
                return {"error": completed.stderr.strip()}
            return completed.stdout.strip()
        git[str(root)] = {"head": command("rev-parse", "HEAD"), "status": command("status", "--short")}
    try:
        ortools = importlib.metadata.version("ortools")
    except importlib.metadata.PackageNotFoundError:
        ortools = None
    return {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "ortools": ortools,
            "git": git, "source_hashes": source_hashes(solver_root)}


def block_key(block):
    return f"{block['instance_id']}-t{block['seconds']:g}-s{block['seed']}"


def planned_blocks(instance_ids):
    ids = set(instance_ids)
    blocks = []
    rng = random.Random(81173)
    for seed in RUN_SEEDS:
        for gid in sorted(i for i in ids if i.startswith("G")):
            engines = list(ENGINES)
            rng.shuffle(engines)
            blocks.append({"instance_id": gid, "seconds": 1.0, "seed": seed, "engines": engines})
    for run_seed in RUN_SEEDS:
        for seconds in BUDGETS:
            for generated_seed in GENERATOR_SEEDS:
                groups = list((*GROUPS, "PROVIDER"))
                rng.shuffle(groups)
                for group in groups:
                    for n in SCALES:
                        iid = f"n{n}-{group}-g{generated_seed}"
                        if iid in ids:
                            engines = list(ENGINES)
                            rng.shuffle(engines)
                            blocks.append({"instance_id": iid, "seconds": seconds, "seed": run_seed, "engines": engines})
    return blocks


def generate_suite(output, solver_root, *, profile="standard", baseline_dir=None):
    if profile == "long-quick":
        from .profiles import prepare_long_quick
        return prepare_long_quick(output, solver_root, baseline_dir, environment=environment, source_files=source_files)
    if profile != "standard":
        raise ValueError(f"Unknown experiment profile: {profile}")
    if baseline_dir is not None:
        raise ValueError("--baseline is only used with --profile long-quick")
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise ValueError("Output already contains a frozen suite; use run/report, or choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    ids, files = [], {}
    for n in SCALES:
        for seed in GENERATOR_SEEDS:
            for group in (*GROUPS, "PROVIDER"):
                item = make_instance(n, group, seed)
                iid = item["metadata"]["id"]
                write_json(output / "instances" / f"{iid}.json", item)
                ids.append(iid)
                files[iid] = digest(item)
        print(f"Generated and certified scale {n}", flush=True)
    for item in golden_instances():
        iid = item["metadata"]["id"]
        write_json(output / "instances" / f"{iid}.json", item)
        ids.append(iid)
        files[iid] = digest(item)
    config = {"suite_version": SUITE_VERSION, "scales": list(SCALES), "groups": list(GROUPS),
              "generator_seeds": list(GENERATOR_SEEDS), "run_seeds": list(RUN_SEEDS), "budgets_seconds": list(BUDGETS),
              "golden_budget_seconds": 1, "engines": list(ENGINES), "batch_limit_seconds": 28800,
              "oracle_limit_seconds": 1800, "oracle_per_instance_seconds": 90, "oracle_max_states": 200000,
              "transition_protection_limit": 2**60, "max_solutions": 20, "cp_sat_workers": 1,
              "memory_metric": "OS peak process working set including imports", "startup_timeout_seconds": 30,
              "candidate_observation": "unique exact-content candidates replayed synchronously; validation counts toward budget",
              "objective": "minimum_makespan", "business_scenarios": False}
    write_json(output / "config.json", config)
    blocks = planned_blocks(ids)
    write_json(output / "schedule.json", blocks)
    with zipfile.ZipFile(output / "source-snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in source_files(solver_root):
            archive.write(path, name)
    manifest = {"suite_version": SUITE_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "solver_root": str(Path(solver_root).resolve()), "instance_ids": ids, "instance_hashes": files,
                "config_hash": digest(config), "schedule_hash": digest(blocks), "planned_run_count": len(blocks) * 3,
                "environment": environment(solver_root)}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_frozen(output, solver_root):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest["environment"]["source_hashes"] != source_hashes(solver_root):
        raise ValueError("Source changed since suite generation; regenerate into a new output directory")
    if manifest["config_hash"] != digest(read_json(output / "config.json")):
        raise ValueError("Frozen configuration changed")
    if manifest["schedule_hash"] != digest(read_json(output / "schedule.json")):
        raise ValueError("Frozen schedule changed")
    for iid, expected in manifest["instance_hashes"].items():
        if digest(read_json(output / "instances" / f"{iid}.json")) != expected:
            raise ValueError(f"Frozen instance changed: {iid}")
    if manifest.get("reference_hashes"):
        from .profiles import file_hash
        for relative, expected in manifest["reference_hashes"].items():
            if file_hash(output / relative) != expected:
                raise ValueError(f"Frozen baseline evidence changed: {relative}")
    if manifest["environment"]["python_executable"] != sys.executable:
        raise ValueError("Use the same Python environment as generation")
    current = environment(solver_root)
    if (current["python"], current["ortools"]) != (manifest["environment"]["python"], manifest["environment"]["ortools"]):
        raise ValueError("Python or OR-Tools version changed")
    return manifest


def supervise(target, arguments, directory, seconds, *, startup_timeout=30, hard_deadline=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=target, args=(*arguments, str(directory), child))
    wall_start = time.perf_counter()
    process.start()
    child.close()
    started, ready, outcome, error = None, None, None, None
    peak = 0
    last_memory = None
    try:
        while True:
            now = time.perf_counter()
            memory = process_memory(process.pid)
            if memory:
                last_memory = memory
                peak = max(peak, memory["peak_working_set_bytes"])
            while True:
                try:
                    if not parent.poll():
                        break
                    kind, value = parent.recv()
                except (EOFError, OSError):
                    break
                if kind == "ready":
                    ready = value
                    parent.send("go")
                elif kind == "started":
                    started = value
                elif kind == "done":
                    outcome = value
                elif kind == "error":
                    error = value
            now = time.perf_counter()
            if outcome is not None or error is not None:
                break
            if hard_deadline is not None and now >= hard_deadline:
                error = "BATCH_HARD_DEADLINE"
                break
            if started is not None and now >= started + seconds:
                error = "TIME_LIMIT"
                break
            if started is None and now - wall_start >= startup_timeout:
                error = "STARTUP_TIMEOUT"
                break
            if not process.is_alive():
                error = "WORKER_EXITED_WITHOUT_RESULT"
                break
            time.sleep(0.01)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        parent.close()
    return {"process_status": "COMPLETED" if outcome is not None else error,
            "startup_seconds": None if started is None else started - wall_start,
            "call_wall_seconds": time.perf_counter() - wall_start,
            "measured_elapsed_seconds": None if started is None else min(seconds, time.perf_counter() - started),
            "peak_working_set_bytes": peak or None, "last_memory_sample": last_memory,
            "ready": ready, "outcome": outcome, "error": error, "exitcode": process.exitcode}


def run_one(output, solver_root, block, engine, *, hard_deadline=None, warmup=False):
    output = Path(output)
    directory = output / ("warmup" if warmup else "runs") / block_key(block) / engine
    if (directory / "run.json").exists():
        return read_json(directory / "run.json")
    instance_path = output / "instances" / f"{block['instance_id']}.json"
    # A process interrupted before run.json is not evidence of a completed run.
    # Preserve it under a numbered attempt rather than mixing event streams.
    if directory.exists() and any(directory.iterdir()):
        count = 1
        while directory.with_name(f"{engine}.interrupted-{count}").exists():
            count += 1
        directory.rename(directory.with_name(f"{engine}.interrupted-{count}"))
    run = supervise(engine_worker, (str(instance_path), engine, block["seconds"], block["seed"], str(solver_root)),
                    directory, block["seconds"], hard_deadline=hard_deadline)
    run.update(instance_id=block["instance_id"], engine=engine, budget_seconds=block["seconds"], seed=block["seed"], warmup=warmup)
    instance = read_json(instance_path)
    events = read_events(directory / "candidates.jsonl")
    valid = [e for e in events if e["valid"] and e["elapsed_seconds"] <= block["seconds"]]
    run["valid_candidate_count"] = len(valid)
    run["rejected_candidate_count"] = sum(not e["valid"] for e in events)
    run["first_solution_seconds"] = min((e["elapsed_seconds"] for e in valid), default=None)
    best = min(valid, key=lambda e: (e["makespan"], e["elapsed_seconds"]), default=None)
    run["best_makespan"] = best["makespan"] if best else None
    if best:
        # Reconstruct evidence after timing without crediting a late candidate.
        candidate = candidate_from_actions(instance["scenario"], best["actions"], algorithm=engine,
                                           run_id=block_key(block), seed=block["seed"] if engine == "GA" else None)
        if candidate.metrics.makespan != best["makespan"]:
            raise AssertionError("Streamed candidate differs from post-run replay")
        from ..models import to_primitive
        write_json(directory / "best_candidate.json", to_primitive(candidate))
    result_path = directory / "engine_result.json"
    if result_path.exists():
        raw = read_json(result_path)
        run["engine_status"] = raw["status"]
        run["engine_stats"] = raw["stats"]
        run["diagnosis"] = raw.get("diagnosis", {})
    else:
        run["engine_status"] = None
    write_json(directory / "run.json", run)
    return run


def run_oracles(output, hard_deadline, *, per_instance=90):
    output = Path(output)
    started = time.perf_counter()
    manifest = read_json(output / "manifest.json")
    selected = [iid for iid in manifest["instance_ids"]
                if iid.startswith(("n8-", "n16-")) and any(f"-{g}-" in iid for g in ("C3", "C4", "C6"))]
    for iid in selected:
        directory = output / "oracles" / iid
        if (directory / "run.json").exists():
            continue
        if time.perf_counter() + per_instance + 32 >= min(hard_deadline, started + 1800):
            break
        path = output / "instances" / f"{iid}.json"
        run = supervise(oracle_worker, (str(path), per_instance), directory, per_instance,
                        hard_deadline=min(hard_deadline, started + 1800))
        write_json(directory / "run.json", run)
        print(f"Oracle {iid}: {run['process_status']}", flush=True)


@contextmanager
def batch_lock(output):
    path = Path(output) / ".batch.lock"
    stream = path.open("a+b")
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another runner owns this experiment") from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def run_suite(output, solver_root, *, hours=8, smoke=False, skip_oracle=False, report_builder=None):
    with batch_lock(output):
        return _run_suite(output, solver_root, hours=hours, smoke=smoke, skip_oracle=skip_oracle,
                          report_builder=report_builder)


def _run_suite(output, solver_root, *, hours=8, smoke=False, skip_oracle=False, report_builder=None):
    output = Path(output).resolve()
    if report_builder is None:
        from .report import build_report
        report_builder = build_report
    if not 0 < hours <= 8:
        raise ValueError("hours must be in (0, 8]")
    manifest = verify_frozen(output, solver_root)
    config = read_json(output / "config.json")
    state_path = output / ("smoke_state.json" if smoke else "batch_state.json")
    previous = read_json(state_path) if state_path.exists() else {}
    consumed = previous.get("consumed_seconds", 0)
    if previous.get("status") == "RUNNING":
        checkpoint = previous.get("updated_at_utc", previous["started_at_utc"])
        consumed += max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(checkpoint)).total_seconds())
    batch_limit = effective_batch_limit(config, hours, previous)
    remaining = batch_limit - consumed
    if remaining <= 0:
        raise ValueError("Persisted batch budget exhausted; report existing results")
    began = time.perf_counter()
    hard_deadline = began + remaining
    state = {"status": "RUNNING", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "consumed_seconds": consumed, "budget_seconds": batch_limit, "smoke": smoke,
             "profile": config.get("profile", "standard"), "phase": "PREPARING",
             "oracle_mode": config.get("oracle_mode") or ("baseline_reuse" if config.get("reuse_baseline_answers") else "skipped" if skip_oracle else "compute"),
             "oracle_skipped_by_request": skip_oracle,
             "completed_runs": previous.get("completed_runs", 0)}
    write_json(state_path, state)
    blocks = read_json(output / "schedule.json")
    if smoke:
        if config.get("profile") == "long-quick":
            blocks = [{"phase": "SMOKE", "instance_id": config.get("warmup_instance_id", "G01"),
                       "seconds": 0.2, "seed": 0, "engines": list(ENGINES)}]
        else:
            blocks = [b for b in blocks if b["instance_id"] in ("G01", "G03", "G09", "G11", "n8-C0-g101", "n8-C3-g101", "n8-C6-g101", "n8-PROVIDER-g101")
                      and b["seed"] == 11 and b["seconds"] == 1]
        # Smoke results are kept separate, never counted as formal repetitions.
        warmup = True
    else:
        warmup = False
    state["completed_runs"] = sum((output / ("warmup" if warmup else "runs") / block_key(b) / e / "run.json").exists()
                                  for b in blocks for e in b["engines"])
    write_json(state_path, state)
    try:
        if not smoke and not skip_oracle and not config.get("reuse_baseline_answers") and not config.get("answers_ready"):
            run_oracles(output, hard_deadline)
        if not smoke:
            for engine in config.get("engines", ENGINES):
                run_one(output, solver_root, {"instance_id": config.get("warmup_instance_id", "G01"), "seconds": 0.2, "seed": 0}, engine,
                        warmup=True, hard_deadline=hard_deadline)
        for index, block in enumerate(blocks):
            pending = [e for e in block["engines"]
                       if not (output / ("warmup" if warmup else "runs") / block_key(block) / e / "run.json").exists()]
            if not pending:
                continue
            # Include bounded startup, termination and evidence reconstruction.
            reserve = len(pending) * (block["seconds"] + 35) + 60
            if time.perf_counter() + reserve >= hard_deadline:
                state["status"] = "BUDGET_STOPPED"
                break
            state["phase"] = block.get("phase", "FORMAL")
            for engine in pending:
                state["active_run"] = {"block": block_key(block), "engine": engine, "budget_seconds": block["seconds"]}
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                state["consumed_seconds"] = consumed + time.perf_counter() - began
                write_json(state_path, state)
                run = run_one(output, solver_root, block, engine, warmup=warmup, hard_deadline=hard_deadline)
                state["completed_runs"] += 1
                state["last_run"] = {"block": block_key(block), "engine": engine,
                                     "status": run["process_status"], "best_makespan": run["best_makespan"]}
                state["consumed_seconds"] = consumed + time.perf_counter() - began
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(state_path, state)
                print(f"{index+1}/{len(blocks)} {block_key(block)} {engine}: {run['process_status']} best={run['best_makespan']}", flush=True)
            if index % 24 == 0 or config.get("profile") == "long-quick" or config.get("report_each_block"):
                report_builder(output)
        else:
            state["status"] = "COMPLETED"
    except BaseException as exc:
        state["status"] = "INTERRUPTED"
        state["error"] = repr(exc)
        raise
    finally:
        state["consumed_seconds"] = consumed + time.perf_counter() - began
        state["active_run"] = None
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
        report_builder(output)
        state["consumed_seconds"] = consumed + time.perf_counter() - began
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
    return state


def effective_batch_limit(config, hours, previous=None):
    """CLI and resumed runs can tighten a profile's cap, never extend it."""
    if not 0 < hours <= 8:
        raise ValueError("hours must be in (0, 8]")
    limits = [hours * 3600, float(config.get("batch_limit_seconds", 28800))]
    if previous and previous.get("budget_seconds") is not None:
        limits.append(float(previous["budget_seconds"]))
    if min(limits) <= 0:
        raise ValueError("Batch budget must be positive")
    return min(limits)
