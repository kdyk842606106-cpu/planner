"""Long-budget studies reuse frozen inputs and never tune the engines."""
from __future__ import annotations

import hashlib
import random
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import ENGINES, SCALES, SUITE_VERSION
from .io import digest, read_json, write_json

# These are the only code changes permitted relative to the original batch.
# Worker, simulator, validator and every solver module remain protected.
HARNESS_CHANGES = frozenset(f"planner/planner_experiment/benchmark_suite/{name}.py"
                            for name in ("runner", "report", "__main__", "profiles", "long_report"))


def protected_sources(hashes: dict[str, str]) -> dict[str, str]:
    return {name: value for name, value in hashes.items() if name not in HARNESS_CHANGES}


def assert_same_engines(baseline: dict, current: dict) -> None:
    if protected_sources(baseline) != protected_sources(current):
        changed = sorted(name for name in set(baseline) | set(current)
                         if name not in HARNESS_CHANGES and baseline.get(name) != current.get(name))
        raise ValueError(f"Protected algorithm or execution code changed: {changed}")


def long_quick_blocks() -> list[dict]:
    rng = random.Random(81173)
    blocks = []
    for phase, seconds, scales, groups in (
        ("A", 30.0, SCALES, ("C0", "C4", "C6")),
        ("B", 60.0, (128, 256), ("C0",)),
        ("C", 300.0, (128, 256), ("C0",)),
    ):
        # Alternate scale and structure instead of leaving all large models last.
        pairs = [(n, groups[(index + turn) % len(groups)])
                 for turn in range(len(groups)) for index, n in enumerate(scales)]
        for n, group in pairs:
            engines = list(ENGINES)
            rng.shuffle(engines)
            blocks.append({"phase": phase, "instance_id": f"n{n}-{group}-g101", "seconds": seconds,
                               "seed": 11, "engines": engines})
    return blocks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_long_quick(output, solver_root, baseline_dir, *, environment, source_files) -> dict:
    if baseline_dir is None:
        raise ValueError("long-quick requires --baseline pointing to the completed original experiment")
    output, baseline = Path(output).resolve(), Path(baseline_dir).resolve()
    if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("New output must be separate from the baseline directory")
    if (output / "manifest.json").exists():
        raise ValueError("Output already contains a frozen suite; choose a new directory")
    old = read_json(baseline / "manifest.json")
    old_state = read_json(baseline / "batch_state.json")
    if old_state["status"] != "COMPLETED" or old_state["completed_runs"] != old["planned_run_count"]:
        raise ValueError("Baseline batch must be fully completed")
    current = environment(solver_root)
    for key in ("python", "python_executable", "ortools"):
        if current[key] != old["environment"][key]:
            raise ValueError(f"Baseline environment changed: {key}")
    assert_same_engines(old["environment"]["source_hashes"], current["source_hashes"])
    blocks = long_quick_blocks()
    ids = list(dict.fromkeys(b["instance_id"] for b in blocks))
    output.mkdir(parents=True, exist_ok=True)
    copied, reference_hashes = {}, {}

    def copy_frozen(source: Path, relative: str):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        reference_hashes[relative] = file_hash(destination)

    for iid in (*ids, "G01"):
        source = baseline / "instances" / f"{iid}.json"
        item = read_json(source)
        if digest(item) != old["instance_hashes"].get(iid):
            raise ValueError(f"Baseline input changed: {iid}")
        destination = output / "instances" / f"{iid}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[iid] = digest(item)
        oracle = baseline / "oracles" / iid
        if iid in ids and oracle.exists():
            for evidence in oracle.iterdir():
                if evidence.is_file():
                    copy_frozen(evidence, f"oracles/{iid}/{evidence.name}")
        if iid not in ids:
            continue
        for engine in ENGINES:
            source_dir = baseline / "runs" / f"{iid}-t10-s11" / engine
            baseline_run = read_json(source_dir / "run.json")
            if (baseline_run["instance_id"], baseline_run["engine"], baseline_run["budget_seconds"], baseline_run["seed"]) != (iid, engine, 10, 11):
                raise ValueError(f"Baseline pairing does not match: {iid}/{engine}")
            for name in ("run.json", "candidates.jsonl", "best_candidate.json", "engine_result.json"):
                path = source_dir / name
                if path.exists():
                    copy_frozen(path, f"reference/{iid}/{engine}/{name}")

    config = {"suite_version": SUITE_VERSION, "profile": "long-quick", "scales": list(SCALES),
              "groups": ["C0", "C4", "C6"], "generator_seeds": [101], "run_seeds": [11],
              "budgets_seconds": [30, 60, 300], "engines": list(ENGINES), "batch_limit_seconds": 4200,
              "warmup_instance_id": "G01", "reuse_baseline_answers": True, "baseline_budget_seconds": 10,
              "baseline_directory": str(baseline), "phase_run_counts": {"A": 54, "B": 6, "C": 6},
              "transition_protection_limit": 2**60, "max_solutions": 20, "cp_sat_workers": 1,
              "memory_metric": "OS peak process working set including imports", "startup_timeout_seconds": 30,
              "candidate_observation": old.get("candidate_observation", "same frozen worker as baseline"),
              "objective": "minimum_makespan", "business_scenarios": False,
              "study_note": "A has 18 models; B/C have two C0 models. Compare matching inputs, never pooled phase rates."}
    write_json(output / "config.json", config)
    write_json(output / "schedule.json", blocks)
    copy_frozen(baseline / "manifest.json", "reference/baseline-manifest.json")
    with zipfile.ZipFile(output / "source-snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in source_files(solver_root):
            archive.write(path, name)
    manifest = {"suite_version": SUITE_VERSION, "profile": "long-quick",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "solver_root": str(Path(solver_root).resolve()),
                "instance_ids": ids, "auxiliary_instance_ids": ["G01"], "instance_hashes": copied,
                "reference_hashes": reference_hashes, "config_hash": digest(config), "schedule_hash": digest(blocks),
                "planned_run_count": len(blocks) * len(ENGINES), "environment": current,
                "protected_source_hashes": protected_sources(current["source_hashes"]),
                "baseline_manifest_hash": file_hash(baseline / "manifest.json"),
                "baseline_directory": str(baseline)}
    write_json(output / "manifest.json", manifest)
    return manifest
