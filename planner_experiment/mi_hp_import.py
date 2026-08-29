from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_FILENAME = "011_mechanical_integration_high_parallel_seed.sql"
SCENARIO_FILENAME = "solver_demo_mi_hp_core_parallel.json"
PACKAGE_SUBSYSTEMS = {
    "MI_HP_PREP_ACT": "PREPARATION",
    "MI_HP_STRUCTURE_ACT": "STRUCTURE",
    "MI_HP_TRANSFER_ACT": "TRANSFER",
    "MI_HP_UTILITY_ACT": "UTILITY",
    "MI_HP_DEBUG_ACT": "COMMISSIONING",
    "MI_HP_ACCEPTANCE_ACT": "ACCEPTANCE",
}


class ImportError(ValueError):
    pass


def _insert_bodies(source: str, table: str) -> list[str]:
    pattern = re.compile(
        rf"INSERT\s+INTO\s+{re.escape(table)}\s*\([^;]+?\)\s*VALUES\s*(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    bodies = pattern.findall(source)
    if not bodies:
        raise ImportError(f"missing INSERT data for {table}")
    return bodies


def project_seed(source_path: str | Path) -> dict[str, Any]:
    path = Path(source_path)
    raw = path.read_bytes()
    source = raw.decode("utf-8")

    resource_rows: list[tuple[str, str, str, int]] = []
    for body in _insert_bodies(source, "mi_hp_resource_plan"):
        resource_rows.extend(
            (resource_type, code, name, int(capacity))
            for resource_type, code, name, capacity in re.findall(
                r"\('([^']+)',\s*'([^']+)',\s*'([^']+)',\s*(\d+)\)", body
            )
        )

    activity_body = _insert_bodies(source, "mi_hp_activity_plan")[0]
    activity_rows = [
        {
            "seq": int(seq),
            "id": code,
            "name": name,
            "package": package,
            "effect_name": effect_name,
            "resource_type": resource_type,
            "duration": int(duration),
        }
        for seq, code, name, package, effect_name, resource_type, duration in re.findall(
            r"\((\d+),\s*'([^']+)',\s*'([^']+)',\s*'([^']+)',\s*'([^']+)',\s*'([^']+)',\s*(\d+)\)",
            activity_body,
        )
    ]

    dependency_body = _insert_bodies(source, "mi_hp_dependency_plan")[0]
    dependency_rows = re.findall(r"\('(MI_A\d+)',\s*'(MI_A\d+)'\)", dependency_body)
    dependencies: dict[str, list[str]] = {item["id"]: [] for item in activity_rows}
    for activity_id, dependency_id in dependency_rows:
        if activity_id not in dependencies:
            raise ImportError(f"dependency references unknown activity {activity_id}")
        dependencies[activity_id].append(dependency_id)

    extra_rows: list[tuple[str, str]] = []
    for body in _insert_bodies(source, "mi_hp_extra_resource_req_plan"):
        extra_rows.extend(re.findall(r"\('(MI_A\d+)',\s*'([A-Z_]+)'\)", body))
    extras: dict[str, list[str]] = {item["id"]: [] for item in activity_rows}
    for activity_id, resource_type in extra_rows:
        extras[activity_id].append(resource_type)

    _validate_projection_inputs(activity_rows, dependency_rows, resource_rows, extra_rows)
    facts = {f"mi_hp_{item['id'].lower()}_done": "false" for item in activity_rows}
    activities = []
    for item in sorted(activity_rows, key=lambda value: value["seq"]):
        activity_id = str(item["id"])
        resource_reqs = {str(item["resource_type"]): 1}
        resource_reqs.update({resource_type: 1 for resource_type in extras[activity_id]})
        activities.append(
            {
                "id": activity_id,
                "name": item["name"],
                "duration": item["duration"],
                # Each source row is a named, one-off DAG task. Preserve that
                # dataset contract explicitly now that planner activities are
                # repeatable by default.
                "max_instances": 1,
                "preconditions": {
                    f"mi_hp_{dependency.lower()}_done": "true"
                    for dependency in sorted(dependencies[activity_id])
                },
                "effects": {f"mi_hp_{activity_id.lower()}_done": "true"},
                "resource_reqs": dict(sorted(resource_reqs.items())),
                "source_metadata": {
                    "package_code": item["package"],
                    "responsible_subsystem": PACKAGE_SUBSYSTEMS[str(item["package"])],
                    "effect_name": item["effect_name"],
                    "source_order": item["seq"],
                },
            }
        )
    return {
        "id": "solver_demo_mi_hp_core_parallel",
        "name": "Solver Demo MI-HP-001 Core Parallel Projection",
        "execution_mode": "parallel",
        "start_time": 0,
        "initial_facts": facts,
        "goal": {"required": {"mi_hp_mi_a036_done": "true"}},
        "resources": [
            {"id": resource_type, "capacity": capacity, "source_code": code, "name": name}
            for resource_type, code, name, capacity in sorted(resource_rows)
        ],
        "activities": activities,
        "max_steps": 36,
        "default_budget": {"time_limit_seconds": 30, "transition_limit": 500_000},
        "provenance": {
            "source_project": "solver_demo_project",
            "source_dataset": "MI-HP-001",
            "source_file": f"seeds/{SOURCE_FILENAME}",
            "source_sha256": hashlib.sha256(raw).hexdigest().upper(),
            "projection": "core_parallel_v1",
            "excluded_rules": "工作日历/白班限制、行吊对全计划独占、责任子系统连续、功能调测独占",
            "comparison_note": "原项目 1170/1316 分钟结果包含本投影排除的日历或调度规则，不能与核心并行 makespan 直接比较。",
            "source_required_only_makespan": "1170",
            "source_all_rules_makespan": "1316",
            "expected_overlap_pairs": "MI_A002+MI_A003,MI_A004+MI_A005,MI_A011+MI_A012,MI_A018+MI_A019,MI_A020+MI_A021,MI_A024+MI_A025,MI_A027+MI_A028,MI_A027+MI_A020",
            "expected_non_overlap_pairs": "MI_A013+MI_A014",
        },
    }


def _validate_projection_inputs(activities, dependencies, resources, extras) -> None:
    issues: list[str] = []
    codes = [item["id"] for item in activities]
    expected_codes = [f"MI_A{index:03d}" for index in range(1, 37)]
    if codes != expected_codes:
        issues.append("activities must be the ordered MI_A001..MI_A036 sequence")
    if len(dependencies) != 52:
        issues.append(f"expected 52 dependencies, got {len(dependencies)}")
    if len(resources) != 12:
        issues.append(f"expected 12 resources, got {len(resources)}")
    if len(extras) != 6:
        issues.append(f"expected 6 extra resource requirements, got {len(extras)}")
    duration = sum(item["duration"] for item in activities)
    if duration != 880:
        issues.append(f"expected 880 total minutes, got {duration}")
    if len(set(codes)) != len(codes):
        issues.append("activity codes must be unique")
    if len({row[0] for row in resources}) != len(resources):
        issues.append("resource types must be unique")
    if issues:
        raise ImportError("; ".join(issues))


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def default_source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "solver_demo_project" / "seeds" / SOURCE_FILENAME


def default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scenarios" / SCENARIO_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project solver_demo MI-HP-001 into planner schema v4")
    parser.add_argument("--source", default=str(default_source_path()))
    parser.add_argument("--output", default=str(default_output_path()))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    generated = canonical_json(project_seed(args.source))
    output = Path(args.output)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"projection drift detected: {output}")
        print(f"projection is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
