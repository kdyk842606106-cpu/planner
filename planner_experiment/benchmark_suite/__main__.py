from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import generate_suite, run_suite
from .report import build_report


def main():
    parser = argparse.ArgumentParser(description="Six-scale, three-engine correctness/performance experiment")
    parser.add_argument("command", choices=("generate", "run", "smoke", "report"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--solver-root", default=str(Path(__file__).resolve().parents[3].parent / "Solver_demo_project_planner_exploration"))
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--skip-oracle", action="store_true", help="Explicit diagnostic override; saved in batch state")
    parser.add_argument("--profile", choices=("standard", "long-quick"), default="standard")
    parser.add_argument("--baseline", help="Completed frozen baseline directory for long-quick")
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_suite(args.output, args.solver_root, profile=args.profile, baseline_dir=args.baseline)
        build_report(args.output)
        print(json.dumps({"instances": len(result["instance_ids"]), "planned_runs": result["planned_run_count"]}, ensure_ascii=False))
    elif args.command == "report":
        result = build_report(args.output)
        print(json.dumps({k: result[k] for k in ("planned_runs", "executed_runs", "unrun_runs")}, ensure_ascii=False))
    else:
        print(json.dumps(run_suite(args.output, args.solver_root, hours=args.hours, smoke=args.command == "smoke",
                                   skip_oracle=args.skip_oracle), ensure_ascii=False))


if __name__ == "__main__":
    main()
