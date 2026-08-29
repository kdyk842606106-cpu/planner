from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .models import Action, Budget
from .runner import run_benchmark, run_compare
from .scenario import load_scenario
from .simulator import PathSimulator
from .validator import ResultValidator


DEFAULT_SEEDS = [11, 23, 37, 53, 71]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Anytime A* vs GA path exploration experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Replay and validate an action sequence")
    validate.add_argument("--scenario", required=True)
    validate.add_argument("--actions", required=True)
    validate.add_argument("--output")
    validate.add_argument("--execution-mode", choices=("serial", "parallel"))

    compare = subparsers.add_parser("compare", help="Run A* and GA concurrently once")
    _add_run_arguments(compare)
    compare.add_argument("--seed", type=int, default=11)

    benchmark = subparsers.add_parser("benchmark", help="Repeat concurrent comparisons for fixed GA seeds")
    _add_run_arguments(benchmark)
    benchmark.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--transition-limit", type=int)
    parser.add_argument("--max-solutions", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = load_scenario(args.scenario)
    if args.execution_mode:
        scenario = replace(scenario, execution_mode=args.execution_mode)
    if args.command == "validate":
        return _validate_command(scenario, args.actions, args.output)

    budget = Budget(
        time_limit_seconds=args.time_limit or scenario.default_time_limit,
        transition_limit=args.transition_limit or scenario.default_transition_limit,
        max_solutions=args.max_solutions,
    )
    base = Path(args.output) if args.output else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.command == "compare":
        comparison = run_compare(scenario, budget, seed=args.seed, output_dir=base)
        print(json.dumps({"output": str(base.resolve()), "best_algorithm": comparison["best_algorithm"]}, ensure_ascii=False))
        return 0
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    summary = run_benchmark(scenario, budget, seeds=seeds, output_dir=base)
    print(json.dumps({"output": str(base.resolve()), "recommendation": summary["recommendation"]}, ensure_ascii=False))
    return 0


def _validate_command(scenario, actions_path: str, output: str | None) -> int:
    payload = json.loads(Path(actions_path).read_text(encoding="utf-8"))
    actions = tuple(
        Action(kind=item["kind"], activity_id=item.get("activity_id"), target_time=item.get("target_time"))
        for item in payload
    )
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    try:
        for action in actions:
            state = simulator.transition(state, action)
        candidate = simulator.candidate_from_state(
            state,
            algorithm="MANUAL",
            run_id="manual-validation",
            seed=None,
            discovered_at_seconds=0.0,
            normalize=False,
        )
        result = ResultValidator(scenario).validate(candidate)
        response = {
            "valid": result.valid,
            "code": result.code,
            "message": result.message,
            "metrics": None if result.candidate is None else result.candidate.metrics.__dict__,
            "signature": None if result.candidate is None else result.candidate.structure_signature,
        }
    except Exception as exc:
        response = {"valid": False, "code": "REPLAY_ERROR", "message": str(exc)}
    encoded = json.dumps(response, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if response["valid"] else 1
