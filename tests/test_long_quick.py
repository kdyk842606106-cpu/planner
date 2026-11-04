from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from planner_experiment.benchmark_suite import ENGINES, SCALES
from planner_experiment.benchmark_suite.generator import make_instance
from planner_experiment.benchmark_suite.io import digest, read_json, write_json
from planner_experiment.benchmark_suite.long_report import classify_change
from planner_experiment.benchmark_suite.profiles import (assert_same_engines, file_hash,
                                                        long_quick_blocks, prepare_long_quick)
from planner_experiment.benchmark_suite.report import build_report, observation_checkpoints
from planner_experiment.benchmark_suite.runner import (block_key, effective_batch_limit,
                                                      run_one, _run_suite, verify_frozen)


class LongQuickTests(unittest.TestCase):
    def setUp(self):
        task_tmp = Path(__file__).resolve().parents[1] / "tmp"
        task_tmp.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=task_tmp)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        baseline, output = self.root / "baseline", self.root / "new"
        ids = list(dict.fromkeys(b["instance_id"] for b in long_quick_blocks()))
        case = make_instance(8, "C0", 101)
        hashes = {}
        env = {"python": "test", "python_executable": sys.executable, "ortools": "test",
               "source_hashes": {"planner/planner_experiment/astar.py": "fixed",
                                 "planner/planner_experiment/benchmark_suite/runner.py": "old"}}
        for iid in (*ids, "G01"):
            item = copy.deepcopy(case)
            item["answer"] = {"status": "PROVEN_OPTIMAL", "optimal_makespan": 10, "lower_bound": 10}
            if iid != "G01":
                scale, group = iid.split("-")[:2]
                item["metadata"].update(id=iid, scale=int(scale[1:]), group=group)
            write_json(baseline / "instances" / f"{iid}.json", item)
            hashes[iid] = digest(item)
            if iid == "G01":
                continue
            write_json(baseline / "oracles" / iid / "answer.json", item["answer"])
            write_json(baseline / "oracles" / iid / "run.json", {"proof_runtime": 12})
            for engine in ENGINES:
                write_json(baseline / "runs" / f"{iid}-t10-s11" / engine / "run.json",
                           {"instance_id": iid, "engine": engine, "seed": 11, "budget_seconds": 10,
                            "best_makespan": 20, "first_solution_seconds": 0.5,
                            "ready": {"shared_input_hash": "same"}})
        write_json(baseline / "manifest.json", {"environment": env, "instance_hashes": hashes,
                                               "planned_run_count": 3996})
        write_json(baseline / "batch_state.json", {"status": "COMPLETED", "completed_runs": 3996})
        current = copy.deepcopy(env)
        current["source_hashes"]["planner/planner_experiment/benchmark_suite/runner.py"] = "new"
        current["source_hashes"]["planner/planner_experiment/benchmark_suite/profiles.py"] = "new"
        return baseline, output, current

    def prepare(self):
        baseline, output, env = self.fixture()
        manifest = prepare_long_quick(output, "unused", baseline,
                                      environment=lambda _: env, source_files=lambda _: [])
        return baseline, output, env, manifest

    def test_exact_schedule_and_time_budget(self):
        blocks = long_quick_blocks()
        self.assertEqual(blocks, long_quick_blocks())
        self.assertEqual(len(blocks), 22)
        self.assertEqual(Counter(b["phase"] for b in blocks), {"A": 18, "B": 2, "C": 2})
        self.assertEqual(sum(len(b["engines"]) for b in blocks), 66)
        self.assertEqual(sum(b["seconds"] * len(b["engines"]) for b in blocks), 3780)
        self.assertEqual(len({b["instance_id"] for b in blocks}), 18)
        self.assertEqual({(b["instance_id"], b["seconds"]) for b in blocks if b["phase"] == "A"},
                         {(f"n{n}-{g}-g101", 30) for n in SCALES for g in ("C0", "C4", "C6")})
        for b in blocks:
            self.assertEqual(set(b["engines"]), set(ENGINES))
            self.assertEqual(b["seed"], 11)
        for phase in ("B", "C"):
            self.assertEqual({b["instance_id"] for b in blocks if b["phase"] == phase},
                             {"n128-C0-g101", "n256-C0-g101"})

    def test_protection_allows_only_named_harness_changes(self):
        base = {"planner/planner_experiment/astar.py": "fixed",
                "planner/planner_experiment/benchmark_suite/report.py": "old"}
        changed = {**base, "planner/planner_experiment/benchmark_suite/report.py": "new"}
        assert_same_engines(base, changed)
        for bad in ({**changed, "planner/planner_experiment/astar.py": "changed"},
                    {**changed, "solver/app/simulator.py": "new"},
                    {"planner/planner_experiment/benchmark_suite/report.py": "new"}):
            with self.assertRaisesRegex(ValueError, "Protected"):
                assert_same_engines(base, bad)

    def test_copy_frozen_inputs_proofs_and_pairing_without_generation(self):
        baseline, output, env, manifest = self.prepare()
        self.assertEqual(manifest["planned_run_count"], 66)
        self.assertEqual(len(manifest["instance_ids"]), 18)
        self.assertEqual(manifest["auxiliary_instance_ids"], ["G01"])
        self.assertEqual(read_json(output / "config.json")["batch_limit_seconds"], 4200)
        for iid in manifest["instance_hashes"]:
            self.assertEqual(file_hash(output / "instances" / f"{iid}.json"),
                             file_hash(baseline / "instances" / f"{iid}.json"))
        self.assertEqual(read_json(output / "oracles/n8-C0-g101/run.json"), {"proof_runtime": 12})
        with patch("planner_experiment.benchmark_suite.runner.source_hashes", return_value=env["source_hashes"]), \
             patch("planner_experiment.benchmark_suite.runner.environment", return_value=env):
            verify_frozen(output, "unused")
            write_json(output / "reference/n8-C0-g101/ASTAR/run.json", {"tampered": True})
            with self.assertRaisesRegex(ValueError, "baseline evidence"):
                verify_frozen(output, "unused")
        self.assertEqual(read_json(baseline / "runs/n8-C0-g101-t10-s11/ASTAR/run.json")["budget_seconds"], 10)

    def test_reject_changed_pairing_and_incomplete_baseline(self):
        baseline, output, env = self.fixture()
        path = baseline / "runs/n8-C0-g101-t10-s11/ASTAR/run.json"
        run = read_json(path)
        run["seed"] = 23
        write_json(path, run)
        with self.assertRaisesRegex(ValueError, "pairing"):
            prepare_long_quick(output, "unused", baseline, environment=lambda _: env, source_files=lambda _: [])
        write_json(baseline / "batch_state.json", {"status": "RUNNING", "completed_runs": 3995})
        with self.assertRaisesRegex(ValueError, "fully completed"):
            prepare_long_quick(output, "unused", baseline, environment=lambda _: env, source_files=lambda _: [])

    def test_config_cli_and_resume_cannot_extend_cap(self):
        config = {"batch_limit_seconds": 4200}
        self.assertEqual(effective_batch_limit(config, 8), 4200)
        self.assertEqual(effective_batch_limit(config, 0.5), 1800)
        self.assertEqual(effective_batch_limit(config, 8, {"budget_seconds": 1800}), 1800)
        self.assertEqual(effective_batch_limit({}, 8), 28800)
        for hours in (0, -1, 9):
            with self.assertRaises(ValueError):
                effective_batch_limit(config, hours)

    def test_long_budget_checkpoints_and_pair_categories(self):
        self.assertEqual(observation_checkpoints(10), [0.1, 0.25, 0.5, 1, 3, 10])
        for seconds in (30, 60, 300):
            self.assertEqual(observation_checkpoints(seconds)[-1], seconds)
            self.assertTrue(all(t <= seconds for t in observation_checkpoints(seconds)))
        for before, after, expected in ((None, 1, "NEW_SOLUTION"), (None, None, "STILL_NO_SOLUTION"),
                                         (20, None, "LOST_SOLUTION"), (20, 10, "IMPROVED"),
                                         (20, 20, "UNCHANGED"), (20, 30, "WORSENED")):
            self.assertEqual(classify_change(before, after), expected)
            self.assertEqual(classify_change(before, after, executed=False), "UNRUN")

    def test_report_long_profile_has_66_unrun_and_paired_evidence(self):
        _, output, _, _ = self.prepare()
        summary = build_report(output)
        self.assertEqual((summary["planned_runs"], summary["executed_runs"], summary["unrun_runs"]), (66, 0, 66))
        self.assertEqual(summary["budget_options"], [30, 60, 300])
        self.assertEqual([p["planned_runs"] for p in summary["long_quick"]["phases"]], [54, 6, 6])
        block = read_json(output / "schedule.json")[0]
        directory = output / "runs" / block_key(block) / "ASTAR"
        actions = [{"kind": "test"}]
        write_json(directory / "run.json", {"instance_id": block["instance_id"], "engine": "ASTAR",
                   "budget_seconds": 30, "seed": 11, "process_status": "TIME_LIMIT", "engine_status": None,
                   "call_wall_seconds": 31, "best_makespan": 15, "first_solution_seconds": 12,
                   "engine_stats": {"maximal_start_seed": {"valid": True, "makespan": 15}},
                   "ready": {"shared_input_hash": "same"}})
        write_json(directory / "best_candidate.json", {"actions": actions,
                   "executions": [{"activity_id": "a"}, {"activity_id": "a"}],
                   "metrics": {"non_causal_action_count": 1}})
        events = [{"valid": True, "elapsed_seconds": 12, "makespan": 15, "actions": actions},
                  {"valid": True, "elapsed_seconds": 31, "makespan": 1, "actions": []}]
        (directory / "candidates.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        summary = build_report(output)
        paired = next(r for r in summary["long_quick"]["comparisons"] if r["executed"])
        self.assertEqual((paired["change"], paired["makespan_delta"]), ("IMPROVED", -5))
        self.assertEqual(paired["new_repeat_count"], 1)
        self.assertTrue(paired["seed_equals_final"])
        self.assertTrue(paired["input_hash_match"])
        self.assertEqual(paired["new_first_solution_seconds"], 12)
        self.assertFalse(summary["proof_violations"])
        # Equal makespan alone does not establish that the actions stayed equal.
        write_json(directory / "best_candidate.json", {"actions": [], "executions": [],
                   "metrics": {"non_causal_action_count": 0}})
        paired = next(r for r in build_report(output)["long_quick"]["comparisons"] if r["executed"])
        self.assertFalse(paired["seed_equals_final"])
        # Unknown optimum must never turn a matching lower bound into an optimal hit.
        iid = block["instance_id"]
        write_json(output / "oracles" / iid / "answer.json", {"status": "UNKNOWN", "lower_bound": 15})
        item = read_json(output / "instances" / f"{iid}.json")
        item["answer"] = {"status": "UNKNOWN", "lower_bound": 15}
        write_json(output / "instances" / f"{iid}.json", item)
        paired = next(r for r in build_report(output)["long_quick"]["comparisons"] if r["executed"])
        self.assertIsNone(paired["new_optimal"])

    def test_budget_reaches_worker_and_late_candidate_is_ignored(self):
        output = self.root
        item = make_instance(8, "C0", 101)
        write_json(output / "instances/x.json", item)
        block = {"instance_id": "x", "seconds": 300, "seed": 11}
        def fake_supervise(target, args, directory, seconds, **kwargs):
            self.assertEqual((args[2], seconds), (300, 300))
            directory.mkdir(parents=True)
            (directory / "candidates.jsonl").write_text(
                json.dumps({"valid": True, "elapsed_seconds": 301, "makespan": 0}) + "\n", encoding="utf-8")
            return {"process_status": "TIME_LIMIT"}
        with patch("planner_experiment.benchmark_suite.runner.supervise", side_effect=fake_supervise):
            run = run_one(output, "unused", block, "GA")
        self.assertIsNone(run["best_makespan"])
        self.assertEqual(run["valid_candidate_count"], 0)

    def test_resume_recounts_committed_runs_and_reuses_oracles(self):
        _, output, _, manifest = self.prepare()
        blocks = read_json(output / "schedule.json")
        for block in blocks:
            for engine in block["engines"]:
                write_json(output / "runs" / block_key(block) / engine / "run.json", {})
        write_json(output / "batch_state.json", {"status": "INTERRUPTED", "completed_runs": 65,
                   "consumed_seconds": 123, "budget_seconds": 4200})
        with patch("planner_experiment.benchmark_suite.runner.verify_frozen", return_value=manifest), \
             patch("planner_experiment.benchmark_suite.runner.run_one") as run, \
             patch("planner_experiment.benchmark_suite.runner.run_oracles") as oracle, \
             patch("planner_experiment.benchmark_suite.report.build_report"):
            state = _run_suite(output, "unused")
        self.assertEqual(state["completed_runs"], 66)
        self.assertEqual(state["status"], "COMPLETED")
        self.assertEqual(state["budget_seconds"], 4200)
        self.assertGreaterEqual(state["consumed_seconds"], 123)
        oracle.assert_not_called()
        self.assertEqual(run.call_count, 3)  # separate warmup only

    def test_insufficient_time_does_not_start_a_partial_paired_block(self):
        _, output, _, manifest = self.prepare()
        write_json(output / "batch_state.json", {"status": "INTERRUPTED", "completed_runs": 0,
                   "consumed_seconds": 4190, "budget_seconds": 4200})
        with patch("planner_experiment.benchmark_suite.runner.verify_frozen", return_value=manifest), \
             patch("planner_experiment.benchmark_suite.runner.run_one") as run, \
             patch("planner_experiment.benchmark_suite.report.build_report"):
            state = _run_suite(output, "unused")
        self.assertEqual(state["status"], "BUDGET_STOPPED")
        self.assertEqual(state["completed_runs"], 0)
        self.assertTrue(all(call.kwargs.get("warmup") for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
