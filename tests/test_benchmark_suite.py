from __future__ import annotations

import copy
import json
import tempfile
import time
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from planner_experiment.benchmark_suite.complexity import dag_properties
from planner_experiment.benchmark_suite.generator import (candidate_from_actions, independent_resource_check,
                                                         make_instance)
from planner_experiment.benchmark_suite.golden import golden_instances
from planner_experiment.benchmark_suite.io import digest, read_events, write_json
from planner_experiment.benchmark_suite.oracle import exact_answer
from planner_experiment.benchmark_suite.report import aggregate, describe_run
from planner_experiment.benchmark_suite.runner import batch_lock, planned_blocks, supervise, verify_frozen
from planner_experiment.benchmark_suite import SCALES, GROUPS, GENERATOR_SEEDS
from planner_experiment.instrumentation import observe_candidates
from planner_experiment.archive import PathArchive


def delayed_worker(directory, pipe):
    pipe.send(("ready", {}))
    pipe.recv()
    stamp = time.perf_counter()
    pipe.send(("started", stamp))
    with (Path(directory) / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"valid": True, "elapsed_seconds": 0.001, "makespan": 10}) + "\n")
        stream.flush()
    time.sleep(5)


def completed_worker(directory, pipe):
    pipe.send(("ready", {}))
    pipe.recv()
    pipe.send(("started", time.perf_counter()))
    pipe.send(("done", {"status": "OK"}))
    pipe.close()


class ComplexityTests(unittest.TestCase):
    def test_reachability_counts_transitive_pairs_not_direct_arcs(self):
        nodes = list("abcd")
        durations = {n: 1 for n in nodes}
        chain = dag_properties(nodes, [["a", "b"], ["b", "c"], ["c", "d"]], durations)
        self.assertEqual((chain["comparable_pairs"], chain["width"], chain["depth"]), (6, 1, 4))
        redundant = dag_properties(nodes, [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]], durations)
        self.assertEqual(redundant["order_strength"], chain["order_strength"])
        independent = dag_properties(nodes, [], durations)
        self.assertEqual((independent["order_strength"], independent["width"]), (0, 4))

    def test_exact_antichain_width_and_cycle_rejection(self):
        # Width 2 although this topological ordering could expose three ranks.
        graph = dag_properties(list("abcd"), [["a", "c"], ["b", "c"], ["c", "d"]], dict.fromkeys("abcd", 1))
        self.assertEqual(graph["width"], 2)
        with self.assertRaises(ValueError):
            dag_properties(list("ab"), [["a", "b"], ["b", "a"]], dict.fromkeys("ab", 1))


class GenerationTests(unittest.TestCase):
    def test_all_scale_group_witnesses_and_constructive_optima(self):
        for n in SCALES:
            for group in (*GROUPS, "PROVIDER"):
                with self.subTest(n=n, group=group):
                    item = make_instance(n, group, 101)
                    expected = n + n // 4 if group == "PROVIDER" else n
                    self.assertEqual(item["complexity"]["activity_count"], expected)
                    self.assertEqual(item["complexity"]["structurally_relevant_count"], expected)
                    width = item["metadata"]["width_parameter"]
                    self.assertEqual(item["complexity"]["skeleton"]["width"], width)
                    self.assertEqual(item["complexity"]["skeleton"]["depth"], n // width)
                    candidate = candidate_from_actions(item["scenario"], item["answer"]["actions"])
                    if item["answer"]["status"] == "PROVEN_OPTIMAL":
                        self.assertEqual(candidate.metrics.makespan, item["answer"]["lower_bound"])
                    if group in ("C3", "C6"):
                        self.assertEqual(item["complexity"]["witness_repeat_count"], n // 8)

    def test_determinism_pairing_and_no_input_mutation(self):
        base = make_instance(16, "C0", 202)
        self.assertEqual(base, make_instance(16, "C0", 202))
        for group in GROUPS[1:]:
            paired = make_instance(16, group, 202)
            for a, b in zip(base["scenario"]["activities"], paired["scenario"]["activities"]):
                self.assertEqual((a["id"], a["duration"], a["resource_reqs"]), (b["id"], b["duration"], b["resource_reqs"]))
        provider = make_instance(16, "PROVIDER", 202)
        self.assertEqual(base["metadata"]["edges"], provider["metadata"]["edges"])
        self.assertEqual(base["scenario"]["goal_state_ids"], provider["scenario"]["goal_state_ids"])
        snapshot = copy.deepcopy(base)
        candidate_from_actions(base["scenario"], base["answer"]["actions"])
        self.assertEqual(base, snapshot)

    def test_frozen_schedule_counts_and_balance(self):
        ids = [f"n{n}-{g}-g{s}" for n in SCALES for g in (*GROUPS, "PROVIDER") for s in GENERATOR_SEEDS]
        ids += [f"G{i:02d}" for i in range(1, 13)]
        blocks = planned_blocks(ids)
        self.assertEqual(len(blocks) * 3, 3996)
        self.assertEqual(sum(b["seconds"] * 3 for b in blocks), 18252)
        self.assertTrue(all(set(b["engines"]) == {"LEGACY", "ASTAR", "GA"} for b in blocks))
        self.assertEqual(blocks, planned_blocks(ids))


class CertificateTests(unittest.TestCase):
    def test_all_golden_proofs_match_full_enumeration(self):
        cases = golden_instances()
        self.assertEqual(len(cases), 12)
        for case in cases:
            with self.subTest(case=case["metadata"]["id"]):
                answer = exact_answer(case["scenario"], 3)
                self.assertEqual(answer["status"], case["answer"]["status"])
                self.assertEqual(answer.get("optimal_makespan"), case["answer"].get("optimal_makespan"))

    def test_oracle_truncation_is_unknown(self):
        item = make_instance(8, "C0", 101)
        self.assertEqual(exact_answer(item["scenario"], 0)["status"], "UNKNOWN")
        self.assertEqual(exact_answer(item["scenario"], 3, max_states=1)["status"], "UNKNOWN")

    def test_independent_sweep_rejects_overlapping_resource_use(self):
        case = golden_instances()[2]
        candidate = candidate_from_actions(case["scenario"], case["answer"]["actions"])
        altered = replace(candidate, executions=(candidate.executions[0], replace(candidate.executions[1], start_time=2)))
        with self.assertRaises(ValueError):
            independent_resource_check(case["scenario"], altered)

    def test_observation_does_not_change_archive_and_restores_default(self):
        case = golden_instances()[0]
        candidate = candidate_from_actions(case["scenario"], case["answer"]["actions"])
        seen = []
        archive = PathArchive()
        with observe_candidates(seen.append, trace_memory=False):
            archive.add(candidate)
        archive.add(candidate)
        self.assertEqual(seen, [candidate])
        self.assertEqual(archive.best.metrics, candidate.metrics)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        self.temporary_root.mkdir(exist_ok=True)

    def test_deadline_kills_worker_and_keeps_flushed_candidate(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            began = time.perf_counter()
            run = supervise(delayed_worker, (), directory, 0.15)
            self.assertEqual(run["process_status"], "TIME_LIMIT")
            self.assertLess(time.perf_counter() - began, 4)
            self.assertEqual(len(read_events(Path(directory) / "candidates.jsonl")), 1)

    def test_partial_json_line_is_not_counted(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"valid":true}\n{"valid":', encoding="utf-8")
            self.assertEqual(read_events(path), [{"valid": True}])

    def test_completed_worker_can_close_pipe_immediately(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            run = supervise(completed_worker, (), directory, 1)
            self.assertEqual(run["process_status"], "COMPLETED")

    def test_batch_lock_prevents_concurrent_runs(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            with batch_lock(directory):
                with self.assertRaises(RuntimeError):
                    with batch_lock(directory):
                        pass
            with batch_lock(directory):
                pass

    def test_frozen_inputs_configuration_and_source_cannot_drift(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            root = Path(directory)
            item, config, schedule = {"input": 1}, {"budget": 1}, []
            env = {"python_executable": sys.executable, "source_hashes": {"module": "one"}, "python": "test", "ortools": "test"}
            write_json(root / "instances" / "x.json", item)
            write_json(root / "config.json", config)
            write_json(root / "schedule.json", schedule)
            write_json(root / "manifest.json", {"environment": env, "instance_hashes": {"x": digest(item)},
                       "config_hash": digest(config), "schedule_hash": digest(schedule)})
            with patch("planner_experiment.benchmark_suite.runner.source_hashes", return_value=env["source_hashes"]), \
                 patch("planner_experiment.benchmark_suite.runner.environment", return_value=env):
                verify_frozen(root, "unused")
                write_json(root / "instances" / "x.json", {"input": 2})
                with self.assertRaisesRegex(ValueError, "instance changed"):
                    verify_frozen(root, "unused")
                write_json(root / "instances" / "x.json", item)
                write_json(root / "config.json", {"budget": 2})
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    verify_frozen(root, "unused")
            with patch("planner_experiment.benchmark_suite.runner.source_hashes", return_value={"module": "two"}):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    verify_frozen(root, "unused")


class ReportingTests(unittest.TestCase):
    def run_fixture(self, case, events, engine_status="TIMEOUT_EMPTY"):
        run = {"instance_id": case["metadata"]["id"], "engine": "ASTAR", "budget_seconds": 1, "seed": 11,
               "process_status": "TIME_LIMIT", "engine_status": engine_status,
               "rejected_candidate_count": 0, "peak_working_set_bytes": None, "call_wall_seconds": 1}
        return describe_run(run, case, case["answer"], events)

    def test_late_candidates_are_excluded_and_failures_are_penalized(self):
        row = self.run_fixture(golden_instances()[0], [{"valid": True, "elapsed_seconds": 1.1, "makespan": 10}])
        self.assertFalse(row["success"])
        self.assertEqual(row["par2_seconds"], 2)
        self.assertIsNone(row["first_solution_seconds"])

    def test_zero_optimum_and_unproven_infeasibility(self):
        row = self.run_fixture(golden_instances()[8], [{"valid": True, "elapsed_seconds": 0.01, "makespan": 0}])
        self.assertTrue(row["optimal_hit"])
        self.assertEqual(row["optimal_absolute_gap"], 0)
        self.assertIsNone(row["optimal_gap"])
        infeasible = self.run_fixture(golden_instances()[10], [], "EXHAUSTED_EMPTY")
        self.assertEqual(infeasible["verdict"], "UNKNOWN_INFEASIBLE")
        false_proof = self.run_fixture(golden_instances()[0], [], "PROVEN_INFEASIBLE")
        self.assertEqual(false_proof["verdict"], "FALSE_INFEASIBLE")

    def test_independent_instances_are_equally_weighted(self):
        case = golden_instances()[0]
        success = self.run_fixture(case, [{"valid": True, "elapsed_seconds": 0.1, "makespan": 10}])
        failure = self.run_fixture(case, [])
        failure["instance_id"] = "different-model"
        _, cells = aggregate([success, success, success, failure])
        self.assertEqual(cells[0]["success_rate"], 0.5)
        self.assertEqual(cells[0]["independent_instances"], 2)


if __name__ == "__main__":
    unittest.main()
