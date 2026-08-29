from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planner_experiment.models import Budget
from planner_experiment.runner import run_benchmark, run_compare
from planner_experiment.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]


class RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = load_scenario(ROOT / "scenarios" / "module_x.json")

    def test_parallel_compare_writes_contract_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison = run_compare(
                self.scenario,
                Budget(3.0, 10_000),
                seed=11,
                output_dir=temp_dir,
                run_id="parallel-test",
            )
            self.assertIsNotNone(comparison["astar"]["best_path"])
            self.assertIsNotNone(comparison["ga"]["best_path"])
            for name in ("manifest.json", "astar_result.json", "ga_result.json", "comparison.json", "report.md"):
                self.assertTrue((Path(temp_dir) / name).exists(), name)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 4)
            self.assertTrue(manifest["engines_parallel"])
            self.assertFalse(manifest["engines_share_mutable_state"])
            self.assertEqual(manifest["gantt_renderer_version"], "svg-gantt-v1")
            self.assertTrue(comparison["gantt"])
            report = (Path(temp_dir) / "report.md").read_text(encoding="utf-8")
            for item in comparison["gantt"]:
                self.assertTrue((Path(temp_dir) / item["svg_file"]).exists())
                self.assertIn(f"({item['svg_file']})", report)
            self.assertEqual(set(manifest["gantt_files"]), {item["svg_file"] for item in comparison["gantt"]})

    def test_one_worker_failure_preserves_other_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison = run_compare(
                self.scenario,
                Budget(2.0, 5_000),
                seed=11,
                output_dir=temp_dir,
                run_id="failure-test",
                force_failure="GA",
            )
            self.assertIsNotNone(comparison["astar"]["best_path"])
            self.assertEqual(comparison["ga"]["status"], "ERROR")
            self.assertTrue(all(item["engine"] == "ASTAR" for item in comparison["gantt"]))

    def test_benchmark_report_links_to_representative_run_gantt(self):
        scenario = load_scenario(ROOT / "scenarios" / "parallel_workflow.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_benchmark(
                scenario,
                Budget(1.5, 20_000),
                seeds=[11],
                output_dir=temp_dir,
            )
            self.assertTrue(summary["gantt"])
            report = (Path(temp_dir) / "report.md").read_text(encoding="utf-8")
            for item in summary["gantt"]:
                self.assertTrue((Path(temp_dir) / item["svg_file"]).exists())
                self.assertIn(f"({item['svg_file']})", report)


if __name__ == "__main__":
    unittest.main()
