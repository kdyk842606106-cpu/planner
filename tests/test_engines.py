from __future__ import annotations

import unittest
from pathlib import Path

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Budget
from planner_experiment.scenario import load_scenario
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = load_scenario(ROOT / "scenarios" / "module_x.json")

    def test_astar_finds_both_golden_structures(self):
        result = AnytimeAStar(self.scenario).run(
            self.scenario,
            Budget(3.0, 20_000),
            run_id="astar-test",
        )
        signatures = {path.structure_signature for path in result.paths}
        self.assertTrue(set(self.scenario.expected_signatures).issubset(signatures))
        self.assertEqual(result.paths[0].metrics.makespan, 90)
        self.assertTrue(all(ResultValidator(self.scenario).validate(path).valid for path in result.paths))

    def test_ga_fixed_seed_is_reproducible(self):
        budget = Budget(10.0, 6_000)
        first = GeneticExplorer(self.scenario).run(self.scenario, budget, run_id="ga-one", seed=23)
        second = GeneticExplorer(self.scenario).run(self.scenario, budget, run_id="ga-two", seed=23)
        self.assertTrue(first.paths)
        self.assertTrue(second.paths)
        self.assertEqual(first.paths[0].structure_signature, second.paths[0].structure_signature)
        self.assertEqual(first.paths[0].metrics, second.paths[0].metrics)
        self.assertEqual(
            [(point.score, point.signature) for point in first.improvements],
            [(point.score, point.signature) for point in second.improvements],
        )

    def test_ga_seed_set_is_valid_and_collectively_diverse(self):
        signatures = set()
        for seed in (11, 23, 37, 53, 71):
            result = GeneticExplorer(self.scenario).run(
                self.scenario,
                Budget(10.0, 6_000),
                run_id=f"ga-{seed}",
                seed=seed,
            )
            self.assertTrue(result.paths, f"seed {seed} produced no valid path")
            self.assertTrue(all(ResultValidator(self.scenario).validate(path).valid for path in result.paths))
            signatures.update(path.structure_signature for path in result.paths)
        self.assertTrue(set(self.scenario.expected_signatures).issubset(signatures))


if __name__ == "__main__":
    unittest.main()
