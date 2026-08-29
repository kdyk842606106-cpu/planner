from __future__ import annotations

import unittest
from pathlib import Path

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Budget
from planner_experiment.scenario import load_scenario
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]


class SyntheticPressureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = load_scenario(ROOT / "scenarios" / "synthetic_pressure.json")

    def test_both_engines_find_valid_synthetic_solution(self):
        budget = Budget(5.0, 50_000)
        astar = AnytimeAStar(self.scenario).run(self.scenario, budget, run_id="synthetic-a")
        ga = GeneticExplorer(self.scenario).run(self.scenario, budget, run_id="synthetic-g", seed=11)
        self.assertTrue(astar.paths)
        self.assertTrue(ga.paths)
        validator = ResultValidator(self.scenario)
        self.assertTrue(validator.validate(astar.paths[0]).valid)
        self.assertTrue(validator.validate(ga.paths[0]).valid)


if __name__ == "__main__":
    unittest.main()

