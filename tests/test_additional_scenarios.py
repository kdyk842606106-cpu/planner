from __future__ import annotations

import unittest
from pathlib import Path

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Action, Budget
from planner_experiment.scenario import load_scenario
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]


def manual_candidate(scenario, activity_ids, *, waits=(), run_id="manual"):
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    wait_by_position = dict(waits)
    for index, activity_id in enumerate(activity_ids):
        if index in wait_by_position:
            state = simulator.transition(state, Action("WAIT", target_time=wait_by_position[index]))
        state = simulator.transition(state, Action("EXECUTE", activity_id=activity_id))
    if len(activity_ids) in wait_by_position:
        state = simulator.transition(
            state, Action("WAIT", target_time=wait_by_position[len(activity_ids)])
        )
    return simulator.candidate_from_state(
        state,
        algorithm="MANUAL",
        run_id=run_id,
        seed=None,
        discovered_at_seconds=0.0,
        normalize=False,
    )


class AdditionalScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.approval = load_scenario(ROOT / "scenarios" / "approval_release.json")
        cls.thermal = load_scenario(ROOT / "scenarios" / "thermal_validation.json")

    def test_approval_productive_and_waiting_baselines(self):
        efficient = manual_candidate(
            self.approval,
            (
                "prepare_document",
                "risk_check",
                "incorporate_legal",
                "submit_review",
                "approve",
                "unlock_environment",
                "deploy",
                "lock_environment",
            ),
            run_id="approval-efficient",
        )
        waiting = manual_candidate(
            self.approval,
            (
                "incorporate_legal",
                "prepare_document",
                "risk_check",
                "submit_review",
                "approve",
                "unlock_environment",
                "deploy",
                "lock_environment",
            ),
            waits=((0, 25),),
            run_id="approval-waiting",
        )
        validator = ResultValidator(self.approval)
        self.assertTrue(validator.validate(efficient).valid)
        self.assertTrue(validator.validate(waiting).valid)
        self.assertEqual(efficient.metrics.makespan, 75)
        self.assertEqual(waiting.metrics.makespan, 100)
        self.assertLess(efficient.metrics.sort_key, waiting.metrics.sort_key)

    def test_approval_reversible_domains_are_generic(self):
        graph = PathSimulator(self.approval).transition_graph
        self.assertIn(("workflow", "draft"), graph.reversible_nodes)
        self.assertIn(("workflow", "review"), graph.reversible_nodes)
        self.assertIn(("access", "locked"), graph.reversible_nodes)
        self.assertIn(("access", "unlocked"), graph.reversible_nodes)
        self.assertNotIn(("deployed", "true"), graph.reversible_nodes)

    def test_thermal_combined_and_staged_baselines(self):
        combined = manual_candidate(
            self.thermal,
            (
                "prepare_rig",
                "calibrate_sensor",
                "install_specimen",
                "combined_climate_test",
                "compile_report",
                "quality_review",
            ),
            waits=((3, 55),),
            run_id="thermal-combined",
        )
        staged = manual_candidate(
            self.thermal,
            (
                "prepare_rig",
                "calibrate_sensor",
                "install_specimen",
                "heat_up",
                "hot_test",
                "cool_to_ambient",
                "chill_down",
                "cold_test",
                "warm_to_ambient",
                "compile_report",
                "quality_review",
            ),
            run_id="thermal-staged",
        )
        validator = ResultValidator(self.thermal)
        self.assertTrue(validator.validate(combined).valid)
        self.assertTrue(validator.validate(staged).valid)
        self.assertEqual(combined.metrics.makespan, 105)
        self.assertEqual(staged.metrics.makespan, 117)
        self.assertLess(combined.metrics.sort_key, staged.metrics.sort_key)
        self.assertEqual(staged.metrics.transition_count, 4)
        self.assertGreaterEqual(staged.metrics.state_revisit_count, 2)
        self.assertGreaterEqual(staged.metrics.goal_regression_count, 2)

    def test_both_engines_return_valid_paths_on_new_scenarios(self):
        budget = Budget(3.0, 30_000)
        for scenario in (self.approval, self.thermal):
            astar = AnytimeAStar(scenario).run(
                scenario, budget, run_id=f"{scenario.id}-astar"
            )
            ga = GeneticExplorer(scenario).run(
                scenario, budget, run_id=f"{scenario.id}-ga", seed=11
            )
            self.assertTrue(astar.paths, f"A* found no path for {scenario.id}")
            self.assertTrue(ga.paths, f"GA found no path for {scenario.id}")
            validator = ResultValidator(scenario)
            self.assertTrue(all(validator.validate(path).valid for path in astar.paths))
            self.assertTrue(all(validator.validate(path).valid for path in ga.paths))


if __name__ == "__main__":
    unittest.main()
