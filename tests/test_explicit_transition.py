from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Action, Budget
from planner_experiment.scenario import ScenarioError, load_scenario, scenario_from_dict
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]


class ExplicitTransitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_scenario(ROOT / "scenarios" / "module_x_explicit_transition.json")

    def test_activity_relationship_roles_and_target_activity_goal(self):
        install = self.scenario.activity_by_id["install_x"]
        self.assertEqual(install.output_state_id, "state:x_installed")
        self.assertEqual(set(install.transition_state_ids), {"state:x_uninstalled"})
        self.assertEqual(set(install.required_state_ids), {"state:power_off"})
        self.assertEqual(set(self.scenario.goal_state_ids), {"state:x_installed"})
        self.assertFalse(any(state.legacy_key for state in self.scenario.state_definitions))
        self.assertTrue(self.scenario.activity_by_id["inspect_offline"].output_state_id.startswith("activity:"))

    def test_install_and_uninstall_replace_only_transition_state(self):
        cycle_scenario = replace(
            self.scenario,
            goal_state_ids=("state:power_on",),
            goal_required=(("state:power_on", "active"),),
        )
        simulator = PathSimulator(cycle_scenario)
        initial = simulator.initial_state()
        installed = simulator.transition(initial, Action("EXECUTE", "install_x"))
        self.assertEqual(
            set(installed.active_state_ids),
            {"state:x_installed", "state:power_off"},
        )
        removed = simulator.transition(installed, Action("EXECUTE", "uninstall_x"))
        self.assertEqual(
            set(removed.active_state_ids),
            {"state:x_uninstalled", "state:power_off"},
        )
        self.assertNotIn("state:x_installed", removed.active_state_ids)

    def test_transition_graph_is_explicit_and_cyclic(self):
        graph = PathSimulator(self.scenario).transition_graph
        self.assertIn(
            ("state:x_uninstalled", "state:x_installed", "install_x"),
            graph.edges,
        )
        self.assertIn(
            ("state:x_installed", "state:x_uninstalled", "uninstall_x"),
            graph.edges,
        )
        self.assertIn("state:x_installed", graph.reversible_state_ids)

    def test_validator_checks_state_id_snapshot(self):
        simulator = PathSimulator(self.scenario)
        state = simulator.transition(simulator.initial_state(), Action("EXECUTE", "install_x"))
        candidate = simulator.candidate_from_state(
            state, algorithm="MANUAL", run_id="explicit", seed=None, discovered_at_seconds=0
        )
        self.assertTrue(ResultValidator(self.scenario).validate(candidate).valid)
        tampered = replace(candidate, final_state_ids=("state:x_uninstalled", "state:power_off"))
        result = ResultValidator(self.scenario).validate(tampered)
        self.assertFalse(result.valid)
        self.assertEqual(result.code, "FINAL_STATES_MISMATCH")

    def test_both_engines_use_explicit_state_ids(self):
        budget = Budget(1.0, 5000)
        astar = AnytimeAStar(self.scenario).run(self.scenario, budget, run_id="explicit-a")
        ga = GeneticExplorer(self.scenario).run(self.scenario, budget, run_id="explicit-g", seed=11)
        self.assertTrue(astar.paths)
        self.assertTrue(ga.paths)
        self.assertEqual(astar.paths[0].metrics.makespan, 10)
        self.assertEqual(ga.paths[0].metrics.makespan, 10)
        self.assertTrue(all(ResultValidator(self.scenario).validate(path).valid for path in (*astar.paths, *ga.paths)))

    def test_native_activity_requires_transition_or_milestone(self):
        with self.assertRaises(ScenarioError):
            scenario_from_dict({
                "id": "invalid-native",
                "initial_state_ids": ["state:ready"],
                "target_activity_ids": ["finish"],
                "activities": [{
                    "id": "finish",
                    "duration": 1,
                    "preconditions": [{"state_id": "state:ready", "relation_role": "required"}]
                }]
            })

    def test_parallel_transition_source_is_exclusive(self):
        scenario = scenario_from_dict({
            "id": "parallel-transition-lock",
            "execution_mode": "parallel",
            "initial_state_ids": ["state:ready"],
            "target_activity_ids": ["consume_a"],
            "activities": [
                {
                    "id": "consume_a", "duration": 5, "output_state_id": "state:a",
                    "preconditions": [{"state_id": "state:ready", "relation_role": "transition"}]
                },
                {
                    "id": "consume_b", "duration": 5, "output_state_id": "state:b",
                    "preconditions": [{"state_id": "state:ready", "relation_role": "transition"}]
                }
            ],
            "max_steps": 2
        })
        simulator = PathSimulator(scenario)
        running = simulator.transition(simulator.initial_state(), Action("START", "consume_a"))
        enabled = {action.activity_id for action in simulator.enabled_actions(running) if action.kind == "START"}
        self.assertNotIn("consume_b", enabled)


if __name__ == "__main__":
    unittest.main()
