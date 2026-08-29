from __future__ import annotations

import unittest
from pathlib import Path

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Action, Budget
from planner_experiment.scenario import ScenarioError, load_scenario, scenario_from_dict
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator, VALIDATOR_VERSION


ROOT = Path(__file__).resolve().parents[1]


class GeneralizedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generic = load_scenario(ROOT / "scenarios" / "generic_modes.json")
        cls.module_x = load_scenario(ROOT / "scenarios" / "module_x.json")

    def test_transition_graph_is_inferred_from_network(self):
        graph = PathSimulator(self.generic).transition_graph
        reversible = graph.reversible_nodes
        self.assertIn(("mode", "assembly"), reversible)
        self.assertIn(("mode", "test"), reversible)
        self.assertIn(("fixture", "general"), reversible)
        self.assertNotIn(("x_installed", "true"), reversible)

    def test_external_event_applies_when_activity_crosses_it(self):
        scenario = scenario_from_dict(
            {
                "id": "cross-event",
                "initial_facts": {"available": "false", "done": "false"},
                "goal": {"required": {"done": "true"}},
                "external_events": [{"id": "ready", "time": 5, "effects": {"available": "true"}}],
                "activities": [{"id": "work", "duration": 10, "effects": {"done": "true"}}],
            }
        )
        simulator = PathSimulator(scenario)
        state = simulator.transition(simulator.initial_state(), Action("EXECUTE", activity_id="work"))
        self.assertEqual(dict(state.facts)["available"], "true")
        self.assertEqual(dict(state.facts)["done"], "true")

    def test_event_only_goal_is_a_valid_waiting_plan(self):
        scenario = scenario_from_dict(
            {
                "id": "event-only-goal",
                "states": [{"id": "seed"}, {"id": "ready"}],
                "initial_state_ids": ["seed"],
                "goal_state_ids": ["ready"],
                "external_events": [
                    {
                        "id": "ready-event",
                        "time": 12,
                        "add_state_ids": ["ready"],
                        "remove_state_ids": [],
                    }
                ],
                "activities": [],
            }
        )
        simulator = PathSimulator(scenario)
        state = simulator.initial_state()
        wait = next(action for action in simulator.enabled_actions(state) if action.kind in {"WAIT", "ADVANCE"})
        state = simulator.transition(state, wait)
        candidate = simulator.candidate_from_state(
            state,
            algorithm="MANUAL",
            run_id="event-only-goal",
            seed=None,
            discovered_at_seconds=0.0,
        )
        self.assertEqual(candidate.executions, ())
        self.assertEqual(candidate.metrics.makespan, 12)
        self.assertTrue(ResultValidator(scenario).validate(candidate).valid)

    def test_conflicting_same_time_events_are_rejected(self):
        with self.assertRaises(ScenarioError):
            scenario_from_dict(
                {
                    "id": "conflict",
                    "initial_facts": {"flag": "none"},
                    "goal": {"required": {"flag": "a"}},
                    "external_events": [
                        {"id": "a", "time": 5, "effects": {"flag": "a"}},
                        {"id": "b", "time": 5, "effects": {"flag": "b"}},
                    ],
                    "activities": [{"id": "keepalive", "duration": 1, "effects": {"flag": "a"}}],
                }
            )

    def test_schedule_normalization_moves_wait_to_the_real_blocker(self):
        simulator = PathSimulator(self.module_x)
        actions = (
            Action("EXECUTE", activity_id="power_on"),
            Action("EXECUTE", activity_id="test_a"),
            Action("WAIT", target_time=45),
            Action("EXECUTE", activity_id="power_off"),
            Action("EXECUTE", activity_id="install_x"),
            Action("EXECUTE", activity_id="power_on"),
            Action("EXECUTE", activity_id="test_b"),
        )
        state = simulator.initial_state()
        for action in actions:
            state = simulator.transition(state, action)
        candidate = simulator.candidate_from_state(
            state, algorithm="TEST", run_id="normalize", seed=None, discovered_at_seconds=0.0
        )
        self.assertEqual(candidate.metrics.makespan, 90)
        self.assertIn("power_off>WAIT@45>install_x", candidate.schedule_signature)
        self.assertEqual(VALIDATOR_VERSION, "temporal-event-validator-v4")
        self.assertTrue(ResultValidator(self.module_x).validate(candidate).valid)

    def test_generic_known_structures_and_metrics(self):
        simulator = PathSimulator(self.generic)
        productive = (
            Action("EXECUTE", activity_id="prepare"),
            Action("EXECUTE", activity_id="fixture_special"),
            Action("WAIT", target_time=30),
            Action("EXECUTE", activity_id="install_x"),
            Action("EXECUTE", activity_id="calibrate"),
            Action("EXECUTE", activity_id="install_y"),
            Action("EXECUTE", activity_id="fixture_general"),
            Action("EXECUTE", activity_id="mode_test"),
            Action("EXECUTE", activity_id="test_combo"),
            Action("EXECUTE", activity_id="quality"),
        )
        batch = (
            Action("EXECUTE", activity_id="prepare"),
            Action("EXECUTE", activity_id="fixture_special"),
            Action("EXECUTE", activity_id="calibrate"),
            Action("WAIT", target_time=50),
            Action("EXECUTE", activity_id="install_combo"),
            Action("EXECUTE", activity_id="fixture_general"),
            Action("EXECUTE", activity_id="mode_test"),
            Action("EXECUTE", activity_id="test_combo"),
            Action("EXECUTE", activity_id="quality"),
        )
        candidates = []
        for index, actions in enumerate((productive, batch)):
            state = simulator.initial_state()
            for action in actions:
                state = simulator.transition(state, action)
            candidates.append(
                simulator.candidate_from_state(
                    state,
                    algorithm="MANUAL",
                    run_id=f"generic-{index}",
                    seed=None,
                    discovered_at_seconds=0.0,
                    normalize=False,
                )
            )
        self.assertEqual(candidates[0].metrics.makespan, 111)
        self.assertEqual(candidates[1].metrics.makespan, 117)
        self.assertLess(candidates[0].metrics.sort_key, candidates[1].metrics.sort_key)
        self.assertGreaterEqual(candidates[0].metrics.transition_count, 3)
        self.assertGreaterEqual(candidates[0].metrics.state_revisit_count, 1)

    def test_both_engines_find_111_on_generic_scenario(self):
        budget = Budget(3.0, 30_000)
        astar = AnytimeAStar(self.generic).run(self.generic, budget, run_id="generic-a")
        ga = GeneticExplorer(self.generic).run(self.generic, budget, run_id="generic-g", seed=11)
        self.assertEqual(astar.paths[0].metrics.makespan, 111)
        self.assertEqual(ga.paths[0].metrics.makespan, 111)
        self.assertTrue(astar.stats["frontier_reused"])
        validator = ResultValidator(self.generic)
        self.assertTrue(all(validator.validate(path).valid for path in (*astar.paths, *ga.paths)))

    def test_multi_switch_structure_is_valid(self):
        simulator = PathSimulator(self.generic)
        actions = tuple(
            Action("EXECUTE", activity_id=activity_id)
            for activity_id in (
                "prepare",
                "calibrate",
                "fixture_special",
                "install_x",
                "fixture_general",
                "mode_test",
                "test_a",
                "mode_assembly",
                "fixture_special",
                "install_y",
                "fixture_general",
                "mode_test",
                "test_b",
                "quality",
            )
        )
        state = simulator.initial_state()
        for action in actions:
            state = simulator.transition(state, action)
        candidate = simulator.candidate_from_state(
            state, algorithm="MANUAL", run_id="multi-switch", seed=None, discovered_at_seconds=0.0
        )
        self.assertTrue(ResultValidator(self.generic).validate(candidate).valid)
        self.assertGreater(candidate.metrics.transition_count, 4)
        self.assertGreater(candidate.metrics.state_revisit_count, 1)


if __name__ == "__main__":
    unittest.main()
