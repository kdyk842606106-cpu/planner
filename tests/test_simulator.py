from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from planner_experiment.models import Action
from planner_experiment.scenario import load_scenario
from planner_experiment.simulator import InvalidAction, PathSimulator
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]


def load_actions(name: str) -> tuple[Action, ...]:
    payload = json.loads((ROOT / "scenarios" / name).read_text(encoding="utf-8"))
    return tuple(Action(item["kind"], item.get("activity_id"), item.get("target_time")) for item in payload)


class SimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = load_scenario(ROOT / "scenarios" / "module_x.json")
        self.simulator = PathSimulator(self.scenario)

    def replay(self, actions: tuple[Action, ...]):
        state = self.simulator.initial_state()
        for action in actions:
            state = self.simulator.transition(state, action)
        return state

    def test_initial_state_is_immutable_and_transition_is_deterministic(self):
        initial = self.simulator.initial_state()
        action = Action("EXECUTE", activity_id="power_on")
        left = self.simulator.transition(initial, action)
        right = self.simulator.transition(initial, action)
        self.assertEqual(left, right)
        self.assertEqual(dict(initial.facts)["power"], "off")
        self.assertEqual(dict(left.facts)["power"], "on")

    def test_effect_is_recorded_at_activity_end(self):
        state = self.simulator.transition(self.simulator.initial_state(), Action("EXECUTE", activity_id="power_on"))
        record = state.executions[0]
        self.assertEqual(record.start_time, 0)
        self.assertEqual(record.end_time, 5)
        self.assertEqual(dict(record.before_facts)["power"], "off")
        self.assertEqual(dict(record.after_facts)["power"], "on")

    def test_material_and_wait_semantics(self):
        initial = self.simulator.initial_state()
        labels = {action.label for action in self.simulator.enabled_actions(initial)}
        self.assertNotIn("install_x", labels)
        self.assertIn("WAIT_UNTIL_45", labels)
        with self.assertRaises(InvalidAction):
            self.simulator.transition(initial, Action("WAIT", target_time=44))
        waited = self.simulator.transition(initial, Action("WAIT", target_time=45))
        self.assertIn("install_x", {action.label for action in self.simulator.enabled_actions(waited)})

    def test_repeat_limits_are_enforced(self):
        actions = (
            Action("EXECUTE", activity_id="power_on"),
            Action("EXECUTE", activity_id="power_off"),
            Action("EXECUTE", activity_id="power_on"),
        )
        state = self.replay(actions)
        self.assertEqual(dict(state.counts)["power_on"], 2)
        self.assertNotIn("power_on", {action.label for action in self.simulator.enabled_actions(state)})

    def test_golden_manual_paths_validate_and_rank(self):
        wait_state = self.replay(load_actions("module_x_wait_actions.json"))
        early_state = self.replay(load_actions("module_x_early_actions.json"))
        wait_candidate = self.simulator.candidate_from_state(
            wait_state, algorithm="MANUAL", run_id="wait", seed=None, discovered_at_seconds=0
        )
        early_candidate = self.simulator.candidate_from_state(
            early_state, algorithm="MANUAL", run_id="early", seed=None, discovered_at_seconds=0
        )
        self.assertTrue(ResultValidator(self.scenario).validate(wait_candidate).valid)
        self.assertTrue(ResultValidator(self.scenario).validate(early_candidate).valid)
        self.assertEqual(wait_candidate.metrics.makespan, 120)
        self.assertEqual(early_candidate.metrics.makespan, 90)
        self.assertLess(early_candidate.metrics.sort_key, wait_candidate.metrics.sort_key)
        self.assertEqual(sum(item.activity_id == "power_on" for item in early_candidate.executions), 2)

    def test_validator_rejects_tampered_result(self):
        state = self.replay(load_actions("module_x_wait_actions.json"))
        candidate = self.simulator.candidate_from_state(
            state, algorithm="MANUAL", run_id="tampered", seed=None, discovered_at_seconds=0
        )
        tampered = replace(candidate, final_facts=tuple((key, "false") if key == "x_installed" else (key, value) for key, value in candidate.final_facts))
        result = ResultValidator(self.scenario).validate(tampered)
        self.assertFalse(result.valid)
        self.assertEqual(result.code, "FINAL_FACTS_MISMATCH")


if __name__ == "__main__":
    unittest.main()

