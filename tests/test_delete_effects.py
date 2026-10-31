from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Action, Budget
from planner_experiment.scenario import ScenarioError, scenario_from_dict
from planner_experiment.simulator import InvalidAction, PathSimulator
from planner_experiment.validator import ResultValidator


def payload(mode="serial", present=True):
    return {
        "id": "four-relations", "schema_version": 2, "execution_mode": mode,
        "states": [{"id": "valid"}],
        "initial_state_ids": ["power-off", "pending"] + (["valid"] if present else []),
        "goal_state_ids": ["adjusted"], "forbidden_state_ids": ["valid"], "max_steps": 8,
        "activities": [{"id": "adjust", "duration": 3, "output_state_id": "adjusted",
                        "preconditions": [{"state_id": "power-off", "relation_role": "required"},
                                          {"state_id": "pending", "relation_role": "transition"}],
                        "delete_effect_state_ids": ["valid", "valid"]}],
    }


class DeleteEffectsTest(unittest.TestCase):
    def test_four_relations_and_absent_delete_are_independent(self):
        for mode in ("serial", "parallel"):
            for present in (True, False):
                data = payload(mode, present)
                original = copy.deepcopy(data)
                scenario = scenario_from_dict(data)
                simulator = PathSimulator(scenario)
                initial = simulator.initial_state()
                final = simulator.transition(initial, Action("EXECUTE", "adjust"))
                self.assertEqual(set(final.active_state_ids), {"power-off", "adjusted"})
                self.assertEqual(data, original)
                self.assertEqual(scenario.activities[0].delete_effect_state_ids, ("valid",))
                candidate = simulator.candidate_from_state(final, algorithm="MANUAL", run_id="four", seed=None, discovered_at_seconds=0)
                self.assertTrue(ResultValidator(scenario).validate(candidate).valid)
                self.assertFalse(ResultValidator(scenario).validate(replace(candidate, final_state_ids=("valid",))).valid)

    def test_negative_goal_only_provider_is_retained_by_both_engines(self):
        for mode in ("serial", "parallel"):
            data = payload(mode)
            data["goal_state_ids"] = ["power-off"]  # already true; only deletion is needed
            scenario = scenario_from_dict(data)
            for engine in (AnytimeAStar(scenario), GeneticExplorer(scenario, population_size=16, max_generations=10)):
                result = engine.run(scenario, Budget(1, 2000), run_id="negative", seed=11)
                self.assertTrue(result.paths, (mode, engine.algorithm, result.diagnosis))
                for candidate in result.paths:
                    self.assertTrue(ResultValidator(scenario).validate(candidate).valid)
                    self.assertIn("adjust", [r.activity_id for r in candidate.executions])
                    self.assertIn("false:valid", candidate.provider_signature)

    def test_deleted_result_can_be_restored(self):
        data = payload()
        data["goal_state_ids"] = ["valid", "adjusted"]
        data["forbidden_state_ids"] = []
        data["activities"].append({"id": "recalibrate", "duration": 2, "preconditions": [{"state_id": "adjusted", "relation_role": "required"}], "output_state_id": "valid"})
        scenario = scenario_from_dict(data)
        for engine in (AnytimeAStar(scenario), GeneticExplorer(scenario, population_size=16, max_generations=10)):
            result = engine.run(scenario, Budget(1, 2000), run_id="restore", seed=11)
            self.assertTrue(result.paths)
            self.assertEqual([r.activity_id for r in result.paths[0].executions], ["adjust", "recalibrate"])

    def test_parallel_delete_locks_readers_and_writers(self):
        for role in ("required", "transition"):
            data = payload("parallel")
            data["activities"].append({"id": "z-read", "duration": 1, "output_state_id": "read", "preconditions": [{"state_id": "valid", "relation_role": role}]})
            simulator = PathSimulator(scenario_from_dict(data))
            state = simulator.transition(simulator.initial_state(), Action("START", "adjust"))
            with self.assertRaises(InvalidAction):
                simulator.transition(state, Action("START", "z-read"))
        data = payload("parallel")
        data["activities"].append({"id": "z-write", "duration": 1, "output_state_id": "valid", "preconditions": []})
        simulator = PathSimulator(scenario_from_dict(data))
        state = simulator.transition(simulator.initial_state(), Action("START", "adjust"))
        self.assertNotIn("z-write", [a.activity_id for a in simulator.enabled_actions(state)])

    def test_conflicts_and_unknown_delete_rejected(self):
        for state_id in ("power-off", "pending", "adjusted", "unknown"):
            data = payload()
            data["activities"][0]["delete_effect_state_ids"] = [state_id]
            with self.assertRaises(ScenarioError):
                scenario_from_dict(data)

    def test_delete_contract_is_not_dominated_by_faster_non_delete(self):
        data = payload()
        alternative = copy.deepcopy(data["activities"][0])
        alternative.update(id="fast", duration=1, delete_effect_state_ids=[])
        data["activities"].append(alternative)
        engine = AnytimeAStar(scenario_from_dict(data))
        self.assertNotIn("adjust", engine._dominated_activities)

    def test_v1_and_v2_without_deletes_replay_same_final_state(self):
        old = payload()
        old["schema_version"] = 1
        old["activities"][0].pop("delete_effect_state_ids")
        old["forbidden_state_ids"] = []
        new = copy.deepcopy(old)
        new["schema_version"] = 2
        new["activities"][0]["delete_effect_state_ids"] = []
        states = []
        for data in (old, new):
            sim = PathSimulator(scenario_from_dict(data))
            states.append(sim.transition(sim.initial_state(), Action("EXECUTE", "adjust")))
        self.assertEqual(states[0], states[1])

    def test_all_checked_in_legacy_scenarios_match_v2_with_empty_deletes(self):
        for path in sorted((Path(__file__).resolve().parents[1] / "scenarios").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "activities" not in data:
                continue
            old = scenario_from_dict(data)
            updated = copy.deepcopy(data)
            updated["schema_version"] = 2
            for activity in updated["activities"]:
                activity["delete_effect_state_ids"] = []
            new = scenario_from_dict(updated)
            with self.subTest(scenario=path.name):
                self.assertEqual(old, replace(new, input_schema_version=old.input_schema_version))
                left, right = PathSimulator(old), PathSimulator(new)
                ls, rs = left.initial_state(), right.initial_state()
                for _ in range(12):
                    actions = left.enabled_actions(ls)
                    self.assertEqual(actions, right.enabled_actions(rs))
                    if not actions:
                        break
                    for action in actions:
                        self.assertEqual(left.transition(ls, action), right.transition(rs, action))
                    action = next((a for a in actions if a.kind in {"EXECUTE", "START"}), actions[0])
                    ls, rs = left.transition(ls, action), right.transition(rs, action)


if __name__ == "__main__":
    unittest.main()
