from __future__ import annotations

import unittest

from planner_experiment.astar import AnytimeAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Action, Budget
from planner_experiment.scenario import ScenarioError, scenario_from_dict
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator


def repeat_scenario(*, max_instances=None, execution_mode="serial"):
    inspect: dict[str, object] = {
        "id": "inspect",
        "name": "检查",
        "duration": 2,
        "output_state_id": "state:inspected",
        "is_milestone": True,
        "preconditions": [
            {"state_id": "state:ready", "relation_role": "required"}
        ],
    }
    if max_instances is not None:
        inspect["max_instances"] = max_instances
    return scenario_from_dict(
        {
            "id": f"repeat-{execution_mode}-{max_instances}",
            "execution_mode": execution_mode,
            "initial_state_ids": ["state:ready"],
            "goal_state_ids": ["state:processed", "state:inspected"],
            "activities": [
                inspect,
                {
                    "id": "process",
                    "name": "处理检查结果",
                    "duration": 3,
                    "output_state_id": "state:processed",
                    "preconditions": [
                        {"state_id": "state:ready", "relation_role": "required"},
                        {"state_id": "state:inspected", "relation_role": "transition"},
                    ],
                },
            ],
            "max_steps": 4,
            "default_budget": {"time_limit_seconds": 1, "transition_limit": 5000},
        }
    )


class RepeatableActivityTest(unittest.TestCase):
    def test_default_is_unlimited_and_ordinary_activity_can_repeat(self):
        scenario = repeat_scenario()
        self.assertIsNone(scenario.activity_by_id["inspect"].max_instances)
        simulator = PathSimulator(scenario)
        state = simulator.initial_state()
        state = simulator.transition(state, Action("EXECUTE", "inspect"))
        state = simulator.transition(state, Action("EXECUTE", "process"))
        self.assertNotIn("state:inspected", state.active_state_ids)
        self.assertIn(Action("EXECUTE", "inspect"), simulator.enabled_actions(state))
        state = simulator.transition(state, Action("EXECUTE", "inspect"))
        self.assertTrue(simulator.is_goal(state))
        self.assertEqual([item.instance_id for item in state.executions], ["inspect#1", "process#1", "inspect#2"])

    def test_redundant_manual_repeat_is_legal_but_same_business_state(self):
        simulator = PathSimulator(repeat_scenario())
        first = simulator.transition(simulator.initial_state(), Action("EXECUTE", "inspect"))
        second = simulator.transition(first, Action("EXECUTE", "inspect"))
        self.assertEqual(first.active_state_ids, second.active_state_ids)
        self.assertEqual(simulator.business_state_key(first), simulator.business_state_key(second))
        self.assertEqual(second.executions[-1].instance_id, "inspect#2")
        state = simulator.transition(second, Action("EXECUTE", "process"))
        state = simulator.transition(state, Action("EXECUTE", "inspect"))
        candidate = simulator.candidate_from_state(
            state,
            algorithm="MANUAL",
            run_id="redundant-replay",
            seed=None,
            discovered_at_seconds=0,
            normalize=False,
        )
        self.assertTrue(ResultValidator(simulator.scenario).validate(candidate).valid)

    def test_explicit_instance_limit_overrides_repeatable_default(self):
        scenario = repeat_scenario(max_instances=1)
        simulator = PathSimulator(scenario)
        state = simulator.transition(simulator.initial_state(), Action("EXECUTE", "inspect"))
        state = simulator.transition(state, Action("EXECUTE", "process"))
        self.assertNotIn(Action("EXECUTE", "inspect"), simulator.enabled_actions(state))
        self.assertEqual(scenario.activity_by_id["inspect"].max_instances, 1)

    def test_parallel_duplicate_output_write_is_locked(self):
        scenario = repeat_scenario(execution_mode="parallel")
        simulator = PathSimulator(scenario)
        running = simulator.transition(simulator.initial_state(), Action("START", "inspect"))
        enabled = [
            action
            for action in simulator.enabled_actions(running)
            if action.kind == "START" and action.activity_id == "inspect"
        ]
        self.assertEqual(enabled, [])

    def test_engines_keep_required_repeat_and_prune_redundant_loops(self):
        scenario = repeat_scenario()
        budget = Budget(1.5, 10_000)
        astar = AnytimeAStar(scenario).run(scenario, budget, run_id="repeat-a")
        ga = GeneticExplorer(scenario).run(scenario, budget, run_id="repeat-g", seed=11)
        for result in (astar, ga):
            self.assertTrue(result.paths)
            best = result.paths[0]
            self.assertEqual(best.structure_signature, "inspect>process>inspect")
            self.assertEqual(best.metrics.execution_count, 3)
            self.assertTrue(ResultValidator(scenario).validate(best).valid)

    def test_max_steps_must_be_positive(self):
        with self.assertRaises(ScenarioError):
            scenario_from_dict(
                {
                    "id": "invalid-step-bound",
                    "initial_state_ids": ["state:ready"],
                    "goal_state_ids": ["state:done"],
                    "activities": [
                        {
                            "id": "finish",
                            "duration": 1,
                            "output_state_id": "state:done",
                            "is_milestone": True,
                            "preconditions": [],
                        }
                    ],
                    "max_steps": 0,
                }
            )


if __name__ == "__main__":
    unittest.main()
