from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from planner_experiment.astar import AnytimeAStar, RestartedWeightedAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.models import Action, Budget, ResourceDefinition
from planner_experiment.scenario import load_scenario, scenario_from_dict
from planner_experiment.simulator import InvalidAction, PathSimulator
from planner_experiment.validator import ResultValidator, VALIDATOR_VERSION


ROOT = Path(__file__).resolve().parents[1]


def replay(simulator: PathSimulator, actions: tuple[Action, ...]):
    state = simulator.initial_state()
    for action in actions:
        state = simulator.transition(state, action)
    return state


def workflow_actions() -> tuple[Action, ...]:
    return (
        Action("START", "prepare"),
        Action("ADVANCE", target_time=10),
        Action("START", "inspection"),
        Action("START", "test_a"),
        Action("START", "test_b"),
        Action("ADVANCE", target_time=25),
        Action("ADVANCE", target_time=30),
        Action("ADVANCE", target_time=40),
        Action("START", "integrate"),
        Action("ADVANCE", target_time=50),
    )


class ParallelSimulatorTest(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(ROOT / "scenarios" / "parallel_workflow.json")
        self.simulator = PathSimulator(self.scenario)

    def test_independent_activities_overlap_and_metrics(self):
        state = replay(self.simulator, workflow_actions())
        candidate = self.simulator.candidate_from_state(
            state, algorithm="MANUAL", run_id="parallel", seed=None, discovered_at_seconds=0
        )
        self.assertTrue(ResultValidator(self.scenario).validate(candidate).valid)
        self.assertEqual(candidate.metrics.makespan, 50)
        self.assertEqual(candidate.metrics.serial_baseline_makespan, 85)
        self.assertEqual(candidate.metrics.parallel_savings, 35)
        self.assertEqual(candidate.metrics.peak_parallelism, 3)
        self.assertIn("t10:[inspection,test_a,test_b]", candidate.schedule_signature)
        self.assertEqual(VALIDATOR_VERSION, "temporal-event-validator-v4")

    def test_resource_capacity_one_yields_seventy(self):
        resources = tuple(
            replace(item, capacity=1) if item.id == "engineer" else item
            for item in self.scenario.resources
        )
        scenario = replace(self.scenario, resources=resources)
        simulator = PathSimulator(scenario)
        actions = (
            Action("START", "prepare"), Action("ADVANCE", target_time=10),
            Action("START", "inspection"), Action("START", "test_a"),
            Action("ADVANCE", target_time=25), Action("ADVANCE", target_time=40),
            Action("START", "test_b"), Action("ADVANCE", target_time=60),
            Action("START", "integrate"), Action("ADVANCE", target_time=70),
        )
        candidate = simulator.candidate_from_state(
            replay(simulator, actions), algorithm="MANUAL", run_id="capacity-one", seed=None, discovered_at_seconds=0
        )
        self.assertEqual(candidate.metrics.makespan, 70)

    def test_capacity_and_canonical_start_order_are_enforced(self):
        state = replay(self.simulator, (Action("START", "prepare"), Action("ADVANCE", target_time=10)))
        state = self.simulator.transition(state, Action("START", "test_a"))
        state = self.simulator.transition(state, Action("START", "test_b"))
        self.assertNotIn("test_a", {item.activity_id for item in self.simulator.enabled_actions(state)})
        with self.assertRaises(InvalidAction):
            self.simulator.transition(state, Action("START", "inspection"))

    def test_goal_requires_empty_running_set_and_advance_is_next_boundary(self):
        state = replay(self.simulator, (Action("START", "prepare"),))
        self.assertFalse(self.simulator.is_goal(state))
        with self.assertRaises(InvalidAction):
            self.simulator.transition(state, Action("ADVANCE", target_time=9))
        self.assertEqual(
            [item.target_time for item in self.simulator.enabled_actions(state) if item.kind == "ADVANCE"],
            [10],
        )

    def test_validator_rejects_tampered_temporal_trace(self):
        state = replay(self.simulator, workflow_actions())
        candidate = self.simulator.candidate_from_state(
            state, algorithm="MANUAL", run_id="tamper", seed=None, discovered_at_seconds=0
        )
        changed = replace(candidate.executions[1], start_time=11)
        tampered = replace(candidate, executions=(candidate.executions[0], changed, *candidate.executions[2:]))
        result = ResultValidator(self.scenario).validate(tampered)
        self.assertFalse(result.valid)
        self.assertEqual(result.code, "EXECUTION_TRACE_MISMATCH")


class TemporalConflictTest(unittest.TestCase):
    def scenario(self, *, event_time=99):
        return scenario_from_dict({
            "id": "locks", "execution_mode": "parallel",
            "initial_facts": {"mode": "ready", "a": "false", "b": "false"},
            "goal": {"required": {"a": "true"}},
            "external_events": [{"id": "mode_change", "time": event_time, "effects": {"mode": "blocked"}}],
            "activities": [
                {"id": "reader", "duration": 10, "preconditions": {"mode": "ready"}, "effects": {"a": "true"}},
                {"id": "writer", "duration": 5, "effects": {"mode": "blocked"}},
                {"id": "same_writer", "duration": 5, "effects": {"a": "true"}},
                {"id": "z_other", "duration": 10, "preconditions": {"mode": "ready"}, "effects": {"b": "true"}}
            ],
            "max_steps": 6
        })

    def test_read_write_and_write_write_overlap_are_rejected(self):
        simulator = PathSimulator(self.scenario())
        state = simulator.transition(simulator.initial_state(), Action("START", "reader"))
        enabled = {item.activity_id for item in simulator.enabled_actions(state) if item.kind == "START"}
        self.assertNotIn("writer", enabled)
        self.assertNotIn("same_writer", enabled)
        self.assertIn("z_other", enabled)

    def test_activity_cannot_cross_breaking_event_but_may_end_at_event(self):
        crossing = PathSimulator(self.scenario(event_time=5))
        self.assertNotIn("reader", {item.activity_id for item in crossing.enabled_actions(crossing.initial_state())})
        exact = PathSimulator(self.scenario(event_time=10))
        state = exact.transition(exact.initial_state(), Action("START", "reader"))
        state = exact.transition(state, Action("ADVANCE", target_time=10))
        self.assertEqual(dict(state.facts)["a"], "true")
        self.assertEqual(dict(state.facts)["mode"], "blocked")


class ParallelEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_scenario(ROOT / "scenarios" / "parallel_workflow.json")

    def test_astar_and_reference_find_fifty(self):
        budget = Budget(3, 50_000, 20)
        result = AnytimeAStar(self.scenario).run(self.scenario, budget, run_id="parallel-a")
        reference = RestartedWeightedAStar(self.scenario).run(self.scenario, budget, run_id="parallel-r")
        self.assertEqual(result.paths[0].metrics.makespan, 50)
        self.assertEqual(reference.paths[0].metrics.sort_key, result.paths[0].metrics.sort_key)
        self.assertTrue(all(ResultValidator(self.scenario).validate(path).valid for path in result.paths))

    def test_ga_five_seeds_find_fifty(self):
        for seed in (11, 23, 37, 53, 71):
            result = GeneticExplorer(self.scenario, population_size=24, max_generations=30).run(
                self.scenario, Budget(2, 12_000, 20), run_id=f"parallel-g-{seed}", seed=seed
            )
            self.assertTrue(result.paths)
            self.assertEqual(result.paths[0].metrics.makespan, 50)
            self.assertTrue(all(ResultValidator(self.scenario).validate(path).valid for path in result.paths))

    def test_parallel_pressure_both_engines_return_valid_candidates(self):
        scenario = load_scenario(ROOT / "scenarios" / "parallel_pressure.json")
        budget = Budget(5, 50_000, 20)
        astar = AnytimeAStar(scenario).run(scenario, budget, run_id="parallel-pressure-a")
        ga = GeneticExplorer(scenario, population_size=32, max_generations=50).run(
            scenario, budget, run_id="parallel-pressure-g", seed=11
        )
        for result in (astar, ga):
            self.assertTrue(result.paths)
            self.assertTrue(all(ResultValidator(scenario).validate(path).valid for path in result.paths))


if __name__ == "__main__":
    unittest.main()
