from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from planner_experiment.astar import AnytimeAStar, RestartedWeightedAStar
from planner_experiment.ga import GeneticExplorer
from planner_experiment.gantt import render_gantt_svg
from planner_experiment.mi_hp_import import canonical_json, project_seed
from planner_experiment.models import Budget
from planner_experiment.scenario import load_scenario
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "scenarios" / "solver_demo_mi_hp_core_parallel.json"
SOURCE_PATH = ROOT.parent / "solver_demo_project" / "seeds" / "011_mechanical_integration_high_parallel_seed.sql"
EXPECTED_SOURCE_HASH = "60A34D6EBDF9397BDCE69D56F5C89F317A1202A97AB6581D7CABF154A169D1CD"
EXPECTED_OVERLAPS = (
    ("MI_A002", "MI_A003"),
    ("MI_A004", "MI_A005"),
    ("MI_A011", "MI_A012"),
    ("MI_A018", "MI_A019"),
    ("MI_A020", "MI_A021"),
    ("MI_A024", "MI_A025"),
    ("MI_A027", "MI_A028"),
    ("MI_A027", "MI_A020"),
)


def maximal_start_candidate(scenario):
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    while not simulator.is_goal(state):
        enabled = simulator.enabled_actions(state)
        starts = [action for action in enabled if action.kind == "START"]
        if starts:
            state = simulator.transition(state, starts[0])
            continue
        advances = [action for action in enabled if action.kind == "ADVANCE"]
        if not advances:
            raise AssertionError("ASAP replay reached a dead end")
        state = simulator.transition(state, advances[0])
    return simulator.candidate_from_state(
        state,
        algorithm="ASAP_REFERENCE",
        run_id="mi-hp-asap",
        seed=None,
        discovered_at_seconds=0,
    )


def overlaps(candidate, left: str, right: str) -> bool:
    records = {record.activity_id: record for record in candidate.executions}
    return (
        records[left].start_time < records[right].end_time
        and records[right].start_time < records[left].end_time
    )


class SolverDemoMIHPDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_scenario(SCENARIO_PATH)

    def test_projection_integrity_and_provenance(self):
        scenario = self.scenario
        self.assertEqual(scenario.execution_mode, "parallel")
        self.assertEqual(len(scenario.activities), 36)
        self.assertEqual(sum(len(item.preconditions) for item in scenario.activities), 52)
        self.assertEqual(len(scenario.resources), 12)
        self.assertEqual(sum(item.duration for item in scenario.activities), 880)
        self.assertEqual(sum(max(0, len(item.resource_reqs) - 1) for item in scenario.activities), 6)
        self.assertEqual(dict(scenario.goal_required), {"mi_hp_mi_a036_done": "true"})
        self.assertEqual(scenario.provenance_dict["source_sha256"], EXPECTED_SOURCE_HASH)
        self.assertEqual(scenario.provenance_dict["projection"], "core_parallel_v1")

    def test_checked_in_projection_matches_source_when_available(self):
        if not SOURCE_PATH.exists():
            self.skipTest("adjacent solver_demo_project is not available")
        generated = canonical_json(project_seed(SOURCE_PATH))
        self.assertEqual(SCENARIO_PATH.read_text(encoding="utf-8"), generated)

    def test_asap_reference_replays_expected_parallel_structure(self):
        candidate = maximal_start_candidate(self.scenario)
        validation = ResultValidator(self.scenario).validate(candidate)
        self.assertTrue(validation.valid, validation.message)
        self.assertEqual(candidate.metrics.makespan, 490)
        self.assertEqual(candidate.metrics.serial_baseline_makespan, 880)
        self.assertEqual(candidate.metrics.parallel_savings, 390)
        self.assertAlmostEqual(candidate.metrics.compression_ratio, 390 / 880)
        self.assertEqual(candidate.metrics.peak_parallelism, 5)
        self.assertEqual(candidate.metrics.execution_count, 36)
        self.assertEqual(len({item.activity_id for item in candidate.executions}), 36)
        for left, right in EXPECTED_OVERLAPS:
            self.assertTrue(overlaps(candidate, left, right), f"expected overlap: {left}/{right}")
        self.assertFalse(overlaps(candidate, "MI_A013", "MI_A014"))

    def test_asap_reference_never_exceeds_resource_capacity(self):
        candidate = maximal_start_candidate(self.scenario)
        for resource in self.scenario.resources:
            events = []
            for record in candidate.executions:
                quantity = dict(self.scenario.activity_by_id[record.activity_id].resource_reqs).get(resource.id, 0)
                if quantity:
                    events.extend(((record.start_time, quantity), (record.end_time, -quantity)))
            used = 0
            for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
                used += delta
                self.assertLessEqual(used, resource.capacity, resource.id)

    def test_asap_reference_gantt_contains_all_thirty_six_activities(self):
        candidate = maximal_start_candidate(self.scenario)
        root = ET.fromstring(render_gantt_svg(self.scenario, candidate, title="MI-HP-001 ASAP"))
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        bars = root.findall(".//svg:g[@id='activity-bars']/svg:rect[@data-activity-id]", namespace)
        self.assertEqual(len(bars), 36)
        self.assertEqual(root.attrib["viewBox"].split()[2], "1440")


class SolverDemoMIHPEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_scenario(SCENARIO_PATH)

    def assert_valid_490_or_better(self, result):
        self.assertTrue(result.paths)
        self.assertLessEqual(result.paths[0].metrics.makespan, 490)
        for path in result.paths:
            validation = ResultValidator(self.scenario).validate(path)
            self.assertTrue(validation.valid, validation.message)
            self.assertEqual(path.metrics.execution_count, 36)

    def test_astar_and_restarted_reference_find_asap_quality_or_better(self):
        budget = Budget(3, 100_000, 20)
        astar = AnytimeAStar(self.scenario).run(self.scenario, budget, run_id="mi-hp-astar")
        reference = RestartedWeightedAStar(self.scenario).run(
            self.scenario, budget, run_id="mi-hp-reference"
        )
        self.assert_valid_490_or_better(astar)
        self.assert_valid_490_or_better(reference)
        self.assertTrue(astar.stats["maximal_start_seed"]["valid"])

    def test_ga_five_seeds_find_asap_quality_and_repeat(self):
        results = []
        for seed in (11, 23, 37, 53, 71):
            result = GeneticExplorer(self.scenario, population_size=8, max_generations=1).run(
                self.scenario,
                Budget(2, 20_000, 20),
                run_id=f"mi-hp-ga-{seed}",
                seed=seed,
            )
            self.assert_valid_490_or_better(result)
            self.assertTrue(result.stats["maximal_start_rule_seed"])
            results.append(result)
        repeated = GeneticExplorer(self.scenario, population_size=8, max_generations=1).run(
            self.scenario,
            Budget(2, 20_000, 20),
            run_id="mi-hp-ga-repeat",
            seed=23,
        )
        self.assertEqual(results[1].paths[0].metrics, repeated.paths[0].metrics)
        self.assertEqual(results[1].paths[0].schedule_signature, repeated.paths[0].schedule_signature)


if __name__ == "__main__":
    unittest.main()
