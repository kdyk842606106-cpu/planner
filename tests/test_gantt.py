from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from planner_experiment.gantt import render_gantt_svg, select_gantt_candidates, temporal_distance
from planner_experiment.models import Action, EngineResult
from planner_experiment.scenario import load_scenario
from planner_experiment.simulator import PathSimulator


ROOT = Path(__file__).resolve().parents[1]
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def candidate_for(actions, *, run_id):
    scenario = load_scenario(ROOT / "scenarios" / "parallel_workflow.json")
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    for action in actions:
        state = simulator.transition(state, action)
    return scenario, simulator.candidate_from_state(
        state, algorithm="TEST", run_id=run_id, seed=None, discovered_at_seconds=0
    )


BEST_ACTIONS = (
    Action("START", "prepare"), Action("ADVANCE", target_time=10),
    Action("START", "inspection"), Action("START", "test_a"), Action("START", "test_b"),
    Action("ADVANCE", target_time=25), Action("ADVANCE", target_time=30), Action("ADVANCE", target_time=40),
    Action("START", "integrate"), Action("ADVANCE", target_time=50),
)
ALTERNATIVE_ACTIONS = (
    Action("START", "prepare"), Action("ADVANCE", target_time=10),
    Action("START", "test_a"), Action("START", "test_b"),
    Action("ADVANCE", target_time=30), Action("START", "inspection"),
    Action("ADVANCE", target_time=40), Action("ADVANCE", target_time=45),
    Action("START", "integrate"), Action("ADVANCE", target_time=55),
)


class GanttRendererTest(unittest.TestCase):
    def test_svg_is_deterministic_valid_and_contains_activity_metadata(self):
        scenario, candidate = candidate_for(BEST_ACTIONS, run_id="gantt-best")
        first = render_gantt_svg(scenario, candidate, title="并行甘特图 <验证>")
        second = render_gantt_svg(scenario, candidate, title="并行甘特图 <验证>")
        self.assertEqual(first, second)
        root = ET.fromstring(first)
        bars = root.findall(".//svg:g[@id='activity-bars']/svg:rect[@data-activity-id]", SVG_NS)
        self.assertEqual(len(bars), len(candidate.executions))
        self.assertEqual({item.attrib["data-activity-id"] for item in bars}, {item.activity_id for item in candidate.executions})
        test_a = next(item for item in bars if item.attrib["data-activity-id"] == "test_a")
        test_b = next(item for item in bars if item.attrib["data-activity-id"] == "test_b")
        self.assertEqual(test_a.attrib["data-start"], test_b.attrib["data-start"])
        self.assertEqual(test_a.attrib["data-resource"], "engineer")
        self.assertIn("engineer×1", "".join(test_a.itertext()))
        self.assertIn("并行甘特图 <验证>", "".join(root.itertext()))

    def test_temporal_distance_and_alternative_selection_are_stable(self):
        _, best = candidate_for(BEST_ACTIONS, run_id="best")
        _, alternative = candidate_for(ALTERNATIVE_ACTIONS, run_id="alternative")
        self.assertGreater(temporal_distance(best, alternative), 0)
        left = EngineResult("TEST", "selection", None, "OK", (alternative, best))
        right = EngineResult("TEST", "selection", None, "OK", (best, alternative))
        left_selected = select_gantt_candidates(left)
        right_selected = select_gantt_candidates(right)
        self.assertEqual([(role, path.schedule_signature) for role, path, _ in left_selected], [(role, path.schedule_signature) for role, path, _ in right_selected])
        self.assertEqual(left_selected[0][0], "best")
        self.assertEqual(left_selected[1][0], "alternative")
        self.assertEqual(left_selected[1][1].metrics.makespan, 55)

    def test_single_schedule_and_empty_result_degrade_cleanly(self):
        _, candidate = candidate_for(BEST_ACTIONS, run_id="single")
        single = EngineResult("TEST", "single", None, "OK", (candidate,))
        empty = EngineResult("TEST", "empty", None, "TIMEOUT_EMPTY")
        self.assertEqual(len(select_gantt_candidates(single)), 1)
        self.assertEqual(select_gantt_candidates(empty), ())


if __name__ == "__main__":
    unittest.main()
