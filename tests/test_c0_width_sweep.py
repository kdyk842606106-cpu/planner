from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.c0_width_sweep import build_report, narrower_chains
from planner_experiment.benchmark_suite.generator import candidate_from_actions, make_instance
from planner_experiment.benchmark_suite.io import read_json, write_json
from planner_experiment.benchmark_suite.runner import _run_suite


class WidthSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = make_instance(256, 'C0', 101)
        cls.variants = {w: narrower_chains(cls.baseline, w) for w in (8, 4, 2, 1)}

    def test_only_chain_links_are_added_and_original_input_is_unchanged(self):
        old = make_instance(256, 'C0', 101)
        self.assertEqual(old, self.baseline)
        for width, item in self.variants.items():
            original = {tuple(e) for e in old['metadata']['edges']}
            edges = {tuple(e) for e in item['metadata']['edges']}
            self.assertTrue(original <= edges)
            self.assertEqual(len(edges - original), 16 - width)
            self.assertEqual(item['scenario']['goal_state_ids'], old['scenario']['goal_state_ids'])
            self.assertEqual(item['scenario']['initial_state_ids'], old['scenario']['initial_state_ids'])
            self.assertEqual(item['scenario']['resources'], old['scenario']['resources'])
            for before, after in zip(old['scenario']['activities'], item['scenario']['activities']):
                self.assertEqual({k: v for k, v in before.items() if k != 'preconditions'},
                                 {k: v for k, v in after.items() if k != 'preconditions'})

    def test_width_depth_and_width_specific_proof(self):
        activities = self.baseline['scenario']['activities']
        durations = [a['duration'] for a in activities]
        lane_sums = [sum(durations[lane::16]) for lane in range(16)]
        for width, item in self.variants.items():
            c = item['complexity']
            self.assertEqual((c['skeleton']['width'], c['skeleton']['depth']), (width, 256 // width))
            self.assertEqual((c['state_count'], c['precondition_count']), (512, 256 - width))
            self.assertEqual(c['structurally_relevant_count'], 256)
            self.assertEqual(c['witness_repeat_count'], 0)
            expected = max(sum(lane_sums[lane::width]) for lane in range(width))
            self.assertEqual(item['answer']['optimal_makespan'], expected)
            candidate = candidate_from_actions(item['scenario'], item['answer']['actions'])
            self.assertEqual(candidate.metrics.makespan, expected)
            self.assertEqual(len(candidate.executions), 256)
        self.assertEqual(self.variants[1]['answer']['optimal_makespan'], sum(durations))

    def test_constraints_increase_transitive_order_without_changing_size(self):
        previous = self.baseline['complexity']['skeleton']['order_strength']
        for width in (8, 4, 2, 1):
            value = self.variants[width]['complexity']['skeleton']['order_strength']
            self.assertGreater(value, previous)
            previous = value
        self.assertEqual(previous, 1)
        for width in (0, 3, 16):
            with self.assertRaises(ValueError):
                narrower_chains(self.baseline, width)

    def test_report_keeps_widths_and_historical_reference_separate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as name:
            root = Path(name)
            write_json(root / 'reference/n256-C0-g101.json', self.baseline)
            identifiers = []
            for item in self.variants.values():
                iid = item['metadata']['id']
                identifiers.append(iid)
                write_json(root / 'instances' / f'{iid}.json', item)
            write_json(root / 'manifest.json', {'instance_ids': identifiers})
            summary = build_report(root)
            self.assertEqual((summary['completed_runs'], summary['planned_runs']), (0, 8))
            self.assertEqual(len(summary['rows']), 10)
            self.assertEqual({r['width'] for r in summary['rows']}, {1, 2, 4, 8, 16})
            self.assertEqual(sum(r['reference'] for r in summary['rows']), 2)
            self.assertTrue(all(r['best_makespan'] is None for r in summary['rows']))

    def test_runner_uses_supplied_reporter_and_only_configured_warmup_engines(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as name:
            root = Path(name)
            write_json(root / 'config.json', {'profile': 'c0-width-sweep', 'engines': ['GA', 'ASTAR'],
                       'answers_ready': True, 'oracle_mode': 'precomputed_analytic_certificates',
                       'batch_limit_seconds': 3000})
            write_json(root / 'schedule.json', [])
            reporter_calls = []
            with patch('planner_experiment.benchmark_suite.runner.verify_frozen'), \
                 patch('planner_experiment.benchmark_suite.runner.run_one') as run, \
                 patch('planner_experiment.benchmark_suite.runner.run_oracles') as oracle:
                result = _run_suite(root, 'unused', report_builder=lambda output: reporter_calls.append(output))
            self.assertEqual([call.args[3] for call in run.call_args_list], ['GA', 'ASTAR'])
            oracle.assert_not_called()
            self.assertEqual(result['oracle_mode'], 'precomputed_analytic_certificates')
            self.assertEqual(result['budget_seconds'], 3000)
            self.assertEqual(len(reporter_calls), 1)


if __name__ == '__main__':
    unittest.main()
