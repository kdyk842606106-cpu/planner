from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path

from experiments.business256.model import make_instance, reduce_edges, structure
from experiments.business256.observe import cpu_seconds, snapshot, supervise
from planner_experiment.benchmark_suite.generator import candidate_from_actions
from planner_experiment.benchmark_suite.complexity import dag_properties
from planner_experiment.models import Action
from planner_experiment.scenario import scenario_from_dict
from planner_experiment.simulator import PathSimulator


def timed_worker(directory, pipe):
    pipe.send(('ready', {}))
    pipe.recv()
    pipe.send(('cpu_start', time.process_time()))
    pipe.send(('started', time.perf_counter()))
    Path(directory, 'candidate-marker.txt').write_text('durable before deadline')
    while True:
        sum(range(1000))


class BusinessModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = make_instance(16, 2, 'T0', 101)
        cls.restore = make_instance(16, 2, 'T2', 101)
        cls.resource = make_instance(16, 2, 'T3', 101)

    def test_exact_width_and_nonredundant_density(self):
        for seed in (101, 202):
            for width in (8, 16, 32):
                for density in (2, 4):
                    nodes, edges, chains, _ = structure(width, density, seed)
                    graph = dag_properties(nodes, [list(e) for e in edges], {n: 1 for n in nodes})
                    self.assertEqual(graph['width'], width)
                    self.assertEqual(len(edges), 224*density)
                    self.assertEqual(set(edges), reduce_edges(nodes, edges))
                    self.assertEqual(sorted(n for chain in chains for n in chain), sorted(nodes))

    def test_size_no_alternatives_and_witness(self):
        for item in (self.base, self.restore, self.resource):
            self.assertEqual(len(item['scenario']['activities']), 256)
            self.assertEqual(len(item['metadata']['task_activities']), 224)
            self.assertTrue(all(len(v) == 1 for v in item['metadata']['task_activities'].values()))
            candidate = candidate_from_actions(item['scenario'], item['answer']['actions'])
            executed = {x.activity_id for x in candidate.executions}
            self.assertTrue(set(item['metadata']['task_activities']) <= executed)
            self.assertLessEqual(item['answer']['lower_bound'], candidate.metrics.makespan)

    def test_resource_pair_only_changes_capacity(self):
        left, right = copy.deepcopy(self.restore['scenario']), copy.deepcopy(self.resource['scenario'])
        for payload in (left, right):
            for key in ('id', 'name', 'resources'):
                payload.pop(key)
        self.assertEqual(left, right)

    def test_retrofit_preserves_history_and_invalidates_current_certificate(self):
        simulator = PathSimulator(scenario_from_dict(self.restore['scenario']))
        state = simulator.initial_state()
        found = False
        for raw in self.restore['answer']['actions']:
            previous = state
            state = simulator.transition(state, Action(**raw))
            if 'history:m00:install_b' in state.active_state_ids and 'history:m00:install_b' not in previous.active_state_ids:
                self.assertIn('valid:m00:test_a', previous.active_state_ids)
                self.assertIn('history:m00:test_a', state.active_state_ids)
                self.assertNotIn('valid:m00:test_a', state.active_state_ids)
                self.assertIn('invalid:m00:test_a', state.active_state_ids)
                found = True
        self.assertTrue(found)
        self.assertTrue(simulator.is_goal(state))
        self.assertIn('valid:m00:test_a', state.active_state_ids)
        self.assertGreaterEqual(dict(state.counts)['m00:test_a'], 2)

    def test_instrumentation_reads_without_modifying_search(self):
        import os
        import threading
        self.assertGreaterEqual(cpu_seconds(os.getpid()), 0)
        self.assertEqual(snapshot(threading.get_ident()), [])

    def test_external_deadline_retains_evidence_and_cpu(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            result = supervise(timed_worker, (), directory, .25)
            self.assertEqual(result['process_status'], 'TIME_LIMIT')
            self.assertGreater(result['cpu_seconds_measured'], 0)
            self.assertIsNotNone(result['peak_working_set_bytes'])
            self.assertTrue(Path(directory, 'candidate-marker.txt').exists())


if __name__ == '__main__':
    unittest.main()
