import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.joint_cpsat.cases import small_case
from experiments.mainline_joint.case import make_case, describe, witness
from experiments.mainline_joint.__main__ import unlimited, fixed_witness_check, write_json
from planner_experiment.benchmark_suite.io import read_json, read_events


class MainlineTests(unittest.TestCase):
    def test_atomic_write_retries_windows_sharing_conflict(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'runs') as directory:
            path = Path(directory) / 'state.json'
            original = Path.replace
            attempts = []
            def replace(source, destination):
                attempts.append(destination)
                if len(attempts) == 1:
                    raise PermissionError('transient reader lock')
                return original(source, destination)
            with patch.object(Path, 'replace', replace):
                write_json(path, {'status': 'RUNNING'})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(read_json(path)['status'], 'RUNNING')

    def test_shape_and_delays(self):
        payload, meta = make_case()
        self.assertEqual((payload, meta), make_case())
        metrics = describe(payload, meta)
        self.assertEqual(metrics['activity_count'], 256)
        self.assertEqual(metrics['skeleton']['task_count'], 240)
        self.assertEqual(metrics['skeleton']['depth'], 180)
        self.assertEqual(metrics['skeleton']['width'], 4)
        self.assertEqual(metrics['alternative_task_fraction'], 0)
        self.assertEqual(payload['max_steps'], sum(a['max_instances'] for a in payload['activities']))
        self.assertEqual({r['id']: r['capacity'] for r in payload['resources']},
                         {'electricians': 2, 'instrument': 1, 'install_team': 1})
        for i, event in enumerate(payload['external_events'], 1):
            self.assertEqual(event['time'] - meta['reference_install_starts'][f'B{i}_08'], meta['uniform_delay'])
            self.assertFalse(event['add_state_ids'] or event['remove_state_ids'])
        self.assertTrue({'mode:on', 'valid:cal', 'valid:test'} <= set(payload['goal_state_ids']))

    def test_both_paths_and_compiler_accept_restoration(self):
        payload, _ = make_case()
        for recover in (False, True):
            proof = witness(payload, recover=recover)
            self.assertEqual(proof['execution_counts']['M080'], 2 if recover else 1)
            self.assertEqual(proof['execution_counts']['M081'], 2 if recover else 1)
            self.assertEqual(len(proof['invalidation_checks']), 3)
            self.assertTrue(all(x['both_qualifications_invalid'] for x in proof['invalidation_checks']))
            self.assertIn(fixed_witness_check(payload, proof['rows'])['status'], ('OPTIMAL', 'FEASIBLE'))

    def test_unlimited_callback_proof_and_files(self):
        # Independently hand/exact-checked regression case: late arrival requires
        # early test plus restoration for the 13-unit optimum.
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'runs') as directory:
            path = Path(directory)
            result = unlimited(small_case(arrival=8), path)
            self.assertEqual(result['status'], 'PROVEN_OPTIMAL', result)
            self.assertEqual(result['best_makespan'], 13)
            self.assertEqual(result['lower_bound'], 13)
            self.assertFalse(result['rejections'])
            self.assertIsNone(result['time_limit_seconds'])
            self.assertEqual(read_json(path / 'model.json')['wall_time_limit'], 'infinity')
            self.assertEqual(read_json(path / 'best.json')['certificate']['makespan'], 13)
            self.assertGreater(len(read_events(path / 'candidates.jsonl')), 0)
            self.assertIsNotNone(result['first_solution_seconds'])
            self.assertIsNotNone(result['first_optimal_seconds'])


if __name__ == '__main__':
    unittest.main()
