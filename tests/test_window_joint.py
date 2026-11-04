import copy
import tempfile
import unittest
from pathlib import Path

from ortools.sat.python import cp_model
from experiments.joint_cpsat.cases import activity, base_payload
from experiments.joint_cpsat.solver import rows_from
from experiments.window_joint.adapter import build, certificate, check_windows, effective_payload
from experiments.window_joint.case import make_case
from experiments.window_joint.__main__ import unlimited
from planner_experiment.benchmark_suite.io import read_json


def tiny():
    p = base_payload('window-tiny', [activity('a', 2)], [], ['done:a'], [], {}, 1)
    p['time_windows'] = [{'id': 'w', 'activity_id': 'a', 'earliest_start': 5, 'latest_finish': 7}]
    return p


class WindowTests(unittest.TestCase):
    def test_exact_boundaries_and_normalization(self):
        p = tiny()
        model, _, slots, _ = build(p)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        self.assertEqual(solver.solve(model), cp_model.OPTIMAL)
        cert = certificate(p, rows_from(solver, slots))
        self.assertEqual(cert['makespan'], 7)
        self.assertEqual(cert['window_checks'][0]['start'], 5)
        self.assertEqual(cert['window_checks'][0]['end'], 7)
        self.assertEqual(len(effective_payload(p)['external_events']), 1)

    def test_both_bounds_and_all_occurrences_checked(self):
        p = tiny()
        for start, end in [(4,6), (6,8)]:
            with self.assertRaises(ValueError):
                certificate(p, [{'activity_id':'a','ordinal':1,'start':start,'end':end}])
        with self.assertRaises(ValueError):
            check_windows(p, [{'activity_id':'a','start':5,'end':7}, {'activity_id':'a','start':7,'end':9}])

    def test_dependency_deadline_infeasible(self):
        p = base_payload('window-unsat', [activity('a', 4), activity('b', 2, required=['done:a'])], [], ['done:b'], [], {}, 2)
        p['time_windows'] = [{'id':'w','activity_id':'b','earliest_start':0,'latest_finish':5}]
        model, _, _, _ = build(p)
        solver = cp_model.CpSolver()
        self.assertEqual(solver.solve(model), cp_model.INFEASIBLE)

    def test_window_does_not_force_optional_activity(self):
        p = tiny()
        p['activities'].append(activity('optional', 2, required=['missing']))
        p['time_windows'].append({'id':'w2','activity_id':'optional','earliest_start':5,'latest_finish':7})
        model, _, slots, _ = build(p)
        solver = cp_model.CpSolver()
        self.assertEqual(solver.solve(model), cp_model.OPTIMAL)
        self.assertEqual([r['activity_id'] for r in rows_from(solver, slots)], ['a'])

    def test_paired_input_and_attaining_proof(self):
        baseline = Path('runs/mainline-joint-cpsat-unlimited-20261103')
        old = read_json(baseline/'input.json')
        saved = copy.deepcopy(old)
        p, evidence = make_case(old, read_json(baseline/'attempt/best.json')['certificate'])
        self.assertEqual(old, saved)
        self.assertEqual(len(p['activities']), 256)
        self.assertEqual(len(p['time_windows']), 30)
        self.assertEqual(evidence['witness_certificate']['makespan'], 1451)
        self.assertEqual(evidence['answer']['lower_bound'], 1451)
        self.assertEqual(len(evidence['witness_certificate']['window_checks']), 30)
        self.assertEqual(len(effective_payload(p)['external_events']), 33)
        for key in old:
            if key not in ('id','name'):
                self.assertEqual(p[key], old[key])

    def test_unlimited_runner_uses_window_adapter(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'runs') as directory:
            result = unlimited(tiny(), Path(directory))
            self.assertEqual(result['status'], 'PROVEN_OPTIMAL')
            self.assertEqual(result['best_makespan'], 7)
            self.assertIsNone(result['time_limit_seconds'])
            self.assertFalse(result['rejections'])
            self.assertEqual(read_json(Path(directory)/'best.json')['certificate']['window_validator'], 'VALID')


if __name__ == '__main__':
    unittest.main()
