import copy
import inspect
import tempfile
import unittest
from pathlib import Path

from experiments.recovery_sweep.case import make_case,audit_cycles,REPAIRS
from experiments.recovery_sweep.runtime import unlimited
from experiments.window_joint.__main__ import unlimited as frozen_unlimited
from experiments.window_joint.adapter import certificate
from experiments.joint_cpsat.cases import activity,base_payload
from planner_experiment.benchmark_suite.io import read_json


class RecoverySweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r=Path('runs/mainline-joint-cpsat-forced-recovery1-20261103')
        cls.old=read_json(r/'input.json');saved=copy.deepcopy(cls.old)
        cls.cert=read_json(r/'attempt/best.json')['certificate']
        cls.cases={n:make_case(cls.old,cls.cert,n) for n in (2,3)}
        assert saved==cls.old

    def test_runtime_only_parameterizes_seed(self):
        expected=inspect.getsource(frozen_unlimited).replace('def unlimited(payload, directory, *, cpu=4):','def unlimited(payload, directory, *, cpu=4, seed=11):').replace('solver.parameters.random_seed = 11','solver.parameters.random_seed = seed')
        self.assertEqual(inspect.getsource(unlimited),expected)

    def test_input_changes_are_nested_and_windows_resources_limits_unchanged(self):
        oldacts={a['id']:a for a in self.old['activities']}
        for n,(payload,_,_) in self.cases.items():
            restored=copy.deepcopy(payload)
            changed={a for a,_ in REPAIRS[1:n]}|{f'M{m+1:03d}' for _,m in REPAIRS[1:n]}
            for a in restored['activities']:
                self.assertEqual(a['resource_reqs'],oldacts[a['id']]['resource_reqs'])
                self.assertEqual(a['duration'],oldacts[a['id']]['duration'])
                self.assertEqual(a['max_instances'],oldacts[a['id']]['max_instances'])
            restored['activities']=[copy.deepcopy(oldacts[a['id']]) if a['id'] in changed else a for a in restored['activities']]
            restored['id'],restored['name']=self.old['id'],self.old['name']
            self.assertEqual(restored,self.old)

    def test_r2_attains_bound_with_separate_episodes(self):
        p,m,e=self.cases[2]
        cert=certificate(p,e['witness_raw_rows'])
        self.assertEqual(cert['makespan'],1494)
        self.assertEqual(len(cert['window_checks']),30)
        audit=audit_cycles(p,cert,2)
        self.assertEqual(audit['execution_counts']['M080'],3)
        self.assertEqual(audit['execution_counts']['M081'],3)
        self.assertEqual(audit['execution_counts']['power_off'],2)
        self.assertEqual(m['skeleton']['task_count'],242)

    def test_r3_is_deadline_conflict_not_invalid_input(self):
        p,m,e=self.cases[3]
        self.assertEqual(e['independent_proof']['completion_lower_bound'],1528)
        self.assertEqual(e['independent_proof']['deadline_excess'],32)
        with self.assertRaises(ValueError):certificate(p,e['relaxed_diagnostic']['witness_raw_rows'])
        relaxed=copy.deepcopy(p)
        for w in relaxed['time_windows']:
            if w['activity_id']=='M180':w['latest_finish']=1528
        cert=certificate(relaxed,e['relaxed_diagnostic']['witness_raw_rows'])
        self.assertEqual(cert['makespan'],1528)
        audit=audit_cycles(relaxed,cert,3)
        self.assertEqual(audit['execution_counts']['M080'],4)
        self.assertEqual(audit['execution_counts']['M081'],4)
        self.assertLessEqual(cert['execution_count'],p['max_steps'])

    def test_missing_one_restore_rejected(self):
        p,_,e=self.cases[2]
        rows=[r for r in e['witness_raw_rows'] if not(r['activity_id']=='M080' and r['ordinal']==3)]
        with self.assertRaises((AssertionError,ValueError,RuntimeError)):certificate(p,rows)

    def test_seeds_recorded_and_infeasible_distinct(self):
        p=base_payload('tiny',[activity('a',2)],[],['done:a'],[],{},1)
        p['time_windows']=[]
        for seed in (11,23,37):
            with tempfile.TemporaryDirectory(dir=Path.cwd()/'runs') as d:
                result=unlimited(p,Path(d),seed=seed)
                self.assertEqual(result['best_makespan'],2)
                self.assertIn(f'random_seed: {seed}',read_json(Path(d)/'model.json')['parameters_text'])
                self.assertIsNone(result['time_limit_seconds'])
        p['activities'][0]['preconditions']=[{'state_id':'missing','relation_role':'required'}]
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'runs') as d:
            result=unlimited(p,Path(d),seed=11)
            self.assertEqual(result['status'],'PROVEN_INFEASIBLE')
            self.assertIsNone(result['best_makespan'])


if __name__=='__main__':unittest.main()
