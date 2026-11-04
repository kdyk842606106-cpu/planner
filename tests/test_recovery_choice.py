import copy
import inspect
import unittest
from pathlib import Path

from experiments.recovery_choice.case import make_case,audit_cycles,REPAIRS
from experiments.recovery_choice.__main__ import unlimited
from experiments.recovery_sweep.runtime import unlimited as frozen_unlimited
from experiments.window_joint.adapter import certificate
from planner_experiment.benchmark_suite.io import read_json


class RecoveryChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=read_json(Path('runs/mainline-joint-cpsat-dense-prerequisites-20261103/input.json'))
        cls.r1=read_json(Path('runs/mainline-joint-cpsat-forced-recovery1-20261103/attempt/best.json'))['certificate']
        cls.r2=read_json(Path('runs/mainline-joint-cpsat-recovery-r2-r3-20261103/attempts/R2-s11/best.json'))['certificate']
        saved=copy.deepcopy(cls.base)
        cls.cases={v:make_case(cls.base,cls.r1,cls.r2,v) for v in ('R4','R5')}
        assert cls.base==saved

    def test_only_pair_difference_is_second_repair_opening(self):
        p=copy.deepcopy(self.cases['R4'][0]);other=self.cases['R5'][0]
        p['id'],p['name']=other['id'],other['name']
        self.assertEqual(p['time_windows'][-1]['earliest_start'],830)
        p['time_windows'][-1]['earliest_start']=1150
        self.assertEqual(p,other)
        self.assertEqual(other['time_windows'][:30],self.base['time_windows'])
        self.assertEqual(len(other['activities']),256)
        self.assertEqual(len(other['time_windows']),32)

    def test_original_activities_and_edges_retained_except_two_repairs(self):
        baseacts={a['id']:a for a in self.base['activities']}
        for payload,_,_ in self.cases.values():
            for a in payload['activities']:
                if a['id'] not in REPAIRS:self.assertEqual(a,baseacts[a['id']])
                for key in ('duration','resource_reqs','max_instances','event_reqs'):
                    self.assertEqual(a[key],baseacts[a['id']][key])
            for key in ('resources','external_events','initial_state_ids','max_steps'):
                self.assertEqual(payload[key],self.base[key])
            self.assertEqual(payload['goal_state_ids'],self.base['goal_state_ids']+['done:'+a for a in REPAIRS])

    def test_r4_has_real_choice_and_merge_saves32(self):
        p,_,e=self.cases['R4']
        merged=certificate(p,e['witness_raw_rows'])
        split=certificate(p,e['split_witness_raw_rows'])
        self.assertEqual(audit_cycles(p,merged)['repair_episode_count'],1)
        self.assertEqual(audit_cycles(p,split)['repair_episode_count'],2)
        self.assertEqual(merged['makespan'],1462)
        self.assertEqual(split['makespan']-merged['makespan'],32)

    def test_r5_forces_restored_online_separator(self):
        p,_,e=self.cases['R5']
        cert=certificate(p,e['witness_raw_rows'])
        audit=audit_cycles(p,cert)
        self.assertEqual(cert['makespan'],1494)
        self.assertEqual(audit['calibration_count'],3)
        self.assertEqual(audit['testing_count'],3)
        self.assertEqual(audit['repair_episode_count'],2)
        middle=next(w for w in p['time_windows'] if w['activity_id']=='M130')
        self.assertLess(p['time_windows'][-2]['latest_finish'],middle['earliest_start'])
        self.assertLess(middle['latest_finish'],p['time_windows'][-1]['earliest_start'])
        with self.assertRaises(ValueError):certificate(p,self.cases['R4'][2]['witness_raw_rows'])

    def test_missing_repair_or_restore_rejected(self):
        p,_,e=self.cases['R4']
        for aid,ordinal in [(REPAIRS[1],1),('M081',2)]:
            rows=[r for r in e['witness_raw_rows'] if not(r['activity_id']==aid and r['ordinal']==ordinal)]
            with self.assertRaises((ValueError,AssertionError,RuntimeError)):certificate(p,rows)

    def test_no_runtime_or_search_modification(self):
        self.assertIs(unlimited,frozen_unlimited)
        self.assertEqual(inspect.getsource(unlimited),inspect.getsource(frozen_unlimited))


if __name__=='__main__':unittest.main()
