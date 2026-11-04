import copy
import unittest
from pathlib import Path

from experiments.cycle_joint.case import make_case, cycle_audit, REPAIR
from experiments.window_joint.adapter import certificate
from planner_experiment.benchmark_suite.io import read_json


class CycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path('runs/mainline-joint-cpsat-dense-prerequisites-20261103')
        cls.old=read_json(root/'input.json')
        cls.oldcert=read_json(root/'attempt/best.json')['certificate']
        saved=copy.deepcopy(cls.old)
        cls.payload,cls.metrics,cls.evidence=make_case(cls.old,cls.oldcert,read_json(root/'complexity.json'))
        assert saved==cls.old

    def test_exact_intervention(self):
        p=copy.deepcopy(self.payload)
        oldacts={a['id']:a for a in self.old['activities']}
        for a in p['activities']:
            if a['id'] in (REPAIR,'M101'):
                self.assertEqual(a['duration'],oldacts[a['id']]['duration'])
                self.assertEqual(a['resource_reqs'],oldacts[a['id']]['resource_reqs'])
                self.assertEqual(a['max_instances'],oldacts[a['id']]['max_instances'])
        p=copy.deepcopy(self.payload)
        p['activities']=[copy.deepcopy(oldacts[a['id']]) if a['id'] in (REPAIR,'M101') else a for a in p['activities']]
        p['id'],p['name']=self.old['id'],self.old['name']
        self.assertEqual(p,self.old)

    def test_old_schedule_no_longer_valid(self):
        raw=[{'activity_id':e['activity_id'],'ordinal':e['ordinal'],'start':e['start_time'],'end':e['end_time']} for e in self.oldcert['executions']]
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):
            certificate(self.payload,raw)

    def test_new_witness_actually_recovers(self):
        cert=certificate(self.payload,self.evidence['witness_raw_rows'])
        audit=cycle_audit(cert)
        self.assertEqual(audit['execution_counts']['M080'],2)
        self.assertEqual(audit['execution_counts']['M081'],2)
        self.assertEqual(audit['execution_counts']['power_off'],1)
        self.assertEqual(len(cert['window_checks']),30)
        self.assertGreater(self.evidence['answer']['lower_bound'],1451)

    def test_omitting_recovery_is_rejected(self):
        raw=[r for r in self.evidence['witness_raw_rows'] if not (r['activity_id']=='M081' and r['ordinal']==2)]
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):
            certificate(self.payload,raw)


if __name__=='__main__': unittest.main()
