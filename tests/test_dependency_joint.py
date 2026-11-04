import copy
import unittest
from pathlib import Path

from experiments.dependency_joint.case import make_case, reduction, reachability
from experiments.window_joint.adapter import certificate
from planner_experiment.benchmark_suite.io import read_json


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path('runs/mainline-joint-cpsat-windows30-20261103')
        cls.old=read_json(root/'input.json')
        saved=copy.deepcopy(cls.old)
        cls.payload,cls.metrics,cls.evidence=make_case(cls.old,read_json(root/'attempt/best.json')['certificate'])
        assert cls.old==saved

    def test_only_adds_prerequisites(self):
        restored=copy.deepcopy(self.payload)
        restored['id'],restored['name']=self.old['id'],self.old['name']
        acts={a['id']:a for a in restored['activities']}
        for a,b in self.evidence['added_edges']:
            acts[b]['preconditions'].remove({'state_id':'done:'+a,'relation_role':'required'})
        self.assertEqual(restored,self.old)
        self.assertEqual(len(self.payload['activities']),256)
        self.assertEqual(len(self.payload['time_windows']),30)

    def test_new_edges_are_real_orderings_and_finally_nonredundant(self):
        meta=self.evidence['metadata']; nodes=list(meta['task_activities'])
        edges=set(map(tuple,meta['edges'])); added=set(map(tuple,self.evidence['added_edges']))
        durations={a['id']:a['duration'] for a in self.payload['activities']}
        before=reachability(nodes,edges-added,durations)
        self.assertTrue(all(b not in before[a] for a,b in added))
        self.assertTrue(added <= reduction(nodes,edges,durations))
        self.assertEqual(len(added),87)
        self.assertEqual(self.metrics['skeleton']['width'],4)
        self.assertEqual(self.metrics['skeleton']['direct_edge_count'],332)
        self.assertEqual(self.metrics['skeleton']['depth'],192)

    def test_attaining_schedule_and_windows(self):
        cert=certificate(self.payload,self.evidence['witness_raw_rows'])
        self.assertEqual(cert['makespan'],1451)
        self.assertEqual(cert['window_validator'],'VALID')
        self.assertEqual(len(cert['window_checks']),30)

    def test_hand_graph_reduction(self):
        nodes=list('abcd'); durations={n:1 for n in nodes}
        edges={('a','b'),('a','c'),('b','d'),('c','d'),('a','d')}
        self.assertEqual(reduction(nodes,edges,durations),edges-{('a','d')})


if __name__=='__main__': unittest.main()
