from __future__ import annotations

import copy
import unittest

from ortools.sat.python import cp_model

from experiments.business256_oracle import build, extract, certificate
from experiments.business256.model import witness
from planner_experiment.benchmark_suite.io import read_json
from planner_experiment.benchmark_suite.generator import candidate_from_actions
from planner_experiment.benchmark_suite.complexity import dag_properties


def small_item(condition):
    item=copy.deepcopy(read_json(f'D:/planner/runs/business256-20261102/instances/b256-w32-d2-{condition}-g101.json'))
    p=item['scenario']
    p.pop('states',None)
    p['activities']=[a for a in p['activities'] if a['id'].startswith(('m00:','p00:'))]
    for a in p['activities']:
        a['duration']=1
        a['preconditions']=[r for r in a['preconditions'] if not r['state_id'].startswith('history:') or r['state_id'].startswith('history:m00:')]
    p['initial_state_ids']=[s for s in p['initial_state_ids'] if s.startswith(('p00:', 'invalid:m00:'))]
    p['goal_state_ids']=['valid:m00:accept','p00:on']
    p['external_events']=[dict(e,time=0) for e in p['external_events'] if e['id']=='arrival:00']
    meta=item['metadata']
    meta['task_activities']={k:v for k,v in meta['task_activities'].items() if k.startswith('m00:')}
    meta['edges']=[e for e in meta['edges'] if all(n.startswith('m00:') for n in e)]
    actions=witness(p,meta)
    bound=dag_properties(list(meta['task_activities']),meta['edges'],{k:1 for k in meta['task_activities']})['critical_path_minutes']
    item['answer']={'lower_bound':bound,'witness_makespan':candidate_from_actions(p,actions).metrics.makespan,'actions':actions}
    return item


class OracleTests(unittest.TestCase):
    def test_hand_proved_one_module_base_and_restoration(self):
        # T0: config/base 2 + installation branches 3 + switch 2 +
        # communication/calibration/test 3 + acceptance 1 = 11.
        # T2: mandatory A pretest by 10 + off/ground 2 + B work 3 +
        # switch 2 + parallel restore/tests 2 + acceptance 1 = 20.
        for condition,optimum in (('T0',11),('T2',20)):
            item=small_item(condition)
            model,data=build(item)
            solver=cp_model.CpSolver()
            solver.parameters.num_search_workers=1
            solver.parameters.max_time_in_seconds=10
            status=solver.solve(model)
            self.assertEqual(status,cp_model.OPTIMAL)
            self.assertEqual(solver.objective_value,optimum)
            plan=certificate(item,extract(solver,data),data)
            self.assertEqual(plan['makespan'],optimum)

    def test_repeat_writes_are_not_treated_as_inert(self):
        item=small_item('T2')
        model,data=build(item)
        self.assertIn(('m00:cal_a','m00:test_a#pre'),data[8])
        # Both first-completion histories and current-valid flags are protected.
        self.assertTrue(any('m00:test_a#pre' in p and 'm00:install_b' in p for p in data[8]))


if __name__=='__main__':unittest.main()
