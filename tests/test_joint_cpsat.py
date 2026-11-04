import copy
import random
import tempfile
import time
from pathlib import Path
import unittest

from experiments.joint_cpsat.cases import activity, base_payload, small_case, large_case, serial_witness
from experiments.joint_cpsat.solver import build, solve, UnsupportedModel
from planner_experiment.benchmark_suite.oracle import exact_answer


class JointTests(unittest.TestCase):
    def compare(self, payload, expected=None):
        oracle = exact_answer(payload, seconds=20)
        self.assertIn(oracle['status'], ('PROVEN_OPTIMAL','PROVEN_INFEASIBLE'), oracle)
        result = solve(payload, 20)
        self.assertFalse(result['rejections'], result)
        if oracle['status'] == 'PROVEN_INFEASIBLE':
            self.assertEqual(result['status'], 'PROVEN_INFEASIBLE')
        else:
            self.assertTrue(result['proven_optimal'], result)
            self.assertEqual(result['best']['certificate']['makespan'], oracle['optimal_makespan'])
            if expected is not None:
                self.assertEqual(oracle['optimal_makespan'], expected)
        return result

    def test_time_changes_path(self):
        # Unit durations: wait cost=max(1,arrival)+downstream+6.
        # Early test completes at 4; dispatch at 5; restore test at
        # max(5,arrival)+4; final at max(5+downstream,restore_end)+1.
        for arrival, optimum, tests in [(0,9,1),(8,13,2)]:
            result = self.compare(small_case(arrival=arrival), optimum)
            executions = result['best']['certificate']['executions']
            self.assertEqual(sum(x['activity_id']=='test' for x in executions), tests)
            self.assertFalse(any(x['activity_id'].startswith('optional_') for x in executions))

    def test_negative_goal_and_independent_deletion(self):
        p = base_payload('negative', [activity('remove', delete=['bad'])], ['bad'], [], [], {}, 1)
        p['forbidden_state_ids']=['bad']
        self.compare(p,1)

    def test_competing_transition_and_impossible_cycle(self):
        p=base_payload('transition', [activity('a',transition=['token']),activity('b',transition=['token'])],
                       ['token'], ['done:a','done:b'], [], {}, 2)
        self.compare(p)
        p=base_payload('cycle', [activity('a',required=['done:b']),activity('b',required=['done:a'])],
                       [], ['done:a'], [], {}, 2)
        self.compare(p)

    def test_full_interval_read_write_locks(self):
        p=base_payload('locks', [activity('reader',3,required=['f']),activity('writer',2,outputs=['f'])],
                       ['f'], ['done:reader','done:writer'], [], {}, 2)
        self.compare(p,5)

    def test_input_step_bound_and_unlimited_activity(self):
        p=base_payload('count', [activity('a'),activity('b',required=['done:a'],limit=None)],
                       [], ['done:b'], [], {}, 1)
        self.compare(p)
        p['max_steps']=2
        self.compare(p,2)

    def test_rejects_metadata_and_state_events(self):
        p=small_case()
        with self.assertRaises(UnsupportedModel): build({'scenario':p,'answer':{}})
        p['external_events'][0]['add_state_ids']=['off']
        with self.assertRaises(UnsupportedModel): build(p)

    def test_expired_budget_is_not_a_solver_failure(self):
        with self.assertRaises(TimeoutError):
            solve(small_case(),1,started=time.perf_counter()-2)

    def test_names_do_not_supply_business_knowledge(self):
        p=small_case(arrival=8)
        from planner_experiment.scenario import scenario_from_dict
        s=scenario_from_dict(p)
        states={x:f'f{i}' for i,x in enumerate(s.state_by_id)}
        ids={a['id']:f'x{i}' for i,a in enumerate(p['activities'])}
        for a in p['activities']:
            a['id']=ids[a['id']];a['name']=a['id']
            a['output_state_id']=states[a['output_state_id']]
            for key in ('additional_output_state_ids','delete_effect_state_ids'):
                a[key]=[states[x] for x in a[key]]
            for pre in a['preconditions']: pre['state_id']=states[pre['state_id']]
        for key in ('initial_state_ids','goal_state_ids','forbidden_state_ids'):
            p[key]=[states[x] for x in p[key]]
        p['activities'].reverse()
        self.compare(p,13)

    def test_small_random_models_against_complete_enumeration(self):
        for seed in range(12):
            rng=random.Random(seed)
            acts=[]
            for i in range(4):
                pre=rng.choice([f'f{j}' for j in range(4) if j!=i])
                transition=rng.choice((False,True))
                delete=rng.choice([f'f{j}' for j in range(4) if j!=i and f'f{j}'!=pre])
                acts.append(activity(f'a{i}',rng.randint(1,3),required=[] if transition else [pre],
                                     transition=[pre] if transition else [],outputs=[f'f{i}'],
                                     delete=[delete] if rng.choice((False,True)) else [],limit=2,
                                     resources={'r':1}))
            p=base_payload(f'random-{seed}',acts,['f0'],['f3'],[],{'r':2},6)
            if seed%2: p['forbidden_state_ids']=['f1']
            with self.subTest(seed=seed): self.compare(p)

    def test_supervisor_records_actual_worker_memory_and_budget(self):
        from experiments.joint_cpsat.__main__ import supervise
        from planner_experiment.benchmark_suite.io import write_json,read_json
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'runs') as temporary:
            root=Path(temporary)
            assert root.resolve().is_relative_to((Path.cwd()/'runs').resolve())
            write_json(root/'inputs'/'small.json',small_case())
            result=supervise(root,'small',2,root/'normal')
            ready=read_json(root/'normal'/'ready.json')
            self.assertEqual(result['solver_pid'],ready['pid'])
            self.assertGreaterEqual(result['peak_working_set_bytes'],result['memory']['peak_working_set_bytes'])
            self.assertTrue(result['best'])
            late=supervise(root,'small',.001,root/'expired')
            self.assertEqual(late['accepted_candidates'],0)
            self.assertFalse(late.get('best'))

    def test_large_pair_integrity_and_private_witness(self):
        a, meta=large_case(0,False,'short',101)
        b, _=large_case(2,True,'long',101)
        self.assertEqual(len(a['activities']),256)
        self.assertEqual(a['initial_state_ids'],b['initial_state_ids'])
        for x,y in zip(a['activities'],b['activities']):
            if x['id'].endswith(':downstream'):
                y=copy.deepcopy(y);y['duration']=x['duration']
            self.assertEqual(x,y)
        self.assertGreater(serial_witness(a)['makespan'],0)
        self.assertGreater(serial_witness(b)['makespan'],0)


if __name__=='__main__': unittest.main()
