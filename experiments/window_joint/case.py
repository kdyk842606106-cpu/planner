"""Paired input, witness and exact lower bound for one 30-window condition."""
import copy

from planner_experiment.benchmark_suite.io import digest
from experiments.window_joint.adapter import certificate

MAIN_IDS = [f'M{i:03d}' for i in (10, 20, 30, 40, 50, 60, 70, 90, 100, 110, 120, 130, 140, 160, 180)]
BRANCH_IDS = [f'B{b}_{i:02d}' for b in range(1, 4) for i in (3, 8, 12, 16, 20)]


def make_case(baseline_payload, baseline_certificate):
    assert baseline_certificate['makespan'] == 1386
    payload = copy.deepcopy(baseline_payload)
    payload['id'] += '-windows30'
    payload['name'] = payload['id']
    base = {e['activity_id']: e for e in baseline_certificate['executions']}
    cut1, cut2 = base['M090']['start_time'], base['M130']['start_time']

    def shift(t):
        return (40 if t >= cut1 else 0) + (25 if t >= cut2 else 0)

    rows = [{'activity_id': e['activity_id'], 'ordinal': e['ordinal'],
             'start': e['start_time']+shift(e['start_time']),
             'end': e['end_time']+shift(e['start_time'])}
            for e in baseline_certificate['executions']]
    by_aid = {r['activity_id']: r for r in rows}
    windows, details = [], []
    for i, aid in enumerate(MAIN_IDS + BRANCH_IDS, 1):
        e, shifted = base[aid], by_aid[aid]
        driver = aid in ('M090', 'M130')
        opening = shifted['start'] if driver else max(0, e['start_time']-(30 if aid.startswith('M') else 20))
        closing = shifted['end'] + (45 if aid.startswith('M') else 20)
        w = {'id': f'W{i:02d}', 'activity_id': aid, 'earliest_start': opening, 'latest_finish': closing}
        windows.append(w)
        details.append({**w, 'placement': 'mainline' if aid.startswith('M') else 'branch',
                        'purpose': 'delayed_opening_driver' if driver else 'local_window',
                        'baseline_start': e['start_time'], 'baseline_end': e['end_time'],
                        'witness_start': shifted['start'], 'witness_end': shifted['end']})
    payload['time_windows'] = windows
    assert len(windows) == len({w['activity_id'] for w in windows}) == 30
    # No changes to activities, resource constraints, events, state or count bounds.
    for key in baseline_payload:
        if key not in ('id', 'name'):
            assert payload[key] == baseline_payload[key], key
    cert = certificate(payload, rows)
    assert len(cert['window_checks']) == 30
    acts = {a['id']: a for a in payload['activities']}
    suffix = sum(acts[f'M{i:03d}']['duration'] for i in range(130, 181))
    release = next(w['earliest_start'] for w in windows if w['activity_id'] == 'M130')
    # Unique completion-state chain and final goal make this lower bound valid.
    assert 'done:M180' in payload['goal_state_ids']
    for i in range(130, 181):
        aid, state = f'M{i:03d}', f'done:M{i:03d}'
        producers = [a['id'] for a in payload['activities'] if state in [a['output_state_id'], *a['additional_output_state_ids']]]
        assert producers == [aid] and state not in payload['initial_state_ids']
        if i > 130:
            assert {'state_id': f'done:M{i-1:03d}', 'relation_role': 'required'} in acts[aid]['preconditions']
    optimum = release + suffix
    assert cert['makespan'] == optimum == 1451
    evidence = {'baseline_input_hash': digest(baseline_payload), 'window_design': details,
                'generation_policy': 'deterministic windows anchored to the existing validated 1386 schedule; feasibility-controlled case, not an independent random sample',
                'shift_policy': {'cut_1': cut1, 'shift_1': 40, 'cut_2': cut2, 'additional_shift_2': 25},
                'witness_raw_rows': rows, 'witness_certificate': cert,
                'answer': {'status': 'PROVEN_OPTIMAL_BY_RELEASE_CHAIN_AND_WITNESS', 'optimal_makespan': optimum,
                           'lower_bound': optimum, 'release_activity': 'M130', 'release_time': release,
                           'mandatory_suffix_duration': suffix,
                           'proof': 'M130 release plus unique mandatory M130..M180 serial chain is a lower bound; independently window-checked and Simulator/Validator/resource-checked witness attains it'},
                'formal_solver_policy': 'raw activity/state/resource/event/window input only; no witness, answer, bounds or hints'}
    return payload, evidence
