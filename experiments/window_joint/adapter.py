"""Window constraints and independent window validation, isolated from engines."""
import copy

from experiments.joint_cpsat.solver import build as base_build, certificate as base_certificate


def effective_payload(payload):
    """Compile opening times into release events so left-shift respects them."""
    effective = copy.deepcopy(payload)
    windows = effective.pop('time_windows', None)
    if not isinstance(windows, list):
        raise ValueError('time_windows must be a list')
    acts = {a['id']: a for a in effective['activities']}
    seen_activities, seen_ids = set(), set()
    event_ids = {e['id'] for e in effective.get('external_events', [])}
    for w in windows:
        aid = w['activity_id']
        if aid not in acts or aid in seen_activities or w['id'] in seen_ids:
            raise ValueError('Unknown or duplicate time-window activity/id')
        seen_activities.add(aid)
        seen_ids.add(w['id'])
        opening, closing = w['earliest_start'], w['latest_finish']
        if type(opening) is not int or type(closing) is not int or opening < effective.get('start_time', 0) or closing < opening + acts[aid]['duration']:
            raise ValueError('Invalid hard time window')
        eid = 'window_open:' + w['id']
        if eid in event_ids:
            raise ValueError('Synthetic release event collision')
        event_ids.add(eid)
        effective.setdefault('external_events', []).append({'id': eid, 'time': opening,
                                                           'add_state_ids': [], 'remove_state_ids': []})
        acts[aid].setdefault('event_reqs', []).append(eid)
    return effective


def check_windows(payload, rows):
    """Independent arithmetic sweep on every selected execution, inclusive ends."""
    windows = {w['activity_id']: w for w in payload['time_windows']}
    evidence = []
    for row in rows:
        if row['activity_id'] not in windows:
            continue
        w = windows[row['activity_id']]
        start = row['start'] if 'start' in row else row['start_time']
        end = row['end'] if 'end' in row else row['end_time']
        if start < w['earliest_start'] or end > w['latest_finish']:
            raise ValueError(f"Window violation {w['id']}: {row['activity_id']} [{start},{end}] outside [{w['earliest_start']},{w['latest_finish']}]")
        evidence.append({'window_id': w['id'], 'activity_id': row['activity_id'], 'start': start, 'end': end,
                         'earliest_start': w['earliest_start'], 'latest_finish': w['latest_finish'],
                         'opening_slack': start-w['earliest_start'], 'closing_slack': w['latest_finish']-end})
    return evidence


def build(payload, *, deadline=None):
    effective = effective_payload(payload)
    model, scenario, slots, stats = base_build(effective, deadline=deadline)
    windows = {w['activity_id']: w for w in payload['time_windows']}
    bounded_slots = 0
    for slot in slots:
        if slot.aid in windows:
            # Opening is already enforced by the corresponding release event.
            model.add(slot.end <= windows[slot.aid]['latest_finish']).only_enforce_if(slot.present)
            bounded_slots += 1
    error = model.validate()
    if error:
        raise ValueError(error)
    stats.update(time_window_count=len(windows), window_bounded_slots=bounded_slots,
                 original_external_event_count=len(payload.get('external_events', [])),
                 effective_release_event_count=len(scenario.external_events),
                 constraints=len(model.proto.constraints), variables=len(model.proto.variables))
    return model, scenario, slots, stats


def certificate(payload, rows):
    check_windows(payload, rows)
    result = base_certificate(effective_payload(payload), rows)
    result['window_checks'] = check_windows(payload, result['executions'])
    result['window_validator'] = 'VALID'
    return result
