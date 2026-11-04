"""Paired deterministic inputs, legal witnesses and constructive certificates."""
from __future__ import annotations

import copy
import math
import random

from ..models import Action, to_primitive
from ..scenario import scenario_from_dict, scenario_hash
from ..simulator import PathSimulator
from ..validator import ResultValidator
from .complexity import complexity, dag_properties


def task_id(index: int) -> str:
    return f"task:{index:04d}"


def activity_id(index: int) -> str:
    return f"a{index:04d}"


def task_state(task: str) -> str:
    return f"state:{task}"


def make_instance(n: int, group: str, seed: int) -> dict:
    if n not in (8, 16, 32, 64, 128, 256) or group not in (*[f"C{i}" for i in range(7)], "PROVIDER"):
        raise ValueError("Unsupported scale or group")
    rng = random.Random(seed)
    durations = [rng.randint(1, 20) for _ in range(n)]
    resource_ids = [f"r{rng.randrange(2)}" for _ in range(n)]
    w0 = 2 ** (int(math.log2(n)) // 2)
    width = 2 * w0 if group in ("C2", "C6") else w0
    layers = n // width
    edges = [[task_id(i - width), task_id(i)] for i in range(width, n)]
    base_edges = [[task_id(i - w0), task_id(i)] for i in range(w0, n)]
    reference_h0 = dag_properties([task_id(i) for i in range(n)], base_edges,
                                  {task_id(i): durations[i] for i in range(n)})["critical_path_minutes"]
    if group in ("C1", "C6"):
        choices = [[task_id((layer - 1) * width + left), task_id(layer * width + right)]
                   for layer in range(1, layers) for left in range(width) for right in range(width) if left != right]
        edges.extend(random.Random(seed + 10000).sample(choices, len(choices) // 2))
    predecessors = {task_id(i): [] for i in range(n)}
    for left, right in edges:
        predecessors[right].append(left)
    tasks = {task_id(i): [activity_id(i)] for i in range(n)}
    activities = []
    for i in range(n):
        activities.append({"id": activity_id(i), "name": activity_id(i), "duration": durations[i],
                           "output_state_id": f"done:{activity_id(i)}", "additional_output_state_ids": [task_state(task_id(i))],
                           "preconditions": [{"state_id": task_state(p), "relation_role": "required"}
                                             for p in sorted(predecessors[task_id(i)])],
                           "delete_effect_state_ids": [], "resource_reqs": {resource_ids[i]: 1},
                           "max_instances": 2, "is_milestone": True})
    goals = [task_state(task_id(i)) for i in range(n - width, n)]
    initial, restoration_pairs = [], []
    if group in ("C3", "C6"):
        choices = [(level * width + lane, (level + 1) * width + lane)
                   for lane in range(width) for level in range(0, layers - 1, 2)]
        for pair_index, (left, right) in enumerate(choices[:n // 8]):
            zero, one = f"mode:{pair_index}:0", f"mode:{pair_index}:1"
            initial.append(zero)
            activities[left]["preconditions"].append({"state_id": zero, "relation_role": "transition"})
            activities[left]["additional_output_state_ids"].append(one)
            activities[left]["is_milestone"] = False
            activities[right]["preconditions"].append({"state_id": one, "relation_role": "transition"})
            # Independent deletes cannot also be required preconditions. The
            # immutable completion fact preserves the same task precedence.
            for relation in activities[right]["preconditions"]:
                if relation["state_id"] == task_state(task_id(left)):
                    relation["state_id"] = activities[left]["output_state_id"]
            activities[right]["additional_output_state_ids"].append(zero)
            activities[right]["delete_effect_state_ids"].append(task_state(task_id(left)))
            activities[right]["is_milestone"] = False
            goals.append(task_state(task_id(left)))
            restoration_pairs.append([activity_id(left), activity_id(right)])
    events = []
    if group in ("C5", "C6"):
        selected = random.Random(seed + 20000).sample(range(n), math.ceil(n / 8))
        for j, index in enumerate(selected):
            event = {"id": f"event:{j}", "time": math.ceil(reference_h0 * (1 + j % 3) / 4),
                     "add_state_ids": [], "remove_state_ids": []}
            events.append(event)
            activities[index]["event_reqs"] = [event["id"]]
    if group == "PROVIDER":
        for i in random.Random(seed + 30000).sample(range(n), math.ceil(n / 4)):
            alternate = copy.deepcopy(activities[i])
            alternate["id"] = f"b{i:04d}"
            alternate["name"] = alternate["id"]
            alternate["output_state_id"] = f"done:{alternate['id']}"
            alternate["duration"] = max(1, durations[i] // 2)
            activities.append(alternate)
            tasks[task_id(i)].append(alternate["id"])
    capacity = max(1, math.ceil(width / 4)) if group in ("C4", "C6") else n
    payload = {"schema_version": 2, "id": f"n{n}-{group}-g{seed}", "name": f"N={n} {group} seed={seed}",
               "execution_mode": "parallel", "start_time": 0, "initial_state_ids": initial,
               "goal_state_ids": sorted(set(goals)), "forbidden_state_ids": [], "activities": activities,
               "resources": [{"id": f"r{i}", "capacity": capacity} for i in range(2)], "external_events": events,
               "max_steps": 2 * n, "default_budget": {"time_limit_seconds": 10, "transition_limit": 2**60},
               "provenance": {"source": "three-engine/v1"}}
    metadata = {"id": payload["id"], "scale": n, "group": group, "generator_seed": seed,
                "task_activities": tasks, "edges": sorted(edges), "reference_activity_ids": [activity_id(i) for i in range(n)],
                "reference_h0_minutes": reference_h0, "reference_width": w0, "width_parameter": width,
                "restoration_pairs": restoration_pairs,
                "paired_baseline": f"n{n}-C0-g{seed}" if group == "PROVIDER" else None}
    return finish_instance(payload, metadata)


def candidate_from_actions(payload, actions, *, algorithm="CERTIFICATE", run_id="certificate", seed=None):
    scenario = scenario_from_dict(payload)
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    for item in actions:
        state = simulator.transition(state, Action(**item))
    candidate = simulator.candidate_from_state(state, algorithm=algorithm, run_id=run_id, seed=seed,
                                               discovered_at_seconds=0, normalize=False)
    checked = ResultValidator(scenario).validate(candidate)
    if not checked.valid:
        raise ValueError(f"Invalid certificate: {checked.code}")
    independent_resource_check(payload, checked.candidate)
    return checked.candidate


def independent_resource_check(payload, candidate):
    """Half-open intervals: releases precede acquisitions at the same minute."""
    activity_map = {a["id"]: a for a in payload["activities"]}
    for resource in payload.get("resources", []):
        boundaries = {}
        for execution in candidate.executions:
            quantity = activity_map[execution.activity_id].get("resource_reqs", {}).get(resource["id"], 0)
            boundaries[execution.start_time] = boundaries.get(execution.start_time, 0) + quantity
            boundaries[execution.end_time] = boundaries.get(execution.end_time, 0) - quantity
        current = 0
        for _, delta in sorted(boundaries.items()):
            current += delta
            if current < 0 or current > resource["capacity"]:
                raise ValueError("Independent resource sweep failed")
        if current:
            raise ValueError("Resource sweep did not release all allocations")


def serial_witness(payload, metadata):
    scenario = scenario_from_dict(payload)
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    order = dag_properties(list(metadata["task_activities"]), metadata["edges"],
                           {task: 1 for task in metadata["task_activities"]})["topological_order"]
    restore = {right: left for left, right in metadata.get("restoration_pairs", [])}
    sequence = []
    for task in order:
        aid = metadata["task_activities"][task][0]
        sequence.append(aid)
        if aid in restore:
            sequence.append(restore[aid])
    for aid in sequence:
        kind = "START" if scenario.execution_mode == "parallel" else "EXECUTE"
        desired = Action(kind, aid)
        while desired not in simulator.enabled_actions(state):
            advances = [a for a in simulator.enabled_actions(state) if a.kind in ("ADVANCE", "WAIT")]
            if not advances:
                raise ValueError(f"Witness blocked at {aid}")
            state = simulator.transition(state, advances[0])
        state = simulator.transition(state, desired)
        while state.running:
            advance = next(a for a in simulator.enabled_actions(state) if a.kind == "ADVANCE")
            state = simulator.transition(state, advance)
    while not simulator.is_goal(state):
        advances = [a for a in simulator.enabled_actions(state) if a.kind in ("ADVANCE", "WAIT")]
        if not advances:
            raise ValueError("Witness did not reach goal")
        state = simulator.transition(state, advances[0])
    return [to_primitive(a) for a in state.actions]


def analytic_certificate(payload, metadata):
    activity_map = {a["id"]: a for a in payload["activities"]}
    choices = {task: min(ids, key=lambda aid: (activity_map[aid]["duration"], aid))
               for task, ids in metadata["task_activities"].items()}
    durations = {task: activity_map[aid]["duration"] for task, aid in choices.items()}
    graph = dag_properties(list(choices), metadata["edges"], durations)
    predecessors = {task: [] for task in choices}
    for left, right in metadata["edges"]:
        predecessors[right].append(left)
    event_times = {event["id"]: event["time"] for event in payload.get("external_events", [])}
    ends, schedule = {}, {}
    for task in graph["topological_order"]:
        aid = choices[task]
        start = max([0, *[ends[p] for p in predecessors[task]],
                     *[event_times[e] for e in activity_map[aid].get("event_reqs", [])]])
        schedule[aid] = start
        ends[task] = start + durations[task]
    simulator = PathSimulator(scenario_from_dict(payload))
    state = simulator.initial_state()
    pending = dict(schedule)
    while pending or state.running:
        for aid in sorted(aid for aid, start in pending.items() if start == state.time):
            state = simulator.transition(state, Action("START", aid))
            del pending[aid]
        if pending or state.running:
            advance = next(a for a in simulator.enabled_actions(state) if a.kind == "ADVANCE")
            state = simulator.transition(state, advance)
    actions = [to_primitive(a) for a in state.actions]
    candidate = candidate_from_actions(payload, actions)
    lower_bound = max(ends.values(), default=0)
    if candidate.metrics.makespan != lower_bound:
        raise ValueError("Constructive schedule does not attain the analytical bound")
    return {"status": "PROVEN_OPTIMAL", "method": "task-dag-release-lower-bound+attaining-replay",
            "optimal_makespan": lower_bound, "lower_bound": lower_bound, "actions": actions,
            "proof_scope": "all mandatory tasks; nonbinding resources; shortest implementation per task"}


def finish_instance(payload, metadata, answer=None):
    payload["states"] = [to_primitive(state) for state in scenario_from_dict(payload).state_definitions]
    if answer is None:
        actions = serial_witness(payload, metadata)
        witness = candidate_from_actions(payload, actions)
        answer = {"status": "UNKNOWN", "lower_bound": dag_properties(list(metadata["task_activities"]), metadata["edges"],
                  {task: min(a["duration"] for a in payload["activities"] if a["id"] in ids)
                   for task, ids in metadata["task_activities"].items()})["critical_path_minutes"],
                  "witness_makespan": witness.metrics.makespan, "actions": actions}
        if metadata["group"] in ("C0", "C1", "C2", "C5", "PROVIDER"):
            answer = analytic_certificate(payload, metadata)
        metadata["witness_activity_ids"] = [a["activity_id"] for a in answer["actions"] if a["kind"] in ("START", "EXECUTE")]
    elif answer.get("actions") is not None:
        candidate_from_actions(payload, answer["actions"])
        metadata["witness_activity_ids"] = [a["activity_id"] for a in answer["actions"] if a["kind"] in ("START", "EXECUTE")]
    metadata["scenario_hash"] = scenario_hash(scenario_from_dict(payload))
    return {"scenario": payload, "metadata": metadata, "complexity": complexity(payload, metadata), "answer": answer}
