"""Input properties, independent of the engine and the resulting plan."""
from __future__ import annotations

import statistics
from collections import Counter

from ..relevance import relevant_activity_ids
from ..scenario import scenario_from_dict
from ..transition_graph import StateTransitionGraph


def dag_properties(nodes: list[str], edges: list[list[str]], durations: dict[str, int]) -> dict:
    """Reachability, longest chain, weighted CP and exact width via Dilworth."""
    successors = {node: set() for node in nodes}
    predecessors = {node: set() for node in nodes}
    for left, right in edges:
        if left == right or left not in successors or right not in successors:
            raise ValueError("Invalid skeleton edge")
        successors[left].add(right)
        predecessors[right].add(left)
    remaining = {node: len(predecessors[node]) for node in nodes}
    ready = sorted(node for node in nodes if not remaining[node])
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(successors[node]):
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("Task skeleton must be acyclic")
    reach = {node: set() for node in nodes}
    depth, end = {}, {}
    for node in order:
        depth[node] = 1 + max((depth[p] for p in predecessors[node]), default=0)
        end[node] = durations[node] + max((end[p] for p in predecessors[node]), default=0)
    for node in reversed(order):
        for child in successors[node]:
            reach[node].add(child)
            reach[node].update(reach[child])
    matched = {}

    def augment(node, seen):
        for child in sorted(reach[node]):
            if child in seen:
                continue
            seen.add(child)
            if child not in matched or augment(matched[child], seen):
                matched[child] = node
                return True
        return False

    matching = sum(augment(node, set()) for node in nodes)
    pairs = len(nodes) * (len(nodes) - 1) // 2
    comparable = sum(map(len, reach.values()))
    return {"task_count": len(nodes), "direct_edge_count": sum(map(len, successors.values())),
            "comparable_pairs": comparable, "all_pairs": pairs,
            "order_strength": comparable / pairs if pairs else 0.0,
            "depth": max(depth.values(), default=0), "width": len(nodes) - matching,
            "critical_path_minutes": max(end.values(), default=0), "topological_order": order}


def complexity(payload: dict, metadata: dict) -> dict:
    scenario = scenario_from_dict(payload)
    activities = scenario.activities
    n = len(activities)
    task_map = metadata["task_activities"]
    original = {a.id: a for a in activities}
    reference = metadata["reference_activity_ids"]
    durations = {task: original[ids[0]].duration for task, ids in task_map.items()}
    skeleton = dag_properties(list(task_map), metadata["edges"], durations)
    h0 = metadata["reference_h0_minutes"]
    providers = [len(ids) for ids in task_map.values()]
    preconditions = sum(len(a.precondition_bindings) for a in activities)
    transitions = sum(len(a.transition_state_ids) for a in activities)
    graph = StateTransitionGraph.build(scenario)
    values = [a.duration for a in activities]
    resources = {
        r.id: {"capacity": r.capacity,
               "reference_load": sum(original[aid].duration * dict(original[aid].resource_reqs).get(r.id, 0)
                                     for aid in reference) / (r.capacity * h0) if h0 else None}
        for r in scenario.resources
    }
    counts = Counter(metadata.get("witness_activity_ids", []))
    return {"activity_count": n, "state_count": len(scenario.state_definitions),
            "precondition_count": preconditions, "mean_preconditions": preconditions / n if n else 0,
            "skeleton": skeleton, "mean_implementations": statistics.mean(providers) if providers else 0,
            "max_implementations": max(providers, default=0),
            "alternative_task_fraction": sum(p > 1 for p in providers) / len(providers) if providers else 0,
            "transition_precondition_fraction": transitions / preconditions if preconditions else 0,
            "independent_delete_activity_fraction": sum(bool(a.delete_effect_state_ids) for a in activities) / n if n else 0,
            "reversible_component_sizes": [len(c) for c in graph.reversible_components],
            "max_steps": scenario.max_steps, "max_instances": {a.id: a.max_instances for a in activities},
            "witness_repeat_count": sum(max(0, value - 1) for value in counts.values()),
            "resources": resources, "resource_activity_fraction": sum(bool(a.resource_reqs) for a in activities) / n if n else 0,
            "event_count": len(scenario.external_events),
            "event_activity_fraction": sum(bool(a.required_events) for a in activities) / n if n else 0,
            "event_time_over_h0": [e.time / h0 for e in scenario.external_events] if h0 else [],
            "positive_goal_count": len(scenario.goal_state_ids), "negative_goal_count": len(scenario.forbidden_state_ids),
            "structurally_relevant_count": len(relevant_activity_ids(scenario)),
            "durations": {"min": min(values, default=0), "max": max(values, default=0),
                          "mean": statistics.mean(values) if values else 0,
                          "cv": statistics.pstdev(values) / statistics.mean(values) if values else 0}}
