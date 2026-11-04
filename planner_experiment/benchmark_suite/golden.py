"""Twelve small semantic certificates, including two well-formed infeasible inputs."""
from __future__ import annotations

from .generator import finish_instance


def activity(aid, duration, *, required=(), transition=(), outputs=(), deletes=(), resource=False):
    return {"id": aid, "name": aid, "duration": duration, "output_state_id": f"done:{aid}",
            "additional_output_state_ids": list(outputs), "delete_effect_state_ids": list(deletes),
            "preconditions": [{"state_id": s, "relation_role": "required"} for s in required]
                             + [{"state_id": s, "relation_role": "transition"} for s in transition],
            "resource_reqs": {"r": 1} if resource else {}, "max_instances": 2,
            "is_milestone": not transition}


def execute(aid):
    return {"kind": "EXECUTE", "activity_id": aid}


def start(aid):
    return {"kind": "START", "activity_id": aid}


def advance(minute):
    return {"kind": "ADVANCE", "target_time": minute}


def golden_instances():
    result = []

    def add(number, title, activities, goals, actions, optimum, proof, *, mode="serial", initial=(),
            forbidden=(), events=(), capacity=None, max_steps=12, edges=(), task_map=None):
        payload = {"schema_version": 2, "id": f"G{number:02d}", "name": title,
                   "execution_mode": mode, "start_time": 0, "initial_state_ids": list(initial),
                   "goal_state_ids": list(goals), "forbidden_state_ids": list(forbidden),
                   "activities": activities, "external_events": list(events), "max_steps": max_steps,
                   "resources": [] if capacity is None else [{"id": "r", "capacity": capacity}],
                   "default_budget": {"time_limit_seconds": 1, "transition_limit": 2**60}}
        tasks = task_map or {a["id"]: [a["id"]] for a in activities}
        metadata = {"id": payload["id"], "scale": len(activities), "group": "GOLDEN", "generator_seed": None,
                    "task_activities": tasks, "edges": [list(e) for e in edges],
                    "reference_activity_ids": [a["id"] for a in activities],
                    "reference_h0_minutes": max(1, sum(a["duration"] for a in activities)),
                    "title": title}
        answer = {"status": "PROVEN_INFEASIBLE" if optimum is None else "PROVEN_OPTIMAL",
                  "method": "hand-proof+replay+independent-resource-sweep", "proof": proof}
        if optimum is not None:
            answer.update(optimal_makespan=optimum, lower_bound=optimum, actions=actions)
        result.append(finish_instance(payload, metadata, answer))

    add(1, "串行依赖链", [activity("a", 2), activity("b", 3, required=["done:a"]),
                          activity("c", 5, required=["done:b"])], ["done:c"],
        [execute("a"), execute("b"), execute("c")], 10, "Necessary chain durations: 2+3+5=10.", edges=[("a", "b"), ("b", "c")])
    add(2, "独立并行", [activity("a", 3), activity("b", 5)], ["done:a", "done:b"],
        [start("a"), start("b"), advance(3), advance(5)], 5, "Longest necessary activity is 5; both start at 0.", mode="parallel")
    add(3, "单容量资源", [activity("a", 3, resource=True), activity("b", 5, resource=True)], ["done:a", "done:b"],
        [start("a"), advance(3), start("b"), advance(8)], 8, "Capacity-one workload bound: 3+5=8.", mode="parallel", capacity=1)
    gated = activity("a", 3)
    gated["event_reqs"] = ["arrival"]
    add(4, "事件等待", [gated], ["done:a"], [advance(7), start("a"), advance(10)], 10,
        "Release 7 plus necessary duration 3.", mode="parallel",
        events=[{"id": "arrival", "time": 7, "add_state_ids": [], "remove_state_ids": []}])
    add(5, "可选提供者", [activity("fast", 4, outputs=["goal"]), activity("slow", 9, outputs=["goal"])],
        ["goal"], [execute("fast")], 4, "Every provider lasts at least 4; fast attains 4.",
        task_map={"task": ["fast", "slow"]})
    add(6, "模式往返与重复", [activity("a", 2, transition=["off"], outputs=["on", "ready"]),
                              activity("b", 3, required=["done:a"], transition=["on"], outputs=["off"], deletes=["ready"])],
        ["ready", "done:b"], [execute("a"), execute("b"), execute("a")], 7,
        "b requires on from a and deletes ready; a must run twice: 2+3+2=7.", initial=["off"], edges=[("a", "b")])
    add(7, "独立删除与负目标", [activity("clean", 2, deletes=["bad"])], [], [execute("clean")], 2,
        "Initially bad; only clean deletes it and lasts 2.", initial=["bad"], forbidden=["bad"])
    add(8, "删除后重新校准", [activity("cal", 2, outputs=["calibrated"]),
                              activity("inspect", 1, required=["calibrated"]),
                              activity("repair", 3, required=["done:inspect"], deletes=["calibrated"])],
        ["calibrated", "done:repair"], [execute("cal"), execute("inspect"), execute("repair"), execute("cal")], 8,
        "Serial necessary sequence cal,inspect,repair,cal costs 2+1+3+2.", edges=[("cal", "inspect"), ("inspect", "repair")])
    add(9, "初始已达目标", [activity("unused", 1)], ["ready"], [], 0,
        "Initial state satisfies goal; no negative durations.", initial=["ready"])
    add(10, "仅事件实现目标", [activity("unused", 1)], ["arrived"], [advance(6)], 6,
        "Only event at 6 provides arrived.", mode="parallel",
        events=[{"id": "arrival", "time": 6, "add_state_ids": ["arrived"], "remove_state_ids": []}])
    add(11, "不可解：唯一转移状态竞争", [activity("a", 2, transition=["token"]), activity("b", 3, transition=["token"])],
        ["done:a", "done:b"], None, None, "Both activities consume the sole token; no provider restores it. Parallel writers conflict.",
        mode="parallel", initial=["token"])
    add(12, "不可解：执行总数不足", [activity("a", 2), activity("b", 3, required=["done:a"]),
                                        activity("c", 5, required=["done:b"])], ["done:c"], None, None,
        "Goal requires three distinct executions but max_steps=2.", max_steps=2, edges=[("a", "b"), ("b", "c")])
    return result
