from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .models import (
    ActivityDefinition,
    ActivityPrecondition,
    ExternalEvent,
    ResourceDefinition,
    Scenario,
    StateDefinition,
    freeze_mapping,
    freeze_state_ids,
    to_primitive,
)


class ScenarioError(ValueError):
    pass


def legacy_state_id(key: str, value: str) -> str:
    """Stable state ID used only while importing the legacy key/value schema."""
    return f"fact:{quote(str(key), safe='')}={quote(str(value), safe='')}"


def generated_output_state_id(activity_id: str) -> str:
    return f"activity:{activity_id}:output"


def scenario_from_dict(payload: dict[str, Any]) -> Scenario:
    execution_mode = str(payload.get("execution_mode", "serial"))
    resources = tuple(
        ResourceDefinition(id=str(item["id"]), capacity=int(item["capacity"]))
        for item in payload.get("resources", [])
    )
    legacy_materials = {str(key): int(value) for key, value in payload.get("materials", {}).items()}
    state_defs: dict[str, StateDefinition] = {}

    def add_state(
        state_id: str,
        *,
        name: str | None = None,
        source_activity_id: str | None = None,
        legacy_key: str | None = None,
        legacy_value: str | None = None,
    ) -> str:
        state_id = str(state_id)
        candidate = StateDefinition(
            id=state_id,
            name=str(name or state_id),
            source_activity_id=source_activity_id,
            legacy_key=legacy_key,
            legacy_value=legacy_value,
        )
        existing = state_defs.get(state_id)
        if existing is None:
            state_defs[state_id] = candidate
        elif (
            candidate.legacy_key is not None
            and existing.legacy_key is not None
            and (existing.legacy_key, existing.legacy_value) != (candidate.legacy_key, candidate.legacy_value)
        ):
            raise ScenarioError(f"state {state_id} has conflicting definitions")
        elif existing.legacy_key is None and candidate.legacy_key is not None:
            state_defs[state_id] = candidate
        elif existing.source_activity_id is None and source_activity_id is not None:
            state_defs[state_id] = StateDefinition(
                id=existing.id,
                name=existing.name,
                source_activity_id=source_activity_id,
                legacy_key=existing.legacy_key,
                legacy_value=existing.legacy_value,
            )
        return state_id

    def add_legacy_fact(key: Any, value: Any) -> str:
        key_text, value_text = str(key), str(value)
        return add_state(
            legacy_state_id(key_text, value_text),
            name=f"{key_text}={value_text}",
            legacy_key=key_text,
            legacy_value=value_text,
        )

    legacy_initial = {str(key): str(value) for key, value in payload.get("initial_facts", {}).items()}
    legacy_goal_required = {
        str(key): str(value) for key, value in payload.get("goal", {}).get("required", {}).items()
    }
    legacy_goal_forbidden = {
        str(key): str(value) for key, value in payload.get("goal", {}).get("forbidden", {}).items()
    }
    for mapping in (legacy_initial, legacy_goal_required, legacy_goal_forbidden):
        for key, value in mapping.items():
            add_legacy_fact(key, value)

    activities: list[ActivityDefinition] = []
    output_by_activity: dict[str, str] = {}
    for item in payload.get("activities", []):
        activity_id = str(item["id"])
        is_native = isinstance(item.get("preconditions"), list) or "output_state_id" in item
        if is_native:
            output_state_id = add_state(
                str(item.get("output_state_id") or generated_output_state_id(activity_id)),
                name=str(item.get("output_state_name") or f"{item.get('name') or activity_id}完成"),
                source_activity_id=activity_id,
            )
            bindings: list[ActivityPrecondition] = []
            for relation in item.get("preconditions", []):
                if not isinstance(relation, dict) or "state_id" not in relation:
                    raise ScenarioError(f"activity {activity_id} native preconditions must be relation objects")
                state_id = add_state(str(relation["state_id"]), name=relation.get("state_name"))
                bindings.append(
                    ActivityPrecondition(
                        state_id=state_id,
                        relation_role=str(relation.get("relation_role", "required")),
                    )
                )
            additional_outputs = freeze_state_ids(
                add_state(str(state_id), source_activity_id=activity_id)
                for state_id in item.get("additional_output_state_ids", [])
            )
            compatibility_preconditions = freeze_mapping(
                {relation.state_id: "active" for relation in bindings}
            )
            compatibility_effects = freeze_mapping(
                {state_id: "active" for state_id in (output_state_id, *additional_outputs)}
            )
            is_milestone = bool(item.get("is_milestone", False))
        else:
            legacy_preconditions = {
                str(key): str(value) for key, value in item.get("preconditions", {}).items()
            }
            legacy_effects = {str(key): str(value) for key, value in item.get("effects", {}).items()}
            bindings = [
                ActivityPrecondition(
                    state_id=add_legacy_fact(key, value),
                    relation_role=(
                        "transition" if key in legacy_effects and legacy_effects[key] != value else "required"
                    ),
                )
                for key, value in legacy_preconditions.items()
            ]
            effect_state_ids = [add_legacy_fact(key, value) for key, value in legacy_effects.items()]
            if len(effect_state_ids) == 1:
                output_state_id = effect_state_ids[0]
                additional_outputs = ()
            else:
                output_state_id = add_state(
                    generated_output_state_id(activity_id),
                    name=f"{item.get('name') or activity_id}完成",
                    source_activity_id=activity_id,
                )
                additional_outputs = freeze_state_ids(effect_state_ids)
            compatibility_preconditions = freeze_mapping(legacy_preconditions)
            compatibility_effects = freeze_mapping(legacy_effects)
            is_milestone = not any(binding.relation_role == "transition" for binding in bindings)

        output_by_activity[activity_id] = output_state_id
        activities.append(
            ActivityDefinition(
                id=activity_id,
                name=str(item.get("name") or activity_id),
                duration=int(item["duration"]),
                preconditions=compatibility_preconditions,
                effects=compatibility_effects,
                material_reqs=tuple(sorted(str(req) for req in item.get("material_reqs", []))),
                event_reqs=tuple(sorted(str(req) for req in item.get("event_reqs", []))),
                resource_reqs=tuple(
                    sorted((str(key), int(value)) for key, value in item.get("resource_reqs", {}).items())
                ),
                max_instances=(
                    None
                    if item.get("max_instances") is None
                    else int(item["max_instances"])
                ),
                precondition_bindings=tuple(sorted(bindings, key=lambda value: value.state_id)),
                output_state_id=output_state_id,
                additional_output_state_ids=additional_outputs,
                is_milestone=is_milestone,
            )
        )

    initial_state_ids = set(str(item) for item in payload.get("initial_state_ids", []))
    for state_id in tuple(initial_state_ids):
        add_state(state_id)
    initial_state_ids.update(add_legacy_fact(key, value) for key, value in legacy_initial.items())

    goal_state_ids = set(str(item) for item in payload.get("goal_state_ids", []))
    goal_state_ids.update(str(item) for item in payload.get("goal", {}).get("state_ids", []))
    goal_state_ids.update(add_legacy_fact(key, value) for key, value in legacy_goal_required.items())
    forbidden_state_ids = set(str(item) for item in payload.get("forbidden_state_ids", []))
    forbidden_state_ids.update(add_legacy_fact(key, value) for key, value in legacy_goal_forbidden.items())
    target_activity_ids = tuple(sorted(str(item) for item in payload.get("target_activity_ids", [])))
    for activity_id in target_activity_ids:
        if activity_id in output_by_activity:
            goal_state_ids.add(output_by_activity[activity_id])
    for state_id in (*goal_state_ids, *forbidden_state_ids):
        add_state(state_id)

    legacy_values_by_key: dict[str, set[str]] = {}
    for state in state_defs.values():
        if state.legacy_key is not None:
            legacy_values_by_key.setdefault(state.legacy_key, set()).add(state.id)

    external_events_by_id: dict[str, ExternalEvent] = {}
    legacy_event_effects_by_time: dict[int, dict[str, str]] = {}
    for item in payload.get("external_events", []):
        event_id, event_time = str(item["id"]), int(item["time"])
        legacy_effects = {str(key): str(value) for key, value in item.get("effects", {}).items()}
        time_bucket = legacy_event_effects_by_time.setdefault(event_time, {})
        for key, value in legacy_effects.items():
            previous = time_bucket.get(key)
            if previous is not None and previous != value:
                raise ScenarioError(f"events at time {event_time} set conflicting values for {key}")
            time_bucket[key] = value
        add_ids = set(str(value) for value in item.get("add_state_ids", []))
        remove_ids = set(str(value) for value in item.get("remove_state_ids", []))
        for state_id in (*add_ids, *remove_ids):
            add_state(state_id)
        for key, value in legacy_effects.items():
            added = add_legacy_fact(key, value)
            legacy_values_by_key.setdefault(key, set()).add(added)
            add_ids.add(added)
            remove_ids.update(legacy_values_by_key.get(key, set()) - {added})
        external_events_by_id[event_id] = ExternalEvent(
            id=event_id,
            time=event_time,
            effects=freeze_mapping(legacy_effects),
            add_state_ids=freeze_state_ids(add_ids),
            remove_state_ids=freeze_state_ids(remove_ids),
        )
    for material_id, available_at in legacy_materials.items():
        existing = external_events_by_id.get(material_id)
        if existing is not None and existing.time != available_at:
            raise ScenarioError(f"event {material_id} conflicts with legacy material availability")
        external_events_by_id.setdefault(material_id, ExternalEvent(material_id, available_at))

    # Legacy SET effects overwrote the prior value even when the old value was
    # not declared as a precondition. Preserve that behavior only in the import
    # adapter; native transition scenarios never receive implicit removals.
    legacy_states_by_key: dict[str, set[str]] = {}
    for state in state_defs.values():
        if state.legacy_key is not None:
            legacy_states_by_key.setdefault(state.legacy_key, set()).add(state.id)
    activities = [
        replace(
            activity,
            compatibility_remove_state_ids=freeze_state_ids(
                state_id
                for key, value in activity.effects
                for state_id in legacy_states_by_key.get(key, ())
                if state_defs[state_id].legacy_value != value
            ),
        )
        for activity in activities
    ]

    def compatibility_mapping(state_ids: set[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for state_id in state_ids:
            key, value = state_defs[state_id].compatibility_fact
            result[key] = value
        return result

    compatibility_initial = compatibility_mapping(initial_state_ids)
    compatibility_goal = compatibility_mapping(goal_state_ids)
    compatibility_forbidden = compatibility_mapping(forbidden_state_ids)
    budget = payload.get("default_budget", {})
    scenario = Scenario(
        id=str(payload["id"]),
        name=str(payload.get("name") or payload["id"]),
        start_time=int(payload.get("start_time", 0)),
        initial_facts=freeze_mapping(compatibility_initial),
        goal_required=freeze_mapping(compatibility_goal),
        goal_forbidden=freeze_mapping(compatibility_forbidden),
        materials=tuple(sorted(legacy_materials.items())),
        external_events=tuple(sorted(external_events_by_id.values(), key=lambda value: (value.time, value.id))),
        activities=tuple(activities),
        max_steps=int(payload.get("max_steps", 20)),
        default_time_limit=float(budget.get("time_limit_seconds", 5.0)),
        default_transition_limit=int(budget.get("transition_limit", 20_000)),
        expected_signatures=tuple(str(item) for item in payload.get("expected_signatures", [])),
        resources=resources,
        execution_mode=execution_mode,
        provenance=freeze_mapping(payload.get("provenance")),
        state_definitions=tuple(sorted(state_defs.values(), key=lambda value: value.id)),
        initial_state_ids=freeze_state_ids(initial_state_ids),
        goal_state_ids=freeze_state_ids(goal_state_ids),
        forbidden_state_ids=freeze_state_ids(forbidden_state_ids),
        target_activity_ids=target_activity_ids,
    )
    validate_scenario(scenario)
    return scenario


def load_scenario(path: str | Path) -> Scenario:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return scenario_from_dict(payload)


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return to_primitive(scenario)


def scenario_hash(scenario: Scenario) -> str:
    encoded = json.dumps(scenario_to_dict(scenario), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_scenario(scenario: Scenario) -> None:
    issues: list[str] = []
    if not scenario.activities and not scenario.external_events:
        issues.append("activities must not be empty")
    ids = [activity.id for activity in scenario.activities]
    if len(ids) != len(set(ids)):
        issues.append("activity ids must be unique")
    if scenario.execution_mode not in {"serial", "parallel"}:
        issues.append("execution_mode must be serial or parallel")

    state_ids = set(scenario.state_by_id)
    if len(state_ids) != len(scenario.state_definitions):
        issues.append("state ids must be unique")
    resource_ids = [resource.id for resource in scenario.resources]
    if len(resource_ids) != len(set(resource_ids)):
        issues.append("resource ids must be unique")
    for resource in scenario.resources:
        if resource.capacity <= 0:
            issues.append(f"resource {resource.id} has non-positive capacity")

    event_ids = set(scenario.event_by_id)
    producers = set(scenario.initial_state_ids)
    for event in scenario.external_events:
        producers.update(event.add_state_ids)
    for activity in scenario.activities:
        if activity.duration <= 0:
            issues.append(f"activity {activity.id} has non-positive duration")
        if activity.max_instances is not None and activity.max_instances <= 0:
            issues.append(f"activity {activity.id} has non-positive max_instances")
        if not activity.output_state_id:
            issues.append(f"activity {activity.id} has no output_state_id")
        if len(activity.precondition_state_ids) != len(set(activity.precondition_state_ids)):
            issues.append(f"activity {activity.id} repeats a precondition state")
        invalid_roles = {
            item.relation_role
            for item in activity.precondition_bindings
            if item.relation_role not in {"required", "transition"}
        }
        if invalid_roles:
            issues.append(f"activity {activity.id} has invalid relation roles {sorted(invalid_roles)}")
        if not activity.transition_state_ids and not activity.is_milestone:
            issues.append(f"activity {activity.id} must declare a transition state or be a milestone")
        referenced_states = set(activity.precondition_state_ids) | set(activity.output_state_ids)
        missing_states = referenced_states - state_ids
        if missing_states:
            issues.append(f"activity {activity.id} references unknown states {sorted(missing_states)}")
        producers.update(activity.output_state_ids)
        missing_events = set(activity.required_events) - event_ids
        if missing_events:
            issues.append(f"activity {activity.id} references unknown events {sorted(missing_events)}")
        for resource_id, quantity in activity.resource_reqs:
            resource = scenario.resource_by_id.get(resource_id)
            if resource is None:
                issues.append(f"activity {activity.id} references unknown resource {resource_id}")
            elif quantity <= 0 or quantity > resource.capacity:
                issues.append(f"activity {activity.id} has invalid resource requirement {resource_id}={quantity}")

    for state_id in (*scenario.initial_state_ids, *scenario.goal_state_ids, *scenario.forbidden_state_ids):
        if state_id not in state_ids:
            issues.append(f"scenario references unknown state {state_id}")
    for state_id in scenario.goal_state_ids:
        if state_id not in producers:
            issues.append(f"goal state {state_id} has no provider")
    missing_targets = set(scenario.target_activity_ids) - set(ids)
    if missing_targets:
        issues.append(f"unknown target activities {sorted(missing_targets)}")
    if scenario.max_steps <= 0:
        issues.append("max_steps must be positive")
    if scenario.default_time_limit <= 0 or scenario.default_transition_limit <= 0:
        issues.append("default budgets must be positive")
    for event in scenario.external_events:
        missing_states = (set(event.add_state_ids) | set(event.remove_state_ids)) - state_ids
        if missing_states:
            issues.append(f"event {event.id} references unknown states {sorted(missing_states)}")
        if set(event.add_state_ids) & set(event.remove_state_ids):
            issues.append(f"event {event.id} both adds and removes the same state")
    if issues:
        raise ScenarioError("; ".join(issues))
