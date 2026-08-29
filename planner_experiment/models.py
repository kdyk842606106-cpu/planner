from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


FactItems = tuple[tuple[str, str], ...]
StateIdItems = tuple[str, ...]
CountItems = tuple[tuple[str, int], ...]
ResourceItems = tuple[tuple[str, int], ...]


def freeze_mapping(value: dict[str, Any] | None) -> FactItems:
    return tuple(sorted((str(key), str(item)) for key, item in (value or {}).items()))


def thaw_mapping(value: FactItems) -> dict[str, str]:
    return dict(value)


def freeze_state_ids(value: Any) -> StateIdItems:
    return tuple(sorted({str(item) for item in (value or ()) if str(item)}))


def count_get(counts: CountItems, key: str) -> int:
    return dict(counts).get(key, 0)


def count_increment(counts: CountItems, key: str) -> CountItems:
    updated = dict(counts)
    updated[key] = updated.get(key, 0) + 1
    return tuple(sorted(updated.items()))


@dataclass(frozen=True)
class StateDefinition:
    id: str
    name: str
    source_activity_id: str | None = None
    legacy_key: str | None = None
    legacy_value: str | None = None

    @property
    def compatibility_fact(self) -> tuple[str, str]:
        if self.legacy_key is not None and self.legacy_value is not None:
            return self.legacy_key, self.legacy_value
        return self.id, "active"


@dataclass(frozen=True)
class ActivityPrecondition:
    state_id: str
    relation_role: Literal["required", "transition"] = "required"


@dataclass(frozen=True)
class ActivityDefinition:
    id: str
    name: str
    duration: int
    preconditions: FactItems = ()
    effects: FactItems = ()
    material_reqs: tuple[str, ...] = ()
    event_reqs: tuple[str, ...] = ()
    resource_reqs: ResourceItems = ()
    # None means no activity-level business limit. Scenario.max_steps remains
    # the hard safety boundary for every generated plan.
    max_instances: int | None = None
    precondition_bindings: tuple[ActivityPrecondition, ...] = ()
    output_state_id: str = ""
    additional_output_state_ids: StateIdItems = ()
    is_milestone: bool = False
    compatibility_remove_state_ids: StateIdItems = ()

    @property
    def required_events(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.material_reqs) | set(self.event_reqs)))

    @property
    def precondition_state_ids(self) -> StateIdItems:
        return tuple(item.state_id for item in self.precondition_bindings)

    @property
    def required_state_ids(self) -> StateIdItems:
        return tuple(item.state_id for item in self.precondition_bindings if item.relation_role == "required")

    @property
    def transition_state_ids(self) -> StateIdItems:
        return tuple(item.state_id for item in self.precondition_bindings if item.relation_role == "transition")

    @property
    def output_state_ids(self) -> StateIdItems:
        return freeze_state_ids((self.output_state_id, *self.additional_output_state_ids))


@dataclass(frozen=True)
class ExternalEvent:
    id: str
    time: int
    effects: FactItems = ()
    add_state_ids: StateIdItems = ()
    remove_state_ids: StateIdItems = ()


@dataclass(frozen=True)
class ResourceDefinition:
    id: str
    capacity: int


@dataclass(frozen=True)
class RunningActivity:
    instance_id: str
    activity_id: str
    ordinal: int
    start_time: int
    end_time: int
    start_facts: FactItems
    resource_allocations: ResourceItems = ()
    start_state_ids: StateIdItems = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    start_time: int
    initial_facts: FactItems
    goal_required: FactItems
    goal_forbidden: FactItems
    materials: tuple[tuple[str, int], ...]
    external_events: tuple[ExternalEvent, ...]
    activities: tuple[ActivityDefinition, ...]
    max_steps: int
    default_time_limit: float
    default_transition_limit: int
    expected_signatures: tuple[str, ...] = ()
    resources: tuple[ResourceDefinition, ...] = ()
    execution_mode: Literal["serial", "parallel"] = "serial"
    provenance: FactItems = ()
    state_definitions: tuple[StateDefinition, ...] = ()
    initial_state_ids: StateIdItems = ()
    goal_state_ids: StateIdItems = ()
    forbidden_state_ids: StateIdItems = ()
    target_activity_ids: tuple[str, ...] = ()

    @property
    def provenance_dict(self) -> dict[str, str]:
        return dict(self.provenance)

    @property
    def activity_by_id(self) -> dict[str, ActivityDefinition]:
        return {activity.id: activity for activity in self.activities}

    @property
    def material_by_id(self) -> dict[str, int]:
        return dict(self.materials)

    @property
    def event_by_id(self) -> dict[str, ExternalEvent]:
        return {event.id: event for event in self.external_events}

    @property
    def resource_by_id(self) -> dict[str, ResourceDefinition]:
        return {resource.id: resource for resource in self.resources}

    @property
    def state_by_id(self) -> dict[str, StateDefinition]:
        return {state.id: state for state in self.state_definitions}

    def compatibility_facts(self, state_ids: StateIdItems) -> FactItems:
        values: dict[str, str] = {}
        definitions = self.state_by_id
        for state_id in state_ids:
            state = definitions.get(state_id)
            key, value = state.compatibility_fact if state is not None else (state_id, "active")
            values[key] = value
        return freeze_mapping(values)


@dataclass(frozen=True)
class Action:
    kind: Literal["EXECUTE", "WAIT", "START", "ADVANCE"]
    activity_id: str | None = None
    target_time: int | None = None

    @property
    def label(self) -> str:
        if self.kind in {"EXECUTE", "START"}:
            return str(self.activity_id)
        return f"{self.kind}_UNTIL_{self.target_time}"


@dataclass(frozen=True)
class FactsDelta:
    time: int
    activity_id: str
    changes: FactItems
    added_state_ids: StateIdItems = ()
    removed_state_ids: StateIdItems = ()


@dataclass(frozen=True)
class ExecutionRecord:
    instance_id: str
    activity_id: str
    activity_name: str
    ordinal: int
    start_time: int
    end_time: int
    before_facts: FactItems
    after_facts: FactItems
    trigger_reason: str
    before_state_ids: StateIdItems = ()
    after_state_ids: StateIdItems = ()


@dataclass(frozen=True)
class PlanEdge:
    from_instance: str
    to_instance: str
    edge_type: Literal["state_causal", "serial_experiment", "resource_order", "state_mutex_order"]
    fact_key: str | None = None


@dataclass(frozen=True)
class SimState:
    facts: FactItems
    time: int
    counts: CountItems = ()
    actions: tuple[Action, ...] = ()
    executions: tuple[ExecutionRecord, ...] = ()
    total_wait: int = 0
    running: tuple[RunningActivity, ...] = ()
    last_started_id: str | None = None
    active_state_ids: StateIdItems = ()

    @property
    def state_key(self) -> tuple[Any, ...]:
        running_key = tuple(
            (item.instance_id, item.activity_id, item.start_time, item.end_time, item.resource_allocations)
            for item in self.running
        )
        return self.active_state_ids, self.time, self.counts, running_key, self.last_started_id

    @property
    def prefix_signature(self) -> str:
        return ">".join(record.activity_id for record in self.executions)


@dataclass(frozen=True)
class PathMetrics:
    goal_check: bool
    makespan: int
    execution_count: int
    transition_count: int
    state_revisit_count: int
    goal_regression_count: int
    non_causal_action_count: int
    total_wait: int
    missing_goal_facts: int = 0
    activity_duration_sum: int = 0
    serial_baseline_makespan: int = 0
    parallel_savings: int = 0
    compression_ratio: float = 0.0
    peak_parallelism: int = 1
    average_parallelism: float = 1.0
    resource_peak: tuple[tuple[str, int], ...] = ()
    resource_utilization: tuple[tuple[str, float], ...] = ()
    critical_path_length: int = 0
    idle_wait_time: int = 0

    @property
    def sort_key(self) -> tuple[int, ...]:
        return (
            0 if self.goal_check else 1,
            self.makespan,
            self.execution_count,
            self.transition_count,
            self.state_revisit_count,
            self.goal_regression_count,
            self.non_causal_action_count,
            self.total_wait,
        )


@dataclass(frozen=True)
class CandidatePath:
    path_id: str
    algorithm: str
    run_id: str
    seed: int | None
    actions: tuple[Action, ...]
    executions: tuple[ExecutionRecord, ...]
    partial_order: tuple[PlanEdge, ...]
    state_trajectory: tuple[FactsDelta, ...]
    final_facts: FactItems
    structure_signature: str
    schedule_signature: str
    causal_core_signature: str
    provider_signature: str
    strategy_signature: str
    metrics: PathMetrics
    raw_makespan: int
    normalization_saved_time: int
    discovered_at_seconds: float
    schema_version: int = 4
    validator_status: str = "UNVALIDATED"
    validator_version: str | None = None
    archive_tier: Literal["quality", "strategy", "unclassified"] = "unclassified"
    final_state_ids: StateIdItems = ()


@dataclass(frozen=True)
class ImprovementPoint:
    elapsed_seconds: float
    score: tuple[int, ...]
    signature: str


@dataclass(frozen=True)
class EngineResult:
    algorithm: str
    run_id: str
    seed: int | None
    status: str
    paths: tuple[CandidatePath, ...] = ()
    first_solution_seconds: float | None = None
    improvements: tuple[ImprovementPoint, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    code: str
    message: str
    candidate: CandidatePath | None = None


@dataclass(frozen=True)
class Budget:
    time_limit_seconds: float
    transition_limit: int
    max_solutions: int = 20


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    return value
