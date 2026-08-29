from __future__ import annotations

from dataclasses import replace

from .models import CandidatePath, Scenario, ValidationResult
from .simulator import InvalidAction, PathSimulator


VALIDATOR_VERSION = "temporal-event-validator-v4"


class ResultValidator:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def validate(self, candidate: CandidatePath) -> ValidationResult:
        if candidate.schema_version != 4:
            return ValidationResult(False, "UNSUPPORTED_SCHEMA", "Only schema v4 candidates can be validated")
        simulator = PathSimulator(self.scenario)
        state = simulator.initial_state()
        try:
            for action in candidate.actions:
                state = simulator.transition(state, action)
        except (InvalidAction, KeyError, ValueError) as exc:
            return ValidationResult(False, "REPLAY_INVALID_ACTION", str(exc))
        if not simulator.is_goal(state):
            return ValidationResult(False, "GOAL_NOT_REACHED", "Replayed final state does not satisfy the goal")
        rebuilt = simulator.candidate_from_state(
            state,
            algorithm=candidate.algorithm,
            run_id=candidate.run_id,
            seed=candidate.seed,
            discovered_at_seconds=candidate.discovered_at_seconds,
            normalize=False,
        )
        if rebuilt.final_facts != candidate.final_facts:
            return ValidationResult(False, "FINAL_FACTS_MISMATCH", "Candidate final facts differ from replay")
        if rebuilt.final_state_ids != candidate.final_state_ids:
            return ValidationResult(False, "FINAL_STATES_MISMATCH", "Candidate final state IDs differ from replay")
        if rebuilt.executions != candidate.executions:
            return ValidationResult(False, "EXECUTION_TRACE_MISMATCH", "Candidate execution trace differs from replay")
        if rebuilt.metrics != candidate.metrics:
            return ValidationResult(False, "METRICS_MISMATCH", "Candidate metrics differ from replay")
        for attribute in (
            "structure_signature",
            "schedule_signature",
            "causal_core_signature",
            "provider_signature",
            "strategy_signature",
        ):
            if getattr(rebuilt, attribute) != getattr(candidate, attribute):
                return ValidationResult(False, "FINGERPRINT_MISMATCH", f"Candidate {attribute} differs from replay")
        if candidate.raw_makespan < candidate.metrics.makespan:
            return ValidationResult(False, "NORMALIZATION_METRICS_INVALID", "Raw makespan is shorter than normalized makespan")
        if candidate.normalization_saved_time != candidate.raw_makespan - candidate.metrics.makespan:
            return ValidationResult(False, "NORMALIZATION_METRICS_INVALID", "Normalization saving is inconsistent")
        validated = replace(candidate, validator_status="VALID", validator_version=VALIDATOR_VERSION)
        return ValidationResult(True, "OK", "Candidate replay passed", validated)
