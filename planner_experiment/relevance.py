"""Conservative provider closure for both positive and negative goals."""

from .models import Scenario


def relevant_activity_ids(scenario: Scenario, active: set[str] | None = None) -> frozenset[str]:
    needed = set(scenario.goal_state_ids) - (active or set())
    # Retain delete providers even while a forbidden fact is absent: another
    # necessary activity or timed event may introduce it later.
    negative = set(scenario.forbidden_state_ids)
    relevant: set[str] = set()
    changed = True
    while changed:
        changed = False
        for activity in scenario.activities:
            if activity.id in relevant:
                continue
            if not (set(activity.output_state_ids) & needed or set(activity.removed_state_ids) & negative):
                continue
            relevant.add(activity.id)
            needed.update(set(activity.precondition_state_ids) - (active or set()))
            changed = True
    return frozenset(relevant)
