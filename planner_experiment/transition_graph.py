from __future__ import annotations

from dataclasses import dataclass

from .models import Scenario


@dataclass(frozen=True)
class StateTransitionGraph:
    """Explicit state transition graph built from relation_role=transition."""

    edges: tuple[tuple[str, str, str], ...]
    reversible_components: tuple[tuple[str, ...], ...]
    compatibility_nodes: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def build(cls, scenario: Scenario) -> "StateTransitionGraph":
        edges: set[tuple[str, str, str]] = set()
        nodes: set[str] = set()
        for activity in scenario.activities:
            for old_state_id in activity.transition_state_ids:
                for new_state_id in activity.output_state_ids:
                    if old_state_id == new_state_id:
                        continue
                    nodes.update((old_state_id, new_state_id))
                    edges.add((old_state_id, new_state_id, activity.id))

        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        for old_state_id, new_state_id, _ in edges:
            adjacency[old_state_id].add(new_state_id)
        components = _strongly_connected_components(adjacency)
        reversible = tuple(
            sorted((tuple(sorted(component)) for component in components if len(component) > 1))
        )
        compatibility = tuple(
            sorted(
                (
                    state.id,
                    state.compatibility_fact[0],
                    state.compatibility_fact[1],
                )
                for state in scenario.state_definitions
            )
        )
        return cls(tuple(sorted(edges)), reversible, compatibility)

    @property
    def reversible_state_ids(self) -> frozenset[str]:
        return frozenset(node for component in self.reversible_components for node in component)

    @property
    def reversible_nodes(self) -> frozenset[tuple[str, str]]:
        """Legacy key/value view retained for existing reports and tests."""
        lookup = {state_id: (key, value) for state_id, key, value in self.compatibility_nodes}
        return frozenset(lookup.get(state_id, (state_id, "active")) for state_id in self.reversible_state_ids)

    def is_reversible_transition(self, old_state_id: str, new_state_id: str) -> bool:
        if old_state_id == new_state_id:
            return False
        return any(old_state_id in component and new_state_id in component for component in self.reversible_components)

    def is_reversible_change(self, key: str, old_value: str, new_value: str) -> bool:
        """Compatibility adapter for legacy scoring code."""
        by_fact = {(fact_key, value): state_id for state_id, fact_key, value in self.compatibility_nodes}
        old_state_id = by_fact.get((key, old_value))
        new_state_id = by_fact.get((key, new_value))
        return bool(
            old_state_id
            and new_state_id
            and self.is_reversible_transition(old_state_id, new_state_id)
        )

    def summary(self) -> dict[str, object]:
        definitions = {
            state_id: (key, value) for state_id, key, value in self.compatibility_nodes
        }
        dimensions: dict[str, set[str]] = {}
        for component in self.reversible_components:
            for state_id in component:
                key, value = definitions.get(state_id, (state_id, "active"))
                dimensions.setdefault(key, set()).add(value)
        return {
            "edge_count": len(self.edges),
            "reversible_dimensions": {
                key: sorted(values) for key, values in sorted(dimensions.items())
            },
        }


def _strongly_connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in adjacency.get(node, ()):
            if neighbour not in indices:
                visit(neighbour)
                lowlink[node] = min(lowlink[node], lowlink[neighbour])
            elif neighbour in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbour])
        if lowlink[node] == indices[node]:
            component: set[str] = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            result.append(component)

    for node in adjacency:
        if node not in indices:
            visit(node)
    return result
