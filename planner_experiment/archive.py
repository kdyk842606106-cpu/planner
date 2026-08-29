from __future__ import annotations

import math
from dataclasses import replace

from .models import CandidatePath


class PathArchive:
    """Bounded two-tier archive: quality first, then distinct strategy families."""

    def __init__(
        self,
        max_paths: int = 20,
        *,
        quality_ratio: float = 0.75,
        quality_bound: float = 1.10,
        strategy_bound: float = 1.25,
        pinned_signatures: tuple[str, ...] = (),
    ) -> None:
        self.max_paths = max_paths
        self.quality_capacity = min(max_paths, max(1, math.ceil(max_paths * quality_ratio)))
        self.strategy_capacity = max(0, max_paths - self.quality_capacity)
        self.quality_bound = quality_bound
        self.strategy_bound = strategy_bound
        self.pinned_signatures = frozenset(pinned_signatures)
        self._paths: dict[str, CandidatePath] = {}

    def add(self, candidate: CandidatePath) -> tuple[bool, bool]:
        previous_best = self.best.metrics.sort_key if self.best else None
        before = self._paths.get(candidate.structure_signature)
        pool = dict(self._paths)
        if before is None or candidate.metrics.sort_key < before.metrics.sort_key:
            pool[candidate.structure_signature] = candidate
        self._paths = self._rebalance(pool)
        selected = self._paths.get(candidate.structure_signature)
        changed = selected is not None and (
            before is None or selected.metrics.sort_key < before.metrics.sort_key
        )
        current_best = self.best.metrics.sort_key if self.best else None
        improved = current_best is not None and (previous_best is None or current_best < previous_best)
        return changed, improved

    @property
    def best(self) -> CandidatePath | None:
        return min(self._paths.values(), key=lambda item: item.metrics.sort_key, default=None)

    def paths(self) -> tuple[CandidatePath, ...]:
        return tuple(sorted(self._paths.values(), key=lambda item: (item.metrics.sort_key, item.structure_signature)))

    @property
    def tier_counts(self) -> dict[str, int]:
        return {
            tier: sum(path.archive_tier == tier for path in self._paths.values())
            for tier in ("quality", "strategy")
        }

    def _rebalance(self, pool: dict[str, CandidatePath]) -> dict[str, CandidatePath]:
        if not pool or self.max_paths <= 0:
            return {}
        ordered = sorted(pool.values(), key=lambda item: (item.metrics.sort_key, item.structure_signature))
        incumbent = ordered[0].metrics.makespan
        quality_limit = math.floor(incumbent * self.quality_bound)
        strategy_limit = math.floor(incumbent * self.strategy_bound)
        pinned = [item for item in ordered if item.structure_signature in self.pinned_signatures][: self.max_paths]
        selected: dict[str, CandidatePath] = {
            item.structure_signature: replace(item, archive_tier="strategy") for item in pinned
        }
        quality = [
            item for item in ordered
            if item.metrics.makespan <= quality_limit and item.structure_signature not in selected
        ][: min(self.quality_capacity, max(0, self.max_paths - len(selected)))]
        selected.update(
            (item.structure_signature, replace(item, archive_tier="quality")) for item in quality
        )
        represented = {item.strategy_signature for item in selected.values()}
        best_by_family: dict[str, CandidatePath] = {}
        for item in ordered:
            if item.metrics.makespan > strategy_limit or item.strategy_signature in represented:
                continue
            current = best_by_family.get(item.strategy_signature)
            if current is None or item.metrics.sort_key < current.metrics.sort_key:
                best_by_family[item.strategy_signature] = item
        strategy = sorted(
            best_by_family.values(), key=lambda item: (item.metrics.sort_key, item.strategy_signature)
        )[: min(self.strategy_capacity, max(0, self.max_paths - len(selected)))]
        selected.update(
            (item.structure_signature, replace(item, archive_tier="strategy")) for item in strategy
        )
        if len(selected) < self.max_paths:
            for item in ordered:
                if item.structure_signature in selected or item.metrics.makespan > quality_limit:
                    continue
                selected[item.structure_signature] = replace(item, archive_tier="quality")
                if len(selected) >= self.max_paths:
                    break
        return selected
