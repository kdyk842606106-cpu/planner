from __future__ import annotations

import time
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from planner_experiment.archive import PathArchive
from planner_experiment.astar import AnytimeAStar, RestartedWeightedAStar
from planner_experiment.ga import GeneticExplorer, _EvaluatedGene, _PhenotypeRecord
from planner_experiment.models import Action, Budget
from planner_experiment.scenario import load_scenario
from planner_experiment.simulator import PathSimulator
from planner_experiment.validator import ResultValidator


ROOT = Path(__file__).resolve().parents[1]


def replay_candidate(scenario, activity_ids, *, wait_before=None, run_id="manual"):
    simulator = PathSimulator(scenario)
    state = simulator.initial_state()
    for index, activity_id in enumerate(activity_ids):
        if wait_before and index in wait_before:
            state = simulator.transition(state, Action("WAIT", target_time=wait_before[index]))
        state = simulator.transition(state, Action("EXECUTE", activity_id=activity_id))
    return simulator.candidate_from_state(
        state,
        algorithm="MANUAL",
        run_id=run_id,
        seed=None,
        discovered_at_seconds=0.0,
        normalize=False,
    )


class Phase2FinalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module_x = load_scenario(ROOT / "scenarios" / "module_x.json")
        cls.thermal = load_scenario(ROOT / "scenarios" / "thermal_validation.json")

    def thermal_combined(self, *, with_cycle=False):
        prefix = ["prepare_rig", "calibrate_sensor", "install_specimen"]
        if with_cycle:
            prefix.extend(("heat_up", "cool_to_ambient"))
        return replay_candidate(
            self.thermal,
            (*prefix, "combined_climate_test", "compile_report", "quality_review"),
            wait_before={3 + (2 if with_cycle else 0): 55},
            run_id="combined-cycle" if with_cycle else "combined",
        )

    def thermal_staged(self):
        return replay_candidate(
            self.thermal,
            (
                "prepare_rig",
                "calibrate_sensor",
                "install_specimen",
                "heat_up",
                "hot_test",
                "cool_to_ambient",
                "chill_down",
                "cold_test",
                "warm_to_ambient",
                "compile_report",
                "quality_review",
            ),
            run_id="staged",
        )

    def test_strategy_fingerprint_ignores_reversible_repair_cycle(self):
        plain = self.thermal_combined()
        cycled = self.thermal_combined(with_cycle=True)
        self.assertEqual(plain.strategy_signature, cycled.strategy_signature)
        self.assertNotEqual(plain.causal_core_signature, cycled.causal_core_signature)

    def test_combined_and_staged_are_different_business_strategies(self):
        combined = self.thermal_combined()
        staged = self.thermal_staged()
        self.assertNotEqual(combined.strategy_signature, staged.strategy_signature)
        self.assertIn("combined_climate_test", combined.strategy_signature)
        self.assertIn("hot_test", staged.strategy_signature)
        self.assertIn("cold_test", staged.strategy_signature)

    def test_two_tier_archive_keeps_125_percent_strategy(self):
        base = self.thermal_combined()
        archive = PathArchive(20)
        for index in range(20):
            metrics = replace(base.metrics, makespan=100 + index % 5)
            archive.add(
                replace(
                    base,
                    structure_signature=f"quality-{index}",
                    strategy_signature="family-quality",
                    metrics=metrics,
                )
            )
        reserve = replace(
            base,
            structure_signature="reserve-124",
            strategy_signature="family-reserve",
            metrics=replace(base.metrics, makespan=124),
        )
        rejected = replace(
            base,
            structure_signature="reserve-126",
            strategy_signature="family-too-slow",
            metrics=replace(base.metrics, makespan=126),
        )
        archive.add(reserve)
        archive.add(rejected)
        by_signature = {path.structure_signature: path for path in archive.paths()}
        self.assertEqual(len(by_signature), 20)
        self.assertEqual(by_signature["reserve-124"].archive_tier, "strategy")
        self.assertNotIn("reserve-126", by_signature)

    def test_astar_parent_rebuild_and_strategy_search(self):
        result = AnytimeAStar(self.thermal).run(
            self.thermal, Budget(3.0, 30_000), run_id="parent-rebuild"
        )
        self.assertEqual(result.paths[0].metrics.makespan, 105)
        self.assertFalse(result.stats["stores_full_histories_per_node"])
        self.assertGreater(result.stats["trajectory_reconstruction_count"], 0)
        self.assertTrue(any("hot_test" in path.strategy_signature for path in result.paths))
        validator = ResultValidator(self.thermal)
        self.assertTrue(all(validator.validate(path).valid for path in result.paths))

    def test_ara_and_restarted_reference_have_same_best_rank(self):
        budget = Budget(3.0, 20_000)
        ara = AnytimeAStar(self.module_x).run(self.module_x, budget, run_id="ara")
        reference = RestartedWeightedAStar(self.module_x).run(
            self.module_x, budget, run_id="restart"
        )
        self.assertEqual(ara.paths[0].metrics.sort_key, reference.paths[0].metrics.sort_key)
        self.assertTrue(ara.stats["frontier_reused"])
        self.assertFalse(reference.stats["frontier_reused"])

    def test_ga_two_branch_decode_and_phenotype_reuse(self):
        ga = GeneticExplorer(self.thermal)
        genes = tuple(0.5 for _ in range(ga.gene_count))
        item, used, frames = ga._decode(
            genes,
            run_id="decode",
            seed=11,
            wall_start=time.perf_counter(),
            transition_allowance=100,
        )
        self.assertLessEqual(len(item.decision_sequences), 2)
        self.assertLessEqual(used, 100)
        reused = ga._reuse_phenotype(
            genes,
            (_PhenotypeRecord(frames, item),),
            run_id="decode-reused",
            seed=11,
        )
        self.assertIsNotNone(reused)
        self.assertEqual(item.fitness, reused.fitness)
        self.assertEqual(item.decision_sequences, reused.decision_sequences)

    def test_ga_elites_cap_each_strategy_at_half(self):
        ga = GeneticExplorer(self.thermal, elite_count=4)
        first = self.thermal_combined()
        second = self.thermal_staged()
        state = ga.simulator.initial_state()
        evaluated = []
        for index in range(6):
            candidate = replace(
                first if index < 3 else second,
                structure_signature=f"candidate-{index}",
            )
            evaluated.append(
                _EvaluatedGene(
                    genes=tuple(float(index) for _ in range(ga.gene_count)),
                    fitness=(index,),
                    candidate=candidate,
                    final_state=state,
                    terminal_reason="GOAL",
                    candidates=(candidate,),
                )
            )
        elites = ga._diverse_elites(evaluated)
        counts = Counter(item.candidate.strategy_signature for item in elites)
        self.assertEqual(len(elites), 4)
        self.assertLessEqual(max(counts.values()), 2)


if __name__ == "__main__":
    unittest.main()
