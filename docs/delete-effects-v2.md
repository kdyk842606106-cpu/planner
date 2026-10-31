# Four activity relations and compatibility

Input schema V2 adds independent `delete_effect_state_ids`. Deletion targets must be known state IDs but need not be active or have a provider. A native V2 activity may have no transitions and need not be a milestone. The stable automatic output is restored on every completion. Historical execution records remain immutable when a result becomes invalid.

Completion semantics: `(active - transition - independent_delete - legacy_compatibility_remove) union outputs`. Simulator, GA predictions, state locks, execution metrics, causal/provider fingerprints and replay share the derived `removed_state_ids` / `written_state_ids`. Independent deletes are not reversible transitions. Negative goals are part of provider relevance even if a necessary activity later introduces the forbidden fact. A* dominance compares removal contracts.

V1 and legacy key/value loaders remain supported. Missing independent deletes default to empty. Output shape remains schema v4, while the Validator version is v5. Existing snapshots are never upgraded in place. The exploration bridge checks `ENGINE_CAPABILITIES` and rejects outdated engines rather than silently ignoring this field.

Tests: `python -m unittest tests.test_delete_effects -v`; full regression: `python -m unittest discover -s tests -v`. Exploration supplies three-engine POP/CP-SAT/A*/GA tests, JSON/Excel/draft reference tests and browser binding tests.

2026-10-28 acceptance: full engine suite 73 passed; ten checked-in scenarios and 315 sampled successors matched V1 versus V2 with empty deletes. Isolated Git HEAD replays also matched candidate traces, metrics and fingerprints for all ten scenarios; a completed historical v4 candidate replayed through the new Validator. Only scenarios with independent deletes enable the new deletion-provider fingerprints. Exploration backend 460 passed, current Planner browser 16 passed, production build passed. Existing untracked `output/` was preserved.
