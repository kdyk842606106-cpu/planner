# R2/R3: recovery counts and the unchanged-window feasibility boundary

This stage follows the completed R1 experiment. R2 requires separate repairs
after M100 and M140; R3 additionally requires a repair after M160. Each offline
repair invalidates calibration/testing and gates the following mainline task.
There are still 256 candidate definitions, the same 30 windows, resource
requests/capacities, durations, material events, 282 execution bound, and four
allowed instances each of calibration/testing. Existing optional checks are
made required repairs; unrelated existing instances are not regenerated.

R2 has independently proven optimum 1494 and a replayed valid witness.
R3 has lower bound 1528 but M180 must finish by 1496: it is infeasible by 32.
R3 is reported as an infeasibility-boundary verification, never a feasible
performance failure. A private diagnostic extends only M180 to 1528 and
validates a three-recovery witness. That input is NOT used by formal workers.

Six formal runs use seeds 11/23/37, R2 then R3 per seed, fresh processes on CPU4,
one CP-SAT worker, no time limit. The measurement/search loop is copied from
the frozen window runner with only its seed parameterized; a source-equality
test enforces this. All compiler/normalizer/validator/algorithm sources remain
unchanged. Formal workers read raw inputs, never answer evidence or hints.

`python -m experiments.recovery_sweep prepare`, `verify`, `run`, `summarize`.
Completed successful attempts can be skipped on resume. Existing incomplete
attempts are never automatically restarted. Raw resource samples, candidates,
bounds, status, proofs, manifests and source snapshot are retained. R4/R5 are
a later stage after reviewing these outcomes; current R0/R1 baselines have
only the existing seed11 run, not three matched repeats.
