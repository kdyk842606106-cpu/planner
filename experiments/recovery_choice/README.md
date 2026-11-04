# R4/R5: choose merged repairs versus window-forced separate recovery

Both cases have 256 definitions, the original 332 process edges, unchanged
durations, resource capacities/requests, material events and execution limits.
CHECK_B1_10 and CHECK_B2_10 become required offline repairs after completed
M100, delete valid:cal/test, and produce dirty:cal/test. Their completion states
are goals, with no immediate mainline successor gate. Timing and grouping are
solver decisions. Original 30 windows are retained exactly; two repair windows
are added in BOTH cases, for 32 total.

R4: repair1 [830,900], repair2 [830,1200]. Both merged (1462) and split (1494)
witnesses pass Simulator/Validator/resource/window checks. A merged cycle costs
32 overhead +4 repair work. Independent lower bound755+671+36=1462 is attained.

R5 differs ONLY in repair2 earliest start: 830 ->1150. Its latest finish remains
1200. M130 requires on/valid-test and must execute within [1079,1131], strictly
between repair windows. This forces restoration between repairs; simply leaving
the system off and waiting cannot satisfy M130. Two episodes cost64+4=68, giving
attained lower bound755+671+68=1494. Merely using disjoint windows without an
online separator would not prove two recoveries necessary.

Run seeds11/23/37, one worker, CPU4, fresh sequential processes, no time limit.
Reuse frozen recovery_sweep.runtime directly. Private witnesses, episode counts
and lower bounds are never given to formal solvers. Instance construction is
feasibility-controlled synthetic validation, not a sampled business workload.

Commands: `python -m experiments.recovery_choice prepare`, `verify`, `run`,
`summarize`. Existing experiments and frozen sources are unchanged. Summary
records actual repair grouping, power-off/calibration/testing counts on each
finished run alongside timing, CPU, memory and branch/conflict counts.
