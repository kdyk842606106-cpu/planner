# Long mainline, one unlimited joint CP-SAT run

This isolated experiment reuses the frozen generic `experiments.joint_cpsat.solver`
compiler, normalizer and certificates. It changes no existing engine, Simulator,
Validator or older frozen study. Run from `D:/planner` with the project Python:

```
python -m experiments.mainline_joint prepare
python -m experiments.mainline_joint run
```

Default root: `runs/mainline-joint-cpsat-unlimited-20261103`.
Prepare verifies previous study source hashes, validates two private witnesses,
audits each witness against the compiler with fixed decisions, then freezes
input/config/complexity/evidence and a full source snapshot. Audit timeout (60s)
is separate from the formal experiment. The formal worker reads only input.json;
it receives no mandatory-task map, witness, schedule hint or supplied bound.

## Model

- 256 definitions: 180 mainline + 3 x 20 branch process tasks, 4 power/grounding
  operations, 12 optional extended inspections. No interchangeable providers.
- Branch forks after M020/M040/M060 and joins before M110/M145/M175.
  First online branch operation also requires done:M081. The immutable-history
  skeleton includes these three edges: 240 nodes, depth 180, exact width 4.
- Seed 101, main durations 5–10, branch durations 2–6, model time units.
- M080 calibration / M081 testing: each at most 4 executions. Installation
  B1_08/B2_08/B3_08 removes valid:cal and valid:test and produces dirty flags.
  Testing requires valid calibration; both qualifications must be valid at finish.
- Mode cycle isolated -> ready -> on -> off -> isolated; each control at most 6.
  Switching alone does not erase qualifications. Installation does.
- Main M001..M079 and branch 01..12 need isolated mode; M080 onward and branch
  13..20 need on mode (post-test work also needs valid:test). State read/write
  locks last the full execution interval, as in the current Simulator.
- Resources: electricians 2, instrument 1, install_team 1. All process tasks
  require one electrician; installation/preparation also competes for the team,
  test/selected process tasks compete for the instrument. Checks use instrument.
- Three release-only events; each installation's reference earliest start plus
  ceil(0.1 H0). H0 ignores resources, modes/restoration and events. Identical delay
  amounts do not mean identical absolute event times.
- Ordinary tasks/checks max_instances=1, max_steps=282. Final goals include main
  and branch completion, current on mode, current calibration/test validity.
- All occurrence slots optional in CP-SAT. Required process tasks are implied
  by input goals and dependencies, not forced by private generator metadata.

## Measurement and stop rule

Formal CP-SAT: one worker, logical CPU 2, seed 11, makespan-only objective. Native
wall-time and deterministic-time limits remain infinity. No parent time limit.
Other search parameters remain default. Stops at an optimality/infeasibility
proof, a validation/encoding error, native solver termination or user interruption.
Proof is relative to finite execution-count bounds, not unbounded repetition.

Each candidate is normalized and replayed with Simulator, Validator and an
independent resource-interval sweep. Compact raw and replayed schedules and
candidate timings persist in candidates.jsonl; best.json stores the best full
certificate. Validation and persistence overhead are recorded separately and
included in elapsed time. A validated plan matching the solver's lower bound
may stop the search even if its normalized makespan beats the raw objective.
No normalized upper-bound feedback is injected into search.

attempt/state.json updates every five seconds; bounds.jsonl tracks proof progress.
resources.jsonl samples the actual worker PID every second (Windows venv launcher
PID is not used). OS peak working set includes imports. CPU seconds, first legal
candidate, current makespan/bound, branches/conflicts and repetitions are saved.
Branches/conflicts are last sampled at a candidate or solver completion, explicitly
timestamped; they are not counts of complete Simulator states.

An unlimited search can take arbitrarily long; this command deliberately does
not restart an already-started attempt or pretend to resume its internal search.
