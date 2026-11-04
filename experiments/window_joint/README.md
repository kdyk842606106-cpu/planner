# Thirty hard windows on the 1386 mainline case

This isolated adapter adds 30 inclusive execution windows: earliest_start <=
start, end <= latest_finish. Each selected occurrence must fit wholly inside
its activity's single window. Windows do not force optional activities to run.
The new field is explicitly compiled and independently checked, not silently
passed to the existing parser (which has no native latest-finish support).

15 windows cover mainline tasks and 15 cover branch tasks (5 per branch). The
activity list, durations, resources, initial/final states, original 3 material
events, dependencies, effects, repeat bounds and objective are unchanged from
`runs/mainline-joint-cpsat-unlimited-20261103/input.json`. Windows are based on
its validated 1386 certificate. This is a feasibility-controlled test, not an
independent random business scenario. M090 opening is delayed by 40 relative to
its baseline start; M130 opening is delayed by 65. All other windows surround
the chosen baseline/reference execution with slack. Latest finishes may change
other solver schedules even when they do not increase the optimal makespan.

The original generic compiler and all frozen sources remain unchanged.
`adapter.py` adds an end-bound per optional slot. It also translates each
opening into a release-only event, so the existing event-driven normalizer and
Simulator cannot left-shift a candidate before its window. Thus the effective
model has 3 material events + 30 synthetic window-opening events. The adapter
checks raw and normalized intervals against the original window list. Existing
Simulator, Validator and independent resource sweep checks still apply.

The runner copies the existing unlimited measurement protocol into this separate
package. CP-SAT remains one worker, seed 11, default search parameters and no
wall/deterministic time cap. CPU 4 is used because the prior run occupies CPU 2;
cross-run timing comparisons are exploratory. No existing process is stopped.
The formal worker reads only input.json, not private witnesses or derived bounds.

Private evidence includes an independently replayed, resource/window-checked
witness and the bound M130.earliest_start + sum(duration[M130..M180]). Fixed
witness embedding checks only model feasibility before the formal search; it is
never a hint. The formal CP-SAT must find and prove its own answer.

```
python -m experiments.window_joint prepare
python -m experiments.window_joint run
```

Results live in `runs/mainline-joint-cpsat-windows30-20261103` with immutable
source snapshot, input/window list, raw candidates, best certificate, lower-bound
history, CPU/RSS samples and result status. No automatic reminder is created.
