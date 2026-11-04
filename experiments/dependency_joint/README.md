# Add-only prerequisite density control

Frozen baseline: `runs/mainline-joint-cpsat-windows30-20261103`.
Keep 256 definitions, 180 mainline tasks, three 20-task branches, all 30 exact
windows, three material events, durations, resources (2/1/1), state effects,
goals and execution bounds unchanged. Add 87 required completion-state edges.

The generator is feasibility-controlled using the existing validated 1451
schedule. It does not infer real business dependencies. New edges are all
nonredundant in the final graph; old edges are retained even if some become
redundant. Process edges rise 245 -> 332, exact width stays 4, longest chain
rises 180 -> 192. Adding edges alone cannot increase partial-order width.
Mean process indegree is over 240 process definitions, separately reported
from all state preconditions over 256 candidate definitions.

Formal solving imports the unchanged window experiment's unlimited runner:
single worker, seed 11, CPU 4, no time limit, same compiler and validator.
Only input.json is read by the worker. Private witness/answer evidence is not
passed as a hint or bound. Optimal value remains 1451 by restriction of the
old problem plus a validated attaining schedule. Timing remains an empirical
single-case comparison and does not establish that density generally hurts.

Run `python -m experiments.dependency_joint prepare`, then `verify`, then `run`.
All artifacts go to a separate frozen directory. Existing runs are untouched.
