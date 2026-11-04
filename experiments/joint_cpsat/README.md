# Independent joint CP-SAT validation

This prototype is separate from the frozen three-engine and specialized oracle
experiments. It accepts only a raw Scenario payload. It reads no task map,
phase assignments, witness, optimum, or generator classifications. Every
activity occurrence is optional; input max_instances and max_steps are the only
execution bounds. Initial propagation may infer mandatory actions inside the
solver. This is computation, not privileged input.

## Supported semantics and proof scope

- Positive-duration, non-preemptive, parallel activities.
- Required and transition prerequisites; independent deletions; completion
  outputs; arbitrary local or cross-module activity-induced invalidation.
- Simulator full-interval read/write exclusion, including repeated writes of
  an already true history fact.
- Renewable resources, release-only events, positive and negative final goals.
- Per-definition and total execution limits. An absent per-definition limit
  uses the original total step bound, not an inferred smaller limit.
- State-changing external events, serial execution and zero-duration activities
  are rejected explicitly. No calendar support is claimed.

For each present occurrence and prerequisite, a Boolean selects an initial
fact or earlier producer. A net deletion must precede that producer or follow
the consumer's start. Interval locks protect the entire consumer duration.
Negative final goals use deletion support and protection from later outputs.
Action completion reassertion takes precedence over deletion, as in Simulator.

Coverage argument: embed any legal bounded event-driven plan into prefix slots
of each activity, retaining its times. Choose a most recent producer for every
true prerequisite, or the initial fact if never deleted. The corresponding
threat disjunctions and original resource/lock constraints are satisfied. Thus
the CP model cannot exclude such a plan within the documented scope.

The CP timing domain is a relaxation allowing arbitrary integer starts. To
produce an original-semantics certificate, preserve every pairwise disjoint
interval order from its solution, then start each interval at the earliest
release or predecessor completion. All original disjoint pairs remain disjoint.
Every new overlapping set was a clique of original intervals and therefore had
a common original time (interval Helly property), preserving cumulative demand.
State-changing writers and their readers retain order. The result is replayed
through the original Simulator/Validator and an independent resource sweep.
Only validated candidates before the deadline count. A validated makespan
equal to the CP lower bound proves optimality for the original bounded input.

The objective is makespan only. Equal-makespan solutions can contain unnecessary
operations. This is recorded, not treated as a violation or silently resolved
with a second objective.

## Small gate

Twelve paired scenarios test arrival, downstream duration, capacity,
invalidation, mandatory historical pretest, and multiple invalidators. All
must match exhaustive full-state enumeration before the 256-activity stage.
For the unit-duration core, let r be material release and h the long work:

- Waiting then testing once: max(1,r) + h + 6.
- Early test plus restore: max(5+h, max(5,r)+4) + 1.

These give 9 versus 10 at r=0,h=2, and 16 versus 13 at r=8,h=2.
At r=2, changing h=1 to h=8 changes the preferred route from waiting
(9 versus 10) to early testing (16 versus 14). A one-unit dispatch latches
the test history; the long task holds the dispatch output, not the test history,
so its overlap with retesting respects the original state locks.

Additional regression tests cover negative goals, impossible causal cycles,
competing transition consumers, finite total steps, state write locks, and
input rejection. Enumeration shares Simulator; the hand calculations and
independent resource check are complementary evidence, not a claim of a
machine-checked universal compiler proof.

## Large study

24 inputs = arrival (0, .5, 1.5 times a frozen seed-specific reference)
x capacity (16 or 2) x downstream work (short or long) x seeds (101,202).
Each input has exactly 256 definitions: 16 modules of 16 activities. Thirty-two
activities are real optional extended inspection/archive work, not required
for the selected acceptance scope. Power switching and restoration counts are
chosen by the solver. No interchangeable implementation variants are needed.
Dependency and mode structure remain fixed across each seed's factor pairs.
There are 368 states, 448 optional occurrence slots, and a 512-step input limit.
The witness is stored separately in private-evidence and is not used as a hint
or horizon bound. The generic horizon is last release + max_steps*max_duration.

All cases get 30 seconds. Unproved cases then get independent 300-second runs;
no 30-second schedule hint is injected. Maximum solve budgets total 132 minutes.
The large batch has a 3-hour active-time cap. Input parsing, construction,
search and synchronous validation count; imports/startup are separate. One
worker on logical CPU 2 avoids the existing CPU 0/15 experiment threads;
shared cache and memory-bandwidth interference remain possible. CPU, external
OS peak memory, first validated solution, variables, constraints, branches and
conflicts are recorded. Branch count is not a business-state count.

Commands (use the project venv):

```
python -m experiments.joint_cpsat prepare
python -m experiments.joint_cpsat audit
python -m experiments.joint_cpsat run
python -m experiments.joint_cpsat report
```

`run` requires the frozen small gate, supports completed-attempt resume and
uses an exclusive batch lock. Source/input hashes are checked before each
attempt. It stops on encoding errors rather than converting them to ordinary
solver failures. Raw inputs, private witnesses, source snapshot, small exact
answers, per-attempt schedules/counters and CSV/JSON reports are retained.
