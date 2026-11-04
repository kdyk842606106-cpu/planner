# Mandatory restoration after M100

Baseline: `runs/mainline-joint-cpsat-dense-prerequisites-20261103`.
Retain all 256 definitions, original 332 task edges, durations, resource requests
and capacities (2/1/1), material events, all 30 exact windows, execution limits
and solver settings. Change CHECK_B1_10 from an optional check into an offline
repair after M100, deleting valid:cal and valid:test and producing both dirty
states. Require its completion before M101. Its existing done:B1_10 prerequisite,
two-unit duration and instrument demand remain unchanged. This is an explicit
synthetic repair scenario, not a claim that the original check had those effects.

M100 requires powered-on, valid-test state. Repair requires isolated mode. M101
requires powered-on, valid-test state and repair completion. The unchanged mode
transitions and qualification producers force power_off -> ground -> repair ->
unground -> power_on -> calibration -> testing, with 34 units of serial work.
Calibration/testing therefore each execute at least twice, with at least one
complete return to isolated mode and recovery. The old schedule is invalid.

This first case forces one episode rather than exploring uncertain numbers of
repairs or alternative repair locations. It validates state-cycle correctness
and measures one controlled change; it is not a general state-explosion test.
No new material event or changed material release is introduced.

The inherited unlimited CP-SAT runner uses one worker, seed 11, CPU 4 and no
deadline. Only raw input.json reaches the formal worker. Private witness and
independent release/serial-chain lower bound are never solver hints. A fixed
witness feasibility check and negative tests gate launch. Results remain in a
new directory; no existing experiment is stopped or altered.

`python -m experiments.cycle_joint prepare`, then `verify`, then `run`.
