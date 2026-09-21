# Real measured robot identification: report methods draft

This prospective task asks an agent to identify the effective dynamics of a
seven-joint Panda arm from recorded physical-robot measurements, starting with
geometry and deliberately generic dynamics. It evaluates the same implementation
through Newton MCP, the actual IPython MCP integration, and a process-per-candidate
edit/restart workflow. The agent must produce a physically valid complete model
that explains measured joint torques and predicts short forward motions in Newton.
The independent publisher test recordings are unavailable to the agent and are
used only for final verification. This task is separate from the earlier
`panda_calibration` experiment, whose observations were explicitly synthetic.

## Measurement source and exact split

The measurements come from the real Panda subset of the
[LIP4RobotInverseDynamics dataset](https://zenodo.org/records/12516500), version DOI
[10.5281/zenodo.12516500](https://doi.org/10.5281/zenodo.12516500). The downloaded
`PUB-5510-LIP4RID.zip` contains 541,538,288 bytes and matches the publisher's MD5
`d3e29fbd280fc4cf08d008eb50559338`. The archive also contains simulated data,
manufacturer-model dynamics components and learned models; none of those are
inputs to this task. The publisher's
[source repository](https://github.com/merlresearch/LIP4RobotInverseDynamics/tree/e1e30333f6ec1f90de33f0d75be8b12a20f9b7e8)
was inspected at revision `e1e30333f6ec1f90de33f0d75be8b12a20f9b7e8` to establish
which recorded torque field its real-data estimator uses.

The associated [MERL paper, section V-B](https://www.merl.com/publications/docs/TR2024-077.pdf)
describes position, velocity and torque observations collected through the
manufacturer's ROS interface, filtered at 4 Hz, and acceleration computed by acausal
differentiation of velocity. The released DataFrames contain seven joints,
original timestamps, q/dq/ddq, tau and tau_interp. Our reference uses `q_1..7`,
`dq_1..7`, `ddq_1..7`, `tau_interp_1..7` and `t`; `tau_interp` is the torque field
selected by the publisher's real-data estimator. Units are radians, radians per
second, radians per second squared, newton metres and seconds. No additional
filtering or synthetic response generation is applied.

There is an archive/publication discrepancy: the paper describes 10 training
records, while the released archive and configuration contain 13 training seeds and
16 test seeds. We use the three lowest numbered published training seeds,
**2, 3, 4**, chosen before fitting, and retain **all 16** published test seeds:
**21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38**. Training records are labeled as
50-sinusoid excitations and test records as 100-sinusoid excitations. No record or
window is selected by fitting error. The three training records contain 8,771 joint
observation vectors in total, with irregular timestamp intervals around 17 ms.
The actual timestamps, rather than a presumed sample frequency, determine every
interpolation and window boundary.

The release does not establish the raw collection topic, motor-current conversion
or a reconstructed commanded motor torque. We therefore describe the input as the
publisher's filtered, interpolated **measured generalized joint torque**, not a
recovered motor command. This distinction limits the physical interpretation of
the identified losses and effective inertias.

## Clean initial robot and physical unknowns

Visual geometry and kinematics come from the
[MuJoCo Menagerie Panda asset](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda),
revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`, file `panda_nohand.xml`.
Preparation creates a separate sanitized asset. It keeps meshes, body/joint
transforms, topology and joint ranges while removing authored inertias, actuator
settings, initial configurations and joint losses. Mesh density is zero and
contacts are disabled. Runtime dynamics are assigned explicitly; they are never
inferred from the meshes. The original authored dynamics, manufacturer M/c/g
arrays, identified parameter sets and private feasibility results are outside the
agent's permitted inputs.

All agent contexts start from the same seven moving links: mass 1 kg each, COM at the
body-frame origin, COM inertia `0.01*identity` kg·m², and zero viscous friction,
Coulomb friction, torque bias and armature. These are homogeneous placeholders,
not multiplicative perturbations of authored Panda dynamics. Replicate identifiers
do not change the initial model or dataset split.

The configuration has **98 independent coefficients**: for each of seven links,
mass, three COM coordinates and six independent entries of the symmetric COM
inertia; plus seven each of viscous friction, Coulomb friction, constant torque
bias and reflected joint inertia. The JSON stores full symmetric 3×3 tensors,
so their mirrored entries are redundant rather than additional free parameters.
Mass is bounded to 0.05–10 kg, each COM component to −0.4–0.4 m, viscous and Coulomb terms
to 0–5 in their respective SI units, bias to ±2 N·m, and armature to 0–1 kg·m². Inertia
must be symmetric, have eigenvalues at least 1e-8 kg·m², satisfy physical triangle
inequalities, and obey a broad origin-second-moment bound corresponding to a 0.5 m
length scale. These constraints enforce physically realizable candidate link
properties without supplying nominal masses or a fitted answer.

The inertial parameterization is not uniquely identifiable. On the fixed training
samples, the Newton design matrix has numerical rank 69 at a relative singular-value
cutoff of 1e-5; tighter cutoffs admit additional float32 numerical directions.
Link inertias, distal hardware and joint nuisance terms can trade off in
unobservable combinations. The task therefore assesses predictive effective
dynamics under physical constraints, not recovery of a unique ground-truth set of
individual link parameters.

## Common preprocessing and fitting access

Every condition receives the same numeric training reference, sanitized geometry,
generic construction code and immutable Newton inverse-dynamics regressor. The
regressor is built from geometry and the measured q/qd/qdd, with no fitted parameter
values or manufacturer mass/Coriolis/gravity model. Seventy link basis columns are
computed through Newton's public inverse-dynamics APIs. The remaining 28 columns
are per-joint qd, sign(qd), constant bias and qdd terms. The unit basis probes are
algebraic vectors; submitted forward simulation candidates must independently pass
physical validation.

For each recorded trajectory, 150 equally spaced integer row indices are selected
from 30 through `N-31`. The training matrix is 3150×98: 450 observation vectors,
seven joint torques per observation. Its link coefficient ordering is mass, three
first moments, and six entries of the inertia about the link frame origin. The
mapping uses the parallel-axis theorem from the submitted COM tensor. Numeric
matrix storage is float64; Newton basis calculations use float32. Sidecar manifests
hash the exact reference, geometry and regressor, document field and basis order,
and record preparation time.

Immutable preparation is performed once, measured separately, and excluded from
all agent timers equally. The agent-facing source provides construction and
validation, but no estimation algorithm or solution. All conditions may write
helper programs and use the same available NumPy/SciPy/CVXPY versions. Offline
linear algebra and optimization are allowed equally. A restart agent may batch
such calculations efficiently, but each submitted mutable physical evaluation
runs a new Newton model and solver in a fresh process. Persistent conditions use
the identical scenario and physics checks. The final report should insert the
frozen runner's exact dependency versions, budgets, condition definitions,
execution order and timing boundaries; these are runner-level registration facts,
not assumptions made by this data/scenario implementation.

## Fixed forward evaluation and quality criteria

Each candidate is evaluated on 12 windows from every recording. Starts are the
12 equally spaced integer row indices from 30 through `N-37`. Every window resets
Newton q/qd to the measured state, then runs **50 steps of 2 ms**, a 100 ms horizon.
The filtered measured torque is linearly interpolated at each step midpoint;
position and velocity references are interpolated at the step end using original
recorded timestamps. The fitted constant torque bias is subtracted from the
input. Viscous friction, Coulomb friction and armature are native solver properties.
No tracking controller is used. Training requires 1,800 steps over 36 windows; final
held-out verification requires 9,600 steps over 192 windows.

All six limits must pass **both pooled and for every recorded trajectory**:

| Quantity | Limit |
|---|---:|
| Maximum joint torque RMSE |0.5 N·m|
| Maximum normalized joint torque RMSE |0.5|
| Maximum joint forward-position RMSE |0.025 rad|
| Maximum joint forward-velocity RMSE |0.5 rad/s|
| 95th percentile absolute forward-position error |0.05 rad|
| Maximum simulated joint speed |5 rad/s|

Torque normalization uses each joint's standard deviation across its 150 selected
observations in that recording, with a 0.5 N·m floor. Forward error metrics include
all recorded window steps, not only end states. The candidate must be physically
valid and retain every expected finite sample. A later window reset cannot erase
an earlier nonfinite failure. Direct and persistent paths attempt the same full
fixed sequence after such an observation; native solver exceptions remain visible
errors. Failed configurations, failed quality candidates and infrastructure
failures are retained with their distinct meanings.

The split, sampling rules and six quality criteria were frozen at
**2026-09-20T22:11:16.449381+00:00**, before any held-out DataFrame was deserialized
or scored. The protocol SHA256 is
`e1f82fd69cd3b16f38b6653be6604b30439256181f09d2e679d219787acf553e`.
Training-only feasibility explored physically constrained fits and their forward
stability; those attempts are retained separately and are not agent results or
agent inputs. Test responses do not select thresholds, records or fitting priors.
Final success requires a complete passing training measurement for the submitted
configuration and a fresh physical evaluation of that same configuration across
the complete published test fold after the agent context exits.

## Implementation validation and limits of interpretation

The reusable implementation is independently checked against public Newton inverse
dynamics for arbitrary physical masses, nonzero COMs and full off-diagonal inertia
tensors. Tests check parallel-axis mapping, reference integrity, physical
constraints, incomplete-candidate rejection and the case in which pooled metrics
pass while one recording fails. Full native mass, COM, principal inertia,
orientation, damping, friction and armature updates were compared with a fresh
model. Actual sanitized Panda q/qd samples matched bitwise for the inspected
training prefix and reset/replay checks. A no-op native-notification control makes
the parity regression fail, showing that the check detects stale solver data.

Thirteen scenario tests pass, including a reproduced failure-path regression:
after an injected nonfinite frame, direct and session evaluations now retain the
same number of attempted steps and the same failure. The original mismatch and
corrected result are preserved as red/green evidence. Frame-zero checkpoint/reset
replay is supported. Restoring a nonzero-frame physical checkpoint is explicitly
rejected for this scored scenario because it does not contain accumulated
application measurements; a full reset is required before further steps or
metrics. This prevents incorrectly aligned partial-history diagnostics.

This is an offline recorded-data model-identification experiment, not deployment
on a robot. Short windows reduce accumulated drift and avoid reconstructing an
unknown controller, but do not establish long-horizon open-loop behavior. The
publisher's filtering and acausal differentiation suppress some dynamics and can
introduce consistency errors between torque and state derivatives. Coulomb and
viscous terms are simplified effective losses; they do not identify all actuator,
transmission, temperature or hysteresis behavior. No contacts are evaluated. Any
unmodeled gripper or other distal hardware is represented only through effective
distal-link parameters. Generalization is measured across this publisher's
independent recordings, not across other robots, payloads or collection systems.
A small agent study should not be described as establishing universal tool
superiority; all conditions, unsuccessful trials and preparation/development
results remain part of the report.

## Licensing and distributable provenance

The measured data and derived references/regressors carry
[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Attribution: “Created by Mitsubishi Electric Research Laboratories (MERL), 2024”;
Giacomuzzo, Carli, Romeres and Dalla Libera. Distributed derivatives retain the
source DOI, license, attribution and an explanation of numeric conversion and
selection. The Panda asset's Apache 2.0 license is copied alongside sanitized
meshes. The publisher's estimator code is AGPL; this task independently implements
preparation, Newton feature construction and validation rather than copying or
using that implementation. Public Newton study source uses its repository license.
Raw and derived file digests, source revisions, field schemas and preparation costs
should accompany downloadable reproducibility artifacts.

## Proposed trace figure and video

Use actual final verifier traces, with the selection rule fixed before plotting:
the sixth of 12 windows in training seed 2 and held-out seed 21. Plot all seven joints;
do not choose the joints or windows with the largest improvement. A compact
scientific figure can have seven rows and two columns (training, held-out), showing
position error relative to the measured trace with a zero measurement baseline,
the uniform starting model in gray, and the three submitted conditions in fixed
colors. An adjacent small strip should show measured q and simulated q for joint 2
on the same selected windows to make the absolute trajectory interpretation clear.
Use shared units, time axes from 0 to 0.1 s and comparable y limits; clearly label any
scale differences. Export Matplotlib SVG and PNG plus the exact numeric plotting
input. Include all replicate values in a separate aggregate error plot, rather than
making one visually selected trace stand for all trials.

For video, use the actual sanitized Panda rendered by the tested Newton/MCP path.
Show the same fixed held-out window, with panels for measured-pose replay, uniform
initial dynamics and the three final fitted simulations. The measured-pose panel
must be labeled **kinematic replay of recorded joint measurements**; it is not
camera footage and not a forward simulation. Render simulated panels from saved
numeric verifier states, without added controllers or smoothing. Use one camera,
geometry, frame rate and time mapping for every panel. Because 100 ms is brief,
label any slow motion explicitly and include a visible recorded-time cursor.
Separate windows with a labeled reset transition if a longer montage is desired.
Retain sidecar configuration/reference hashes and parity checks between plotted,
rendered and scored q/qd arrays. The visuals illustrate the measured-data
validation; they do not replace the full held-out numerical results.
