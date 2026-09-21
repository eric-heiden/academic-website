# Physical Panda identification

The final `config.json` contains all seven link masses, centers of mass, full symmetric COM inertia tensors, and joint viscous friction, Coulomb friction, torque bias, and armature. It is the exact configuration measured successfully in `candidate-016.npz`; detailed results are in `evaluation-016.json` and `identification-summary.json`.

Only the supplied three measured recordings, immutable 3150-by-98 regressor, its manifest, task definition, sanitized geometry, and permitted public model/solver sources were used. No nominal Panda dynamic parameters or external recordings were used. Input and geometry hashes were verified unchanged.

## Estimator

`identify.py` implements the estimator and prescribed candidate workflow. Linear coefficients consist of each link's mass, first moment, and origin inertia, followed by the 28 joint coefficients. There are 69 well-observed numerical directions (singular-value threshold 0.01); individual link parameters are not uniquely identified.

Each link's pseudo-inertia matrix is constrained positive semidefinite, with a small positive second-moment margin. Mass, COM, second moment, joint loss, bias, and armature bounds are imposed explicitly. This yields positive COM inertias satisfying triangle inequalities. Every configuration is also passed through the supplied physical validator.

The objective is weighted torque squared error plus scaled quadratic regularization. Final joint residual multipliers are `[1, 1, 1, 1, 1.5, 1, 2]`, and the regularization coefficient is `1e-4`. Coefficient scales are `[3, .5, .5, .5, .2, .2, .2, .1, .1, .1]` per link, 2 for friction and bias coefficients, and .2 for armature. The penalty is centered at zero and contains no nominal dynamics. QR compression removes only a parameter-independent objective constant.

Unregularized armature estimates approached zero, giving good torque agreement but poor forward predictions, especially at the wrist. Physical candidates were used to select armature lower bounds `[.03, .03, .03, .03, .035, .03, .02]` kg m². All remaining coefficients were refitted under these constraints.

## Validation

Sixteen complete physical candidates were evaluated, each through `apply_config`, `dispatch('reset')`, `dispatch('step', {'count': 1800})`, and `metrics()`. All candidate traces and logs, including failures, were retained. Thirty-one offline fits included a regularization sweep and leave-one-recording-out checks. For the latter, fitting used two recordings and the omitted third was checked using both torque and the same prescribed forward evaluation. All three omitted recordings passed with nearby armature floors `[.01, .01, .01, .01, .03, .01, .015]`.

The final candidate passed all six thresholds pooled and separately for every recording. Worst values across recordings:

| Metric | Measured | Limit |
|---|---:|---:|
| Maximum joint torque RMSE, Nm | 0.210264 | 0.5 |
| Maximum normalized joint torque RMSE | 0.300982 | 0.5 |
| Maximum joint position RMSE, rad | 0.0109864 | 0.025 |
| Maximum joint velocity RMSE, rad/s | 0.260956 | 0.5 |
| 95th percentile absolute position error, rad | 0.0127759 | 0.05 |
| Maximum joint speed, rad/s | 2.20826 | 5.0 |

These are effective dynamics for filtered joint measurements. They do not uniquely recover individual physical link properties or raw motor commands. Armature regularization improves forward stability at some cost in torque fit. The independent 16-recording test fold was not accessed and awaits external verification.
