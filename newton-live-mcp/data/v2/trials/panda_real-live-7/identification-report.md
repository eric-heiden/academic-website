The submitted config.json contains a full physical seven-link model fitted from the supplied measured recordings and geometry-only inverse-dynamics regressor. It was applied through scenario.apply_config, reset, and measured for all 1,800 prescribed Newton steps. The final measurement passes every threshold both pooled and in each of the three recordings.

Estimation used CVXPY constrained least squares over the 98 linear physical coefficients. Each link has a positive semidefinite pseudo-inertia matrix, bounded mass and COM, and the prescribed second-moment bound. Joint damping, Coulomb loss, torque bias, and armature have their prescribed bounds. Weak coefficient regularization resolves unobservable directions. Armature and inertia floors were selected using measured forward agreement. Leave-one-recording-out torque fits checked sensitivity to the training recordings. The regressor has 69 well-observed singular directions; individual link parameters therefore should not be interpreted as uniquely measured hardware properties.

A final local refinement adjusted wrist joint losses, biases, and armatures using finite differences of complete authorized candidate rollouts and a minimax fit to the torque and forward-motion thresholds. It changed physical configuration values only. No reference observations, scoring, windows, callbacks, geometry, or solver algorithm were modified. No external data or nominal Panda dynamics were used.

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.1938148 | 0.2087269 | 0.5 |
| max_joint_torque_normalized_rmse | 0.2410431 | 0.2770985 | 0.5 |
| max_joint_position_rmse_rad | 0.01131769 | 0.01210961 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.2541324 | 0.2746694 | 0.5 |
| position_p95_rad | 0.01196494 | 0.01299364 | 0.05 |
| max_joint_speed_rad_s | 2.207163 | 2.207163 | 5.0 |

There were 36 complete physical candidate evaluations, including the final repeat; all candidate traces and live_rollouts.jsonl are retained. Candidates 1, 5, and 6 failed and remain available for audit. The exact final trace is candidate-036.npz and its metrics are in final-training-measurement.json. The selected numerical parameters were first measured as candidate 34 and reproduced as candidate 36.

Supplied recording, regressor, manifest, and geometry hashes were checked against the immutable task/manifest. Configuration SHA-256: 0ae734d1ff233e51a6556f6db438ff451d8b42a372c9b9a8496b6458b6415ee5.

These are effective dynamics for publisher-filtered measured torque. Unmodeled actuator effects and filtering remain limitations. The independent 16-recording test fold was not accessed; its fresh-process verification remains pending.
