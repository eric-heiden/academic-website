# Identified effective Panda dynamics

The exact final numeric model is `config.json` (SHA-256 `e4a5de44ddbeb3c37a775ba722706ad28f05937bfe67164b42135bb4bda4319c`). Fresh-process validation: `candidate-025/metrics.json`, complete 1,800-step trace: `/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-restart-3/candidate-025/metrics.npz`. Every pooled and individual-recording threshold passed. All 25 complete physical candidate evaluations, including failures, remain in their separate candidate directories with process logs, configurations, metrics and traces. Every candidate used a separate invocation of the prescribed `real_rollout` process. No persistent simulator or additional simulator was used.

## Estimation

Only the supplied three measured recordings, immutable 3,150-by-98 numerical regressor, mapping manifest and sanitized geometry were used. The model was estimated from the supplied homogeneous placeholders without nominal Panda dynamic parameters. `fit_model.py` contains the estimator. The measured regressor has 69 well-resolved singular directions; individual physical link parameters are not uniquely identified.

The convex fit estimates all 70 link coefficients and 28 joint coefficients. For each link, its pseudo-inertia matrix [[S,h],[h^T,m]] is positive definite, where S = trace(I_origin)/2 * identity - I_origin and h = mass * COM. This enforces positive COM inertia and the physical triangle inequalities. Explicit constraints also enforce the supplied mass, COM, second-moment and joint-loss bounds. Exported tensors are validated using the public exact physical mapping.

The final fit uses a Huber loss with transition 0.1 after dividing each recording/joint's torque residual by its torque standard deviation clipped to [0.5,1] Nm. A small dimensionless quadratic penalty (coefficient 1e-5) regularizes the parameter representation. Pseudo-inertia second-moment eigenvalues are bounded below by 1e-5. Effective armature lower bounds are [0.03,0.02,0.03,0.02,0.025,0.025,0.015] kg*m^2. These regularization choices were assessed by the prescribed physical candidates; unconstrained torque fitting alone produced inadequate wrist motion. Candidate 016 was selected for its balance of torque and motion error, with reduced armature at joint 7 to limit acceleration-related torque error.

## Final training quality

| Criterion | Pooled | Worst recording | Required maximum |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.197930 | 0.214004 | 0.5 |
| max_joint_torque_normalized_rmse | 0.253078 | 0.282625 | 0.5 |
| max_joint_position_rmse_rad | 0.010349 | 0.011669 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.238553 | 0.271826 | 0.5 |
| position_p95_rad | 0.010870 | 0.012836 | 0.05 |
| max_joint_speed_rad_s | 2.213081 | 2.213081 | 5 |

Leave-one-recording-out refits of the final regularization also passed all thresholds, including each omitted recording (candidates 022–024). These use only the three supplied training recordings and are not the independent published test fold.

## Limitations and provenance

These are effective dynamics estimates for publisher-filtered measured torque, not uniquely recovered physical link properties or raw motor commands. Some weakly identified parameters meet their regularization bounds. The 100 ms prediction checks do not establish long-horizon simulation accuracy. The 16-recording independent fold was not accessed or evaluated; its final verification is external to this run.

Training source: LIP4RobotInverseDynamics, DOI 10.5281/zenodo.12516500, CC-BY-SA-4.0 (as supplied in the immutable input manifest). Supplied hashes of training.npz, training-regressor.npz, training-regressor.manifest.json and sanitized geometry were checked unchanged. No scoring code, reference data, geometry, solver algorithm or shared source files were edited.
