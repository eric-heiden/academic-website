# Measured Panda dynamics identification

Final configuration: `config.json`. Final completed measurement: `final_metrics.json`, trace `candidate-056.npz`. All six thresholds pass both pooled and separately for all three supplied recordings. A rebuilt live model reproduced the final metrics exactly. The exact submitted configuration equals the measured configuration.

Only the supplied training recordings, immutable regressor and manifest, task bounds, sanitized geometry, and permitted shared/public physics sources were used. No nominal Panda dynamics or additional robot recordings were used. No subagents or extra simulators were created. Input and geometry SHA-256 hashes were checked and remain unchanged.

The fit uses the 98-coefficient physical parameterization, weighted convex least squares, and positive semidefinite pseudo-inertia constraints. These enforce positive mass, bounded COM, realizable symmetric COM tensors, and the link second-moment bound. Joint friction, bias and armature satisfy their specified bounds. The objective uses torque weights [1,1,1,1,2,2,3] and ridge coefficient 1e-4, with geometric scaling documented in `fit_model.py`. Armatures were selected by complete Newton measurements and recording-level cross-validation, then the remaining coefficients were refitted. About 69 parameter directions are numerically identifiable in this regressor.

The final model is `refined_a5_0.03.json`: armatures [0.02,0.01,0.02,0.01,0.03,0.01,0.025]. Leave-one-recording-out fits with this procedure passed all checks on each excluded recording: worst torque RMSE 0.213992 Nm, normalized torque RMSE 0.294792, position RMSE 0.012349 rad, velocity RMSE 0.294095 rad/s. Later joint-loss refinements were rejected because their worst excluded-recording velocity error was higher (0.345220 rad/s).

56 complete physical candidates were evaluated, each by apply_config/reset/1800 steps/metrics. All 56 traces, including the failed initial candidate, are preserved. Candidate 1 passed torque checks but failed motion checks because its very small effective wrist inertias amplified residual torque. No candidate trace was overwritten. Final candidate 56 completed after a rebuild. Work finished within the 1200-second budget (approximately nine minutes).

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.194006 | 0.208306 | 0.5 |
| max_joint_torque_normalized_rmse | 0.255470 | 0.284226 | 0.5 |
| max_joint_position_rmse_rad | 0.009800 | 0.011814 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.229098 | 0.279504 | 0.5 |
| position_p95_rad | 0.011865 | 0.013784 | 0.05 |
| max_joint_speed_rad_s | 2.210246 | 2.210246 | 5 |

This is an effective dynamics identification for filtered joint measurements. Individual link parameters are not uniquely recoverable; several fitted inertia directions are close to the physical lower bound. The independent 16-recording test fold was not accessed. Training and cross-validation passes do not establish its result, or behavior beyond the measured 100 ms horizons.
