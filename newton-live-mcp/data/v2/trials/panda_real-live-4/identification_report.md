# Physical Panda identification

Final artifact: `config.json`. This contains all full numeric link masses, COMs, symmetric COM inertias, viscous and Coulomb friction, torque biases and armatures. No nominal dynamic asset or correction factors were used.

## Estimation

Custom CVXPY estimation used the supplied 3150 by 98 immutable regressor. The loss was Huber with threshold 0.2 on torque residuals weighted by the inverse of each recording/joint torque standard deviation clipped to [0.5, 1]. A 1e-5 quadratic regularizer resolved poorly determined directions, with coefficient scales 2 kg for mass, 0.2 kg m for first moments, 0.1 kg m^2 for origin inertias, and 1 for joint terms. All seven pseudo-inertia matrices were constrained positive definite (1e-6 margin), along with the supplied mass, COM, second-moment and joint bounds. Conversion to COM inertia was validated by the supplied physical validator.

The regressor has approximately 69 numerically identifiable directions (SVD cutoff 1e-5 relative). Full physical parameter values are therefore regularized effective estimates, not unique measurements of individual links.

Torque-only estimation produced insufficient effective joint inertia for forward prediction. Armature lower bounds were selected using complete prescribed physical evaluations and leave-one-recording-out validation. Selected bounds are [0.03, 0.01, 0.03, 0.01, 0.025, 0.02, 0.015] kg m^2. The selected fit used all three recordings. The estimation and comparison code is in fit_physical.py, compare_fits.py and refine_fits.py.

## Completed physical verification

31 complete physical candidates were evaluated, each using apply_config, reset, 1800 managed steps, and metrics. All candidate traces and automatic logs were preserved. The final model was also rebuilt through the configured MCP tool and remeasured, with identical scores. The live simulation remains on this exact successful model.

| Metric | Pooled | Worst recording | Threshold |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.197120681 | 0.212568048 | 0.5 |
| max_joint_torque_normalized_rmse | 0.24887568 | 0.277827638 | 0.5 |
| max_joint_position_rmse_rad | 0.010384922 | 0.0120243221 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.240480657 | 0.281088798 | 0.5 |
| position_p95_rad | 0.0111507768 | 0.0133651806 | 0.05 |
| max_joint_speed_rad_s | 2.21050286 | 2.21050286 | 5.0 |

All pooled and per-recording checks passed. Final complete trace: `candidate-031.npz`. Final metrics: `final_training_metrics.json`.

For the selected fitting procedure, every excluded recording passed both torque and forward-motion thresholds under leave-one-recording-out fitting. Worst excluded-recording values were:
- max_joint_torque_rmse_nm: 0.220817239
- max_joint_torque_normalized_rmse: 0.283199587
- max_joint_position_rmse_rad: 0.0127812713
- max_joint_velocity_rmse_rad_s: 0.301363103
- position_p95_rad: 0.0140863105
- max_joint_speed_rad_s: 2.21129131

## Integrity and limitations

All supplied training input SHA-256 hashes were checked against task.json and matched. Shared sources, geometry, task and input archives were not modified. Only the supplied recordings and immutable features were used. No subagents, external data, nominal Panda parameters, alternate simulators or held-out recordings were accessed.

The measurements are filtered and measured joint torque is not a recovered raw motor command. These parameters represent effective dynamics over measured motions and the specified 100 ms prediction horizon. Cross-validation across the three supplied recordings passed; the independent 16-recording test fold remains unobserved and its result is not claimed.

Final config SHA-256: `801ffef0bcae58aaf2924a97de5681fa4ac5cfab720426a3db010c44403af97b`.
