# Physical Panda identification

The submitted full numeric model is `config.json`. Its final complete Newton measurement passed all six thresholds, pooled and in every supplied recording (episodes 2, 3, and 4). Final trace: `candidate-022.npz`; full metrics: `final_measurement.json`. There were 22 complete physical candidate evaluations, including the final repeat. All candidate traces and evaluation logs are retained.

## Estimation

Custom CVXPY estimation fits all 98 physical coefficients using the supplied immutable Newton regressor. Each link is represented by mass, first mass moment, and symmetric inertia about the link origin. A positive semidefinite pseudo-inertia constraint guarantees physical COM inertias and the triangle inequalities. Mass, COM, second moment, friction, torque bias, and armature obey the task bounds. A small positive second-moment margin avoids singular tensors. A weak quadratic regularizer selects a representative among poorly identifiable link parameter combinations; its prior uses only the supplied homogeneous placeholders. No nominal Panda dynamic parameters were used.

The first least-squares candidate fit torque well but failed wrist motion checks. Complete prescribed Newton evaluations were used to select modest joint-armature lower bounds. The final fit uses lower bounds [0.025, 0.01, 0.025, 0.015, 0.03, 0.02, 0.015] kg*m^2, a 1e-4 pseudo-inertia margin, and 1e-4 quadratic regularization. Loss-parameter alternatives and broader armature sweeps were retained but not selected. `fit_physical.py` and `fit_balanced.py` reproduce the selected offline estimate.

## Final measured quality

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.193753 | 0.208669 | 0.5 |
| max_joint_torque_normalized_rmse | 0.262398 | 0.296768 | 0.5 |
| max_joint_position_rmse_rad | 0.009734 | 0.011607 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.228085 | 0.275422 | 0.5 |
| position_p95_rad | 0.011860 | 0.013820 | 0.05 |
| max_joint_speed_rad_s | 2.204657 | 2.204657 | 5.0 |

## Validation and limitations

Three separate fits each omitted one entire training recording. All three omitted-recording torque and forward-motion measurements passed; details are in `cross_validation.json`. This is an internal robustness check, not the independent published test fold. The 16 independent recordings remain inaccessible and unverified until the external verification process runs.

The regressor has approximately 69 numerically resolved directions (singular values above 1e-3), so individual link masses, centers, and inertia tensors are not uniquely determined. The result is an effective physical link/joint model. Publisher-filtered torque and state derivatives introduce model mismatch; measured joint torque is not a recovered raw motor command. The selected armature regularization balances torque agreement with short-horizon motion prediction.

The three immutable input hashes were rechecked successfully after fitting. No shared sources, geometry, target windows, callbacks, or solver settings were modified.
