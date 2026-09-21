# Physical Panda identification

The exact submitted model is `config.json`, independently rerun as `candidate-015`. It passed all six quality thresholds pooled and for each supplied recording. The completed measurement contains 36 independent 100 ms windows, 1,800 Newton steps at 0.002 s. All 15 physical evaluations used separate processes; configurations, logs, provenance, metrics and complete traces are retained in their respective candidate directories, including failures.

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.196194 | 0.210051 | 0.5 |
| max_joint_torque_normalized_rmse | 0.254602 | 0.284281 | 0.5 |
| max_joint_position_rmse_rad | 0.00984095 | 0.0118766 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.229973 | 0.280766 | 0.5 |
| position_p95_rad | 0.0107566 | 0.0129652 | 0.05 |
| max_joint_speed_rad_s | 2.21146 | 2.21146 | 5 |

## Estimation

`fit_model.py` fits the supplied 3,150 by 98 design matrix in absolute physical coefficients. No nominal Panda dynamics or external data were used. The objective is weighted least squares with per-recording/per-joint scale `min(1, max(0.5, std(torque)))`. A quadratic regularizer of 1e-5 selects among weakly identifiable inertial decompositions, using generic scales of 3 kg, 0.3 kg m and 0.1 kg m^2.

The solver constrains each link's pseudo-inertia matrix to be positive semidefinite, bounds mass and COM, enforces the second-moment radius bound, and constrains all joint coefficients to the specified physical intervals. A 1e-6 spatial second-moment margin keeps COM inertias above the required positive eigenvalue limit. A minimum armature of 0.03 kg m^2 was selected using the torque/forward tradeoff. Increasing it further reduced forward errors but eventually violated torque thresholds. Forced high viscous losses and robust objectives did not improve the balanced result.

The final model is the all-recording fit in `fit-006`; candidate 15 exactly repeats its numeric configuration after validation. The original matrix has 69 numerically resolved singular directions at relative cutoff 1e-5. Full link parameters are therefore a feasible effective decomposition, not a unique identification of individual hardware properties.

## Validation and limits

`cross-validation.json` records an offline regularization/armature sweep with one supplied recording excluded at a time. Candidates 12, 13 and 14 additionally evaluate fits excluding recordings 2, 3 and 4 respectively. Each omitted recording passes both torque and forward thresholds; worst omitted-recording torque RMSE is 0.21596 N m, normalized torque RMSE 0.29641, position RMSE 0.012434 rad and velocity RMSE 0.29552 rad/s. These are internal validation splits within the three supplied recordings.

The independent 16-recording test fold was not accessed or evaluated. Its quality remains unknown until the external verification runs. The identified model describes filtered measured torque and short-horizon effective motion; the link decomposition is nonunique and does not recover raw motor commands or establish long-horizon accuracy. Input and geometry hashes were checked against the supplied immutable manifests.
