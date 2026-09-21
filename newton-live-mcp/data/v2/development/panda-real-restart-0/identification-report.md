# Physical Panda identification

Final numeric configuration: `config.json`. Final fresh-process measurement: `candidate-018/metrics.json`; complete traces: `candidate-018/metrics.npz`.

All six thresholds pass both pooled and in each of recordings 2, 3, and 4, across all 36 independent 100 ms windows (1,800 steps).

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.207628 | 0.222723 | 0.5 |
| max_joint_torque_normalized_rmse | 0.250457 | 0.274260 | 0.5 |
| max_joint_position_rmse_rad | 0.009570 | 0.011691 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.222750 | 0.274574 | 0.5 |
| position_p95_rad | 0.010646 | 0.012615 | 0.05 |
| max_joint_speed_rad_s | 2.211584 | 2.211584 | 5 |

## Estimation

Own CVXPY estimator fits all 98 physical coefficients directly to the supplied immutable matrix. Positive semidefinite link pseudo-inertias enforce physically consistent mass, first moment and full symmetric COM inertia; explicit constraints enforce all supplied bounds. The weak squared regularizer (1e-5) selects among poorly identified parameter combinations using only homogeneous placeholders. No authored Panda dynamics were used. The numerical matrix has approximately 69 identifiable directions.

The selected weighted least-squares fit uses joint torque standard deviations, floored at 0.5 N m, and an armature lower bound of 0.03 kg m^2 selected through forward measurements. Candidate 10 was selected for balanced torque and forward-motion margins. Final candidate 18 verifies the identical submitted numeric values in another fresh process.

Three additional fits each omitted one entire recording. Their excluded recordings all passed the complete torque and physical-motion criteria. The worst excluded-recording torque RMSE was 0.228318 N m; velocity RMSE was 0.284585 rad/s.

18 complete physical candidate evaluations were used. Each used the prescribed module in a fresh process and retained configuration, logs, metrics and full traces in a unique directory. Candidate 1 failed forward-motion thresholds; its complete trace and logs are retained. A nonphysical helper indexing error is retained in `armature_sweep.log`; it did not start an additional physical evaluation.

Input archive and geometry hashes match the supplied manifests. Estimation helpers and numerical fitting logs are retained in the workspace.

## Limitations

Individual link parameters are not uniquely identified. These are effective dynamics for the filtered measured joint torque, which is not a recovered motor command. The 16 independent test recordings were neither accessed nor evaluated; final generalization remains for independent verification.
