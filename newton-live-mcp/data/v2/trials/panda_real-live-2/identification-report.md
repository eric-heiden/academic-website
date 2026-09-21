Fitted model saved in config.json.

All 98 coefficients were estimated from the supplied regressor, geometry and measured training data; no nominal Panda dynamics were used. The regressor has 69 numerically identifiable directions above its float32 noise floor. Physical pseudo-inertia constraints enforce positive COM inertia, triangle inequalities, mass/COM bounds and the second-moment bound. Weighted Huber fitting (threshold 0.15) uses ridge 0.0001, coefficient scales in fit_model.py, second-moment eigenvalue floor 0.00025, and selected armature floors [0.02, 0.01, 0.02, 0.01, 0.025, 0.02, 0.015].

The selected model passed all six thresholds pooled and for each of three recordings. Each of three auxiliary fits omitting one recording also passed the complete training measurement, including the omitted recording. These are checks within the supplied training data, not the independent final test.

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.19085255 | 0.20509423 | 0.5 |
| max_joint_torque_normalized_rmse | 0.24810209 | 0.27377473 | 0.5 |
| max_joint_position_rmse_rad | 0.010088949 | 0.011849897 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.23360318 | 0.27680092 | 0.5 |
| position_p95_rad | 0.011647055 | 0.013477984 | 0.05 |
| max_joint_speed_rad_s | 2.2106667 | 2.2106667 | 5 |

29 of 60 physical candidate evaluations completed. Every trace and original live log was preserved. The final candidate, candidate-029.npz, was measured after rebuilding the solver and exactly matches the submitted config.json. Input hashes were checked against the immutable task and manifest.

Only effective parameters are identifiable; recordings have filtered torque. Independent 16-recording fold was not accessed and remains unverified.
