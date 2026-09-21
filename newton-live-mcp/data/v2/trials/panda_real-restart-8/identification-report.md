# Effective Panda dynamics identification

Final configuration: `config.json`, exactly matching the passing configuration in `candidate-005/metrics.json`.

Fitted from the supplied measured recordings and immutable geometry-based regressor only. No nominal dynamics or external data were used. The model includes seven complete link mass/COM/inertia sets and viscous friction, Coulomb friction, torque bias, and armature for every joint.

The estimator uses a constrained convex fit with positive-semidefinite link pseudo-inertias, the prescribed mass/COM/second-moment and joint bounds, a weighted Huber loss, and weak scaled coefficient regularization. Singular values below 0.01 were excluded from fitting to avoid exploiting float32 noise in unidentifiable directions. Small positive armature floors were selected using complete physical rollouts to balance torque error and sensitivity to measured torque residuals.

Eight complete candidates were run in eight fresh rollout processes. All logs and traces are retained, including the failed first candidate. The final candidate completed 1,800 steps and passed all six thresholds pooled and in every recording.

| Metric | Worst across recordings and pooled | Limit |
|---|---:|---:|
| max_joint_torque_rmse_nm | 0.207763 | 0.5 |
| max_joint_torque_normalized_rmse | 0.287401 | 0.5 |
| max_joint_position_rmse_rad | 0.011581 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.272816 | 0.5 |
| position_p95_rad | 0.013414 | 0.05 |
| max_joint_speed_rad_s | 2.210150 | 5 |

Three additional models were each fitted while omitting one recording, then evaluated using the unchanged complete training protocol. Each passed forward and torque checks, including on its omitted recording. These are transfer checks within the supplied training recordings; the independent 16-recording test fold was not accessed or evaluated.

Individual link parameters remain nonunique. The identified model represents effective dynamics under publisher filtering and cannot recover raw motor commands. Good short-horizon agreement does not establish long-horizon prediction accuracy.

Data attribution: published LIP4RobotInverseDynamics recordings, DOI 10.5281/zenodo.12516500, CC-BY-SA-4.0 (as supplied in the manifest).
