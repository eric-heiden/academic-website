# Panda effective dynamics identification

The exact final parameters are in `config.json`. The matching complete training measurement is `candidate-009/metrics.json`, with all 1,800 simulation steps in `candidate-009/metrics.npz`. All thresholds pass pooled and separately for recordings 2, 3, and 4.

Fitted all 98 physical coefficients using the supplied geometry-only regressor. The convex fit enforces positive link pseudoinertia matrices, mass/COM/second-moment bounds, and bounded joint losses, biases and armatures. It uses a small scaled ridge penalty, joint residual weights `[1,1,1,1,4,4,4]`, and armature lower bounds `[0.03,0.03,0.03,0.03,0.03,0.03,0.015]` kg m². No nominal dynamics or external data were used.

| Metric | Worst recording | Limit |
|---|---:|---:|
| max_joint_torque_rmse_nm | 0.212627 | 0.5 |
| max_joint_torque_normalized_rmse | 0.277265 | 0.5 |
| max_joint_position_rmse_rad | 0.011997 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.283332 | 0.5 |
| position_p95_rad | 0.013529 | 0.05 |
| max_joint_speed_rad_s | 2.205052 | 5.0 |

Three further fits each excluded one entire recording. Each passed both torque and motion thresholds on its excluded recording (candidates 010–012). These are checks within the supplied training data; the independent final test fold remains inaccessible.

Twelve complete physical candidates were evaluated in separate fresh processes. All candidate configurations, process logs, metrics and traces remain in `candidate-001` through `candidate-012`, including failed candidates 001 and 005. Immutable training inputs and geometry were checked against their supplied SHA-256 hashes.

The fitted values describe effective dynamics under filtered measured torque. Individual link properties are not uniquely identifiable, and several fitted tensors lie near physical constraint boundaries. This is not a claim to recover the original hardware inertias or raw motor commands. Independent test performance remains unverified.

Data attribution: Giacomuzzo, Carli, Romeres and Dalla Libera / Mitsubishi Electric Research Laboratories, LIP4RobotInverseDynamics, DOI 10.5281/zenodo.12516500; supplied measured data licensed CC-BY-SA-4.0.
