Selected candidate 002; config.json is byte-identical to candidate-002/config.json and numerically identical to its passing completed measurement.

Five fresh-process physical candidate evaluations were completed. Candidate 001 failed motion checks; candidates 002–005 passed all pooled and per-recording thresholds. All outputs are retained.

| Check | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.196055 | 0.210050 | 0.5 |
| max_joint_torque_normalized_rmse | 0.255156 | 0.285174 | 0.5 |
| max_joint_position_rmse_rad | 0.009755 | 0.011775 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.228038 | 0.278394 | 0.5 |
| position_p95_rad | 0.010675 | 0.012730 | 0.05 |
| max_joint_speed_rad_s | 2.211429 | 2.211429 | 5.0 |

Weighted linear least squares with seven positive-semidefinite 4x4 pseudo-inertia constraints, mass/COM/second-moment and joint coefficient bounds; weak scaled ridge (lambda=1e-4) toward homogeneous placeholders. Joint armature floor 0.03 kg*m^2 chosen by comparing torque and prescribed forward motion checks.

Leave-one-recording-out validation used only the three supplied training recordings. Worst validation torque RMSE: 0.21591159710792696 N m.

Effective dynamics for filtered measured torque; individual link parameters are not uniquely identified.
Only the supplied three recordings were used. Independent held-out evaluation is pending and was not accessed.
The positive armature floor trades torque residual against sensitivity of forward motion to residual torque.

LIP4RobotInverseDynamics, MERL (Giacomuzzo, Carli, Romeres, Dalla Libera), 2024, DOI 10.5281/zenodo.12516500; CC-BY-SA-4.0.
