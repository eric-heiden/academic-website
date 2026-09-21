Final configuration: config.json.

Final verification: candidate-020/metrics.json, success=true, 1800 steps across 36 windows.
Physical evaluations: 20; passing: 18. Every trace and process log is retained.

Worst result over the three supplied recordings:

| Metric | Fitted | Limit |
|---|---:|---:|
| max_joint_torque_rmse_nm | 0.209514 | 0.500000 |
| max_joint_torque_normalized_rmse | 0.277193 | 0.500000 |
| max_joint_position_rmse_rad | 0.011946 | 0.025000 |
| max_joint_velocity_rmse_rad_s | 0.282270 | 0.500000 |
| position_p95_rad | 0.013734 | 0.050000 |
| max_joint_speed_rad_s | 2.208515 | 5.000000 |

All pooled and per-recording checks passed. Three leave-one-recording-out refits using the final method also passed on their omitted recordings.

Fit: full 98-coefficient weighted least squares with convex physical moment constraints and weak zero-centered regularization. Small armature floors were selected using the measured forward windows. No nominal Panda dynamic parameters were used.

The link parameters are nonunique effective dynamics. The measured torque is filtered, and verification here covers 100 ms windows. The independent 16-recording test fold remains untested.
