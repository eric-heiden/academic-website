# Measured Panda identification

The final full physical parameter set is `config.json`. It was applied through
`scenario.apply_config`, reset, and measured for 1800 Newton steps across the
prescribed 36 independent windows. `final_measurement.json` reports success
pooled and for each recording; `candidate-018.npz` retains the complete trace.

## Estimation

Only the supplied training observations, immutable 3150-by-98 regressor, and
sanitized geometry were used. No nominal Panda dynamics or other recordings
were used. The estimation code is in `fit_model.py`, `fit_sweep.py`,
`cross_validate.py`, and `finalize_fit.py`.

The model uses mass, first moments, origin inertia, and four joint coefficients
per joint as its linear coordinates. A semidefinite pseudo-inertia constraint
enforces physical inertia triangles and positive COM inertia. Mass, COM, second
moment, joint friction, bias, and armature obey the supplied bounds. A weak,
dimension-scaled quadratic prior (weight 1e-5) selects among equivalent physical
models without any authored dynamics. The objective is mean squared torque
error plus that regularizer. The pseudo-inertia Schur complement has a 1e-5
second-moment margin.

Torque-only fitting produced insufficient effective inertia for accurate
forward motion. Armature lower bounds were selected using the prescribed
physical evaluations, then the full model was refitted. The selected per-joint
lower bounds are [0.02, 0.01, 0.02, 0.01, 0.04, 0.02, 0.02] kg m^2.

Leaving each training recording out in turn produced passing torque and forward
predictions on the omitted recording. The worst omitted-recording errors were
0.21414 N m torque RMSE, 0.33671 normalized torque RMSE, 0.010829 rad position
RMSE, and 0.26091 rad/s velocity RMSE. These are validation within the supplied
training set, not the independent test fold.

## Final training measurement

| Metric (worst pooled or recording value) | Measured | Limit |
|---|---:|---:|
| max_joint_torque_rmse_nm | 0.20755862 | 0.5 |
| max_joint_torque_normalized_rmse | 0.32839124 | 0.5 |
| max_joint_position_rmse_rad | 0.01036468 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.24863760 | 0.5 |
| position_p95_rad | 0.01306419 | 0.05 |
| max_joint_speed_rad_s | 2.20805621 | 5.0 |

All 18 complete physical evaluations and traces are preserved, including failed
candidates. Offline semidefinite fitting and cross-validation do not invoke a
simulator. The final live model and saved numeric configuration match exactly.
Input hashes and geometry hash were checked after fitting and are unchanged.

## Limitations and provenance

The regressor has approximately 69 identifiable directions at relative singular
value threshold 1e-5, so the individual link masses, COMs, inertias, and armatures
are not uniquely identified. This is an effective model of publisher-filtered
joint measurements and 100 ms motion; it does not recover raw motor commands.
The 16-recording independent test fold was not accessed and remains unverified
in this session.

Data: LIP4RobotInverseDynamics, DOI 10.5281/zenodo.12516500, MERL (2024),
Giacomuzzo, Carli, Romeres, Dalla Libera; CC-BY-SA-4.0.
