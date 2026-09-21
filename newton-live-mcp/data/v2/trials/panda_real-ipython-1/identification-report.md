# Measured Panda dynamics identification

The exact submitted configuration is `config.json`. Candidate 020 completed all 1800 prescribed Newton steps and passed every pooled and per-recording threshold. Its full trace is `candidate-020.npz`; its metrics are `final-metrics.json`.

Fitting used only the supplied immutable 3150-by-98 regressor, measured targets, and geometry. A singular-value cutoff of 0.01 retained 69 observable combinations and removed the numerical near-null space. All 98 physical coefficients were estimated, with full positive pseudo-inertia constraints, center-of-mass and mass bounds, and joint-loss/bias/armature bounds. A scaled ridge penalty of 1e-5 regularized the nonunique solution. Residual multipliers were [1,1,1,1,3,2,3]. The selected armature lower bounds were [0.03,0.03,0.03,0.03,0.03,0.03,0.02] kg*m^2; these were selected using the authorized complete motion measurements to balance torque fit and forward stability. The pseudo-inertia positive-definiteness margin was 1e-5. No nominal Panda dynamics were used.

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| Maximum joint torque RMSE (N*m) | 0.198175 | 0.211193 | 0.5 |
| Maximum joint normalized torque RMSE | 0.250347 | 0.277148 | 0.5 |
| Maximum joint position RMSE (rad) | 0.009738 | 0.011844 | 0.025 |
| Maximum joint velocity RMSE (rad/s) | 0.226842 | 0.278985 | 0.5 |
| 95th-percentile absolute position error (rad) | 0.011163 | 0.013243 | 0.05 |
| Maximum joint speed (rad/s) | 2.208387 | 2.208387 | 5.0 |

As a further training-data generalization check, three models were each fitted on two recordings and evaluated through the complete prescribed workflow. Each omitted recording passed all thresholds. The worst omitted-recording torque RMSE was 0.216686 N*m, position RMSE 0.012381 rad, and velocity RMSE 0.293625 rad/s. Hyperparameters were selected using the training recordings, so these are supportive checks, not an independent test-fold result.

Twenty complete physical candidate evaluations were used. All candidate traces, including the two failed candidates, remain preserved. `identification-history.json` records candidate configurations and metrics. `offline-fitting-results.json` preserves algebraic fit and cross-validation results. `fit_model.py` contains the estimator and can reproduce the selected fitting recipe without running a simulator.

The training arrays, regressor, regressor manifest, and geometry hashes were verified unchanged. The final configuration on disk exactly matches the applied and measured configuration.

These parameters describe effective dynamics under publisher-filtered torque measurements. Individual link properties are not uniquely identifiable, and measured torque is not recovered raw motor command. The independent 16-recording test fold was not accessed or evaluated; its verification remains pending.
