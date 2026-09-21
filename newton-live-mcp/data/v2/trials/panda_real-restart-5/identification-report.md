The final full numeric model is in `config.json` and matches the passing completed measurement in `candidate-013/metrics.json`. All three supplied recordings pass every threshold, pooled and separately, over 36 windows and 1,800 prescribed Newton steps. All 13 complete physical evaluations used fresh processes; the failed first candidate and every log and trace are retained.

| Metric (worst recording) | Measured | Limit |
| --- | ---: | ---: |
| Joint torque RMSE (N m) | 0.209965 | 0.5 |
| Normalized joint torque RMSE | 0.284561 | 0.5 |
| Joint position RMSE (rad) | 0.011894 | 0.025 |
| Joint velocity RMSE (rad/s) | 0.281200 | 0.5 |
| 95th percentile absolute position error (rad) | 0.013594 | 0.05 |
| Maximum joint speed (rad/s) | 2.205067 | 5 |

Constrained weighted least squares over all 98 physical coefficients with CVXPY CLARABEL. PSD pseudo-inertias enforce physical COM inertias and triangle inequalities. Broad mass/COM/second-moment/joint bounds enforced. Scale-normalized quadratic regularization 1e-4; armature floors [0.03,0.03,0.03,0.03,0.03,0.02,0.015] kg m^2 selected from prescribed full physical evaluations. No nominal dynamics used.

Each of the three recording-omission fits also passed on its omitted recording. Worst omitted-recording joint torque, position and velocity RMSE were 0.215820 N m, 0.012456 rad and 0.296084 rad/s.

All supplied data and regressor hashes and the geometry hash match their original manifests. Complete trace values were read and independently checked against the reported torque, position and velocity RMSE.

Individual link parameters are not uniquely identifiable. Model represents effective dynamics of filtered measured joint torque, not raw motor commands. Omission checks reuse the supplied training recordings and are not the independent 16-recording test fold. Independent test results remain unknown.
