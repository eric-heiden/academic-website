# Effective Panda dynamics identification

The submitted `config.json` exactly matches completed candidate 018. Its 1,800-step measurement passed every threshold pooled and in each of the three supplied physical recordings. All 18 fresh-process candidates, including the failed initial candidate, retain their numeric configuration, metrics, full NPZ trace, and process log.

## Method

Fitted the supplied 3,150-by-98 Newton regressor directly using weighted least squares and CVXPY/Clarabel. Each link has a full symmetric origin inertia, mass, and first moment. Positive semidefinite pseudo-inertia constraints enforce physically valid COM tensors and triangle inequalities; mass, COM, second-moment, loss, bias, and armature bounds are enforced. A small coefficient regularizer (1e-4) uses only the homogeneous placeholders. Joint loss coefficients are nonnegative.

The torque-only optimum made several effective rotational inertias too small for accurate measured forward motion. Candidate comparisons selected armature lower bounds [0.025, 0.025, 0.025, 0.025, 0.030, 0.025, 0.025] kg m², refitting all other parameters jointly. These constraints balance torque and motion prediction. No nominal Panda dynamics or external data were used.

## Final measured quality

| Metric | Pooled | Worst recording | Required maximum |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.194670 | 0.208646 | 0.5 |
| max_joint_torque_normalized_rmse | 0.254927 | 0.285664 | 0.5 |
| max_joint_position_rmse_rad | 0.009845 | 0.011819 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.229794 | 0.279070 | 0.5 |
| position_p95_rad | 0.011264 | 0.013451 | 0.05 |
| max_joint_speed_rad_s | 2.210194 | 2.210194 | 5 |

## Validation and artifacts

The final fitting procedure was also trained on each pair of recordings (candidates 015–017). Each excluded recording passed every torque and forward-motion threshold. Worst excluded-recording torque RMSE was 0.214469 N m, normalized torque RMSE 0.295323, position RMSE 0.012378 rad, and velocity RMSE 0.293983 rad/s.

- Final configuration: `config.json`
- Final completed measurement and trace: `candidate-018/metrics.json`, `candidate-018/metrics.npz`
- Fitting code: `fit_model.py`; final fit and validation driver: `finalize_fit.py`
- Audit with input hashes and candidate records: `identification-audit.json`

## Limits

These are effective dynamics for filtered measured joint torques, not a unique recovery of individual link properties or raw motor commands. The supplied data do not uniquely identify every link coefficient. Forward validation covers 100 ms windows; longer horizons are untested. The independent 16-recording test fold was not accessed and remains for the post-submission verifier.
