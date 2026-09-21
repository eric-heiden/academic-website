# Panda effective dynamics identification

The final numeric `config.json` passes every training threshold in fresh-process candidate 009: all 1,800 steps, all 36 windows, and each of the three recordings. Nine complete physical candidates were evaluated; every candidate configuration, process log, metric file, and trace is retained.

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
| max_joint_torque_rmse_nm | 0.195546 | 0.210538 | 0.5 |
| max_joint_torque_normalized_rmse | 0.260766 | 0.294759 | 0.5 |
| max_joint_position_rmse_rad | 0.009787 | 0.011651 | 0.025 |
| max_joint_velocity_rmse_rad_s | 0.229036 | 0.276241 | 0.5 |
| position_p95_rad | 0.010620 | 0.012777 | 0.05 |
| max_joint_speed_rad_s | 2.211369 | 2.211369 | 5 |

The estimator in `fit.py` solves a convex, physically constrained least-squares problem using only the supplied immutable design matrix. Each link has a positive semidefinite pseudo-inertia matrix, bounded mass and COM, and bounded second moment. The fitted joint coefficients include viscous friction, Coulomb friction, bias, and armature. A weak scaled ridge term resolves ambiguity without nominal Panda dynamics. COM tensors are recovered from origin tensors using the parallel-axis relation.

Candidate 001 minimized torque error well but failed forward motion because effective inertias approached zero. Armature floors of 0.01, 0.02, 0.03, and 0.05 kg*m^2 were evaluated; all four passed training. The 0.03 floor balances motion and torque margins. Three further fits each omitted one recording; all passed both pooled and per-recording checks, including the omitted recording. Candidate 009 re-evaluates the selected all-recording fit exactly.

Reproduce the final fit with:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-sync --project /home/horde/apps/newton-live-mcp python fit.py --output reproduced-fit.json --armature-min .03
```

The final metric file is `candidate-009/metrics.json`; its complete trace is `candidate-009/metrics.npz`. The exact final configuration equals the configuration embedded in that passing measurement. Supplied data and geometry hashes remain unchanged.

Limitations: approximately 69 independent combinations of 98 coefficients are identifiable. This is an effective physical model of filtered observations; individual link parameters are not uniquely recovered. The 16-recording independent test fold was not accessed and its performance remains unverified.

LIP4RobotInverseDynamics, DOI 10.5281/zenodo.12516500, CC-BY-SA-4.0; MERL (Giacomuzzo, Carli, Romeres, Dalla Libera), 2024.
