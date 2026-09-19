# Synthetic Panda response references

These files contain Newton-generated joint positions, not physical measurements.
The release occurred after all 18 independent agent trials completed. The seed,
generating parameters, and held-out responses were excluded from agent inputs.

- `manifest.json` is the unchanged pre-agent commitment record.
- `release.json` gives release time and verified original SHA-256 digests.
- `truth.json` reveals the NumPy PCG64 seed and three generating configurations.
- `variant-N/training.npz`: `episodes=[0,1]`, `q.shape=(2,1500,7)`.
- `variant-N/heldout.npz`: `episodes=[2]`, `q.shape=(1,1500,7)`.
- `variant-N/generating-config.py` is the corresponding committed configuration.

Positions are radians, recorded after each 0.002-second step. Each episode lasts
3 seconds. The noise is independent Gaussian position noise with standard
deviation 0.0002 rad. Velocity and command traces are candidate outputs, not
fields in these reference NPZ files.

The reproducible forward model and commands are in
[`tools/mcp_evaluation/calibration.py`](https://github.com/eric-heiden/newton/blob/cd9bfdd866459a58e26a3ce6687f741c40d1f26b/tools/mcp_evaluation/calibration.py).
The dataset generator initializes `numpy.random.default_rng(truth['seed'])`,
draws all three configurations first, in the bound order `payload_mass`,
`damping_multiplier`, `joint_friction`, then processes instances 0, 1, 2. For each,
`CalibrationScenario(config).rollout_episode(i)` produces the clean positions for
episodes 0, 1, 2. The generator stacks these into shape `(3,1500,7)`, draws the
Gaussian noise with the same RNG and that shape, and adds it before splitting
training and held-out arrays. No reference instance was rejected or resampled.
Floating-point trajectories can differ across software, devices, or versions;
the supplied files preserve the exact responses used by these trials.

From this extracted directory, verify every original file commitment:

```bash
uv run --no-project python - <<'PY'
import hashlib, json
from pathlib import Path
release = json.loads(Path('release.json').read_text())
for record in release['files']:
    actual = hashlib.sha256(Path(record['file']).read_bytes()).hexdigest()
    assert actual == record['sha256'], record['file']
print('All ten reference, truth, and configuration commitments match.')
PY
```

Use the [evaluation guide](https://github.com/eric-heiden/newton/blob/cd9bfdd866459a58e26a3ce6687f741c40d1f26b/tools/mcp_evaluation/README.md)
for candidate rollout and final verification commands. Agents in a new comparison
must receive training responses only; exposing this post-study reveal would
change the identification task.
