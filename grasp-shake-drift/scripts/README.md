# Scripts for the grasp-shake-drift report

All measurements in the report were produced with these scripts against
[StoneT2000/newton-tests](https://github.com/StoneT2000/newton-tests) on
Newton 1.4.0 / MuJoCo Warp 3.10.0.3 / Warp 1.15.0, on an NVIDIA L40.

Drop them into `src/newtontests/` of the repro checkout and run with `uv run python -m newtontests.<name>`.

| Script | What it does |
| --- | --- |
| `franka_cube_shake.py` | The repro under test, unmodified (vendored here for reference). |
| `franka_cube_shake_fixed.py` | The same file with the two `# FIX` changes applied. Drop-in replacement; measured at 0.31 mm slip vs 6.36 mm, 23 contact rows vs 57, and passes the example's own `test_final()`. |
| `measure.py` | Headless run that logs the cube's pose in the TCP frame every frame. Produces the baseline slip trace. |
| `experiments.py` | `Variant` subclass plus the ~30 single-change configurations. `uv run python -m newtontests.experiments <variant> [...]`. |
| `frames.py` | Decomposes the drift across hand / TCP / finger / cube frames, to prove the articulation is rigid. |
| `contact_dump.py` | Reads live `d.contact` rows and `d.efc.force` mid-shake: normal force, tangential force, friction-cone utilisation, penetration. |
| `friction_probe.py` | Prints the friction / solimp / gap / priority values MuJoCo actually assigns to each geom. |
| `solref_probe.py` | Prints the `solref` on the loaded grasp contacts, used to check the `refsafe` clamp. |
| `efc_probe.py` | Prints per-row `efc_pos` / `efc_vel` / force for a loaded contact, to inspect normal vs friction rows. |
| `sweep.py` | Amplitude and frequency severity sweeps. |
| `bench.py` | Times ms/frame for a set of variants after a warm-up, to check the fix costs nothing. |
| `record.py` | Renders the videos headless (`xvfb-run -a uv run python -m newtontests.record ...`). |
| `make_plots.py` | Builds every figure in the report from `../data`. |

Extra dependencies beyond the repro's own: `matplotlib`, `imageio[ffmpeg]`, and `xvfb` for video capture.
