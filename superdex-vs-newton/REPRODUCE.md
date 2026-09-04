# Reproducing the Newton × SuperDex robot comparison

This bundle corresponds to the public report refreshed on 4 September 2026.
It pins the exact repository states used for the evidence:

- Newton: `d37f4d3d341ccce1e06a1dff21e9a054759b4855`
- Project SuperDex: `777b498122ab4b8581dfecb5f035af134c647eab`

The recorded Newton runs used Warp 1.17.0 on an NVIDIA RTX PRO 6000
Blackwell Server Edition MIG 1g.24gb device. Different supported devices can
reproduce the tasks, but timing and floating-point traces need not be identical.

## Newton robot workloads

From a clean Newton checkout:

```bash
git clone https://github.com/newton-physics/newton.git
cd newton
git checkout d37f4d3d341ccce1e06a1dff21e9a054759b4855
uv sync --extra examples
```

Download `scripts/render_newton_robot_workloads.py` from this report, then run:

```bash
env PYGLET_HEADLESS=1 uv run --with imageio --with imageio-ffmpeg \
  /path/to/render_newton_robot_workloads.py all \
  --output /path/to/results/media
```

The command creates these media files:

- `newton-g1-locomotion.mp4` and `.jpg`
- `newton-panda-pick-place.mp4` and `.jpg`
- `newton-h1-jacket.mp4` and `.jpg`

The JSON traces are written to a `data` directory alongside `media`. Each file
records its Newton revision, package versions, device, timestamp, samples, and
summary values. Each workload also runs the corresponding example's final check.

Individual workloads can be selected with `g1`, `panda`, or `cloth` instead of
`all`. A displayless Linux machine needs a working headless OpenGL setup. When
that is unavailable, the stock examples can still be checked with Newton's null
viewer, but no video will be produced.

## Stock Newton entry points

These commands run the same first-party examples interactively:

```bash
uv run --extra examples -m newton.examples robot_policy --robot g1_29dof
uv run --extra examples -m newton.examples robot_panda_hydro --scene pen --world-count 1
uv run --extra examples -m newton.examples cloth_h1
```

The Panda example also accepts `--scene cube`. The G1 example uses `i`/`k` for
forward/backward motion, `j`/`l` for lateral motion, `u`/`o` for turning, and
`p` to reset.

## SuperDex G1 compatibility probe

From a clean Project SuperDex checkout:

```bash
git clone https://github.com/facebookresearch/project_superdex.git
cd project_superdex
git checkout 777b498122ab4b8581dfecb5f035af134c647eab
CC=clang-17 CXX=clang++-17 \
CMAKE_ARGS='-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=/usr/bin/clang-scan-deps-17 -DCMAKE_C_COMPILER_CLANG_SCAN_DEPS=/usr/bin/clang-scan-deps-17 -DCMAKE_ASM_COMPILER_CLANG_SCAN_DEPS=/usr/bin/clang-scan-deps-17' \
uv sync --extra core
```

The probe takes an existing Unitree G1 URDF. The Newton asset helper can obtain
the exact robot asset used here:

```bash
cd /path/to/newton
G1_URDF=$(uv run --extra examples python -c \
  'import newton.utils; print(newton.utils.download_asset("unitree_g1") / "urdf/g1_29dof.urdf")')
```

Download `scripts/probe_superdex_g1.py`, return to the pinned SuperDex checkout,
and run:

```bash
uv run /path/to/probe_superdex_g1.py \
  --urdf "$G1_URDF" \
  --steps 10 \
  --output /path/to/superdex-g1-probe.json
```

The script deliberately caps the step count at 100. It records the source-build
revision, input checksum and geometry inventory, imported articulation structure,
diagnostics, elapsed setup/step time, and root motion.

## Integrity

`data/manifest.json` lists every report-owned data and media file with SHA-256
checksums. Verify a downloaded file with:

```bash
sha256sum /path/to/file
```

The SuperDex gallery clips embedded in the report are first-party project media,
not outputs of the compatibility probe. Their source URLs and attribution are
recorded in the manifest.
