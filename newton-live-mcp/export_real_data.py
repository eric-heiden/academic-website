"""Package only measured references, regressors, and sanitized geometry after the study."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

NOTICE = """Measured Panda robot references and Newton dynamics features

Created by Mitsubishi Electric Research Laboratories (MERL), 2024.
Attribution: Giacomuzzo, Carli, Romeres and Dalla Libera.
Source: LIP4RobotInverseDynamics, https://doi.org/10.5281/zenodo.12516500
License for measured data and derived references/regressors: CC-BY-SA-4.0,
https://creativecommons.org/licenses/by-sa/4.0/

Adaptation: select the physical Panda recordings with training seeds 2/3/4 and
all 16 publisher test seeds; convert q/dq/ddq/tau_interp/t into numeric arrays;
retain original timestamps and publisher filtering. Construct fixed sampled
inverse-dynamics features using Newton and geometry, without fitted parameters.
The publisher's simulated responses and manufacturer dynamics are not included.

Sanitized Menagerie geometry retains its separate Apache-2.0 license in
references/geometry/ASSET_LICENSE. Its source is
https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda
Authored inertias, actuator settings, initial configurations and losses were
removed. Mesh density is zero; supplied runtime parameters determine dynamics.

No publisher estimator code is included. Newton study software remains under
its repository license; the data license does not relicense that software.
"""

README = """# Measured Panda identification references

Read NOTICE.txt and the included manifests for attribution, field schemas,
exact source digests, selection, and feature construction. This archive contains
physical measurements and immutable features, not fitted candidate parameters.
Independent per-agent fitted configurations accompany the report's trial records.

The reference arrays contain measured states and filtered/interpolated generalized
joint torque. They do not establish reconstructed raw motor commands. The model
starts from geometry and uniform placeholder dynamics; link parameters need not be
uniquely identifiable. Evaluation checks 100 ms forward windows and measured torque
agreement, both pooled and per recording. It does not test grasping or deployment.

Use frozen Newton revision fe4fda0199adae250d0ed1d5e37cb91f98c89f77 and the versions
in CONFIRMATION_ENVIRONMENT.json linked by the report. From the Newton checkout,
with this archive unpacked and a selected numeric candidate config.json:

```sh
uv run --no-sync -m tools.mcp_evaluation.real_rollout \\
  --config /absolute/path/config.json \\
  --reference /absolute/path/references/training.npz \\
  --output /absolute/path/verification/training/metrics.json
uv run --no-sync -m tools.mcp_evaluation.real_rollout \\
  --config /absolute/path/config.json \\
  --reference /absolute/path/references/heldout.npz \\
  --output /absolute/path/verification/heldout/metrics.json
```

Both references share the adjacent geometry directory. The held-out reference is
for post-agent verification; it was excluded from the agents' permitted inputs.
Source and input digests are listed in MANIFEST.json. Connection descriptors,
kernel credentials, fitted development answers and raw publisher pickle objects
are not part of this archive.
"""


def read_regular(path: Path) -> bytes:
    """Reject symlinks in the file and every parent before reading study inputs."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Refusing a symlink in an export input: {path}")
    if not path.is_file():
        raise ValueError(f"Missing regular export input: {path}")
    return path.read_bytes()


def checked_json(path: Path, expected: str) -> dict:
    content = read_regular(path)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Frozen manifest digest mismatch: {path}")
    return json.loads(content)


def export_inputs(data_root: Path, registration: dict, prepared: dict):
    """Bind numerical inputs and geometry to registered tasks and prepared verification."""
    entry = next(
        item for item in registration["trials"] if item["id"] == "panda_real-live-0"
    )
    workspace = Path(entry["workspace"])
    task = checked_json(workspace / "task.json", entry["task_sha256"])
    integrity = checked_json(
        workspace / "integrity-manifest.json", entry["integrity_manifest_sha256"]
    )
    if prepared["task_sha256"] != entry["task_sha256"]:
        raise ValueError(
            "Prepared verifier inputs do not belong to the registered task"
        )
    expected = {
        f"public/{name}": digest for name, digest in task["input_hashes"].items()
    }
    private = prepared["private_input_hashes"]
    private_names = {Path(name).name for name in private}
    if private_names != {
        "heldout.npz",
        "heldout-regressor.npz",
        "heldout-regressor.manifest.json",
    }:
        raise ValueError("Unexpected prepared verifier input membership")
    expected.update(
        {f"private/{Path(name).name}": digest for name, digest in private.items()}
    )
    expected.update(
        {
            f"public/{name}": digest
            for name, digest in integrity["geometry_hashes"].items()
        }
    )
    allowed = {
        "public/training.npz",
        "public/training-regressor.npz",
        "public/training-regressor.manifest.json",
        "private/heldout.npz",
        "private/heldout-regressor.npz",
        "private/heldout-regressor.manifest.json",
    }
    if any(
        name not in allowed and not name.startswith("public/geometry/")
        for name in expected
    ):
        raise ValueError("Unexpected frozen export input membership")
    geometry = data_root / "public/geometry"
    actual_geometry = set()
    for path in geometry.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink in geometry: {path}")
        if path.is_file():
            actual_geometry.add(path.relative_to(data_root).as_posix())
    if actual_geometry != {
        name for name in expected if name.startswith("public/geometry/")
    }:
        raise ValueError("Geometry membership differs from frozen integrity manifest")
    inputs = []
    for name, digest in sorted(expected.items()):
        path = data_root / name
        if ".." in Path(name).parts or Path(name).is_absolute():
            raise ValueError("Invalid input path in frozen manifest")
        content = read_regular(path)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Frozen data digest mismatch: {path}")
        inputs.append(
            (
                f"references/{name.split('/', 1)[1]}",
                content,
                "registered task/integrity manifest"
                if name.startswith("public/")
                else "prepared verifier input hashes",
            )
        )
    # Conversion metadata was not an agent input; disclose its weaker provenance
    # instead of claiming it was individually hash-bound by the registration.
    for partition, split in (("public", "training"), ("private", "heldout")):
        content = read_regular(data_root / partition / f"{split}.manifest.json")
        metadata = json.loads(content)
        if (
            metadata["reference_sha256"] != expected[f"{partition}/{split}.npz"]
            or metadata["dataset_doi"] != "10.5281/zenodo.12516500"
        ):
            raise ValueError(
                "Conversion metadata disagrees with the frozen measured reference"
            )
        inputs.append(
            (
                f"references/{split}.manifest.json",
                content,
                "conversion metadata; reference digest validated at export",
            )
        )
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--prepared-verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    if comparison["completed_trials"] != comparison["registered_trials"] or any(
        row["issues"] for row in comparison["trials"]
    ):
        raise ValueError(
            "Complete and audit every registered context before exporting held-out references"
        )
    registration = checked_json(
        args.registration, comparison["registration_raw_sha256"]
    )
    inputs = export_inputs(
        args.data_root, registration, json.loads(read_regular(args.prepared_verifier))
    )
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:

        def write(name, data):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=6)

        for name, content, provenance in inputs:
            records.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "digest_provenance": provenance,
                }
            )
            write(name, content)
        write("NOTICE.txt", NOTICE.encode())
        write("README.md", README.encode())
        manifest = {
            "source_commit": comparison["trials"][0]["source_commit"],
            "registration_sha256": comparison["registration_raw_sha256"],
            "data_source": "10.5281/zenodo.12516500",
            "data_license": "CC-BY-SA-4.0",
            "geometry_license": "Apache-2.0",
            "files": records,
        }
        write("MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode())
    manifest["archive"] = {
        "filename": args.output.name,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["archive"]))


if __name__ == "__main__":
    main()
