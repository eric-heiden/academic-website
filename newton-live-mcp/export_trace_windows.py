"""Export the registered figure windows without publishing unrelated runtime artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from export_real_data import checked_json, read_regular
from plot_comparison import serializable, window

NOTICE = """Fixed-window excerpts for the Newton live simulation study

Each NPZ contains only the sixth 100 ms prediction window, all seven joints,
from training recording 2 or held-out recording 21, for replicate zero.
These are plot inputs, not complete verifier trajectories. The original full
trace digests and the excerpt digests are recorded in MANIFEST.json. Full-trace
digests are measured at export, not commitments made when trials completed.
Verifier configuration/reference identities are checked against audited rows;
initial metrics are checked against the registered uniform initial configuration.

Measured references: MERL, 2024; Giacomuzzo, Carli, Romeres and Dalla Libera.
Source: https://doi.org/10.5281/zenodo.12516500
License: CC-BY-SA-4.0, https://creativecommons.org/licenses/by-sa/4.0/
Adaptations: measured references are interpolated at fixed 2 ms steps; simulated
trajectories are produced by Newton with the registered initial model or each
agent's submitted parameters. Nonfinite simulated values are preserved.

The initial model's metrics JSON files describe full training/held-out
verification; the adjacent NPZ files deliberately contain only figure excerpts.
Do not recompute full-study quality from these abbreviated NPZ files.

To regenerate the report figures, unpack this archive, obtain comparison.json
and plot_comparison.py from the report, and use the recorded Python environment:

uv run --no-sync python plot_comparison.py --comparison comparison.json \\
  --trial-root confirmation --initial-training initial/training/metrics.npz \\
  --initial-heldout initial/heldout/metrics.npz --output regenerated-figures

All unavailable selected traces remain omissions; another trial is never used
as a replacement. The selection was registered before real-data trial outcomes.
"""


def check_identity(metrics, *, config, reference, source):
    """Reject mislabeled verifier evidence without requiring the model to pass."""
    if (
        metrics.get("scenario") != "panda_real"
        or metrics.get("config") != config
        or metrics.get("reference_sha256") != reference
    ):
        raise ValueError(
            f"Verifier configuration/reference identity mismatch: {source}"
        )
    if Path(metrics.get("trace_path", "")).resolve() != source.resolve():
        raise ValueError(f"Verifier trace association mismatch: {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    if comparison["completed_trials"] != comparison["registered_trials"] or any(
        row["issues"] for row in comparison["trials"]
    ):
        raise ValueError(
            "Complete and audit the study before exporting its held-out traces"
        )
    registration = checked_json(
        args.registration, comparison["registration_raw_sha256"]
    )
    entries = {item["id"]: item for item in registration["trials"]}
    rows = {item["id"]: item for item in comparison["trials"]}
    entry = entries["panda_real-live-0"]
    task = checked_json(Path(entry["workspace"]) / "task.json", entry["task_sha256"])
    references = {
        "training": task["input_hashes"]["training.npz"],
        "heldout": rows[entry["id"]]["quality"]["reference_sha256"],
    }
    files, records, omissions = {}, [], []
    initial_config = (json.dumps(task["initial"], indent=2) + "\n").encode()
    files["initial/config.json"] = initial_config
    records.append(
        {
            "path": "initial/config.json",
            "sha256": hashlib.sha256(initial_config).hexdigest(),
            "bytes": len(initial_config),
            "source_task_sha256": entry["task_sha256"],
        }
    )
    for split, episode in (("training", 2), ("heldout", 21)):
        paths = {
            "initial": (
                args.initial_root / split / "metrics.npz",
                f"initial/{split}/metrics.npz",
            )
        }
        for method in ("live", "ipython", "restart"):
            identifier = f"panda_real-{method}-0"
            if (args.trial_root / identifier).resolve() != Path(
                entries[identifier]["workspace"]
            ).resolve():
                raise ValueError(
                    "Trace root differs from the registered original workspace"
                )
            suffix = f"panda_real-{method}-0/verification/" + (
                "training/metrics.npz" if split == "training" else "metrics.npz"
            )
            paths[method] = (args.trial_root / suffix, f"confirmation/{suffix}")
        for method, (source, target) in paths.items():
            if any(path.is_symlink() for path in (source, *source.parents)):
                raise ValueError(
                    f"Refusing a symlink in fixed-window evidence: {source}"
                )
            metrics_path = source.with_suffix(".json")
            if metrics_path.exists():
                metrics = json.loads(read_regular(metrics_path))
                config = (
                    task["initial"]
                    if method == "initial"
                    else rows[f"panda_real-{method}-0"]["quality"]["config"]
                )
                check_identity(
                    metrics, config=config, reference=references[split], source=source
                )
            elif method == "initial":
                raise ValueError(
                    f"Initial metrics JSON is required for plot regeneration: {metrics_path}"
                )
            else:
                omissions.append(
                    {
                        "split": split,
                        "method": method,
                        "target": target,
                        "reason": "Accompanying verifier metrics JSON unavailable",
                    }
                )
                continue
            try:
                selected = window(source, episode)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                omissions.append(
                    {
                        "split": split,
                        "method": method,
                        "target": target,
                        "reason": str(error),
                    }
                )
                continue
            payload = io.BytesIO()
            arrays = {
                key: value
                for key, value in selected.items()
                if isinstance(value, np.ndarray)
            }
            np.savez_compressed(
                payload,
                **arrays,
                episode_ids=np.full(50, episode),
                window_index=np.full(50, 5),
            )
            files[target] = payload.getvalue()
            records.append(
                {
                    "path": target,
                    "sha256": hashlib.sha256(files[target]).hexdigest(),
                    "bytes": len(files[target]),
                    "original_trace_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                    "original_trace_digest_provenance": "measured at export",
                    "original_trace_bytes": source.stat().st_size,
                    "verifier_metrics_sha256": hashlib.sha256(
                        read_regular(metrics_path)
                    ).hexdigest(),
                    "identity_checks": "Verifier scenario, configuration, measured reference digest, and associated trace path",
                    "episode": episode,
                    "window_index": 5,
                    "samples": 50,
                    "nonfinite_simulated_values": selected[
                        "nonfinite_simulated_values"
                    ],
                }
            )
        metrics_path = args.initial_root / split / "metrics.json"
        metrics = json.loads(read_regular(metrics_path))
        metrics.pop("trace_path", None)
        target = f"initial/{split}/metrics.json"
        files[target] = (
            json.dumps(serializable(metrics), indent=2, allow_nan=False) + "\n"
        ).encode()
        records.append(
            {
                "path": target,
                "sha256": hashlib.sha256(files[target]).hexdigest(),
                "bytes": len(files[target]),
                "content": "Full initial-model verifier metrics; adjacent NPZ is a figure excerpt",
            }
        )
    manifest = {
        "registration_sha256": comparison["registration_raw_sha256"],
        "source_commit": comparison["trials"][0]["source_commit"],
        "selection": "Replicate 0, window index 5, training seed 2 and heldout seed 21",
        "data_license": "CC-BY-SA-4.0",
        "files": records,
        "omissions": omissions,
    }
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    files["NOTICE.txt"] = NOTICE.encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=6)
    manifest["archive"] = {
        "filename": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["archive"]))


if __name__ == "__main__":
    main()
