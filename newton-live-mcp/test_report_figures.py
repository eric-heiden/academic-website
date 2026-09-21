"""Verify that report tables and figures preserve audited selection and measurements."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import export_real_data
import export_trace_windows
import plot_comparison
import render_comparison


class TestReportEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        plt.close("all")
        self.temporary.cleanup()

    def comparison(self):
        rows, pairs = [], []
        for cohort, scenario in (
            ("existing_primary", "panda"),
            ("real_primary", "panda_real"),
        ):
            for variant in (0, 1):
                for method, value in (("live", 20), ("restart", 40), ("ipython", 10)):
                    rows.append(
                        {
                            "id": f"{scenario}-{method}-{variant}",
                            "cohort": cohort,
                            "scenario": scenario,
                            "variant": variant,
                            "condition": method,
                            "startup_inclusive_seconds": value,
                            "input_output_tokens": value * 100,
                            "eligible_success": True,
                        }
                    )
            for other, ratio in (("restart", 2), ("ipython", 0.5)):
                stats = {
                    "pair_ids": [[f"{scenario}-live-0", f"{scenario}-{other}-0"]],
                    "paired_ratios": [ratio],
                    "geometric_ratio": ratio,
                    "bootstrap_95_interval": [ratio, ratio],
                    "n": 1,
                }
                pairs.append(
                    {
                        "cohort": cohort,
                        "scope": "all",
                        "numerator_condition": other,
                        "denominator_condition": "live",
                        "metrics": {
                            key: dict(stats)
                            for key in (
                                "startup_inclusive_seconds",
                                "input_output_tokens",
                                "uncached_input_output_tokens",
                            )
                        },
                        "excluded_pairs": [
                            {
                                "scenario": scenario,
                                "variant": 1,
                                "reason": "Reference identity mismatch",
                            }
                        ],
                    }
                )
        return {"trials": rows, "pairwise": pairs}

    def test_tables_invert_both_point_estimate_and_interval(self):
        result = self.comparison()
        stats = result["pairwise"][0]["metrics"]["startup_inclusive_seconds"]
        stats["bootstrap_95_interval"] = [1.25, 4]
        rendered = render_comparison.ratios(result)
        self.assertIn("Newton MCP / Restart", rendered)
        self.assertIn("0.500 [0.250, 0.800]; n=1", rendered)
        self.assertIn("Newton MCP / IPython MCP", rendered)
        self.assertIn("2.000 [2.000, 2.000]; n=1", rendered)

    def test_narrow_quality_failure_remains_numerical(self):
        key = "max_joint_torque_normalized_rmse"
        quality = {
            "finite": True,
            "sample_count": 9600,
            "expected_frames": 9600,
            "thresholds": {key: 0.5},
            key: 0.38,
            "per_episode": [{"episode": 30, key: 0.5005334121902201}],
        }
        row = {
            "id": "panda_real-ipython-3",
            "scenario": "panda_real",
            "variant": 3,
            "condition": "ipython",
            "eligible_success": False,
            "quality": quality,
            "issues": [],
            "failures": ["Independent physical quality or matching training failed"],
        }
        rendered = render_comparison.failure_details({"trials": [row]})
        self.assertIn("recording 30", rendered)
        self.assertIn("0.500533412 exceeds 0.5", rendered)
        self.assertNotIn("timeout", rendered)
        self.assertNotIn("trajectory not established", rendered)

    def test_figure_respects_audited_exclusions_and_mobile_parity(self):
        with patch.object(plot_comparison, "save"):
            desktop = plot_comparison.performance(self.comparison(), self.root)
            mobile = plot_comparison.performance(
                self.comparison(), self.root, mobile=True
            )
        self.assertEqual(desktop, mobile)
        included = [row for row in desktop["rows"] if row["included"]]
        excluded = [row for row in desktop["rows"] if not row["included"]]
        self.assertEqual(len(included), 8)
        self.assertEqual(len(excluded), 8)
        self.assertTrue(all(row["variant"] == 0 for row in included))
        self.assertTrue(
            all(
                row["omitted_reason"] == "Reference identity mismatch"
                for row in excluded
            )
        )
        self.assertEqual({row["newton_over_comparator"] for row in included}, {0.5, 2})

    def test_figure_rejects_disagreement_with_recorded_costs(self):
        result = self.comparison()
        result["trials"][0]["startup_inclusive_seconds"] = 21
        with (
            patch.object(plot_comparison, "save"),
            self.assertRaisesRegex(ValueError, "disagrees"),
        ):
            plot_comparison.performance(result, self.root)

    def trace(self, path, *, samples=50, nonfinite=False):
        q = np.full((samples, 7), 0.125)
        if nonfinite:
            q[17, 3] = np.nan
        np.savez(
            path,
            q=q,
            qd=np.zeros_like(q),
            reference_q=np.zeros_like(q),
            recorded_time=1 + np.arange(1, samples + 1) * 0.002,
            episode_ids=np.full(samples, 2),
            window_index=np.full(samples, 5),
        )

    def test_trace_requires_registered_window_and_preserves_nonfinite_gap(self):
        path = self.root / "trace.npz"
        self.trace(path, nonfinite=True)
        result = plot_comparison.window(path, 2)
        self.assertEqual(result["nonfinite_simulated_values"], 1)
        encoded = plot_comparison.serializable(result)
        self.assertIsNone(encoded["q"][17][3])
        json.dumps(encoded, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            plot_comparison.window(path, 21)
        self.trace(path, samples=49)
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            plot_comparison.window(path, 2)

    def test_missing_fixed_trace_is_disclosed_without_substitution(self):
        with patch.object(plot_comparison, "save"):
            result = plot_comparison.traces(
                self.root / "trials",
                self.root / "initial-training.npz",
                self.root / "initial-heldout.npz",
                self.root,
            )
        self.assertEqual(set(result), {"training", "heldout"})
        self.assertTrue(
            all(
                "omitted" in record
                for split in result.values()
                for record in split.values()
            )
        )
        self.assertTrue(
            all(
                "-0/verification/" in record["path"]
                for split in result.values()
                for name, record in split.items()
                if name != "initial"
            )
        )

    def test_quality_includes_pooled_gate(self):
        keys = (
            "max_joint_torque_rmse_nm",
            "max_joint_torque_normalized_rmse",
            "max_joint_position_rmse_rad",
            "max_joint_velocity_rmse_rad_s",
            "position_p95_rad",
            "max_joint_speed_rad_s",
        )
        measurement = {key: 0.25 for key in keys}
        measured = dict(
            measurement, thresholds={key: 1 for key in keys}, per_episode=[measurement]
        )
        measured[keys[0]] = 1.1
        rows = [
            {
                "id": "real-live-0",
                "cohort": "real_primary",
                "condition": "live",
                "variant": 0,
                "quality": measured,
                "eligible_success": False,
            }
        ]
        with patch.object(plot_comparison, "save"):
            result = plot_comparison.quality(rows, measured, self.root)
        self.assertEqual(
            result["rows"][0]["normalized_worst_pooled_or_recording_values"][0], 1.1
        )
        self.assertEqual(result["initial_values"][0], 1.1)
        self.assertFalse(result["rows"][0]["eligible_success"])

    def export_fixture(self):
        data = self.root / "data"
        public, private = {}, {}
        for partition, split, hashes in (
            ("public", "training", public),
            ("private", "heldout", private),
        ):
            for suffix in (".npz", "-regressor.npz", "-regressor.manifest.json"):
                name = split + suffix
                path = data / partition / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"inert test fixture")
                hashes[name if partition == "public" else str(path)] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (data / partition / f"{split}.manifest.json").write_text(
                json.dumps(
                    {
                        "reference_sha256": hashlib.sha256(
                            b"inert test fixture"
                        ).hexdigest(),
                        "dataset_doi": "10.5281/zenodo.12516500",
                    }
                )
            )
        geometry = data / "public/geometry/panda_geometry.xml"
        geometry.parent.mkdir()
        geometry.write_text("<mujoco/>")
        workspace = self.root / "trial"
        workspace.mkdir()
        task = workspace / "task.json"
        task.write_text(json.dumps({"input_hashes": public}))
        integrity = workspace / "integrity-manifest.json"
        integrity.write_text(
            json.dumps(
                {
                    "geometry_hashes": {
                        "geometry/panda_geometry.xml": hashlib.sha256(
                            geometry.read_bytes()
                        ).hexdigest()
                    }
                }
            )
        )
        entry = {
            "id": "panda_real-live-0",
            "workspace": str(workspace),
            "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
            "integrity_manifest_sha256": hashlib.sha256(
                integrity.read_bytes()
            ).hexdigest(),
        }
        return (
            data,
            {"trials": [entry]},
            {"task_sha256": entry["task_sha256"], "private_input_hashes": private},
        )

    def test_export_rejects_changed_bytes_and_extra_geometry(self):
        data, registration, prepared = self.export_fixture()
        inputs = export_real_data.export_inputs(data, registration, prepared)
        self.assertEqual(len(inputs), 9)
        extra = data / "public/geometry/extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "membership"):
            export_real_data.export_inputs(data, registration, prepared)
        extra.unlink()
        (data / "public/training.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            export_real_data.export_inputs(data, registration, prepared)

    def test_export_rejects_reference_symlink(self):
        data, registration, prepared = self.export_fixture()
        path = data / "public/training.npz"
        target = self.root / "same-bytes.npz"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            export_real_data.export_inputs(data, registration, prepared)

    def test_trace_export_rejects_wrong_fit_reference_or_association(self):
        source = self.root / "metrics.npz"
        metrics = {
            "scenario": "panda_real",
            "config": {"mass": [1]},
            "reference_sha256": "reference",
            "trace_path": str(source),
        }
        export_trace_windows.check_identity(
            metrics, config={"mass": [1]}, reference="reference", source=source
        )
        for changed in (
            dict(metrics, config={"mass": [2]}),
            dict(metrics, reference_sha256="wrong"),
            dict(metrics, trace_path=str(self.root / "other.npz")),
        ):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                export_trace_windows.check_identity(
                    changed, config={"mass": [1]}, reference="reference", source=source
                )


if __name__ == "__main__":
    unittest.main()
