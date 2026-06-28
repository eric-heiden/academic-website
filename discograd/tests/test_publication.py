from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

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
REPORT_TEMPLATE = Path(__file__).resolve().parents[1] / "report_template.html"
BUILD_SCRIPT = REPORT_TEMPLATE.with_name("build_report.py")
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _RenderedPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[tuple[dict[str, str], str]] = []
        self.figures: list[dict[str, str]] = []
        self.images: list[tuple[dict[str, str], int | None]] = []
        self.data_narratives: list[dict[str, str]] = []
        self._cell_attributes: dict[str, str] | None = None
        self._cell_text: list[str] = []
        self._figure_stack: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        if tag == "td":
            self._cell_attributes = attributes
            self._cell_text = []
        elif tag == "figure":
            self.figures.append(attributes)
            self._figure_stack.append(len(self.figures) - 1)
        elif tag == "img":
            parent = self._figure_stack[-1] if self._figure_stack else None
            self.images.append((attributes, parent))
        elif tag == "p" and "lede" in attributes.get("class", "").split():
            self.data_narratives.append(attributes)

    def handle_data(self, data: str) -> None:
        if self._cell_attributes is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell_attributes is not None:
            self.cells.append((self._cell_attributes, "".join(self._cell_text).strip()))
            self._cell_attributes = None
            self._cell_text = []
        elif tag == "figure" and self._figure_stack:
            self._figure_stack.pop()


def _resolve_json_pointer(value: object, pointer: str) -> object:
    if pointer.startswith("#"):
        pointer = unquote(pointer[1:])
    if not pointer.startswith("/"):
        raise AssertionError(f"not a JSON pointer: {pointer!r}")
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise AssertionError(f"pointer {pointer!r} traverses a scalar")
    return current


def _displayed_numeric_value(text: str) -> int | float | list[object] | None:
    if text.startswith("[") and text.endswith("]"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        return (
            value
            if isinstance(value, list)
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            )
            else None
        )
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


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
            "start_id": "initial",
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
    analytic_anchor = {
        "row_id": "analytic:anchor",
        "scenario_family": "analytic",
        "accepted": True,
    }
    optimization_rows = [
        {
            "row_id": f"path:optimization:score:{seed}",
            "scenario_family": "path_tracer",
            "scenario": "five_spheres",
            "method": "score",
            "final_hard_loss": 0.0,
            "held_out_loss": 0.0,
            "success": True,
        }
        for seed in range(2)
    ]
    comparison_images = {
        "comparison_initial": [[[0.2, 0.2, 0.2]]],
        "comparison_target": [[[0.0, 0.0, 0.0]]],
        "comparison_recovered": [[[0.1, 0.1, 0.1]]],
    }
    comparison_rows = [
        {
            "row_id": f"path:{role}",
            "scenario_family": "path_tracer",
            "scenario": "analytic_five_sphere_render",
            "role": role,
            "accepted": True,
            "image": image,
        }
        for role, image in comparison_images.items()
    ]
    triangle_diagnostic = {
        "row_id": "triangle:0",
        "scenario_family": "triangle_2d",
        "scenario": "edge_slice",
        "edge": 0,
        "rows": [
            {
                "signed_offset": -0.1,
                "analytic_intersection": 0.0,
                "hard_intersection": 0.0,
            },
            {
                "signed_offset": 0.1,
                "analytic_intersection": 1.0,
                "hard_intersection": 1.0,
            },
        ],
    }
    collision_diagnostic = {
        "row_id": "collision:0",
        "scenario_family": "collision_2d",
        "scenario": "pinball_bank",
        "start_id": "start_0",
        "outer_seed": 300,
        "final_positions": [[0.0, 0.0], [0.5, 0.25]],
        "losses": [1.0, 0.5],
    }
    contact_gradient = {
        **next(row for row in method_rows if row["method"] == "soft_ad"),
        "row_id": "contact:soft_ad",
        "scenario_family": "contact_3d",
        "scenario": "three_sphere_floor_ramp",
        "start_id": "initial_launch_velocity",
        "target": "launch_velocity",
    }
    contact_diagnostic = {
        "row_id": "contact:0",
        "scenario_family": "contact_3d",
        "scenario": "three_sphere_floor_ramp",
        "positions": [[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.1]]],
        "max_penetration": 0.01,
        "max_contact_energy_gain": 0.02,
    }
    opaque_diagnostic = {
        "row_id": "mesh:0",
        "scenario_family": "opaque_mesh",
        "scenario": "procedural_cube_silhouette",
        "transform_status": "estimator_only",
        "transformable": False,
        "boundary": {"transformed_sites": 0, "preserved_sites": 1},
    }
    raw: dict[str, list[dict[str, object]]] = {name: [] for name in RAW_NAMES}
    raw["analytic.json"] = [*method_rows, analytic_anchor]
    raw["path_tracer_gradients.json"] = [*score_rows, *comparison_rows]
    raw["path_tracer_optimization.json"] = optimization_rows
    raw["triangle_2d.json"] = [triangle_diagnostic]
    raw["collision_2d.json"] = [collision_diagnostic]
    raw["contact_3d_gradients.json"] = [contact_gradient, contact_diagnostic]
    raw["opaque_mesh.json"] = [opaque_diagnostic]
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

    method_summaries = [
        {
            "scenario": "analytic:step:analytic",
            "target": "x",
            "method": row["method"],
            "samples": 1,
            "mean_gradient": [0.0],
            "relative_error": 0.0,
            "cosine_similarity": 0.0,
            "sign_agreement": 0.0,
            "empirical_bias": [0.0],
            "empirical_variance": [0.0],
            "mean_squared_error": [0.0],
            "source_row_ids": [row["row_id"]],
        }
        for row in method_rows
    ]
    method_summaries.extend(
        (
            {
                "scenario": "path_tracer:five_spheres:initial",
                "target": "initial_parameters",
                "method": "score",
                "samples": 8,
                "mean_gradient": [0.0],
                "relative_error": 0.0,
                "cosine_similarity": 0.0,
                "sign_agreement": 0.0,
                "empirical_bias": [0.0],
                "empirical_variance": [0.0],
                "mean_squared_error": [0.0],
                "source_row_ids": [row["row_id"] for row in score_rows],
            },
            {
                "scenario": "contact_3d:three_sphere_floor_ramp:initial_launch_velocity",
                "target": "launch_velocity",
                "method": "soft_ad",
                "samples": 1,
                "mean_gradient": [0.0],
                "relative_error": 0.0,
                "cosine_similarity": 0.0,
                "sign_agreement": 0.0,
                "empirical_bias": [0.0],
                "empirical_variance": [0.0],
                "mean_squared_error": [0.0],
                "source_row_ids": [contact_gradient["row_id"]],
            },
        )
    )
    method_summaries.sort(
        key=lambda row: (
            str(row["scenario"]),
            str(row["target"]),
            str(row["method"]),
            int(row["samples"]),
        )
    )
    path_summary_index = next(
        index
        for index, row in enumerate(method_summaries)
        if str(row["scenario"]).startswith("path_tracer:")
    )
    contact_summary_index = next(
        index
        for index, row in enumerate(method_summaries)
        if str(row["scenario"]).startswith("contact_3d:")
    )

    plot_rows = {
        "analytic_gates.json": [
            {
                "plot_id": "analytic-gates",
                "values": [0.0, 0.0, 0.0],
                "source_row_ids": [row["row_id"] for row in raw["analytic.json"]],
            }
        ],
        "gradient_quality.json": [
            {
                "plot_id": "triangle-edge-slice-0",
                "kind": "triangle_edge_slice",
                "scenario": "triangle_2d:edge_slice",
                "edge": 0,
                "x_values": [-0.1, 0.1],
                "values": [0.0, 1.0],
                "hard_values": [0.0, 1.0],
                "source_row_ids": [triangle_diagnostic["row_id"]],
            },
            {
                "plot_id": f"path-tracer-gradient-quality-{path_summary_index}",
                "kind": "path_tracer_gradient_quality",
                "scenario": "path_tracer:five_spheres:initial",
                "method": "score",
                "values": [0.0],
                "source_row_ids": [row["row_id"] for row in score_rows],
            },
            {
                "plot_id": f"contact-3d-gradient-quality-{contact_summary_index}",
                "kind": "contact_3d_gradient_quality",
                "scenario": "contact_3d:three_sphere_floor_ramp:initial_launch_velocity",
                "method": "soft_ad",
                "values": [0.0],
                "source_row_ids": [contact_gradient["row_id"]],
            },
        ],
        "bias_variance.json": [
            {
                "plot_id": f"bias-variance-{index}",
                "scenario": row["scenario"],
                "method": row["method"],
                "values": row["mean_squared_error"],
                "source_row_ids": row["source_row_ids"],
            }
            for index, row in enumerate(method_summaries)
        ],
        "optimization.json": [
            {
                "plot_id": "optimization-0",
                "scenario": "path_tracer",
                "method": "score",
                "values": [0.0, 0.0],
                "source_row_ids": [row["row_id"] for row in optimization_rows],
            },
            {
                "plot_id": "path-render-comparison",
                "kind": "path_render_comparison",
                "scenario": "path_tracer",
                "method": "best_recovered_render",
                "values": [0.04000000000000001, 0.010000000000000002],
                "source_row_ids": [row["row_id"] for row in comparison_rows],
                "path_target_image": comparison_images["comparison_target"],
                "path_initial_image": comparison_images["comparison_initial"],
                "path_recovered_image": comparison_images["comparison_recovered"],
            },
        ],
        "validity.json": [
            {
                "plot_id": "collision-pinball_bank-start_0-300",
                "kind": "collision_trajectory",
                "scenario": "collision_2d:pinball_bank",
                "final_positions": collision_diagnostic["final_positions"],
                "values": collision_diagnostic["losses"],
                "source_row_ids": [collision_diagnostic["row_id"]],
            },
            {
                "plot_id": "contact-3d-trajectories",
                "kind": "contact_trajectory",
                "scenario": "contact_3d",
                "positions": contact_diagnostic["positions"],
                "values": [
                    contact_diagnostic["max_penetration"],
                    contact_diagnostic["max_contact_energy_gain"],
                ],
                "source_row_ids": [contact_diagnostic["row_id"]],
            },
            {
                "plot_id": "opaque-mesh-boundary",
                "kind": "opaque_boundary",
                "scenario": "opaque_mesh",
                "values": [0, 1],
                "labels": ["transformed", "preserved"],
                "source_row_ids": [opaque_diagnostic["row_id"]],
            },
        ],
        "performance.json": [
            {
                "plot_id": f"performance-{index}",
                "scenario": row["scenario"],
                "method": row["method"],
                "values": [
                    row["cold_compile_time"],
                    row["warm_median"],
                    row["warm_iqr"],
                ],
                "source_row_ids": [row["row_id"]],
            }
            for index, row in enumerate(raw["performance.json"])
        ],
    }
    for name, rows in plot_rows.items():
        write_fixture_json(
            root / "data/plot_data" / name,
            {
                "schema_version": 1,
                "dataset": name.removesuffix(".json"),
                "rows": rows,
            },
        )

    performance_summaries = [
        {
            "scenario": row["scenario"],
            "method": row["method"],
            "cold_compile_time": row["cold_compile_time"],
            "warm_median": row["warm_median"],
            "warm_iqr": row["warm_iqr"],
            "forward_executions": row["forward_executions"],
            "backward_executions": row["backward_executions"],
            "tracemalloc_peak": row["tracemalloc_peak"],
            "rss_delta": row["rss_delta"],
            "warp_allocation_peak": row["warp_allocation_peak"],
            "device_free_memory_delta": row["device_free_memory_delta"],
            "source_row_ids": [row["row_id"]],
        }
        for row in raw["performance.json"]
    ]
    validity_groups = {
        "analytic": [analytic_anchor],
        "triangle_2d": raw["triangle_2d.json"],
        "collision_2d": raw["collision_2d.json"],
        "path_tracer": comparison_rows,
        "contact_3d": [contact_diagnostic],
        "opaque_mesh": raw["opaque_mesh.json"],
        "references": reference_rows,
    }
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
            "analytic_anchor_accepted": {
                "value": 1,
                "unit": "boolean",
                "source_row_ids": [analytic_anchor["row_id"]],
            },
            "path_best_held_out_loss": {
                "value": 0.0,
                "unit": "mean_squared_error",
                "source_row_ids": [optimization_rows[0]["row_id"]],
            },
        },
        "method_summaries": method_summaries,
        "optimization_summaries": [
            {
                "scenario": "path_tracer",
                "method": "score",
                "final_hard_loss_mean": 0.0,
                "final_hard_loss_ci_low": 0.0,
                "final_hard_loss_ci_high": 0.0,
                "success_rate": 1.0,
                "held_out_loss_mean": 0.0,
                "source_row_ids": [row["row_id"] for row in optimization_rows],
            }
        ],
        "performance_summaries": performance_summaries,
        "scenario_validity": [
            {
                "scenario": scenario,
                "accepted": True,
                "metrics": {
                    "row_count": len(rows),
                    "accepted_count": len(rows),
                },
                "source_row_ids": [row["row_id"] for row in rows],
            }
            for scenario, rows in validity_groups.items()
        ],
    }
    write_fixture_json(root / "data/summary.json", summary)
    for name in FIGURE_NAMES:
        (root / "assets/figures" / name).write_bytes(PNG_BYTES)
    for name in IMAGE_NAMES:
        (root / "assets/images" / name).write_bytes(PNG_BYTES)

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
            "optimization_steps": 10,
            "optimization_schedules": 2,
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

    def test_duplicate_html_attributes_are_rejected_before_mapping(self):
        snippets = (
            '<a href="javascript:alert(1)" href="https://example.com">unsafe</a>',
            '<a HREF="javascript:alert(1)" href="https://example.com">unsafe</a>',
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                root = make_valid_fixture(self)
                page = root / "index.html"
                page.write_text(
                    page.read_text(encoding="utf-8") + snippet,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError, "duplicate.*attribute|attribute.*duplicate"
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
        removed = next(
            index
            for index, row in enumerate(payload["rows"])
            if row.get("method") == "score" and row.get("outer_seed") == 31
        )
        payload["rows"].pop(removed)
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


class ReportBuilderTests(unittest.TestCase):
    @staticmethod
    def _builder():
        from discograd import build_report

        return build_report

    def test_current_producer_gradient_plot_kind_and_ids_are_supported(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        payload = json.loads(
            (root / "data/plot_data/gradient_quality.json").read_text(encoding="utf-8")
        )
        path_row = next(
            row
            for row in payload["rows"]
            if row.get("kind") == "path_tracer_gradient_quality"
        )
        contact_row = next(
            row
            for row in payload["rows"]
            if row.get("kind") == "contact_3d_gradient_quality"
        )

        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE).decode(
            "utf-8"
        )

        self.assertTrue(path_row["plot_id"].startswith("path-tracer-gradient-quality-"))
        self.assertTrue(
            contact_row["plot_id"].startswith("contact-3d-gradient-quality-")
        )
        self.assertIn(path_row["plot_id"], page)
        self.assertIn(contact_row["plot_id"], page)

    def test_every_fixed_asset_rejects_semantically_wrong_same_family_evidence(self):
        builder = self._builder()
        semantic_validator = getattr(builder, "_validate_figure_semantics", None)
        self.assertIsNotNone(
            semantic_validator,
            "the builder must expose and invoke exact per-asset semantic validation",
        )
        assert semantic_validator is not None

        def mutate(case: str, root: Path) -> None:
            if case in {"analytic", "triangle", "path_gradient", "contact_gradient"}:
                relative = "data/plot_data/gradient_quality.json"
                if case == "analytic":
                    relative = "data/plot_data/analytic_gates.json"
                payload = json.loads((root / relative).read_text(encoding="utf-8"))
                if case == "analytic":
                    payload["rows"][0]["source_row_ids"] = ["analytic:anchor"]
                elif case == "triangle":
                    payload["rows"][0]["x_values"] = [-99.0, 99.0]
                elif case == "path_gradient":
                    row = next(
                        item
                        for item in payload["rows"]
                        if item.get("kind") == "path_tracer_gradient_quality"
                    )
                    row["source_row_ids"] = ["path:optimization:score:0"]
                else:
                    row = next(
                        item
                        for item in payload["rows"]
                        if item.get("kind") == "contact_3d_gradient_quality"
                    )
                    row["source_row_ids"] = ["contact:0"]
            elif case in {"collision", "contact_trajectory", "opaque"}:
                relative = "data/plot_data/validity.json"
                payload = json.loads((root / relative).read_text(encoding="utf-8"))
                if case == "collision":
                    row = next(
                        item
                        for item in payload["rows"]
                        if item.get("kind") == "collision_trajectory"
                    )
                    row["final_positions"] = [[99.0, 99.0]]
                elif case == "contact_trajectory":
                    row = next(
                        item
                        for item in payload["rows"]
                        if item.get("kind") == "contact_trajectory"
                    )
                    row["positions"] = [[[99.0, 99.0, 99.0]]]
                else:
                    row = next(
                        item
                        for item in payload["rows"]
                        if item.get("kind") == "opaque_boundary"
                    )
                    row["values"] = [1, 0]
            elif case in {"path_roles", "path_image", "optimization"}:
                relative = "data/plot_data/optimization.json"
                payload = json.loads((root / relative).read_text(encoding="utf-8"))
                if case == "optimization":
                    row = next(
                        item
                        for item in payload["rows"]
                        if str(item.get("plot_id", "")).startswith("optimization-")
                    )
                    row["values"] = [99.0, 99.0]
                else:
                    row = next(
                        item
                        for item in payload["rows"]
                        if item.get("kind") == "path_render_comparison"
                    )
                    if case == "path_roles":
                        row["source_row_ids"] = [
                            "path:score:8:0",
                            "path:score:8:1",
                            "path:score:8:2",
                        ]
                    else:
                        row["path_target_image"] = [[[0.9, 0.9, 0.9]]]
            elif case == "bias_variance":
                relative = "data/plot_data/bias_variance.json"
                payload = json.loads((root / relative).read_text(encoding="utf-8"))
                payload["rows"][0]["values"] = [99.0]
            elif case == "performance":
                relative = "data/plot_data/performance.json"
                payload = json.loads((root / relative).read_text(encoding="utf-8"))
                payload["rows"][0]["values"] = [99.0, 99.0, 99.0]
            else:  # pragma: no cover - guards the mutation table itself
                raise AssertionError(case)
            write_fixture_json(root / relative, payload)
            _refresh_manifest(root)

        cases = (
            "analytic",
            "triangle",
            "collision",
            "path_roles",
            "path_image",
            "path_gradient",
            "contact_trajectory",
            "contact_gradient",
            "opaque",
            "bias_variance",
            "optimization",
            "performance",
        )
        for case in cases:
            with self.subTest(asset=case):
                root = make_valid_fixture(self)
                mutate(case, root)
                with self.assertRaisesRegex(
                    builder.BuildError, "figure|plot|lineage|source|semantic|mismatch"
                ):
                    semantic_validator(builder.load_report(root))

    def test_duplicate_semantic_asset_identities_are_rejected_before_mapping(self):
        builder = self._builder()
        raw_cases = (
            (
                "path-role",
                "data/raw/path_tracer_gradients.json",
                "path_tracer",
                lambda row: row.get("role") == "comparison_target",
            ),
            (
                "triangle-edge",
                "data/raw/triangle_2d.json",
                "triangle_2d",
                lambda row: row.get("scenario") == "edge_slice",
            ),
            (
                "collision-trajectory",
                "data/raw/collision_2d.json",
                "collision_2d",
                lambda row: "final_positions" in row,
            ),
            (
                "contact-trajectory",
                "data/raw/contact_3d_gradients.json",
                "contact_3d",
                lambda row: "positions" in row,
            ),
            (
                "opaque-boundary",
                "data/raw/opaque_mesh.json",
                "opaque_mesh",
                lambda row: row.get("transform_status") == "estimator_only",
            ),
        )
        for case, relative, scenario, predicate in raw_cases:
            with self.subTest(identity=case):
                root = make_valid_fixture(self)
                raw_path = root / relative
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                source = next(row for row in payload["rows"] if predicate(row))
                duplicate = json.loads(json.dumps(source))
                duplicate["row_id"] = f"{source['row_id']}:duplicate"
                payload["rows"].append(duplicate)
                write_fixture_json(raw_path, payload)

                summary_path = root / "data/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                validity = next(
                    row
                    for row in summary["scenario_validity"]
                    if row["scenario"] == scenario
                )
                validity["source_row_ids"].append(duplicate["row_id"])
                validity["metrics"]["row_count"] += 1
                validity["metrics"]["accepted_count"] += 1
                write_fixture_json(summary_path, summary)
                _refresh_manifest(root)

                with self.assertRaisesRegex(builder.BuildError, "duplicate|one-to-one"):
                    builder.render_report(root=root, template_path=REPORT_TEMPLATE)

        root = make_valid_fixture(self)
        plot_path = root / "data/plot_data/bias_variance.json"
        payload = json.loads(plot_path.read_text(encoding="utf-8"))
        payload["rows"].append(json.loads(json.dumps(payload["rows"][0])))
        write_fixture_json(plot_path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(builder.BuildError, "duplicate|one-to-one"):
            builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_path_headline_is_path_only_with_exact_units_and_producer_order_ties(self):
        builder = self._builder()

        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_optimization.json"
        path_rows = json.loads(path.read_text(encoding="utf-8"))
        for row in path_rows["rows"]:
            row["held_out_loss"] = 2.0
        write_fixture_json(path, path_rows)
        contact_row = {
            "row_id": "contact:optimization:soft_ad:0",
            "scenario_family": "contact_3d",
            "scenario": "three_sphere_floor_ramp",
            "method": "soft_ad",
            "final_hard_loss": 0.0,
            "held_out_loss": 1.0,
            "success": True,
        }
        write_fixture_json(
            root / "data/raw/contact_3d_optimization.json",
            {
                "schema_version": 1,
                "dataset": "contact_3d_optimization",
                "rows": [contact_row],
            },
        )
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        path_summary = summary["optimization_summaries"][0]
        path_summary["held_out_loss_mean"] = 2.0
        contact_summary = {
            "scenario": "contact_3d",
            "method": "soft_ad",
            "final_hard_loss_mean": 0.0,
            "final_hard_loss_ci_low": 0.0,
            "final_hard_loss_ci_high": 0.0,
            "success_rate": 1.0,
            "held_out_loss_mean": 1.0,
            "source_row_ids": [contact_row["row_id"]],
        }
        summary["optimization_summaries"] = [contact_summary, path_summary]
        summary["headline_metrics"]["path_best_held_out_loss"].update(
            value=2.0,
            source_row_ids=[path_rows["rows"][0]["row_id"]],
        )
        write_fixture_json(summary_path, summary)
        optimization_plot = root / "data/plot_data/optimization.json"
        plot_payload = json.loads(optimization_plot.read_text(encoding="utf-8"))
        plot_payload["rows"][0].update(
            plot_id="optimization-1",
            values=[0.0, 2.0],
        )
        plot_payload["rows"].insert(
            0,
            {
                "plot_id": "optimization-0",
                "scenario": "contact_3d",
                "method": "soft_ad",
                "values": [0.0, 1.0],
                "source_row_ids": [contact_row["row_id"]],
            },
        )
        write_fixture_json(optimization_plot, plot_payload)
        _refresh_manifest(root)
        builder.render_report(root=root, template_path=REPORT_TEMPLATE)

        root = make_valid_fixture(self)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["headline_metrics"]["path_best_held_out_loss"]["unit"] = "seconds"
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)
        with self.assertRaisesRegex(builder.BuildError, "unit|headline|summary"):
            builder.render_report(root=root, template_path=REPORT_TEMPLATE)

        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_optimization.json"
        path_payload = json.loads(path.read_text(encoding="utf-8"))
        path_payload["rows"].reverse()
        write_fixture_json(path, path_payload)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["headline_metrics"]["path_best_held_out_loss"]["source_row_ids"] = [
            path_payload["rows"][0]["row_id"]
        ]
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)
        builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_render_is_byte_deterministic_and_contains_complete_semantics(self):
        builder = self._builder()
        root = make_valid_fixture(self)

        first = builder.render_report(root=root, template_path=REPORT_TEMPLATE)
        second = builder.render_report(root=root, template_path=REPORT_TEMPLATE)

        self.assertEqual(first, second)
        page = first.decode("utf-8")
        for section_id in REQUIRED_SECTION_IDS:
            with self.subTest(section_id=section_id):
                self.assertIn(f'id="{section_id}"', page)
        for method in REQUIRED_METHOD_IDS:
            with self.subTest(method=method):
                self.assertIn(f'data-method="{method}"', page)
        self.assertIn(
            "surrogate/pseudo-gradient, not the derivative of the hard execution",
            page,
        )
        self.assertIn("hard forward executions", page)
        self.assertIn("soft forward executions", page)

    def test_check_mode_detects_one_byte_edit_and_atomic_write_repairs_it(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        output = root / "generated-report.html"
        arguments = [
            "--root",
            str(root),
            "--template",
            str(REPORT_TEMPLATE),
            "--output",
            str(output),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(builder.main(arguments), 0)
        expected = output.read_bytes()
        self.assertTrue(expected)

        edited = bytearray(expected)
        edited[-1] = (edited[-1] + 1) % 256
        output.write_bytes(edited)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(builder.main([*arguments, "--check"]), 1)
        self.assertEqual(output.read_bytes(), bytes(edited))

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(builder.main(arguments), 0)
        self.assertEqual(output.read_bytes(), expected)
        self.assertEqual(list(root.glob(".generated-report.html.*.tmp")), [])

    def test_every_numeric_cell_and_figure_has_resolvable_provenance(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE)
        parser = _RenderedPageParser()
        parser.feed(page.decode("utf-8"))

        numeric_cells = 0
        payloads: dict[str, object] = {}
        for attributes, text in parser.cells:
            displayed = _displayed_numeric_value(text)
            if displayed is None:
                continue
            numeric_cells += 1
            source_file = attributes.get("data-source-file")
            source_key = attributes.get("data-source-key")
            self.assertIsNotNone(source_file, text)
            self.assertIsNotNone(source_key, text)
            assert source_file is not None
            assert source_key is not None
            if source_file not in payloads:
                payloads[source_file] = json.loads(
                    (root / source_file).read_text(encoding="utf-8")
                )
            source_value = _resolve_json_pointer(payloads[source_file], source_key)
            if isinstance(displayed, list):
                self.assertIsInstance(source_value, list, (source_file, source_key))
                self.assertEqual(
                    text,
                    "["
                    + ", ".join(builder.format_number(value) for value in source_value)
                    + "]",
                )
            elif isinstance(source_value, str):
                self.assertEqual(text, source_value)
            else:
                self.assertIsInstance(
                    source_value, (int, float), (source_file, source_key)
                )
                self.assertNotIsInstance(source_value, bool)
                self.assertEqual(text, builder.format_number(source_value))
        self.assertGreater(numeric_cells, 10)

        self.assertGreaterEqual(len(parser.figures), len(FIGURE_NAMES))
        for attributes in parser.figures:
            source_file = attributes["data-source-file"]
            source_key = attributes.get("data-source-key")
            self.assertIsNotNone(source_key)
            payload = json.loads((root / source_file).read_text(encoding="utf-8"))
            record = _resolve_json_pointer(payload, source_key or "")
            self.assertIsInstance(record, dict)
            self.assertEqual(record["plot_id"], attributes.get("data-plot-id"))
            source_keys = json.loads(attributes["data-source-keys"])
            plot_ids = json.loads(attributes["data-plot-ids"])
            self.assertEqual(len(source_keys), len(plot_ids))
            self.assertGreater(len(source_keys), 0)
            for pointer, plot_id in zip(source_keys, plot_ids, strict=True):
                selected = _resolve_json_pointer(payload, pointer)
                self.assertIsInstance(selected, dict)
                self.assertEqual(selected["plot_id"], plot_id)

    def test_every_evidence_cell_figure_and_data_narrative_has_global_provenance(self):
        builder = self._builder()
        canonical_display = getattr(builder, "_canonical_evidence_display", None)
        self.assertIsNotNone(
            canonical_display,
            "sourced evidence cells need one canonical display-value contract",
        )
        assert canonical_display is not None
        root = make_valid_fixture(self)
        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE)
        parser = _RenderedPageParser()
        parser.feed(page.decode("utf-8"))

        payloads: dict[str, object] = {}
        for attributes, text in parser.cells:
            source_file = attributes.get("data-source-file")
            source_key = attributes.get("data-source-key")
            static_key = attributes.get("data-static-key")
            self.assertTrue(
                (
                    source_file is not None
                    and source_key is not None
                    and static_key is None
                )
                or (
                    source_file is None
                    and source_key is None
                    and static_key is not None
                ),
                (attributes, text),
            )
            if source_file is not None and source_key is not None:
                if source_file not in payloads:
                    payloads[source_file] = json.loads(
                        (root / source_file).read_text(encoding="utf-8")
                    )
                source_value = _resolve_json_pointer(payloads[source_file], source_key)
                self.assertEqual(
                    text,
                    canonical_display(source_value),
                    (source_file, source_key, source_value),
                )
            if static_key is not None:
                self.assertIsNone(_displayed_numeric_value(text), (static_key, text))

        expected_assets = {
            *(f"assets/figures/{name}" for name in FIGURE_NAMES),
            *(f"assets/images/{name}" for name in IMAGE_NAMES),
        }
        self.assertEqual(
            {attributes.get("data-asset") for attributes in parser.figures},
            expected_assets,
        )
        for attributes in parser.figures:
            self.assertIn("data-source-file", attributes)
            self.assertIn("data-source-key", attributes)
            self.assertIn("data-source-keys", attributes)
            self.assertIn("data-plot-id", attributes)
            self.assertIn("data-plot-ids", attributes)

        self.assertEqual(len(parser.images), len(expected_assets))
        image_sources = [attributes.get("src") for attributes, _ in parser.images]
        self.assertEqual(set(image_sources), expected_assets)
        self.assertEqual(len(image_sources), len(set(image_sources)))
        figure_image_counts = [0] * len(parser.figures)
        for attributes, parent in parser.images:
            self.assertIsNotNone(parent, attributes)
            assert parent is not None
            figure_image_counts[parent] += 1
            self.assertEqual(
                attributes.get("src"), parser.figures[parent].get("data-asset")
            )
        self.assertTrue(all(count == 1 for count in figure_image_counts))

        self.assertEqual(len(parser.data_narratives), 1)
        narrative = parser.data_narratives[0]
        self.assertIn("data-source-files", narrative)
        self.assertIn("data-source-keys", narrative)
        self.assertEqual(
            len(json.loads(narrative["data-source-files"])),
            len(json.loads(narrative["data-source-keys"])),
        )

    def test_rendered_contract_rejects_unprovenanced_evidence_nodes(self):
        builder = self._builder()
        provenance_validator = getattr(builder, "_validate_evidence_provenance", None)
        self.assertIsNotNone(
            provenance_validator,
            "the builder must validate evidence provenance independently of tests",
        )
        snippets = {
            "cell": "<table><tbody><tr><td>123</td></tr></tbody></table>",
            "wrong_sourced_value": (
                '<table><tbody><tr><td data-source-file="data/summary.json" '
                'data-source-key="#%2Fschema_version">999</td></tr></tbody></table>'
            ),
            "figure": (
                '<figure><img src="assets/images/path_target.png" '
                'alt="unprovenanced"></figure>'
            ),
            "standalone_duplicate_image": (
                '<img src="assets/images/path_target.png" alt="duplicate standalone">'
            ),
            "narrative": '<p class="lede">measured device: 123</p>',
        }
        for name, snippet in snippets.items():
            with self.subTest(node=name):
                root = make_valid_fixture(self)
                template = root / f"unprovenanced-{name}.html"
                template.write_text(
                    REPORT_TEMPLATE.read_text(encoding="utf-8").replace(
                        "</main>", snippet + "</main>"
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    builder.BuildError, "provenance|source|evidence|figure|narrative"
                ):
                    builder.render_report(root=root, template_path=template)

    def test_unknown_missing_and_duplicate_template_tokens_are_rejected(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        template = REPORT_TEMPLATE.read_text(encoding="utf-8")
        cases = {
            "unknown": template + "{{UNKNOWN_TOKEN}}",
            "lowercase": template + "{{unknown-token}}",
            "missing": template.replace("{{METHOD_TABLE}}", ""),
            "duplicate": template.replace(
                "{{METHOD_TABLE}}", "{{METHOD_TABLE}}{{METHOD_TABLE}}"
            ),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                path = root / f"{name}.html"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(builder.BuildError, "token|Token|template"):
                    builder.render_report(root=root, template_path=path)

    def test_figure_bindings_reject_wrong_plot_identity_or_lineage(self):
        builder = self._builder()
        for case in ("identity", "lineage"):
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path = root / "data/plot_data/gradient_quality.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                record = next(
                    row
                    for row in payload["rows"]
                    if str(row.get("scenario", "")).startswith("path_tracer:")
                )
                if case == "identity":
                    record["plot_id"] = "unrelated-plot"
                else:
                    record["source_row_ids"] = ["contact:0"]
                write_fixture_json(path, payload)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    builder.BuildError, "figure|plot|lineage|source"
                ):
                    builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_summary_aggregates_are_recomputed_and_exactly_aligned(self):
        builder = self._builder()
        cases = (
            "method_value",
            "method_label",
            "method_sources",
            "optimization_value",
            "performance_value",
            "validity_value",
            "headline_value",
        )
        for case in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path = root / "data/summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                path_summary = next(
                    row
                    for row in summary["method_summaries"]
                    if row["scenario"].startswith("path_tracer:")
                )
                if case == "method_value":
                    path_summary["mean_gradient"] = [1.0]
                elif case == "method_label":
                    path_summary["scenario"] = "contact_3d:wrong:start"
                elif case == "method_sources":
                    path_summary["source_row_ids"] = ["analytic:score"]
                elif case == "optimization_value":
                    summary["optimization_summaries"][0]["final_hard_loss_mean"] = 1.0
                elif case == "performance_value":
                    summary["performance_summaries"][0]["warm_median"] = 1.0
                elif case == "validity_value":
                    summary["scenario_validity"][0]["accepted"] = False
                elif case == "headline_value":
                    summary["headline_metrics"]["path_best_held_out_loss"]["value"] = (
                        1.0
                    )
                write_fixture_json(path, summary)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    builder.BuildError, "aggregate|lineage|mismatch|summary"
                ):
                    builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_direct_script_uses_custom_root_defaults(self):
        root = make_valid_fixture(self)
        (root / "report_template.html").write_bytes(REPORT_TEMPLATE.read_bytes())
        live_index = REPORT_TEMPLATE.with_name("index.html")
        live_before = live_index.read_bytes()

        completed = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--root", str(root)],
            cwd=REPORT_TEMPLATE.parents[1],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("WROTE:", completed.stdout)
        self.assertEqual(live_index.read_bytes(), live_before)
        self.assertIn(
            'id="reproducibility"', (root / "index.html").read_text(encoding="utf-8")
        )
        validate_publication(root)

    def test_output_guards_reject_template_and_bundle_overwrites(self):
        builder = self._builder()
        for destination in ("template", "bundle"):
            with self.subTest(destination=destination):
                root = make_valid_fixture(self)
                template = root / "report_template.html"
                template.write_bytes(REPORT_TEMPLATE.read_bytes())
                output = (
                    template
                    if destination == "template"
                    else root / "data/summary.json"
                )
                before = output.read_bytes()

                with self.assertRaisesRegex(
                    builder.BuildError, "output|template|bundle|artifact"
                ):
                    builder.build_report(
                        root=root, output=output, template_path=template
                    )
                self.assertEqual(output.read_bytes(), before)

    def test_atomic_output_failures_are_normalized_to_build_error(self):
        builder = self._builder()
        destination_opener = getattr(builder, "_open_output_destination", None)
        self.assertIsNotNone(
            destination_opener,
            "atomic writes need a retained no-follow destination directory",
        )
        assert destination_opener is not None
        root = make_valid_fixture(self)
        output = root / "already-a-directory"
        output.mkdir()

        destination = destination_opener(
            root=root, output=output, template_path=REPORT_TEMPLATE
        )
        try:
            with self.assertRaisesRegex(builder.BuildError, "write|output|directory"):
                builder._atomic_write(destination, b"rendered")
        finally:
            destination.close()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = builder.main(
                [
                    "--root",
                    str(root),
                    "--template",
                    str(REPORT_TEMPLATE),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_output_ancestor_swap_during_render_cannot_redirect_atomic_write(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        publish = root / "publish"
        publish.mkdir()
        output = publish / "summary.json"
        displaced = root / "publish-before-swap"
        evidence = root / "data/summary.json"
        evidence_before = evidence.read_bytes()
        real_render = builder.render_report

        def render_then_swap(*, root: Path, template_path: Path) -> bytes:
            rendered = real_render(root=root, template_path=template_path)
            publish.rename(displaced)
            publish.symlink_to(root / "data", target_is_directory=True)
            return rendered

        with mock.patch.object(builder, "render_report", side_effect=render_then_swap):
            with self.assertRaisesRegex(
                builder.BuildError, "changed|race|directory|output"
            ):
                builder.build_report(
                    root=root, output=output, template_path=REPORT_TEMPLATE
                )

        self.assertEqual(evidence.read_bytes(), evidence_before)
        self.assertFalse((displaced / "summary.json").exists())

    def test_atomic_cleanup_failure_does_not_mask_primary_write_error(self):
        builder = self._builder()
        destination_opener = getattr(builder, "_open_output_destination", None)
        self.assertIsNotNone(
            destination_opener,
            "atomic writes need a retained no-follow destination directory",
        )
        assert destination_opener is not None
        root = make_valid_fixture(self)
        output = root / "publish" / "report.html"
        destination = destination_opener(
            root=root, output=output, template_path=REPORT_TEMPLATE
        )
        try:
            with (
                mock.patch.object(
                    builder.os,
                    "replace",
                    side_effect=OSError("primary replace failure"),
                ),
                mock.patch.object(
                    builder.os,
                    "unlink",
                    side_effect=OSError("cleanup failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    builder.BuildError, "primary replace failure"
                ) as raised:
                    builder._atomic_write(destination, b"rendered")
            self.assertNotIn("cleanup failure", str(raised.exception))
        finally:
            destination.close()

    def test_output_path_resolution_failures_are_build_errors(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        first = root / "output-loop-a"
        second = root / "output-loop-b"
        first.symlink_to(second, target_is_directory=True)
        second.symlink_to(first, target_is_directory=True)

        with self.assertRaisesRegex(
            builder.BuildError, "resolve|symlink|filesystem|output"
        ):
            builder.build_report(
                root=root,
                output=first / "report.html",
                template_path=REPORT_TEMPLATE,
            )

    def test_direct_render_template_path_failures_are_build_errors(self):
        builder = self._builder()
        root = make_valid_fixture(self)

        with self.assertRaisesRegex(builder.BuildError, "template|filesystem|read"):
            builder.render_report(
                root=root,
                template_path=Path("invalid\0template.html"),
            )

    def test_active_rendered_html_is_rejected_before_write(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        malicious = root / "malicious-template.html"
        malicious.write_text(
            REPORT_TEMPLATE.read_text(encoding="utf-8").replace(
                "</body>", "<script>alert('unsafe')</script></body>"
            ),
            encoding="utf-8",
        )
        output = root / "generated-report.html"
        output.write_bytes(b"existing-safe-page")

        with self.assertRaisesRegex(builder.BuildError, "active|unsafe|script"):
            builder.build_report(root=root, output=output, template_path=malicious)
        self.assertEqual(output.read_bytes(), b"existing-safe-page")

    def test_duplicate_html_attributes_are_rejected_before_render_or_write(self):
        builder = self._builder()
        snippets = (
            '<a href="javascript:alert(1)" href="https://example.com">unsafe</a>',
            '<a HREF="javascript:alert(1)" href="https://example.com">unsafe</a>',
        )
        for snippet in snippets:
            for entry_point in ("render", "build"):
                with self.subTest(snippet=snippet, entry_point=entry_point):
                    root = make_valid_fixture(self)
                    template = root / "duplicate-attribute-template.html"
                    template.write_text(
                        REPORT_TEMPLATE.read_text(encoding="utf-8").replace(
                            "</main>", snippet + "</main>"
                        ),
                        encoding="utf-8",
                    )
                    if entry_point == "render":
                        with self.assertRaisesRegex(
                            builder.BuildError,
                            "duplicate.*attribute|attribute.*duplicate",
                        ):
                            builder.render_report(root=root, template_path=template)
                        continue
                    output = root / "generated-report.html"
                    output.write_bytes(b"existing-safe-page")
                    with self.assertRaisesRegex(
                        builder.BuildError,
                        "duplicate.*attribute|attribute.*duplicate",
                    ):
                        builder.build_report(
                            root=root,
                            output=output,
                            template_path=template,
                        )
                    self.assertEqual(output.read_bytes(), b"existing-safe-page")

    def test_embedded_nul_output_path_is_normalized_to_build_error(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        output = Path(f"{root}/generated\0-report.html")

        with self.assertRaisesRegex(
            builder.BuildError, "resolve|output|filesystem|null|NUL"
        ):
            builder.build_report(
                root=root, output=output, template_path=REPORT_TEMPLATE
            )

    def test_embedded_nul_output_path_has_concise_cli_failure(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        output = f"{root}/generated\0-report.html"
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = builder.main(
                [
                    "--root",
                    str(root),
                    "--template",
                    str(REPORT_TEMPLATE),
                    "--output",
                    output,
                ]
            )

        self.assertEqual(result, 1)
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_loader_failures_are_normalized_to_build_error(self):
        builder = self._builder()

        root = make_valid_fixture(self)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["headline_metrics"]["analytic_anchor_accepted"]["value"] = 10**4000
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)
        with self.assertRaisesRegex(builder.BuildError, "finite|numeric|overflow"):
            builder.render_report(root=root, template_path=REPORT_TEMPLATE)

        root = make_valid_fixture(self)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["performance_summaries"][0]["method"] = "unknown_method"
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)
        with self.assertRaisesRegex(builder.BuildError, "unknown method"):
            builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_report_renders_explicit_estimator_identities_and_tradeoffs(self):
        builder = self._builder()
        page = builder.render_report(
            root=make_valid_fixture(self), template_path=REPORT_TEMPLATE
        ).decode("utf-8")

        for label in (
            "Gaussian-smoothed hard objective",
            "Score estimator identity",
            "Pathwise estimator target",
            "Finite-difference target",
            "Soft surrogate",
            "Straight-through construction",
            "Residual control-variate identity",
        ):
            with self.subTest(label=label):
                self.assertIn(f'aria-label="{label}"', page)
        self.assertIn("∇E[H] = E[∇S] + E[(H−S)u / σ]", page)
        self.assertIn("Exactness", page)
        self.assertIn("Bias", page)
        self.assertIn("Variance", page)
        self.assertIn("DGO backend estimates jump contributions", page)

    def test_report_renders_complete_evidence_and_reproducibility_fields(self):
        builder = self._builder()
        page = builder.render_report(
            root=make_valid_fixture(self), template_path=REPORT_TEMPLATE
        ).decode("utf-8")

        for label in (
            "Mean gradient",
            "Empirical bias",
            "Empirical variance",
            "Mean squared error",
            "CI low",
            "CI high",
            "Tracemalloc peak",
            "RSS delta",
            "Warp allocation peak",
            "Device free-memory delta",
            "Manifest schema",
            "Summary schema",
            "Report tier",
            "Artifact digests",
            "optimization schedules",
            "estimator seeds",
            "https://github.com/NVIDIA/warp",
        ):
            with self.subTest(label=label):
                self.assertIn(label, page)

    def test_residual_execution_table_does_not_truncate_rows(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        raw_path = root / "data/raw/analytic.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        residual = next(
            row
            for row in raw["rows"]
            if row.get("method") == "residual_control_variate"
        )
        residual_rows = [residual]
        for index in range(1, 10):
            clone = dict(residual)
            clone["row_id"] = f"analytic:residual_control_variate:{index}"
            clone["outer_seed"] = index
            raw["rows"].append(clone)
            residual_rows.append(clone)
        write_fixture_json(raw_path, raw)

        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        residual_summary = next(
            row
            for row in summary["method_summaries"]
            if row["scenario"] == "analytic:step:analytic"
            and row["method"] == "residual_control_variate"
        )
        residual_summary["source_row_ids"] = [row["row_id"] for row in residual_rows]
        write_fixture_json(summary_path, summary)
        analytic_plot = root / "data/plot_data/analytic_gates.json"
        analytic_payload = json.loads(analytic_plot.read_text(encoding="utf-8"))
        analytic_payload["rows"][0]["source_row_ids"] = [
            row["row_id"] for row in raw["rows"]
        ]
        analytic_payload["rows"][0]["values"] = [
            row["gradient"][0]
            for row in raw["rows"]
            if row.get("method") in {"score", "soft_ad", "residual_control_variate"}
        ]
        write_fixture_json(analytic_plot, analytic_payload)
        bias_plot = root / "data/plot_data/bias_variance.json"
        bias_payload = json.loads(bias_plot.read_text(encoding="utf-8"))
        residual_plot = next(
            row
            for row in bias_payload["rows"]
            if row.get("scenario") == "analytic:step:analytic"
            and row.get("method") == "residual_control_variate"
        )
        residual_plot["source_row_ids"] = [row["row_id"] for row in residual_rows]
        write_fixture_json(bias_plot, bias_payload)
        _refresh_manifest(root)

        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE).decode(
            "utf-8"
        )
        self.assertEqual(page.count("hard_forward_executions"), 10)

    def test_corrupt_png_is_rejected_even_when_manifest_digest_matches(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        path = root / "assets/figures/analytic_gates.png"
        corrupted = bytearray(path.read_bytes())
        corrupted[corrupted.index(b"IDAT") + 4] ^= 1
        path.write_bytes(corrupted)
        _refresh_manifest(root)

        with self.assertRaisesRegex(builder.BuildError, "PNG|png|image"):
            builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_all_data_derived_strings_are_html_escaped(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dangerous = '<script data-value="&">unsafe'
        summary["method_labels"]["score"] = dangerous
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)

        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE).decode(
            "utf-8"
        )

        self.assertNotIn(dangerous, page)
        self.assertNotIn("<script", page.casefold())
        self.assertIn(
            "&lt;script data-value=&quot;&amp;&quot;&gt;unsafe",
            page,
        )

    def test_typed_model_and_aggregates_are_immutable_and_checked(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        model = builder.load_report(root)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.manifest.tier = "changed"
        with self.assertRaises(TypeError):
            model.summary.method_labels["score"] = "changed"
        self.assertIsInstance(model.gradient_rows, tuple)
        path_score = next(
            aggregate
            for aggregate in model.gradient_aggregates
            if aggregate.scenario_family == "path_tracer"
            and aggregate.method == "score"
        )
        self.assertEqual(path_score.rows, 32)
        self.assertEqual(path_score.mean_wall_time, 0.01)
        self.assertEqual(path_score.wall_time_q25, 0.01)
        self.assertEqual(path_score.wall_time_q75, 0.01)
        with self.assertRaisesRegex(builder.BuildError, "finite|nonempty"):
            builder.checked_fmean(())
        with self.assertRaisesRegex(builder.BuildError, "quantile"):
            builder.checked_quantile((1.0,), 1.5)

    def test_generated_fixture_passes_publication_validator(self):
        builder = self._builder()
        root = make_valid_fixture(self)

        builder.build_report(
            root=root,
            output=root / "index.html",
            template_path=REPORT_TEMPLATE,
        )

        result = validate_publication(root)
        self.assertEqual(result["files"], 31)


if __name__ == "__main__":
    unittest.main(verbosity=2)
