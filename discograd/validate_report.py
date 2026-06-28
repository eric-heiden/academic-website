from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
REQUIRED_SECTION_IDS = (
    "scope",
    "semantics",
    "literature",
    "grammar",
    "protocol",
    "analytic",
    "path-tracing",
    "contact-3d",
    "opaque-mesh",
    "gradient-quality",
    "optimization",
    "performance",
    "limitations",
    "reproducibility",
)
REQUIRED_METHOD_IDS = (
    "crisp_ad",
    "crisp_fd",
    "smoothed_pathwise",
    "score",
    "smoothed_crn_fd",
    "soft_ad",
    "straight_through_ad",
    "residual_control_variate",
)
REQUIRED_SCENARIO_FAMILIES = (
    "analytic",
    "triangle_2d",
    "collision_2d",
    "path_tracer",
    "contact_3d",
    "opaque_mesh",
)
REQUIRED_LITERATURE_URLS = (
    "https://arxiv.org/abs/2109.05143",
    "https://arxiv.org/abs/2310.03585",
    "https://github.com/DiscoGrad/DiscoGrad",
    "https://github.com/a-paulus/softtorch",
    "https://arxiv.org/abs/2603.08824",
)
EXPECTED_REFERENCE_CELLS = (
    "triangle_2d:edge_midpoints",
    "collision_2d:pinball_bank:start_0",
    "collision_2d:pinball_bank:start_1",
    "collision_2d:pinball_bank:start_2",
    "collision_2d:crowded_table:start_0",
    "collision_2d:crowded_table:start_1",
    "collision_2d:crowded_table:start_2",
    "path_tracer:initial_parameters",
    "contact_3d:initial_launch_velocity",
    "opaque_mesh:camera_parameters",
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
EXPECTED_FILES = frozenset(
    ("data/summary.json",)
    + tuple(f"data/raw/{name}" for name in RAW_NAMES)
    + tuple(f"data/plot_data/{name}" for name in PLOT_NAMES)
    + tuple(f"assets/figures/{name}" for name in FIGURE_NAMES)
    + tuple(f"assets/images/{name}" for name in IMAGE_NAMES)
)
LEGACY_FILES = frozenset(
    {
        "results.json",
        "benchmark_speed.png",
        "collision_convergence.png",
        "collision_trajectories.png",
        "gradient_variance.png",
        "triangle_gradients.png",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_WINDOWS_PATH_RE = re.compile(r"(?:^|\s)[A-Za-z]:[\\/]")
_EMBEDDED_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"'<>]+")
_PROGRAM_MODULE = "warp.examples.optim.example_program_smoothing"
_COMMAND_PREFIXES = (
    ("uv", "run", "--extra", "examples", "-m", _PROGRAM_MODULE),
    ("uv", "run", "-m", _PROGRAM_MODULE),
    ("python", "-m", _PROGRAM_MODULE),
)
_REPORT_CONFIG = {
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
_REPORT_REFERENCE_COUNT_RATIONALE = (
    "Accepted after a clean CPU pilot projected 30,807.341991021996 seconds for the report workload; "
    "the nominal reference counts are retained without reduction."
)
_REPORT_REFERENCE_COUNT_DECISION = {
    "status": "accepted",
    "pilot_tier": "pilot",
    "pilot_manifest_sha256": "d6ab265c77e3f08fe759fde02bf4d07e43b8c358a4bb46024da1422ba73dd6cb",
    "pilot_source_commit": "e25b05937fa149a341177559f2fa7f7e2ff3f651",
    "pilot_projected_report_seconds": 30807.341991021996,
    "decided_at_utc": "2026-06-28T14:43:54Z",
    "protocol_fingerprint": "f44b07543db9c793cc0fd16a2fde768cc7c7589b60f9489e9f600c31d729e0e9",
    "path_reference_samples": 32768,
    "contact_reference_samples": 65536,
    "reference_seed_sets": 4,
    "rationale": _REPORT_REFERENCE_COUNT_RATIONALE,
}
_REPORT_RUNTIME_PHASES = (
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
_PATH_CERTIFICATE_FINGERPRINT = (
    "8f4814d92d301575e1d79caa80ddb6dcdf3a89cce666950000b5d24aa3129676"
)
_PATH_GAUGE_TARGET = "gaussian_smoothed_hard_with_numerical_gauge_assumption"
_PATH_GAUGE_TARGET_LABEL = (
    "certified residual control variate for the Gaussian-smoothed box-clipped hard render; "
    "exact Duff and safety-gauge selected-arm derivatives hold almost everywhere"
)
_PATH_ONB_SEAM_SEMANTICS = (
    "z >= 0 uses the positive Duff chart; z < 0 uses the negative Duff chart; "
    "this numerical gauge branch is not smoothed"
)
_PATH_ROOT_CALLABLE_KEYS = (
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
_PATH_CALLABLE_KEYS = (
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
    *_PATH_ROOT_CALLABLE_KEYS,
)


def _css_has_external_or_active_content(value: str) -> bool:
    if "\\" in value:
        return True
    without_comments = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
    normalized = re.sub(r"\s+", "", without_comments).casefold()
    return any(
        token in normalized
        for token in (
            "url(",
            "@import",
            "image-set(",
            "-webkit-image-set(",
            "expression(",
            "javascript:",
            "behavior:",
            "-moz-binding:",
        )
    )


class ValidationError(ValueError):
    """Raised when a publication artifact violates its evidence contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_finite_tree(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, location=f"{location}[{index}]")


def load_json_finite(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read UTF-8 JSON {path}: {error}") from error
    try:
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {path}: {error}") from error
    _validate_finite_tree(value, location=path.as_posix())
    return value


def _load_json_bytes_finite(data: bytes, *, location: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise ValidationError(
            f"cannot decode UTF-8 JSON {location}: {error}"
        ) from error
    try:
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {location}: {error}") from error
    _validate_finite_tree(value, location=location)
    return value


def _safe_relative_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValidationError("manifest file names must be nonempty strings")
    if "\\" in raw:
        raise ValidationError(f"manifest path must use POSIX separators: {raw!r}")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValidationError(f"unsafe manifest path {raw!r}")
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValidationError(f"symlink is not allowed in publication path {raw!r}")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise ValidationError(f"manifest path escapes report root: {raw!r}") from error
    return candidate


def _relative_parts(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValidationError(
            f"artifact path must be a nonempty relative POSIX path: {raw!r}"
        )
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValidationError(f"unsafe manifest path {raw!r}")
    return pure.parts


def _read_artifact_at(root_descriptor: int, relative: str) -> bytes:
    parts = _relative_parts(relative)
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    directory_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        try:
            for component in parts[:-1]:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=directory_descriptor
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            file_descriptor = os.open(
                parts[-1], file_flags, dir_fd=directory_descriptor
            )
        except OSError as error:
            raise ValidationError(
                f"artifact is missing, non-regular, or traverses a symlink: {relative}"
            ) from error
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise ValidationError(f"artifact is not a regular file: {relative}")
        chunks: list[bytes] = []
        while block := os.read(file_descriptor, 1024 * 1024):
            chunks.append(block)
        return b"".join(chunks)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be finite")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValidationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


def _finite_vector(value: Any, *, name: str, nonnegative: bool = False) -> list[float]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a JSON array")
    return [
        _finite_number(item, name=f"{name}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(value)
    ]


def _json_type_strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_type_strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_type_strict_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _validate_report_config_and_decision(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config")
    if not _json_type_strict_equal(config, _REPORT_CONFIG):
        raise ValidationError(
            "report config and sample counts must exactly match the pilot-approved "
            "protocol fingerprint"
        )
    decision = manifest.get("report_reference_count_decision")
    if not _json_type_strict_equal(decision, _REPORT_REFERENCE_COUNT_DECISION):
        raise ValidationError(
            "report reference-count decision must exactly match the accepted pilot evidence and protocol fingerprint"
        )
    return config


def _validate_installed_runtime(runtime: Any) -> None:
    required_keys = {
        "tier",
        "elapsed_seconds",
        "elapsed_measurement",
        "measurement_excludes",
        "measured_finalization_pass_seconds",
        "measured_installation_pass_seconds",
        "phase_seconds",
        "projection_factors",
        "projection_model",
        "projected_report_seconds",
    }
    if not isinstance(runtime, dict) or set(runtime) != required_keys:
        raise ValidationError(
            "summary runtime must publish the exact installed-measurement schema"
        )
    expected_exclusions = [
        "final_install_timing_metadata_rewrite",
        "final_metadata_bearing_reinstall_and_binding_verification",
    ]
    if (
        runtime.get("tier") != "report"
        or runtime.get("elapsed_measurement")
        != "through_one_complete_descriptor_relative_install_pass"
        or runtime.get("measurement_excludes") != expected_exclusions
        or runtime.get("projection_model")
        != "measured_phase_times_scaled_by_exact_report_workload_ratios"
    ):
        raise ValidationError(
            "summary runtime does not describe the canonical installed measurement"
        )
    phases = runtime.get("phase_seconds")
    factors = runtime.get("projection_factors")
    if (
        not isinstance(phases, dict)
        or set(phases) != set(_REPORT_RUNTIME_PHASES)
        or not isinstance(factors, dict)
        or set(factors) != set(_REPORT_RUNTIME_PHASES)
    ):
        raise ValidationError(
            "summary runtime phase or projection coverage is incomplete"
        )
    for name in _REPORT_RUNTIME_PHASES:
        if (
            type(phases[name]) is not float
            or not math.isfinite(phases[name])
            or phases[name] < 0.0
        ):
            raise ValidationError(
                f"summary runtime phase {name!r} must be a finite nonnegative float"
            )
        if type(factors[name]) is not float or factors[name] != 1.0:
            raise ValidationError(
                f"summary runtime report projection factor {name!r} must be 1.0"
            )
    numeric_fields = (
        "elapsed_seconds",
        "measured_finalization_pass_seconds",
        "measured_installation_pass_seconds",
        "projected_report_seconds",
    )
    if any(
        type(runtime.get(name)) is not float or not math.isfinite(runtime[name])
        for name in numeric_fields
    ):
        raise ValidationError("summary runtime durations must be finite floats")
    finalization = runtime["measured_finalization_pass_seconds"]
    installation = runtime["measured_installation_pass_seconds"]
    if (
        runtime["elapsed_seconds"] <= 0.0
        or finalization <= 0.0
        or installation <= 0.0
        or finalization != phases["finalization"]
        or installation != phases["installation"]
    ):
        raise ValidationError(
            "summary runtime finalization or installation measurement is inconsistent"
        )
    projected = sum(phases[name] * factors[name] for name in _REPORT_RUNTIME_PHASES)
    tolerance = 1.0e-12 * max(1.0, abs(projected), abs(runtime["elapsed_seconds"]))
    if (
        abs(runtime["projected_report_seconds"] - projected) > tolerance
        or abs(runtime["elapsed_seconds"] - projected) > tolerance
    ):
        raise ValidationError(
            "summary runtime projected and elapsed durations do not recompute from phases"
        )


def _expected_path_certificate() -> dict[str, Any]:
    return {
        "complete": True,
        "transformed_sites": 7,
        "smoothed_sites": 2,
        "numerical_gauge_sites": 5,
        "fully_smoothed": False,
        "fingerprint": _PATH_CERTIFICATE_FINGERPRINT,
        "root_callable_keys": list(_PATH_ROOT_CALLABLE_KEYS),
        "callable_keys": list(_PATH_CALLABLE_KEYS),
    }


def _validate_path_numerical_gauge_contract(
    method_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    *,
    source_commit: str,
    device: str,
) -> None:
    protocols = [
        row for row in path_rows if row.get("scenario") == "path_randomness_protocol"
    ]
    if len(protocols) != 1:
        raise ValidationError(
            "path numerical-gauge protocol row is missing or duplicated"
        )
    protocol = protocols[0]
    expected_seed_tables = {
        "training": list(range(1000, 1032)),
        "target": list(range(2000, 2016)),
        "held_out": list(range(3000, 3016)),
        "reference_base": 4000,
        "reference_inner_base": 4001,
    }
    if (
        protocol.get("accepted") is not True
        or protocol.get("source_commit") != source_commit
        or protocol.get("device") != device
        or not _json_type_strict_equal(
            protocol.get("seed_tables"), expected_seed_tables
        )
        or protocol.get("estimator_outer_seeds") != list(range(10000, 10032))
        or protocol.get("reference_protocol")
        != {
            "inputs": {"reference_base": 4000, "reference_inner_base": 4001},
            "realized_streams_location": "data/raw/references.json:seeds",
        }
        or protocol.get("render_work")
        != {"paths_per_forward": 768, "sphere_tests_per_forward": 11520}
    ):
        raise ValidationError(
            "path randomness protocol is not canonical or source-bound"
        )
    if not _json_type_strict_equal(
        protocol.get("certificate"), _expected_path_certificate()
    ):
        raise ValidationError(
            "path numerical-gauge certificate projection or fingerprint is not exact"
        )
    control = protocol.get("control_variate")
    expected_control = {
        "unbiased_target": True,
        "target": _PATH_GAUGE_TARGET,
        "certificate_fingerprint": _PATH_CERTIFICATE_FINGERPRINT,
        "hard_forward_executions": 8,
        "soft_forward_executions": 8,
        "numerical_gauge_assumption": True,
        "numerical_gauge_sites": 5,
    }
    if not _json_type_strict_equal(control, expected_control):
        raise ValidationError(
            "path numerical-gauge control protocol is not exact or certificate-bound"
        )

    residual_count = 0
    expected_lower = [-0.8, -0.5, -0.9]
    expected_upper = [0.8, 0.5, 0.3]
    for row in method_rows:
        if row.get("scenario_family") != "path_tracer":
            if row.get("target") == _PATH_GAUGE_TARGET:
                raise ValidationError(
                    "only path residual rows may claim the numerical-gauge target"
                )
            if (
                row.get("numerical_gauge_assumption", False) is not False
                or row.get("numerical_gauge_sites", 0) != 0
            ):
                raise ValidationError(
                    "nonpath rows must not claim a numerical-gauge assumption"
                )
            continue
        config = row.get("config")
        if (
            not isinstance(config, dict)
            or config.get("numerical_gauge_policy")
            != "exact_selected_arm_derivative_almost_everywhere"
            or config.get("numerical_gauge_sites") != 5
            or config.get("parameter_extension")
            != "componentwise_box_clip_before_geometry"
            or not _json_type_strict_equal(
                config.get("parameter_lower"), expected_lower
            )
            or not _json_type_strict_equal(
                config.get("parameter_upper"), expected_upper
            )
            or config.get("onb_seam_semantics") != _PATH_ONB_SEAM_SEMANTICS
        ):
            raise ValidationError(
                "path method row has an incomplete numerical-gauge configuration"
            )
        if row.get("method") == "residual_control_variate":
            residual_count += 1
            if (
                row.get("target") != _PATH_GAUGE_TARGET
                or row.get("target_label") != _PATH_GAUGE_TARGET_LABEL
                or row.get("unbiased_target") is not True
                or row.get("certificate_fingerprint") != _PATH_CERTIFICATE_FINGERPRINT
                or row.get("numerical_gauge_assumption") is not True
                or row.get("numerical_gauge_sites") != 5
            ):
                raise ValidationError(
                    "path residual target is not gauge-qualified and certificate-bound"
                )
        elif (
            row.get("numerical_gauge_assumption") is not False
            or row.get("numerical_gauge_sites") != 0
        ):
            raise ValidationError(
                "nonresidual path rows must not claim a numerical-gauge estimator assumption"
            )
    if residual_count == 0:
        raise ValidationError(
            "path numerical-gauge protocol requires residual-control-variate rows"
        )


def _validate_path_gradient_contract(path_rows: list[dict[str, Any]]) -> None:
    method_specs = {
        "crisp_ad": (
            (1,),
            False,
            None,
            "hard_program",
            "local derivative of the box-clipped hard-render execution path",
            True,
        ),
        "crisp_fd": (
            (1,),
            False,
            0.01,
            "hard_program_central_difference",
            "central finite difference of the box-clipped hard render",
            True,
        ),
        "smoothed_pathwise": (
            (8, 16, 32, 64, 128),
            True,
            None,
            "gaussian_smoothed_hard",
            "pathwise samples of the box-clipped hard render; visibility-boundary terms are omitted",
            False,
        ),
        "score": (
            (8, 16, 32, 64, 128),
            True,
            None,
            "gaussian_smoothed_hard",
            "unbiased Gaussian score estimator of the box-clipped hard render",
            True,
        ),
        "smoothed_crn_fd": (
            (8, 16, 32, 64, 128),
            True,
            0.01,
            "gaussian_smoothed_hard_finite_epsilon",
            "CRN central difference of the Gaussian-smoothed box-clipped hard render",
            True,
        ),
        "soft_ad": (
            (1,),
            False,
            None,
            "local_soft_surrogate",
            "AD of the source-smoothed box-clipped path-tracing surrogate",
            True,
        ),
        "straight_through_ad": (
            (1,),
            False,
            None,
            "hard_primal_local_soft_pseudogradient",
            "box-clipped hard rendered primal with a source-smoothed pseudo-gradient",
            None,
        ),
        "residual_control_variate": (
            (8, 16, 32, 64, 128),
            True,
            None,
            _PATH_GAUGE_TARGET,
            _PATH_GAUGE_TARGET_LABEL,
            True,
        ),
    }
    method_rows = [
        row
        for row in path_rows
        if row.get("scenario_family") == "path_tracer" and "gradient" in row
    ]
    expected_identities = {
        (method, samples, 10000 + seed, 1000 + seed)
        for method, (sample_counts, *_rest) in method_specs.items()
        for samples in sample_counts
        for seed in range(32)
    }
    identities = [
        (
            row.get("method"),
            row.get("samples"),
            row.get("outer_seed"),
            row.get("inner_seed"),
        )
        for row in method_rows
    ]
    if (
        len(method_rows) != 768
        or len(set(identities)) != 768
        or set(identities) != expected_identities
    ):
        raise ValidationError(
            "path gradient rows do not provide exact deterministic and stochastic "
            "coverage; each stratum requires exactly 32 rows with distinct outer "
            "seeds and canonical outer_seed values"
        )

    digest_by_inner_seed: dict[int, str] = {}
    for row in method_rows:
        row_id = row.get("row_id", "path gradient row")
        method = row.get("method")
        sample_counts, antithetic, epsilon, target, target_label, unbiased = (
            method_specs[method]
        )
        inner_seed = row.get("inner_seed")
        config = row.get("config")
        if (
            isinstance(inner_seed, bool)
            or not isinstance(inner_seed, int)
            or inner_seed < 0
            or not isinstance(config, dict)
        ):
            raise ValidationError(
                f"{row_id} cannot establish the path inner_seed digest mapping"
            )
        digest = config.get("inner_random_digest")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValidationError(f"{row_id} has an invalid inner-random digest")
        expected_config = {
            "bounces": 3,
            "epsilon": epsilon,
            "gate_family": "gaussian",
            "gate_width": 0.05,
            "height": 16,
            "inner_random_digest": digest,
            "outer_parameter_sigma": 0.03,
            "numerical_gauge_policy": (
                "exact_selected_arm_derivative_almost_everywhere"
            ),
            "numerical_gauge_sites": 5,
            "onb_seam_semantics": _PATH_ONB_SEAM_SEMANTICS,
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
        if (
            row.get("samples") not in sample_counts
            or row.get("antithetic") is not antithetic
            or row.get("target") != target
            or row.get("target_label") != target_label
            or row.get("unbiased_target") is not unbiased
            or not _json_type_strict_equal(config, expected_config)
        ):
            raise ValidationError(
                f"path gradient method/config contract is not exact for {row_id}"
            )
        previous = digest_by_inner_seed.setdefault(inner_seed, digest)
        if digest != previous:
            raise ValidationError(
                "path inner_seed maps to inconsistent inner-random digests across "
                "methods or sample counts"
            )
    if set(digest_by_inner_seed) != set(range(1000, 1032)):
        raise ValidationError(
            "path inner_seed digest mapping does not cover the canonical 32 seeds"
        )


def _path_optimization_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValidationError(f"path optimization {name} must be a finite float")
    return value


def _path_optimization_history(
    value: Any,
    *,
    name: str,
    length: int,
    dimension: int | None = None,
) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise ValidationError(
            f"path optimization {name} must contain exactly {length} records"
        )
    if dimension is None:
        for index, item in enumerate(value):
            _path_optimization_float(item, name=f"{name}[{index}]")
        return value
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != dimension:
            raise ValidationError(
                f"path optimization {name}[{index}] must be a {dimension}-vector"
            )
        for component, scalar in enumerate(item):
            _path_optimization_float(scalar, name=f"{name}[{index}][{component}]")
    return value


def _validate_path_optimization_contract(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    source_commit: str,
    device: str,
) -> None:
    methods = (
        "crisp_ad",
        "soft_ad",
        "straight_through_ad",
        "residual_control_variate",
    )
    expected_identities = {
        (method, schedule) for schedule in range(16) for method in methods
    }
    identities = [(row.get("method"), row.get("schedule_id")) for row in rows]
    if (
        len(rows) != 64
        or len(set(identities)) != 64
        or set(identities) != expected_identities
    ):
        raise ValidationError(
            "path optimization requires exactly 64 unique method/schedule identities"
        )

    required_keys = {
        "row_id",
        "scenario_family",
        "scenario",
        "method",
        "schedule_id",
        "initial_hard_loss",
        "final_hard_loss",
        "held_out_loss",
        "success",
        "accepted",
        "source_commit",
        "device",
        "losses",
        "parameters",
        "gradients",
        "final_parameter_error",
        "estimator_seeds",
        "target_seed",
        "held_out_seed",
        "deterministic_final_recheck",
        "final_recheck_seed",
        "final_recheck_protocol",
        "held_out_render_evaluations",
        "held_out_render_work",
        "objective_extension",
        "numerical_gauge_policy",
        "box_constraints",
    }
    expected_render_work = {
        "cached_target_renders": 1,
        "shared_initial_candidate_renders": 1,
        "optimization_candidate_renders": 256,
        "deterministic_final_recheck_renders": 4,
        "total_candidate_renders": 261,
        "total_renders": 262,
    }
    expected_box = {"lower": [-0.8, -0.5, -0.9], "upper": [0.8, 0.5, 0.3]}
    for row in rows:
        row_id = row.get("row_id")
        schedule = row.get("schedule_id")
        if set(row) != required_keys:
            raise ValidationError(
                f"path optimization row {row_id!r} does not have the exact producer schema"
            )
        if (
            not isinstance(row_id, str)
            or not row_id
            or type(schedule) is not int
            or row.get("scenario_family") != "path_tracer"
            or row.get("scenario") != "analytic_five_sphere"
            or row.get("source_commit") != source_commit
            or row.get("device") != device
        ):
            raise ValidationError(
                f"path optimization row {row_id!r} is not scenario/source/device bound"
            )
        initial = _path_optimization_float(
            row.get("initial_hard_loss"), name=f"{row_id}.initial_hard_loss"
        )
        final = _path_optimization_float(
            row.get("final_hard_loss"), name=f"{row_id}.final_hard_loss"
        )
        held_out = _path_optimization_float(
            row.get("held_out_loss"), name=f"{row_id}.held_out_loss"
        )
        recheck = _path_optimization_float(
            row.get("deterministic_final_recheck"),
            name=f"{row_id}.deterministic_final_recheck",
        )
        losses = _path_optimization_history(
            row.get("losses"), name=f"{row_id}.losses", length=65
        )
        parameters = _path_optimization_history(
            row.get("parameters"),
            name=f"{row_id}.parameters",
            length=65,
            dimension=3,
        )
        _path_optimization_history(
            row.get("gradients"),
            name=f"{row_id}.gradients",
            length=64,
            dimension=3,
        )
        parameter_error = _path_optimization_float(
            row.get("final_parameter_error"),
            name=f"{row_id}.final_parameter_error",
        )
        if parameter_error < 0.0:
            raise ValidationError(
                f"path optimization {row_id}.final_parameter_error must be nonnegative"
            )
        expected_seeds = list(range(12000 + 100 * schedule, 12064 + 100 * schedule))
        if row.get("estimator_seeds") != expected_seeds:
            raise ValidationError(
                f"path optimization {row_id} estimator seed schedule is not canonical"
            )
        if any(
            parameter[component] < expected_box["lower"][component]
            or parameter[component] > expected_box["upper"][component]
            for parameter in parameters
            for component in range(3)
        ):
            raise ValidationError(
                f"path optimization {row_id} parameter history leaves the box"
            )
        accepted = (
            final == held_out == recheck
            and row.get("target_seed") == 2000 + schedule
            and row.get("held_out_seed") == 3000 + schedule
            and row.get("final_recheck_seed") == 3000 + schedule
            and row.get("final_recheck_protocol")
            == "same_seed_integrity_check_against_cached_immutable_target"
        )
        success = accepted and final < initial
        if (
            row.get("accepted") is not accepted
            or accepted is not True
            or row.get("success") is not success
            or losses[0] != initial
            or losses[-1] != final
        ):
            raise ValidationError(
                f"path optimization {row_id} acceptance, success, or loss history does not recompute"
            )
        if (
            row.get("held_out_render_evaluations") != 66
            or not _json_type_strict_equal(
                row.get("held_out_render_work"), expected_render_work
            )
            or row.get("objective_extension")
            != "componentwise_box_clip_before_geometry"
            or row.get("numerical_gauge_policy")
            != "exact_selected_arm_derivative_almost_everywhere"
            or not _json_type_strict_equal(row.get("box_constraints"), expected_box)
        ):
            raise ValidationError(
                f"path optimization {row_id} render work, box, or gauge protocol is not exact"
            )

    declared = [
        record
        for record in summary.get("optimization_summaries", [])
        if isinstance(record, dict) and record.get("scenario") == "path_tracer"
    ]
    if len(declared) != len(methods):
        raise ValidationError(
            "path optimization_summaries must cover exactly four recovery methods"
        )
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        actual = [record for record in declared if record.get("method") == method]
        if len(actual) != 1:
            raise ValidationError(
                "path optimization_summaries method identity is missing or duplicated"
            )
        final_values = [row["final_hard_loss"] for row in selected]
        held_values = [row["held_out_loss"] for row in selected]
        expected = {
            "scenario": "path_tracer",
            "method": method,
            "final_hard_loss_mean": sum(final_values) / len(final_values),
            "final_hard_loss_ci_low": _linear_quantile(final_values, 0.025),
            "final_hard_loss_ci_high": _linear_quantile(final_values, 0.975),
            "success_rate": sum(row["success"] for row in selected) / len(selected),
            "held_out_loss_mean": sum(held_values) / len(held_values),
            "source_row_ids": [row["row_id"] for row in selected],
        }
        if not _json_type_strict_equal(actual[0], expected):
            raise ValidationError(
                f"path optimization_summaries aggregate for {method} is not exact"
            )


_CONTACT_METHODS = ("soft_ad", "straight_through_ad", "residual_control_variate")
_CONTACT_EVENT_LABELS = (
    "pair_01",
    "pair_02",
    "pair_12",
    "floor_0",
    "floor_1",
    "floor_2",
    "ramp_0",
    "ramp_1",
    "ramp_2",
)


def _contact_event_sequences(value: Any, *, name: str) -> list[list[list[str]]]:
    if not isinstance(value, list) or len(value) != 180:
        raise ValidationError(f"contact {name} event evidence must contain 180 steps")
    normalized = []
    for step in value:
        if not isinstance(step, list) or len(step) != 4:
            raise ValidationError(
                f"contact {name} event evidence must contain four sweeps per step"
            )
        normalized_step = []
        for sweep in step:
            if (
                not isinstance(sweep, list)
                or any(
                    not isinstance(event, str) or event not in _CONTACT_EVENT_LABELS
                    for event in sweep
                )
                or len(sweep) != len(set(sweep))
            ):
                raise ValidationError(f"contact {name} event multiplicity is invalid")
            positions = [_CONTACT_EVENT_LABELS.index(event) for event in sweep]
            if positions != sorted(positions):
                raise ValidationError(
                    f"contact {name} events violate canonical solver order"
                )
            normalized_step.append(list(sweep))
        normalized.append(normalized_step)
    return normalized


def _contact_event_counts(sequences: list[list[list[str]]]) -> dict[str, int]:
    return {
        label: sum(
            event == label for step in sequences for sweep in step for event in sweep
        )
        for label in _CONTACT_EVENT_LABELS
    }


def _validate_contact_physical_validity(value: Any) -> bool:
    expected_keys = {
        "valid",
        "checks",
        "pair_contact_counts",
        "static_contact_counts",
        "pair_correction_counts",
        "static_correction_counts",
        "positive_impulse_event_types_by_step_and_sweep",
        "correction_event_types_by_step_and_sweep",
        "canonical_solver_event_order",
        "event_sequence_semantics",
        "stick_contacts",
        "slide_contacts",
        "zero_limit_slide_contacts",
        "body_steps",
        "contact_sweeps",
        "pair_solver_calls",
        "static_solver_calls",
        "minimum_positive_normal_impulse",
        "positive_impulse_threshold",
        "max_penetration",
        "max_contact_energy_gain",
        "max_pair_momentum_error",
        "max_pair_angular_momentum_error",
        "thresholds",
    }
    check_keys = {
        "pair_01_contact",
        "pair_12_contact",
        "ordered_pair_01_then_pair_12",
        "floor_contact",
        "ramp_contact",
        "stick_mode",
        "slide_mode",
        "no_zero_limit_slide",
        "penetration_bounded",
        "contact_energy_bounded",
        "pair_momentum_conserved",
        "pair_angular_momentum_conserved",
        "meaningful_positive_impulse",
    }
    thresholds = {
        "max_penetration": 0.03,
        "max_contact_energy_gain": 0.02,
        "max_pair_momentum_error": 1.0e-5,
        "max_pair_angular_momentum_error": 1.0e-5,
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValidationError(
            "contact optimization physical-validity schema is not canonical"
        )
    checks = value.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != check_keys
        or any(type(flag) is not bool for flag in checks.values())
        or not _json_type_strict_equal(value.get("thresholds"), thresholds)
    ):
        raise ValidationError(
            "contact optimization physical thresholds or checks are invalid"
        )
    count_schemas = {
        "pair_contact_counts": {"0-1", "0-2", "1-2"},
        "pair_correction_counts": {"0-1", "0-2", "1-2"},
        "static_contact_counts": {"floor", "ramp"},
        "static_correction_counts": {"floor", "ramp"},
    }
    for field, keys in count_schemas.items():
        counts = value.get(field)
        if (
            not isinstance(counts, dict)
            or set(counts) != keys
            or any(type(count) is not int or count < 0 for count in counts.values())
        ):
            raise ValidationError(f"contact optimization {field} is invalid")
    for field in ("stick_contacts", "slide_contacts", "zero_limit_slide_contacts"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValidationError(
                f"contact optimization {field} must be a nonnegative integer"
            )
    for field in (
        "minimum_positive_normal_impulse",
        "positive_impulse_threshold",
        "max_penetration",
        "max_contact_energy_gain",
        "max_pair_momentum_error",
        "max_pair_angular_momentum_error",
    ):
        if (
            type(value.get(field)) is not float
            or not math.isfinite(value[field])
            or value[field] < 0.0
        ):
            raise ValidationError(
                f"contact optimization {field} must be a finite nonnegative float"
            )

    positive = _contact_event_sequences(
        value.get("positive_impulse_event_types_by_step_and_sweep"),
        name="positive-impulse",
    )
    corrections = _contact_event_sequences(
        value.get("correction_event_types_by_step_and_sweep"),
        name="correction",
    )
    if any(
        not set(positive_sweep) <= set(correction_sweep)
        for positive_step, correction_step in zip(positive, corrections, strict=True)
        for positive_sweep, correction_sweep in zip(
            positive_step, correction_step, strict=True
        )
    ):
        raise ValidationError(
            "contact positive events must be a per-sweep subset of correction events"
        )
    positive_counts = _contact_event_counts(positive)
    correction_counts = _contact_event_counts(corrections)
    expected_counts = {
        "pair_contact_counts": {
            "0-1": positive_counts["pair_01"],
            "0-2": positive_counts["pair_02"],
            "1-2": positive_counts["pair_12"],
        },
        "static_contact_counts": {
            "floor": sum(positive_counts[f"floor_{body}"] for body in range(3)),
            "ramp": sum(positive_counts[f"ramp_{body}"] for body in range(3)),
        },
        "pair_correction_counts": {
            "0-1": correction_counts["pair_01"],
            "0-2": correction_counts["pair_02"],
            "1-2": correction_counts["pair_12"],
        },
        "static_correction_counts": {
            "floor": sum(correction_counts[f"floor_{body}"] for body in range(3)),
            "ramp": sum(correction_counts[f"ramp_{body}"] for body in range(3)),
        },
    }
    if any(
        not _json_type_strict_equal(value.get(field), counts)
        for field, counts in expected_counts.items()
    ):
        raise ValidationError(
            "contact event multiplicity disagrees with aggregate counts"
        )
    ordered = [event for step in positive for sweep in step for event in sweep]
    first = [index for index, event in enumerate(ordered) if event == "pair_01"]
    second = [index for index, event in enumerate(ordered) if event == "pair_12"]
    ordered_transfer = bool(first and second and first[-1] < second[0])
    total_positive = sum(positive_counts.values())
    if (
        value.get("canonical_solver_event_order") != list(_CONTACT_EVENT_LABELS)
        or value.get("event_sequence_semantics")
        != "ordered_per_step_per_sweep_solver_call_events_with_multiplicity"
        or value.get("body_steps") != 540
        or value.get("contact_sweeps") != 720
        or value.get("pair_solver_calls") != 2160
        or value.get("static_solver_calls") != 4320
        or value.get("positive_impulse_threshold") != 1.0e-8
        or value["stick_contacts"] + value["slide_contacts"] != total_positive
    ):
        raise ValidationError(
            "contact optimization event work or semantics are inconsistent"
        )
    expected_checks = {
        "pair_01_contact": value["pair_contact_counts"]["0-1"] > 0,
        "pair_12_contact": value["pair_contact_counts"]["1-2"] > 0,
        "ordered_pair_01_then_pair_12": ordered_transfer,
        "floor_contact": value["static_contact_counts"]["floor"] > 0,
        "ramp_contact": value["static_contact_counts"]["ramp"] > 0,
        "stick_mode": value["stick_contacts"] > 0,
        "slide_mode": value["slide_contacts"] > 0,
        "no_zero_limit_slide": value["zero_limit_slide_contacts"] == 0,
        "penetration_bounded": value["max_penetration"] < thresholds["max_penetration"],
        "contact_energy_bounded": value["max_contact_energy_gain"]
        < thresholds["max_contact_energy_gain"],
        "pair_momentum_conserved": value["max_pair_momentum_error"]
        < thresholds["max_pair_momentum_error"],
        "pair_angular_momentum_conserved": (
            value["max_pair_angular_momentum_error"]
            < thresholds["max_pair_angular_momentum_error"]
        ),
        "meaningful_positive_impulse": (
            total_positive > 0
            and value["minimum_positive_normal_impulse"]
            >= value["positive_impulse_threshold"]
        ),
    }
    valid = all(expected_checks.values())
    if (
        not _json_type_strict_equal(checks, expected_checks)
        or value.get("valid") is not valid
    ):
        raise ValidationError(
            "contact optimization event order or physical checks do not recompute"
        )
    return valid


def _linear_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _validate_contact_optimization_contract(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    source_commit: str,
    device: str,
) -> None:
    expected_identities = {
        (method, schedule) for schedule in range(16) for method in _CONTACT_METHODS
    }
    identities = [(row.get("method"), row.get("schedule_id")) for row in rows]
    if (
        len(rows) != 48
        or len(set(identities)) != 48
        or set(identities) != expected_identities
    ):
        raise ValidationError(
            "contact optimization method/schedule identities are incomplete or duplicated"
        )
    expected_hard_work = {
        "initial_forward_executions": 1,
        "line_search_batches": 64,
        "line_search_candidates_per_batch": 6,
        "line_search_forward_executions": 384,
        "final_forward_executions": 1,
        "recheck_forward_executions": 1,
        "total_forward_executions": 387,
    }
    schedule_domains: dict[int, tuple[int, ...]] = {}
    successful_methods = []
    accepted_count = 0
    for row in rows:
        method = row["method"]
        schedule = row["schedule_id"]
        if row.get("source_commit") != source_commit or row.get("device") != device:
            raise ValidationError("contact optimization row is not source/device bound")
        domain = tuple(range(6000 + 64 * schedule, 6000 + 64 * schedule + 64))
        if row.get("realized_outer_seeds") != list(domain):
            raise ValidationError("contact optimization seed domain is not canonical")
        previous = schedule_domains.setdefault(schedule, domain)
        if previous != domain:
            raise ValidationError(
                "contact optimization methods disagree on schedule seed domains"
            )
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
        work = row.get("gradient_work")
        if not isinstance(work, list) or len(work) != 64:
            raise ValidationError(
                "contact optimization gradient work must cover 64 sequential steps"
            )
        for step, record in enumerate(work):
            expected = {
                "step": step,
                "outer_seed": domain[step],
                "inner_seed": None,
                **method_work,
            }
            if not _json_type_strict_equal(record, expected):
                raise ValidationError("contact optimization gradient work is not exact")
        if not _json_type_strict_equal(
            row.get("hard_evaluation_work"), expected_hard_work
        ):
            raise ValidationError(
                "contact optimization hard evaluation work is not exact"
            )
        physical_valid = _validate_contact_physical_validity(
            row.get("final_physical_validity")
        )
        initial = row.get("initial_hard_loss")
        final = row.get("final_hard_loss")
        held_out = row.get("held_out_loss")
        finite_losses = all(
            type(value) is float and math.isfinite(value)
            for value in (initial, final, held_out)
        )
        accepted = finite_losses and final == held_out and physical_valid
        success = accepted and final < initial
        if row.get("accepted") is not accepted or row.get("success") is not success:
            raise ValidationError(
                "contact optimization accepted/success booleans disagree with recomputed losses"
            )
        accepted_count += int(accepted)
        if success:
            successful_methods.append(method)
    flattened_domains = [
        seed for schedule in range(16) for seed in schedule_domains[schedule]
    ]
    if len(flattened_domains) != len(set(flattened_domains)):
        raise ValidationError("contact optimization schedule seed domains overlap")
    if accepted_count != len(rows) or not successful_methods:
        raise ValidationError(
            "contact optimization requires all rows accepted and at least one successful method"
        )

    expected_validity = {
        "scenario": "contact_3d_optimization",
        "accepted": True,
        "metrics": {
            "row_count": len(rows),
            "accepted_count": accepted_count,
            "success_count": len(successful_methods),
            "successful_methods": successful_methods,
        },
        "source_row_ids": [row["row_id"] for row in rows],
    }
    validity_rows = [
        record
        for record in summary.get("scenario_validity", [])
        if isinstance(record, dict)
        and record.get("scenario") == "contact_3d_optimization"
    ]
    if len(validity_rows) != 1 or not _json_type_strict_equal(
        validity_rows[0], expected_validity
    ):
        raise ValidationError(
            "scenario_validity contact optimization summary is not exact"
        )

    summary_rows = [
        record
        for record in summary.get("optimization_summaries", [])
        if isinstance(record, dict) and record.get("scenario") == "contact_3d"
    ]
    if len(summary_rows) != len(_CONTACT_METHODS):
        raise ValidationError(
            "optimization_summaries contact optimization coverage is incomplete"
        )
    for method in _CONTACT_METHODS:
        selected = [row for row in rows if row["method"] == method]
        declared = [record for record in summary_rows if record.get("method") == method]
        if len(declared) != 1:
            raise ValidationError(
                "contact optimization summary method identity is duplicated"
            )
        final_values = [row["final_hard_loss"] for row in selected]
        held_values = [row["held_out_loss"] for row in selected]
        expected = {
            "scenario": "contact_3d",
            "method": method,
            "final_hard_loss_mean": sum(final_values) / len(final_values),
            "final_hard_loss_ci_low": _linear_quantile(final_values, 0.025),
            "final_hard_loss_ci_high": _linear_quantile(final_values, 0.975),
            "success_rate": sum(row["success"] for row in selected) / len(selected),
            "held_out_loss_mean": sum(held_values) / len(held_values),
            "source_row_ids": [row["row_id"] for row in selected],
        }
        actual = declared[0]
        if set(actual) != set(expected):
            raise ValidationError("contact optimization summary schema is not exact")
        for field in expected:
            if field in {
                "final_hard_loss_mean",
                "final_hard_loss_ci_low",
                "final_hard_loss_ci_high",
                "success_rate",
                "held_out_loss_mean",
            }:
                if type(actual[field]) is not float or not math.isclose(
                    actual[field], expected[field], rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise ValidationError(
                        "contact optimization summary numeric aggregate is inconsistent"
                    )
            elif not _json_type_strict_equal(actual[field], expected[field]):
                raise ValidationError(
                    "contact optimization summary lineage or labels are inconsistent"
                )


def _assert_no_local_path(value: Any, *, location: str) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        is_web_url = (
            stripped.startswith(("https://", "http://")) and " " not in stripped
        )
        if (
            stripped.casefold().startswith("file:")
            or stripped.startswith(("/", "\\\\"))
            or _WINDOWS_PATH_RE.search(value)
            or (not is_web_url and _EMBEDDED_POSIX_PATH_RE.search(value))
        ):
            raise ValidationError(f"local filesystem path leaked at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_no_local_path(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_local_path(child, location=f"{location}[{index}]")


def validate_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    if not (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
    ):
        raise ValidationError(
            "secure publication validation requires descriptor-relative O_NOFOLLOW support"
        )
    root_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(os.fspath(root), root_flags)
    except OSError as error:
        raise ValidationError(
            "report root is missing, not a directory, or is a symbolic link"
        ) from error
    try:
        manifest_bytes = _read_artifact_at(root_descriptor, "data/manifest.json")
        manifest = _load_json_bytes_finite(
            manifest_bytes, location="data/manifest.json"
        )
    except ValidationError as error:
        os.close(root_descriptor)
        if "data/manifest.json" not in str(error):
            raise ValidationError(f"data/manifest.json is invalid: {error}") from error
        raise
    if not isinstance(manifest, dict):
        os.close(root_descriptor)
        raise ValidationError("manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        os.close(root_descriptor)
        raise ValidationError("manifest schema_version must be 1")
    if manifest.get("tier") != "report":
        os.close(root_descriptor)
        raise ValidationError("manifest tier must be 'report'")
    files = manifest.get("files")
    if not isinstance(files, dict):
        os.close(root_descriptor)
        raise ValidationError("manifest files must be an object")
    actual_names = frozenset(files)
    if actual_names != EXPECTED_FILES:
        os.close(root_descriptor)
        missing = sorted(EXPECTED_FILES - actual_names)
        extra = sorted(actual_names - EXPECTED_FILES)
        raise ValidationError(
            f"manifest must declare exactly 31 artifacts; missing={missing}, undeclared={extra}"
        )

    loaded: dict[str, Any] = {}
    try:
        for relative, descriptor in files.items():
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "sha256",
                "bytes",
            }:
                raise ValidationError(f"invalid manifest descriptor for {relative}")
            digest = descriptor.get("sha256")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValidationError(f"invalid SHA-256 for {relative}")
            byte_count = _nonnegative_int(
                descriptor.get("bytes"), name=f"{relative}.bytes"
            )
            data = _read_artifact_at(root_descriptor, relative)
            if len(data) != byte_count:
                raise ValidationError(f"byte count mismatch for {relative}")
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValidationError(f"SHA-256 mismatch for {relative}")
            if relative.endswith(".json"):
                loaded[relative] = _load_json_bytes_finite(data, location=relative)
    finally:
        os.close(root_descriptor)

    on_disk: set[str] = set()
    for parent in (root / "data", root / "assets"):
        for directory, directory_names, file_names in os.walk(
            parent, followlinks=False
        ):
            directory_path = Path(directory)
            for name in directory_names:
                if (directory_path / name).is_symlink():
                    raise ValidationError(
                        f"symlink is not allowed in publication tree: {(directory_path / name).relative_to(root)}"
                    )
            for name in file_names:
                path = directory_path / name
                if path.is_symlink():
                    raise ValidationError(
                        f"symlink is not allowed in publication tree: {path.relative_to(root)}"
                    )
                relative = path.relative_to(root).as_posix()
                if relative != "data/manifest.json":
                    on_disk.add(relative)
    undeclared = sorted(on_disk - EXPECTED_FILES)
    if undeclared:
        raise ValidationError(f"undeclared publication artifacts: {undeclared}")
    missing_on_disk = sorted(EXPECTED_FILES - on_disk)
    if missing_on_disk:
        raise ValidationError(f"manifest artifacts missing on disk: {missing_on_disk}")
    _assert_no_local_path(manifest, location="manifest")
    for relative, payload in loaded.items():
        _assert_no_local_path(payload, location=relative)
    return manifest, loaded


def _dataset_rows(payload: Any, *, relative: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"{relative} must use schema_version 1")
    if not isinstance(payload.get("dataset"), str) or not payload["dataset"]:
        raise ValidationError(f"{relative} has no dataset name")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValidationError(f"{relative}.rows must be an array of objects")
    return rows


def _validate_variance_fields(
    row: dict[str, Any], *, row_id: str, dimension: int
) -> None:
    for stem in ("contribution_variance", "gradient_variance"):
        flag = row.get(f"{stem}_available")
        value = row.get(stem)
        if not isinstance(flag, bool):
            raise ValidationError(f"{row_id}.{stem}_available must be boolean")
        if flag:
            vector = _finite_vector(value, name=f"{row_id}.{stem}", nonnegative=True)
            if len(vector) != dimension:
                raise ValidationError(f"{row_id}.{stem} dimension mismatch")
        elif value is not None:
            raise ValidationError(f"{row_id}.{stem} must be null when unavailable")
    low_raw = row.get("ci_low")
    high_raw = row.get("ci_high")
    if low_raw is None or high_raw is None:
        if low_raw is not None or high_raw is not None:
            raise ValidationError(
                f"{row_id} confidence bounds must both be null or arrays"
            )
        if row["gradient_variance_available"] and row["independent_contributions"] >= 2:
            raise ValidationError(
                f"{row_id} confidence interval is required when mean variance is available"
            )
        return
    low = _finite_vector(low_raw, name=f"{row_id}.ci_low")
    high = _finite_vector(high_raw, name=f"{row_id}.ci_high")
    if not row["gradient_variance_available"]:
        raise ValidationError(
            f"{row_id} confidence interval requires gradient variance"
        )
    if len(low) != len(high) or any(first > second for first, second in zip(low, high)):
        raise ValidationError(f"{row_id} confidence interval has invalid bounds")
    if len(low) != dimension:
        raise ValidationError(f"{row_id} confidence interval dimension mismatch")


def _validate_method_row(row: dict[str, Any]) -> None:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValidationError("method row has no row_id")
    method = row.get("method")
    if method not in REQUIRED_METHOD_IDS:
        raise ValidationError(f"{row_id} has unknown method {method!r}")
    samples = _nonnegative_int(row.get("samples"), name=f"{row_id}.samples")
    if samples == 0:
        raise ValidationError(f"{row_id}.samples must be positive")
    counts = {}
    for field in (
        "forward_executions",
        "backward_executions",
        "independent_contributions",
        "parameter_perturbations",
    ):
        counts[field] = _nonnegative_int(row.get(field), name=f"{row_id}.{field}")
    _finite_number(row.get("wall_time"), name=f"{row_id}.wall_time", nonnegative=True)
    gradient = _finite_vector(row.get("gradient"), name=f"{row_id}.gradient")
    if not gradient:
        raise ValidationError(f"{row_id}.gradient must be nonempty")
    dimension = len(gradient)
    antithetic = row.get("antithetic")
    if not isinstance(antithetic, bool):
        raise ValidationError(f"{row_id}.antithetic must be boolean")
    if antithetic and samples % 2:
        raise ValidationError(f"{row_id} antithetic samples must be positive and even")
    _validate_variance_fields(row, row_id=row_id, dimension=dimension)

    hard = row.get("hard_forward_executions")
    soft = row.get("soft_forward_executions")
    independent = samples // 2 if antithetic else samples
    expected_underlying = {
        "crisp_ad": "pathwise",
        "crisp_fd": "finite_difference",
        "smoothed_pathwise": "pathwise",
        "score": "score_function",
        "smoothed_crn_fd": "finite_difference",
        "soft_ad": "pathwise",
        "straight_through_ad": "pathwise",
        "residual_control_variate": "control_variate",
    }[method]
    if row.get("underlying_method") not in {None, expected_underlying}:
        raise ValidationError(f"{row_id}.underlying_method disagrees with method")
    if expected_underlying == "score_function":
        expected = (independent, samples, samples, 0, None, None)
    elif expected_underlying == "pathwise":
        expected = (independent, samples, samples, samples, None, None)
    elif expected_underlying == "finite_difference":
        expected = (1, 2 * dimension, samples * (1 + 2 * dimension), 0, None, None)
    else:
        expected = (independent, samples, 2 * samples, samples, samples, samples)

    if expected_underlying == "control_variate":
        hard_count = _nonnegative_int(hard, name=f"{row_id}.hard_forward_executions")
        soft_count = _nonnegative_int(soft, name=f"{row_id}.soft_forward_executions")
    else:
        hard_count = hard
        soft_count = soft
    if expected_underlying != "control_variate" and (
        hard is not None or soft is not None
    ):
        raise ValidationError(
            f"{row_id} hard/soft forward fields are reserved for residual_control_variate"
        )
    actual = (
        counts["independent_contributions"],
        counts["parameter_perturbations"],
        counts["forward_executions"],
        counts["backward_executions"],
        hard_count,
        soft_count,
    )
    if actual != expected:
        raise ValidationError(
            f"{row_id} estimator accounting mismatch: expected {expected}, got {actual}"
        )


def _reference_interval(
    values: list[list[float]], record: Any, *, name: str
) -> tuple[list[float], list[float], list[float]]:
    if not isinstance(record, dict):
        raise ValidationError(f"{name} interval record must be an object")
    dimension = len(values[0])
    replicates = len(values)
    means: list[float] = []
    variances: list[float] = []
    mean_variances: list[float] = []
    for index in range(dimension):
        components = [Fraction.from_float(row[index]) for row in values]
        exact_total = sum(components, Fraction())
        exact_mean = exact_total / replicates
        exact_variance = sum(
            ((value - exact_mean) ** 2 for value in components),
            Fraction(),
        ) / (replicates - 1)
        exact_mean_variance = exact_variance / replicates
        converted: list[float] = []
        for statistic, exact in (
            ("mean", exact_mean),
            ("variance", exact_variance),
            ("variance of the mean", exact_mean_variance),
        ):
            try:
                binary64 = float(exact)
            except OverflowError as error:
                raise ValidationError(
                    f"{name} has unrepresentable {statistic}"
                ) from error
            if not math.isfinite(binary64) or (exact != 0 and binary64 == 0.0):
                raise ValidationError(f"{name} has unrepresentable {statistic}")
            converted.append(binary64)
        means.append(converted[0])
        variances.append(converted[1])
        mean_variances.append(converted[2])
    raw_half_widths = [3.182446305284263 * math.sqrt(value) for value in mean_variances]
    ci_low = [mean - width for mean, width in zip(means, raw_half_widths)]
    ci_high = [mean + width for mean, width in zip(means, raw_half_widths)]
    half_widths = [0.5 * (high - low) for low, high in zip(ci_low, ci_high)]
    expected_vectors = {
        "mean": means,
        "variance": variances,
        "mean_variance": mean_variances,
        "half_width": half_widths,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    parsed: dict[str, list[float]] = {}
    for field, expected in expected_vectors.items():
        parsed[field] = _finite_vector(record.get(field), name=f"{name}.{field}")
        if len(parsed[field]) != dimension or any(
            not math.isclose(actual, target, rel_tol=1.0e-10, abs_tol=1.0e-12)
            for actual, target in zip(parsed[field], expected)
        ):
            raise ValidationError(f"{name}.{field} disagrees with stored replicates")
    if record.get("replicates") != replicates:
        raise ValidationError(f"{name}.replicates is inconsistent")
    if record.get("degrees_of_freedom") != replicates - 1:
        raise ValidationError(f"{name}.degrees_of_freedom is inconsistent")
    if _finite_number(record.get("confidence"), name=f"{name}.confidence") != 0.95:
        raise ValidationError(f"{name}.confidence must be 0.95")
    return parsed["mean"], parsed["ci_low"], parsed["ci_high"]


def _validate_reference_rows(
    rows: list[dict[str, Any]],
    required_cells: Any,
    *,
    config: dict[str, Any],
    source_commit: str,
    device: str,
) -> None:
    if not isinstance(required_cells, list) or any(
        not isinstance(cell, str) or not cell for cell in required_cells
    ):
        raise ValidationError(
            "reference_required_cells must be a list of nonempty strings"
        )
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cell = row.get("cell_id")
        if isinstance(cell, str):
            by_cell.setdefault(cell, []).append(row)
    for cell in required_cells:
        accepted = [
            row
            for row in by_cell.get(cell, ())
            if isinstance(row.get("accepted"), dict)
            and row["accepted"].get("references") is True
        ]
        if not accepted:
            raise ValidationError(f"required cell {cell!r} has no accepted reference")
        if len(accepted) != 1:
            raise ValidationError(
                f"required cell {cell!r} must have exactly one accepted reference"
            )
        for row in accepted:
            if row.get("tier") != "report" or row.get("device") != device:
                raise ValidationError(
                    f"accepted reference {cell!r} has wrong tier/device"
                )
            if row.get("source_commit") != source_commit:
                raise ValidationError(
                    f"accepted reference {cell!r} has wrong source_commit"
                )
            parameters = _finite_vector(
                row.get("parameters"), name=f"{cell}.parameters"
            )
            if not parameters:
                raise ValidationError(
                    f"accepted reference {cell!r} has empty parameters"
                )
            dimension = len(parameters)
            sigma = _finite_vector(row.get("sigma"), name=f"{cell}.sigma")
            h = _finite_vector(row.get("h"), name=f"{cell}.h")
            h_half = _finite_vector(row.get("h_half"), name=f"{cell}.h_half")
            if any(len(vector) != dimension for vector in (sigma, h, h_half)) or any(
                step <= 0.0 for step in sigma + h + h_half
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} has invalid vector dimensions"
                )
            if any(first != 2.0 * second for first, second in zip(h, h_half)):
                raise ValidationError(
                    f"accepted reference {cell!r} must store exact h and h/2"
                )

            matrices: dict[str, list[list[float]]] = {}
            for field in ("g_h", "g_h2", "score"):
                raw_matrix = row.get(field)
                if not isinstance(raw_matrix, list) or len(raw_matrix) != 4:
                    raise ValidationError(
                        f"accepted reference {cell!r} needs four {field} replicates"
                    )
                matrix = [
                    _finite_vector(vector, name=f"{cell}.{field}[{index}]")
                    for index, vector in enumerate(raw_matrix)
                ]
                if any(len(vector) != dimension for vector in matrix):
                    raise ValidationError(
                        f"accepted reference {cell!r} {field} dimension mismatch"
                    )
                matrices[field] = matrix
            paired = [
                [first - second for first, second in zip(g_h, g_h2)]
                for g_h, g_h2 in zip(matrices["g_h"], matrices["g_h2"])
            ]
            if any(not math.isfinite(value) for vector in paired for value in vector):
                raise ValidationError(
                    f"accepted reference {cell!r} paired differences are non-finite"
                )
            intervals = row.get("intervals")
            if not isinstance(intervals, dict):
                raise ValidationError(
                    f"accepted reference {cell!r} intervals must be an object"
                )
            h_mean, h_low, h_high = _reference_interval(
                matrices["g_h"],
                intervals.get("g_h"),
                name=f"{cell}.intervals.g_h",
            )
            h2_mean, h2_low, h2_high = _reference_interval(
                matrices["g_h2"],
                intervals.get("g_h2"),
                name=f"{cell}.intervals.g_h2",
            )
            _score_mean, score_low, score_high = _reference_interval(
                matrices["score"],
                intervals.get("score"),
                name=f"{cell}.intervals.score",
            )
            _paired_mean, paired_low, paired_high = _reference_interval(
                paired,
                intervals.get("paired_h_minus_h2"),
                name=f"{cell}.intervals.paired_h_minus_h2",
            )
            reference_gradient = _finite_vector(
                row.get("reference_gradient"), name=f"{cell}.reference_gradient"
            )
            if reference_gradient != h2_mean:
                raise ValidationError(
                    f"accepted reference {cell!r} reference_gradient must equal g_h2 mean"
                )

            overlap = [
                max(first, third) <= min(second, fourth)
                for first, second, third, fourth in zip(
                    h2_low, h2_high, score_low, score_high
                )
            ]
            paired_consistency = [
                low <= 0.0 <= high for low, high in zip(paired_low, paired_high)
            ]
            marginal_consistency = [
                abs(first - second)
                <= (first_high - first_low + second_high - second_low) / 2.0
                for first, second, first_low, first_high, second_low, second_high in zip(
                    h_mean, h2_mean, h_low, h_high, h2_low, h2_high
                )
            ]
            if not all(overlap) or not all(paired_consistency):
                raise ValidationError(
                    f"accepted reference {cell!r} fails overlap or paired step consistency"
                )
            diagnostics = row.get("diagnostics")
            expected_diagnostics = {
                "overlap_components": overlap,
                "marginal_step_components": marginal_consistency,
                "paired_step_components": paired_consistency,
            }
            if not isinstance(diagnostics, dict) or any(
                diagnostics.get(name) != expected
                for name, expected in expected_diagnostics.items()
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} diagnostics disagree with replicates"
                )
            accepted_flags = row["accepted"]
            required_flags = {
                "references": True,
                "fd_score_overlap": all(overlap),
                "step_consistency": all(paired_consistency),
                "marginal_step_consistency": all(marginal_consistency),
                "paired_step_consistency": all(paired_consistency),
                "replicate_count_sufficient": True,
                "smoke_only": False,
            }
            if any(
                accepted_flags.get(name) is not value
                for name, value in required_flags.items()
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} acceptance flags are inconsistent"
                )

            counts = row.get("counts")
            if not isinstance(counts, dict):
                raise ValidationError(
                    f"accepted reference {cell!r} counts must be an object"
                )
            sample_field = (
                "contact_reference_samples"
                if cell.startswith("contact_3d:")
                else "path_reference_samples"
            )
            samples = _nonnegative_int(
                config.get(sample_field), name=f"config.{sample_field}"
            )
            replicates = _nonnegative_int(
                config.get("reference_seed_sets"), name="config.reference_seed_sets"
            )
            if replicates != 4 or samples == 0 or samples % 2:
                raise ValidationError(
                    "report reference configuration must use four replicates and even samples"
                )
            expected_single = 4 * dimension * samples * replicates
            expected_counts = {
                "samples": samples,
                "replicates": replicates,
                "h_forward_executions": expected_single,
                "h2_forward_executions": expected_single,
                "five_point_forward_executions": 2 * expected_single,
                "score_forward_executions": samples * replicates,
                "forward_executions": 2 * expected_single + samples * replicates,
            }
            if any(
                counts.get(name) != expected
                for name, expected in expected_counts.items()
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} forward execution counts are inconsistent"
                )

            seeds = row.get("seeds")
            seed_names = (
                "five_point_outer",
                "five_point_inner",
                "score_outer",
                "score_inner",
            )
            if not isinstance(seeds, dict) or set(seeds) != set(seed_names):
                raise ValidationError(
                    f"accepted reference {cell!r} seed streams are invalid"
                )
            domain_size = (2**31) // 4
            flattened: list[int] = []
            offsets: list[list[int]] = []
            for domain, name in enumerate(seed_names):
                stream = seeds[name]
                if not isinstance(stream, list) or len(stream) != replicates:
                    raise ValidationError(
                        f"accepted reference {cell!r} seed stream {name} has wrong length"
                    )
                parsed = [
                    _nonnegative_int(seed, name=f"{cell}.seeds.{name}")
                    for seed in stream
                ]
                if any(
                    seed > 2**31 - 1 or seed // domain_size != domain for seed in parsed
                ):
                    raise ValidationError(
                        f"accepted reference {cell!r} seed stream {name} has wrong domain"
                    )
                flattened.extend(parsed)
                offsets.append([seed % domain_size for seed in parsed])
            if len(flattened) != len(set(flattened)) or any(
                offset != offsets[0] for offset in offsets[1:]
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} seed streams are not CRN-aligned and disjoint"
                )
            if offsets[0] != [
                (offsets[0][0] + index) % domain_size for index in range(replicates)
            ]:
                raise ValidationError(
                    f"accepted reference {cell!r} seed offsets are not consecutive"
                )


def _validate_producer_command(command: Any, *, device: str) -> None:
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValidationError("source command must be a nonempty argv array")
    options: list[str] | None = None
    for prefix in _COMMAND_PREFIXES:
        if tuple(command[: len(prefix)]) == prefix:
            options = command[len(prefix) :]
            break
    if options is None:
        raise ValidationError(
            "source command must be a canonical program-smoothing producer invocation"
        )
    values: dict[str, str] = {}
    headless = False
    index = 0
    while index < len(options):
        option = options[index]
        if option == "--headless":
            if headless:
                raise ValidationError("canonical producer command repeats --headless")
            headless = True
            index += 1
            continue
        if option not in {"--device", "--tier", "--output-dir"} or option in values:
            raise ValidationError(
                "source command must be a canonical report producer invocation"
            )
        if index + 1 >= len(options):
            raise ValidationError(f"source command option {option} has no value")
        values[option] = options[index + 1]
        index += 2
    if values != {
        "--device": device,
        "--tier": "report",
        "--output-dir": "$OUTPUT_DIR",
    }:
        raise ValidationError(
            "source command must select the manifest device, report tier, and $OUTPUT_DIR"
        )


def _validate_seed_tree(value: Any, *, name: str) -> None:
    if isinstance(value, bool):
        raise ValidationError(f"{name} seed metadata must contain nonnegative integers")
    if isinstance(value, int):
        if value < 0 or value > 2**31 - 1:
            raise ValidationError(f"{name} seed is outside Warp's signed 32-bit domain")
        return
    if isinstance(value, list):
        if not value:
            raise ValidationError(f"{name} seed sequence must be nonempty")
        for index, child in enumerate(value):
            _validate_seed_tree(child, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        if not value or any(not isinstance(key, str) or not key for key in value):
            raise ValidationError(
                f"{name} seed mapping must be nonempty with string keys"
            )
        for key, child in value.items():
            _validate_seed_tree(child, name=f"{name}.{key}")
        return
    raise ValidationError(f"{name} seed metadata has unsupported type")


def _validate_source(
    manifest: dict[str, Any], method_rows: list[dict[str, Any]]
) -> tuple[str, str]:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValidationError("manifest source must be an object")
    if source.get("dirty") is not False:
        raise ValidationError("published Warp source must be clean")
    commit = source.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValidationError("source commit must be a full lowercase Git object ID")
    device = source.get("device")
    if not isinstance(device, str) or not device:
        raise ValidationError("source device metadata is required")
    _validate_producer_command(source.get("command"), device=device)
    for field in ("python", "warp", "platform", "compiler", "cpu_model"):
        if not isinstance(source.get(field), str) or not source[field]:
            raise ValidationError(f"source {field} metadata is required")
    if _nonnegative_int(source.get("cpu_threads"), name="source.cpu_threads") == 0:
        raise ValidationError("source.cpu_threads must be positive")
    if not isinstance(source.get("seeds"), dict) or not source["seeds"]:
        raise ValidationError("source.seeds must be a nonempty object")
    _validate_seed_tree(source["seeds"], name="source.seeds")
    expected_seeds = {
        "analytic_estimator_outer": list(range(100, 132)),
        "collision_estimator_outer": list(range(300, 332)),
        "contact_3d": {
            "estimator_outer": list(range(5000, 5032)),
            "optimization_outer": [6000 + 64 * schedule for schedule in range(16)],
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
    }
    if not _json_type_strict_equal(source["seeds"], expected_seeds):
        raise ValidationError(
            "source seed protocol must exactly match the canonical report schedule"
        )
    for row in method_rows:
        row_commit = row.get("source_commit")
        if row_commit != commit:
            raise ValidationError(
                f"{row['row_id']} source_commit disagrees with manifest"
            )
        if row.get("device") != device:
            raise ValidationError(f"{row['row_id']} device disagrees with manifest")
    return commit, device


def _validate_performance_row(row: dict[str, Any]) -> None:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValidationError("performance row has no row_id")
    if row.get("method") not in REQUIRED_METHOD_IDS:
        raise ValidationError(f"{row_id} has an unknown performance method")
    if not isinstance(row.get("scenario"), str) or not row["scenario"]:
        raise ValidationError(f"{row_id}.scenario must be a nonempty string")
    if not isinstance(row.get("device"), str) or not row["device"]:
        raise ValidationError(f"{row_id}.device must be a nonempty string")
    if (
        not isinstance(row.get("source_commit"), str)
        or _COMMIT_RE.fullmatch(row["source_commit"]) is None
    ):
        raise ValidationError(f"{row_id}.source_commit must be a full object ID")
    for field in ("forward_executions", "backward_executions"):
        _nonnegative_int(row.get(field), name=f"{row_id}.{field}")
    _finite_number(row.get("wall_time"), name=f"{row_id}.wall_time", nonnegative=True)
    for field in ("cold_compile_time", "warm_median", "warm_iqr"):
        _finite_number(row.get(field), name=f"{row_id}.{field}", nonnegative=True)
    repeats = _nonnegative_int(row.get("warm_repeats"), name=f"{row_id}.warm_repeats")
    if repeats < 5:
        raise ValidationError(f"{row_id}.warm_repeats must be at least five")
    for field in (
        "tracemalloc_peak",
        "rss_delta",
        "warp_allocation_peak",
        "device_free_memory_delta",
    ):
        availability = row.get(f"{field}_available")
        if not isinstance(availability, bool):
            raise ValidationError(f"{row_id}.{field}_available must be boolean")
        value = row.get(field)
        if availability:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(
                    f"{row_id}.{field} must be an integer when available"
                )
            if field.endswith("peak") and value < 0:
                raise ValidationError(f"{row_id}.{field} must be nonnegative")
        elif value is not None:
            raise ValidationError(f"{row_id}.{field} must be null when unavailable")


def _validate_protocol(
    manifest: dict[str, Any], loaded: dict[str, Any]
) -> tuple[int, str]:
    raw_rows: dict[str, list[dict[str, Any]]] = {
        name: _dataset_rows(loaded[f"data/raw/{name}"], relative=f"data/raw/{name}")
        for name in RAW_NAMES
    }
    plot_rows: dict[str, list[dict[str, Any]]] = {
        name: _dataset_rows(
            loaded[f"data/plot_data/{name}"], relative=f"data/plot_data/{name}"
        )
        for name in PLOT_NAMES
    }
    summary = loaded["data/summary.json"]
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("data/summary.json must use schema_version 1")
    config = _validate_report_config_and_decision(manifest)
    _validate_installed_runtime(summary.get("runtime"))

    all_raw_rows = [row for rows in raw_rows.values() for row in rows]
    row_ids: dict[str, dict[str, Any]] = {}
    for row in all_raw_rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValidationError("raw row_id must be a nonempty string")
        if row_id in row_ids:
            raise ValidationError(f"duplicate raw row_id {row_id!r}")
        row_ids[row_id] = row

    method_rows = [
        row
        for row in all_raw_rows
        if row.get("method") in REQUIRED_METHOD_IDS and "gradient" in row
    ]
    gradient_methods = {row["method"] for row in method_rows}
    missing_methods = sorted(set(REQUIRED_METHOD_IDS) - gradient_methods)
    if missing_methods:
        raise ValidationError(
            f"gradient records omit required methods: {missing_methods}"
        )
    for row in raw_rows["performance.json"]:
        _validate_performance_row(row)
    performance_methods = {row.get("method") for row in raw_rows["performance.json"]}
    missing_performance = sorted(set(REQUIRED_METHOD_IDS) - performance_methods)
    if missing_performance:
        raise ValidationError(
            f"performance records omit required methods: {missing_performance}"
        )
    families = {row.get("scenario_family") for row in all_raw_rows}
    missing_families = sorted(set(REQUIRED_SCENARIO_FAMILIES) - families)
    if missing_families:
        raise ValidationError(f"raw records omit scenario families: {missing_families}")

    for row in method_rows:
        _validate_method_row(row)
    commit, source_device = _validate_source(manifest, method_rows)
    _validate_path_numerical_gauge_contract(
        method_rows,
        raw_rows["path_tracer_gradients.json"],
        source_commit=commit,
        device=source_device,
    )
    _validate_path_gradient_contract(raw_rows["path_tracer_gradients.json"])
    _validate_path_optimization_contract(
        raw_rows["path_tracer_optimization.json"],
        summary,
        source_commit=commit,
        device=source_device,
    )
    _validate_contact_optimization_contract(
        raw_rows["contact_3d_optimization.json"],
        summary,
        source_commit=commit,
        device=source_device,
    )
    for row in raw_rows["performance.json"]:
        if row.get("source_commit") != commit:
            raise ValidationError(
                f"{row['row_id']}.source_commit disagrees with manifest"
            )
        if row.get("device") != source_device:
            raise ValidationError(f"{row['row_id']}.device disagrees with manifest")

    applicability = manifest.get("applicability")
    if not isinstance(applicability, list) or not applicability:
        raise ValidationError("manifest applicability matrix must be nonempty")
    required_cell_keys = {
        "scenario",
        "method",
        "samples",
        "stochastic",
        "antithetic",
        "report_required",
        "estimator_only",
        "applicable",
        "transformable",
        "optimization_enabled",
        "reference_required",
        "reason",
    }
    estimator_seeds = _nonnegative_int(
        config.get("estimator_seeds"), name="config.estimator_seeds"
    )
    if estimator_seeds != 32:
        raise ValidationError("report config.estimator_seeds must be 32")
    smoothing_samples_raw = config.get("smoothing_samples")
    if (
        not isinstance(smoothing_samples_raw, list)
        or not smoothing_samples_raw
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 2
            for value in smoothing_samples_raw
        )
        or len(smoothing_samples_raw) != len(set(smoothing_samples_raw))
    ):
        raise ValidationError(
            "config.smoothing_samples must contain unique positive even values"
        )
    smoothing_samples = set(smoothing_samples_raw)
    stochastic_groups: dict[tuple[str, str], set[int]] = {}
    applicability_identities: set[tuple[str, str, int]] = set()
    matrix_pairs: set[tuple[str, str]] = set()
    for index, cell in enumerate(applicability):
        if not isinstance(cell, dict) or set(cell) != required_cell_keys:
            raise ValidationError(f"applicability[{index}] has the wrong schema")
        for field in (
            "stochastic",
            "antithetic",
            "report_required",
            "estimator_only",
            "applicable",
            "transformable",
            "optimization_enabled",
            "reference_required",
        ):
            if not isinstance(cell[field], bool):
                raise ValidationError(f"applicability[{index}].{field} must be boolean")
        method = cell["method"]
        if method not in REQUIRED_METHOD_IDS:
            raise ValidationError(f"applicability[{index}] has unknown method")
        samples = _nonnegative_int(
            cell["samples"], name=f"applicability[{index}].samples"
        )
        if samples == 0:
            raise ValidationError(f"applicability[{index}].samples must be positive")
        identity = (cell["scenario"], method, samples)
        if identity in applicability_identities:
            raise ValidationError(f"duplicate applicability cell {identity}")
        applicability_identities.add(identity)
        matrix_pairs.add((cell["scenario"], method))
        if cell["estimator_only"] and cell["transformable"]:
            raise ValidationError(
                f"applicability[{index}] estimator-only cell cannot be transformable"
            )
        if not cell["applicable"]:
            if not isinstance(cell["reason"], str) or not cell["reason"]:
                raise ValidationError(f"inapplicable cell {index} needs a reason")
            continue
        if not cell["report_required"]:
            continue
        matching = [
            row
            for row in method_rows
            if row.get("scenario_family") == cell["scenario"]
            and row.get("method") == method
            and row.get("samples") == samples
        ]
        if not matching:
            raise ValidationError(
                f"report-required applicability cell has no row: {cell['scenario']}/{method}/N={samples}"
            )
        if any(row.get("antithetic") is not cell["antithetic"] for row in matching):
            raise ValidationError(
                f"{cell['scenario']}/{method}/N={samples} antithetic declaration disagrees with rows"
            )
        if cell["stochastic"]:
            stochastic_groups.setdefault((cell["scenario"], method), set()).add(samples)
            raw_seeds = [row.get("outer_seed") for row in matching]
            if any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in raw_seeds
            ):
                raise ValidationError(
                    f"{cell['scenario']}/{method}/N={samples} outer_seed values must be nonnegative integers"
                )
            seeds = set(raw_seeds)
            if len(matching) != estimator_seeds or len(seeds) != estimator_seeds:
                raise ValidationError(
                    f"{cell['scenario']}/{method}/N={samples} requires exactly 32 rows with distinct outer seeds; "
                    f"found rows={len(matching)}, seeds={len(seeds)}"
                )
            if cell["scenario"] == "path_tracer":
                expected_pairs = {
                    (10_000 + index, 1_000 + index) for index in range(estimator_seeds)
                }
                observed_pairs = {
                    (row.get("outer_seed"), row.get("inner_seed")) for row in matching
                }
                if observed_pairs != expected_pairs:
                    raise ValidationError(
                        f"path_tracer/{method}/N={samples} does not use the canonical outer/inner seed pairs"
                    )
                starts = {row.get("start_id") for row in matching}
                targets = {row.get("target") for row in matching}
                normalized_configs = []
                inner_digests = set()
                for row in matching:
                    config_row = row.get("config")
                    if not isinstance(config_row, dict):
                        raise ValidationError(
                            f"path_tracer/{method}/N={samples} requires configuration metadata"
                        )
                    digest = config_row.get("inner_random_digest")
                    if (
                        not isinstance(digest, str)
                        or _SHA256_RE.fullmatch(digest) is None
                    ):
                        raise ValidationError(
                            f"path_tracer/{method}/N={samples} has an invalid inner-random digest"
                        )
                    inner_digests.add(digest)
                    normalized_configs.append(
                        {
                            key: value
                            for key, value in config_row.items()
                            if key != "inner_random_digest"
                        }
                    )
                if (
                    len(starts) != 1
                    or len(targets) != 1
                    or len(inner_digests) != estimator_seeds
                    or any(
                        config_row != normalized_configs[0]
                        for config_row in normalized_configs[1:]
                    )
                ):
                    raise ValidationError(
                        f"path_tracer/{method}/N={samples} has inconsistent strata or per-seed randomness"
                    )
                continue
            strata: dict[tuple[Any, ...], list[int]] = {}
            for row, seed in zip(matching, raw_seeds):
                key = (
                    row.get("start_id"),
                    row.get("target"),
                    row.get("inner_seed"),
                    json.dumps(
                        row.get("config"), sort_keys=True, separators=(",", ":")
                    ),
                )
                strata.setdefault(key, []).append(seed)
            for key, stratum_seeds in strata.items():
                if (
                    len(stratum_seeds) != estimator_seeds
                    or len(set(stratum_seeds)) != estimator_seeds
                ):
                    raise ValidationError(
                        f"{cell['scenario']}/{method}/N={samples} duplicate or incomplete stochastic rows in stratum {key}"
                    )
    expected_matrix_pairs = {
        (scenario, method)
        for scenario in REQUIRED_SCENARIO_FAMILIES
        for method in REQUIRED_METHOD_IDS
    }
    if matrix_pairs != expected_matrix_pairs:
        missing = sorted(expected_matrix_pairs - matrix_pairs)
        extra = sorted(matrix_pairs - expected_matrix_pairs)
        raise ValidationError(
            f"applicability matrix is incomplete; missing={missing}, extra={extra}"
        )
    for (scenario, method), declared_samples in stochastic_groups.items():
        if declared_samples != smoothing_samples:
            missing = sorted(smoothing_samples - declared_samples)
            extra = sorted(declared_samples - smoothing_samples)
            raise ValidationError(
                f"{scenario}/{method} stochastic sample cells disagree with config; missing N={missing}, extra={extra}"
            )

    required_reference_cells = manifest.get("reference_required_cells")
    if (
        not isinstance(required_reference_cells, list)
        or set(required_reference_cells) != set(EXPECTED_REFERENCE_CELLS)
        or len(required_reference_cells) != len(EXPECTED_REFERENCE_CELLS)
    ):
        raise ValidationError(
            "reference_required_cells must declare each canonical publication reference exactly once"
        )
    _validate_reference_rows(
        raw_rows["references.json"],
        required_reference_cells,
        config=config,
        source_commit=commit,
        device=source_device,
    )

    rejected_ids = {
        row["row_id"]
        for row in all_raw_rows
        if (
            row.get("accepted") is False
            or (
                isinstance(row.get("accepted"), dict)
                and row["accepted"].get("references") is False
            )
        )
        and "row_id" in row
    }
    referenced_ids: set[str] = set()
    for name, rows in plot_rows.items():
        for index, row in enumerate(rows):
            source_ids = row.get("source_row_ids")
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or any(not isinstance(item, str) for item in source_ids)
            ):
                raise ValidationError(
                    f"data/plot_data/{name}.rows[{index}] needs nonempty source_row_ids"
                )
            if len(source_ids) != len(set(source_ids)):
                raise ValidationError(
                    f"data/plot_data/{name}.rows[{index}] contains duplicate source_row_ids"
                )
            for source_id in source_ids:
                if source_id not in row_ids:
                    raise ValidationError(
                        f"plot lineage references unknown raw row {source_id!r}"
                    )
                if source_id in rejected_ids and name != "validity.json":
                    raise ValidationError(
                        f"plot lineage references rejected diagnostic row {source_id!r}"
                    )
                referenced_ids.add(source_id)
    summary_schemas = {
        "method_summaries": {
            "scenario",
            "target",
            "method",
            "samples",
            "mean_gradient",
            "relative_error",
            "cosine_similarity",
            "sign_agreement",
            "empirical_bias",
            "empirical_variance",
            "mean_squared_error",
            "source_row_ids",
        },
        "optimization_summaries": {
            "scenario",
            "method",
            "final_hard_loss_mean",
            "final_hard_loss_ci_low",
            "final_hard_loss_ci_high",
            "success_rate",
            "held_out_loss_mean",
            "source_row_ids",
        },
        "performance_summaries": {
            "scenario",
            "method",
            "cold_compile_time",
            "warm_median",
            "warm_iqr",
            "forward_executions",
            "backward_executions",
            "tracemalloc_peak",
            "rss_delta",
            "warp_allocation_peak",
            "device_free_memory_delta",
            "source_row_ids",
        },
        "scenario_validity": {
            "scenario",
            "accepted",
            "metrics",
            "source_row_ids",
        },
    }
    for collection in (
        "method_summaries",
        "optimization_summaries",
        "performance_summaries",
        "scenario_validity",
    ):
        values = summary.get(collection)
        if not isinstance(values, list) or not values:
            raise ValidationError(f"summary.{collection} must be a nonempty array")
        for index, row in enumerate(values):
            if not isinstance(row, dict):
                raise ValidationError(
                    f"summary.{collection}[{index}] must be an object"
                )
            if set(row) != summary_schemas[collection]:
                raise ValidationError(
                    f"summary.{collection}[{index}] has the wrong schema"
                )
            if not isinstance(row.get("scenario"), str) or not row["scenario"]:
                raise ValidationError(
                    f"summary.{collection}[{index}].scenario must be nonempty"
                )
            if collection != "scenario_validity":
                if row.get("method") not in REQUIRED_METHOD_IDS:
                    raise ValidationError(
                        f"summary.{collection}[{index}] has unknown method"
                    )
            if collection == "method_summaries":
                if not isinstance(row.get("target"), str) or not row["target"]:
                    raise ValidationError(
                        f"summary.{collection}[{index}].target must be nonempty"
                    )
                if (
                    _nonnegative_int(
                        row.get("samples"),
                        name=f"summary.{collection}[{index}].samples",
                    )
                    == 0
                ):
                    raise ValidationError("method summary samples must be positive")
                vectors = [
                    _finite_vector(
                        row.get(field),
                        name=f"summary.{collection}[{index}].{field}",
                        nonnegative=field
                        in {"empirical_variance", "mean_squared_error"},
                    )
                    for field in (
                        "mean_gradient",
                        "empirical_bias",
                        "empirical_variance",
                        "mean_squared_error",
                    )
                ]
                if not vectors[0] or any(
                    len(vector) != len(vectors[0]) for vector in vectors
                ):
                    raise ValidationError("method summary gradient dimensions disagree")
                _finite_number(
                    row.get("relative_error"),
                    name="method summary relative_error",
                    nonnegative=True,
                )
                cosine = _finite_number(
                    row.get("cosine_similarity"),
                    name="method summary cosine_similarity",
                )
                sign = _finite_number(
                    row.get("sign_agreement"), name="method summary sign_agreement"
                )
                if not -1.0 <= cosine <= 1.0 or not 0.0 <= sign <= 1.0:
                    raise ValidationError(
                        "method summary similarity metrics are out of range"
                    )
            elif collection == "optimization_summaries":
                low = _finite_number(
                    row.get("final_hard_loss_ci_low"),
                    name="optimization CI low",
                    nonnegative=True,
                )
                mean = _finite_number(
                    row.get("final_hard_loss_mean"),
                    name="optimization mean",
                    nonnegative=True,
                )
                high = _finite_number(
                    row.get("final_hard_loss_ci_high"),
                    name="optimization CI high",
                    nonnegative=True,
                )
                _finite_number(
                    row.get("held_out_loss_mean"),
                    name="optimization held-out loss",
                    nonnegative=True,
                )
                success = _finite_number(
                    row.get("success_rate"), name="optimization success rate"
                )
                if not low <= mean <= high or not 0.0 <= success <= 1.0:
                    raise ValidationError(
                        "optimization summary intervals or success rate are invalid"
                    )
            elif collection == "performance_summaries":
                for field in ("cold_compile_time", "warm_median", "warm_iqr"):
                    _finite_number(
                        row.get(field),
                        name=f"performance summary {field}",
                        nonnegative=True,
                    )
                for field in ("forward_executions", "backward_executions"):
                    _nonnegative_int(
                        row.get(field), name=f"performance summary {field}"
                    )
                for field in (
                    "tracemalloc_peak",
                    "rss_delta",
                    "warp_allocation_peak",
                    "device_free_memory_delta",
                ):
                    if row.get(field) is not None and (
                        isinstance(row[field], bool) or not isinstance(row[field], int)
                    ):
                        raise ValidationError(
                            f"performance summary {field} must be integer or null"
                        )
            elif not isinstance(row.get("accepted"), bool) or not isinstance(
                row.get("metrics"), dict
            ):
                raise ValidationError("scenario validity fields are invalid")
            sources = row.get("source_row_ids")
            if not isinstance(sources, list) or not sources:
                raise ValidationError(
                    f"summary.{collection}[{index}] has orphaned aggregate data"
                )
            if len(sources) != len(set(sources)):
                raise ValidationError(
                    f"summary.{collection}[{index}] contains duplicate source_row_ids"
                )
            for source_id in sources:
                if source_id not in row_ids:
                    raise ValidationError(
                        f"summary lineage references unknown raw row {source_id!r}"
                    )
                if source_id in rejected_ids and collection != "scenario_validity":
                    raise ValidationError(
                        f"summary lineage references rejected diagnostic row {source_id!r}"
                    )
                referenced_ids.add(source_id)

    headline_metrics = summary.get("headline_metrics")
    if not isinstance(headline_metrics, dict) or not headline_metrics:
        raise ValidationError("summary.headline_metrics must be a nonempty object")
    for name, record in headline_metrics.items():
        if not isinstance(record, dict):
            raise ValidationError(
                f"headline metric {name!r} must be an aggregate object"
            )
        source_ids = record.get("source_row_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValidationError(f"headline metric {name!r} has no source_row_ids")
        if len(source_ids) != len(set(source_ids)):
            raise ValidationError(
                f"headline metric {name!r} contains duplicate source_row_ids"
            )
        for source_id in source_ids:
            if source_id not in row_ids:
                raise ValidationError(
                    f"headline metric {name!r} references unknown raw row {source_id!r}"
                )
            if source_id in rejected_ids:
                raise ValidationError(
                    f"headline metric {name!r} references rejected raw row {source_id!r}"
                )
            referenced_ids.add(source_id)

    orphaned_ids = sorted(set(row_ids) - referenced_ids)
    if orphaned_ids:
        raise ValidationError(
            f"raw rows have no plot or aggregate lineage: {orphaned_ids}"
        )

    opaque_rows = raw_rows["opaque_mesh.json"]
    if not opaque_rows:
        raise ValidationError("opaque mesh diagnostics are missing")
    opaque_protocol_rows = [
        row
        for row in opaque_rows
        if row.get("method") not in REQUIRED_METHOD_IDS
        and ("transform_status" in row or "transformable" in row)
    ]
    if (
        len(opaque_protocol_rows) != 1
        or opaque_protocol_rows[0].get("transform_status") != "estimator_only"
        or opaque_protocol_rows[0].get("transformable") is not False
    ):
        raise ValidationError(
            "opaque native/BVH branches require a one-to-one estimator-only "
            "protocol diagnostic"
        )
    for cell in applicability:
        if cell["scenario"] == "opaque_mesh" and cell["transformable"] is not False:
            raise ValidationError(
                "opaque mesh applicability cannot claim transformed native branches"
            )

    literature = summary.get("literature_urls")
    if not isinstance(literature, list):
        raise ValidationError("summary literature_urls must be an array")
    for url in REQUIRED_LITERATURE_URLS:
        if url not in literature:
            raise ValidationError(f"required literature URL is missing: {url}")

    accepted = manifest.get("accepted")
    if not isinstance(accepted, dict) or any(
        accepted.get(key) is not True
        for key in ("analytic", "references", "scenario_validity")
    ):
        raise ValidationError("manifest acceptance gates must all pass")
    total_rows = len(all_raw_rows) + sum(len(rows) for rows in plot_rows.values())
    return total_rows, commit


def _validated_html_attributes(
    tag: str, attrs: list[tuple[str, str | None]]
) -> dict[str, str | None]:
    validated: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for name, value in attrs:
        normalized = name.casefold()
        if normalized in seen:
            raise ValidationError(f"duplicate HTML attribute {name!r} on <{tag}>")
        seen.add(normalized)
        validated.append((normalized, value))
    return dict(validated)


class _ReportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: dict[str, tuple[str, bool]] = {}
        self.duplicate_ids: set[str] = set()
        self.method_text: dict[str, list[str]] = {}
        self.section_text: dict[str, list[str]] = {}
        self.references: list[tuple[str, str, str]] = []
        self.active_content_violations: list[str] = []
        self.source_commit: str | None = None
        self.reproducibility_text: list[str] = []
        self.text_fragments: list[str] = []
        self.attribute_values: list[str] = []
        self._element_stack: list[tuple[str, str | None, bool, bool, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = _validated_html_attributes(tag, attrs)
        element_id = attributes.get("id")
        style = (attributes.get("style") or "").replace(" ", "").casefold()
        hidden = (
            any(element[3] for element in self._element_stack)
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
            or tag in {"head", "script", "style", "template"}
        )
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids[element_id] = (tag, hidden)
            if tag == "section":
                self.section_text.setdefault(element_id, [])
        forbidden_elements = {
            "script",
            "iframe",
            "object",
            "embed",
            "base",
            "link",
            "form",
            "input",
            "button",
            "textarea",
            "select",
            "video",
            "audio",
        }
        if tag in forbidden_elements:
            self.active_content_violations.append(f"forbidden element <{tag}>")
        for attribute, value in attrs:
            lowered = attribute.casefold()
            if lowered.startswith("on") or lowered in {
                "srcset",
                "formaction",
                "xlink:href",
                "action",
                "poster",
                "ping",
                "cite",
                "background",
                "manifest",
                "data",
                "codebase",
                "archive",
                "profile",
                "usemap",
                "srcdoc",
            }:
                self.active_content_violations.append(
                    f"forbidden active attribute {attribute}"
                )
            if lowered == "style" and value:
                if _css_has_external_or_active_content(value):
                    self.active_content_violations.append(
                        "unsafe inline style attribute"
                    )
        if tag == "meta" and "http-equiv" in attributes:
            self.active_content_violations.append("active meta http-equiv control")
        method = attributes.get("data-method")
        if method is not None:
            self.method_text.setdefault(method, [])
        starts_repro = element_id == "reproducibility"
        self._element_stack.append(
            (
                tag,
                method,
                starts_repro,
                hidden,
                element_id if tag == "section" else None,
            )
        )
        self.attribute_values.extend(
            value for value in attributes.values() if isinstance(value, str)
        )
        for attribute in ("src", "href"):
            reference = attributes.get(attribute)
            if reference:
                self.references.append((tag, attribute, reference))
        if tag == "body" and attributes.get("data-source-commit"):
            self.source_commit = attributes["data-source-commit"]

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        match = next(
            (
                index
                for index in range(len(self._element_stack) - 1, -1, -1)
                if self._element_stack[index][0] == tag
            ),
            None,
        )
        if match is None:
            return
        del self._element_stack[match:]

    def handle_data(self, data: str) -> None:
        self.text_fragments.append(data)
        if any(element[0] == "style" for element in self._element_stack):
            if _css_has_external_or_active_content(data):
                self.active_content_violations.append(
                    "unsafe resource or active expression in style element"
                )
        if any(element[3] for element in self._element_stack):
            return
        for _tag, method, _starts_repro, _hidden, _section_id in self._element_stack:
            if method is not None:
                self.method_text.setdefault(method, []).append(data)
        for _tag, _method, _starts_repro, _hidden, section_id in self._element_stack:
            if section_id is not None:
                self.section_text.setdefault(section_id, []).append(data)
        if any(element[2] for element in self._element_stack):
            self.reproducibility_text.append(data)


def _validate_html(root: Path, commit: str) -> None:
    root_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(os.fspath(root), root_flags)
        try:
            source_bytes = _read_artifact_at(root_descriptor, "index.html")
        finally:
            os.close(root_descriptor)
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeError, ValidationError) as error:
        raise ValidationError(f"cannot read index.html: {error}") from error
    lower_source = source.casefold()
    for legacy in LEGACY_FILES:
        if legacy.casefold() in lower_source:
            raise ValidationError(f"index.html references legacy artifact {legacy}")
    forbidden_claims = (
        "opaque native branches were transformed",
        "opaque mesh branches were transformed",
        "bvh branches were transformed",
    )
    if any(claim in lower_source for claim in forbidden_claims):
        raise ValidationError(
            "report overclaims transformation of opaque native/BVH branches"
        )

    parser = _ReportHTMLParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as error:
        raise ValidationError(f"invalid report HTML: {error}") from error
    for index, text in enumerate(parser.text_fragments):
        _assert_no_local_path(text, location=f"index.html text[{index}]")
    for index, value in enumerate(parser.attribute_values):
        _assert_no_local_path(value, location=f"index.html attribute[{index}]")
    if parser.duplicate_ids:
        raise ValidationError(
            f"report contains duplicate IDs: {sorted(parser.duplicate_ids)}"
        )
    if parser.active_content_violations:
        raise ValidationError(
            f"report contains unsafe active HTML: {parser.active_content_violations}"
        )
    missing_sections = [
        section
        for section in REQUIRED_SECTION_IDS
        if parser.ids.get(section) != ("section", False)
    ]
    if missing_sections:
        raise ValidationError(
            f"report is missing required sections: {missing_sections}"
        )
    empty_sections = [
        section
        for section in REQUIRED_SECTION_IDS
        if not " ".join(parser.section_text.get(section, ())).strip()
    ]
    if empty_sections:
        raise ValidationError(
            f"report contains empty required sections: {empty_sections}"
        )
    for method in REQUIRED_METHOD_IDS:
        visible = " ".join(parser.method_text.get(method, ())).strip()
        if not visible:
            raise ValidationError(
                f"report is missing a visible label for method {method}"
            )
    if parser.source_commit != commit:
        raise ValidationError("index.html source commit does not match manifest")
    if commit not in " ".join(parser.reproducibility_text):
        raise ValidationError("reproducibility section omits the exact Warp commit")

    for tag, attribute, raw_reference in parser.references:
        split = urlsplit(raw_reference)
        if split.scheme:
            if split.scheme in {"http", "https"} and attribute == "href" and tag == "a":
                continue
            raise ValidationError(f"unsafe HTML reference scheme in {raw_reference!r}")
        if split.netloc:
            raise ValidationError(
                f"protocol-relative HTML reference is unsafe: {raw_reference!r}"
            )
        if not split.path:
            continue
        if Path(split.path).name in LEGACY_FILES:
            raise ValidationError(
                f"report references legacy artifact {Path(split.path).name}"
            )
        pure = PurePosixPath(split.path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in split.path:
            raise ValidationError(f"unsafe relative asset reference {raw_reference!r}")
        relative = pure.as_posix()
        if relative not in EXPECTED_FILES:
            raise ValidationError(
                f"local HTML reference is not declared by the manifest: {raw_reference}"
            )
        if tag not in {"a", "img", "source"}:
            raise ValidationError(
                f"unexpected local reference element {tag!r} for {raw_reference!r}"
            )


def validate_bundle(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], int, str]:
    """Validate and return one descriptor-relative evidence-bundle snapshot."""

    root = Path(root)
    if not root.is_dir():
        raise ValidationError(f"report root does not exist: {root}")
    manifest, loaded = validate_manifest(root)
    rows, commit = _validate_protocol(manifest, loaded)
    return manifest, loaded, rows, commit


def validate_publication(root: Path) -> dict[str, int]:
    root = Path(root)
    _manifest, _loaded, rows, commit = validate_bundle(root)
    _validate_html(root.resolve(), commit)
    return {
        "files": len(EXPECTED_FILES),
        "rows": rows,
        "assets": len(FIGURE_NAMES) + len(IMAGE_NAMES),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the published Warp smoothing report"
    )
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).resolve().parent
    )
    arguments = parser.parse_args(argv)
    try:
        result = validate_publication(arguments.root)
    except ValidationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: files={result['files']} rows={result['rows']} assets={result['assets']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
